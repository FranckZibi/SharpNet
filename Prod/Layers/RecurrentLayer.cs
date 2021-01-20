using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    /// <summary>
    /// input 'x' shape :
    ///     (batchSize, timeSteps, inputSize )
    /// output 'y' shape :
    ///     (batchSize, timeSteps, K*hiddenSize)  if _returnSequences == true
    ///     (batchSize, K*hiddenSize)             if _returnSequences == false
    /// with
    ///      K == 2     if IsBidirectional == true 
    ///      K == 1     if IsBidirectional == false
    /// </summary>
    public class RecurrentLayer : Layer
    {
        #region Fields
        #region GPU Parameters 
        private readonly cudnnRNNDescriptor_t _cudnnRNNDescriptor_t;
        /// <summary>
        /// needed only for training, null for inference
        /// </summary>
        private Tensor _reserveSpace;
        /// <summary>
        /// size of the temporary buffer used internally by the cuDNN API
        /// </summary>
        private size_t _workSpaceBufferSizeInBytes;
        #endregion

        #region trainable parameters
        [NotNull] private Tensor _weightsAndBiases;
        #endregion
        #region gradients
        [NotNull] private Tensor _weightsAndBiasesGradients;
        #endregion
        private readonly bool _returnSequences;
        private readonly RNNDescriptor _rnnDescriptor;
        /// <summary>
        /// when returnSequences is false and we are in training mode (not inference) :
        ///     we'll keep the original 'y' computed by the 'cudnnRNNForward' method to use it during backward propagation
        /// will be null for inference or when returnSequences == true
        /// </summary>
        private Tensor _yIfReturnSequences;
        [NotNull] private readonly Optimizer _optimizer;
        #endregion

        private cudnnRNNMode_t CellMode => _rnnDescriptor.cellMode;
        private cudnnRNNBiasMode_t BiasMode => _rnnDescriptor.biasMode;
        private int InputSize => _rnnDescriptor.inputSize;      // == features
        private int HiddenSize => _rnnDescriptor.hiddenSize;    // == units
        private bool IsBidirectional => _rnnDescriptor.dirMode == cudnnDirectionMode_t.CUDNN_BIDIRECTIONAL;

        #region constructor
        public RecurrentLayer(int hiddenSize, cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode, bool returnSequences, bool isBidirectional, bool trainable, Network network, string layerName) : base(network, layerName)
        {
            int inputSize = PrevLayer.OutputShape(1)[2]; // inputSize == Features
            uint auxFlags = 0;
            auxFlags |= CudnnWrapper.CUDNN_RNN_PADDED_IO_ENABLED;
            _rnnDescriptor = new RNNDescriptor(
                cudnnRNNAlgo_t.CUDNN_RNN_ALGO_STANDARD,
                cellMode,
                biasMode,
                isBidirectional ? cudnnDirectionMode_t.CUDNN_BIDIRECTIONAL : cudnnDirectionMode_t.CUDNN_UNIDIRECTIONAL,
                cudnnRNNInputMode_t.CUDNN_LINEAR_INPUT,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
                cudnnMathType_t.CUDNN_DEFAULT_MATH,
                inputSize,  /* = features */
                hiddenSize, /* hiddenSize */
                hiddenSize, /* projSize*/
                1,          /* numLayers */
                0.0,
                auxFlags);

            _returnSequences = returnSequences;
            Trainable = trainable;

            _weightsAndBiases = GetFloatTensor(new []{ _rnnDescriptor.WeightAndBiasCount});
            _weightsAndBiasesGradients = GetFloatTensor(_weightsAndBiases.Shape);
            _optimizer = GetOptimizer(_weightsAndBiases.Shape, null);

            // ReSharper disable once VirtualMemberCallInConstructor
            ResetParameters(false);

            if (network.UseGPU)
            {
                _cudnnRNNDescriptor_t = Network.GpuWrapper.RNNDesc(_rnnDescriptor, Network.GetRandomNumberGeneratorStatesBuffer());
                CudnnWrapper.cudnnGetRNNWeightSpaceSize(Network.GpuWrapper.CudnnHandle, _cudnnRNNDescriptor_t, out var weightSpaceSize);
                if (_weightsAndBiases.Count != (int)(weightSpaceSize / 4))
                {
                    throw new ArgumentException("expecting 4*" + _weightsAndBiases.Count + " weights and bias but got " + weightSpaceSize);
                }
            }
        }
        public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
        {
            Weights_ax.GlorotUniform(Rand);
            Weights_aa.Orthogonal(Rand);
            Bias?.ZeroMemory();
        }
        #endregion

        public override string LayerType()
        {
            if (IsBidirectional)
            {
                return "Bidirectional";
            }
            switch (CellMode)
            {
                case cudnnRNNMode_t.CUDNN_RNN_RELU:
                case cudnnRNNMode_t.CUDNN_RNN_TANH:
                    return "SimpleRNN";
                case cudnnRNNMode_t.CUDNN_GRU:
                    return "GRU";
                case cudnnRNNMode_t.CUDNN_LSTM:
                    return "LSTM";
                default:
                    throw new NotSupportedException("invalid cellMode: "+CellMode);
            }
        }

        protected override string ComputeLayerName()
        {
            var result = LayerType().ToLowerInvariant().Replace("simplernn", "simple_rnn");
            int countBefore = IsBidirectional
                ? Layers.Count(l => l.LayerIndex < LayerIndex && l is RecurrentLayer layer && layer.IsBidirectional)
                : Layers.Count(l => l.LayerIndex < LayerIndex && l is RecurrentLayer layer && layer.CellMode == CellMode);
            if (countBefore != 0)
            {
                result += "_" + countBefore;
            }
            return result;
        }

        #region forward and backward propagation
        /// <summary>
        /// </summary>
        /// <param name="allX"></param>
        /// input 'x' shape:
        ///     (batchSize, timeSteps, inputSize)
        /// <param name="y">
        /// output 'y' shape:
        ///     (batchSize, timeSteps, K*hiddenSize)      if _returnSequences == true
        ///     (batchSize, K*hiddenSize)                 if _returnSequences == false
        /// with
        ///      K == 2     if IsBidirectional == true 
        ///      K == 1     if IsBidirectional == false
        /// </param>
        /// <param name="isTraining"></param>
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            if (!Network.UseGPU)
            {
                ForwardPropagationCPU(allX, y);
                return;
            }
            Debug.Assert(allX.Count == 1);
            Debug.Assert(_reserveSpace == null);
            Debug.Assert(_yIfReturnSequences == null);
            var x = allX[0];
            var batchSize = x.Shape[0];
            int timeSteps = x.Shape[1];
            Debug.Assert(x.Shape[2] == InputSize);
            Debug.Assert(x.Shape.Length == 3);

            const cudnnDataType_t dataType = cudnnDataType_t.CUDNN_DATA_FLOAT;
            var xRnnDataDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, InputSize);
            var yRnnDataDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, y.Shape.Last());

            var fMode = isTraining
                ? cudnnForwardMode_t.CUDNN_FWD_MODE_TRAINING
                : cudnnForwardMode_t.CUDNN_FWD_MODE_INFERENCE;

            var res = CudnnWrapper.cudnnGetRNNTempSpaceSizes(
                Network.GpuWrapper.CudnnHandle,
                _cudnnRNNDescriptor_t,
                fMode,
                xRnnDataDesc,
                out _workSpaceBufferSizeInBytes,
                out var reserveSpaceSize);
            GPUWrapper.CheckStatus(res);

            var workSpaceBuffer = GetBuffer(_workSpaceBufferSizeInBytes);

            Debug.Assert(reserveSpaceSize == 0 || isTraining);
            _reserveSpace = isTraining ? GetBuffer(reserveSpaceSize) : null;
            var devSeqLengths = Network.GpuWrapper.GetDevSeqLengths(batchSize, timeSteps);

            _yIfReturnSequences = _returnSequences?y:GetFloatTensor(_rnnDescriptor.YRnnData_Shape(timeSteps, batchSize));

            //the tensor shape expected by the cuDNN API is : (batchSize, timeSteps, inputSize or hiddenSize) 
            res = CudnnWrapper.cudnnRNNForward(
                Network.GpuWrapper.CudnnHandle,
                _cudnnRNNDescriptor_t,
                fMode,
                devSeqLengths,
                xRnnDataDesc,
                x,
                yRnnDataDesc,
                _yIfReturnSequences,
                new cudnnTensorDescriptor_t(), IntPtr.Zero, IntPtr.Zero,
                new cudnnTensorDescriptor_t(), IntPtr.Zero, IntPtr.Zero,
                _weightsAndBiases.CapacityInBytes,
                _weightsAndBiases,
                _workSpaceBufferSizeInBytes,
                workSpaceBuffer,
                reserveSpaceSize,       //needed only during training (data will be kept for backward propagation)
                _reserveSpace);        // null for inference
            GPUWrapper.CheckStatus(res);

            //We need to convert 'yRnnData' tensor to output 'y' tensor
            // '_yIfReturnSequences' shape:
            //      (batchSize, timeSteps, hiddenSize)
            // y shape:
            //      (batchSize, timeSteps, hiddenSize)      when _returnSequences == true
            //      (batchSize, hiddenSize)                 when _returnSequences == false
            if (_returnSequences)
            {
            }
            else
            {
                //from yRnnData (batchSize, timeSteps, K*hiddenSize) to y (batchSize, K*hiddenSize)
                int lastDim = _yIfReturnSequences.Shape[2];
                for (int batchId = 0; batchId < batchSize; ++batchId)
                {
                    if (IsBidirectional)
                    {
                        _yIfReturnSequences.CopyTo(_yIfReturnSequences.Idx(batchId, timeSteps - 1, 0), y, y.Idx(batchId, 0), lastDim/2);
                        _yIfReturnSequences.CopyTo(_yIfReturnSequences.Idx(batchId, 0, lastDim/2), y, y.Idx(batchId, lastDim/2), lastDim/2);
                    }
                    else
                    {
                        _yIfReturnSequences.CopyTo(_yIfReturnSequences.Idx(batchId, timeSteps - 1, 0), y, y.Idx(batchId, 0), lastDim);
                    }
                }
            }

            FreeFloatTensor(workSpaceBuffer);

            if (!_returnSequences && !isTraining)
            {
                //no need to keep the '_yIfReturnSequences' tensor
                FreeFloatTensor(ref _yIfReturnSequences);
            }

            if (_returnSequences)
            {
                _yIfReturnSequences = null;
            }
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dxList)
        {
            if (!Network.UseGPU)
            {
                BackwardPropagationCPU(allX, dy);
                return;
            }
            Debug.Assert(allX.Count == 1);
            Debug.Assert(_reserveSpace != null);
            Debug.Assert((_returnSequences && _yIfReturnSequences == null) || (!_returnSequences && _yIfReturnSequences != null));
            var x = allX[0];                // x shape  :     (batchSize, timeSteps, features)
            var batchSize = x.Shape[0];
            var timeSteps = x.Shape[1];
            Debug.Assert(x.Shape[2] == InputSize);

            #region we need to convert 'y' (& 'dy') tensor shape to '_yRnnData' (& 'dyRnnData') shape
            Tensor dyIfReturnSequences;
            if (_returnSequences)
            {
                // y & dy & _yIfReturnSequences & dyIfReturnSequences shape:
                //      (batchSize, timeSteps, K*hiddenSize)
                Debug.Assert(_yIfReturnSequences == null);
                _yIfReturnSequences = y;
                dyIfReturnSequences = dy;
            }
            else
            {
                // y & dy shape:
                //      (batchSize, K*hiddenSize)
                // _yIfReturnSequences & dyIfReturnSequences shape:
                //      (batchSize, timeSteps, K*hiddenSize)
                //we build dyIfReturnSequences from dy
                Debug.Assert(_yIfReturnSequences != null); //_yIfReturnSequences must have been kept from forward propagation
                dyIfReturnSequences = GetFloatTensor(_yIfReturnSequences.Shape);
                dyIfReturnSequences.ZeroMemory();
                // from dy (batchSize, K*hiddenSize) to dyIfReturnSequences (batchSize, timeSteps, K*hiddenSize)
                int lastDim = dyIfReturnSequences.Shape[^1];
                for (int batchId = 0; batchId < batchSize; ++batchId)
                {
                    if (IsBidirectional)
                    {
                        dy.CopyTo(dy.Idx(batchId, 0), dyIfReturnSequences, dyIfReturnSequences.Idx(batchId, timeSteps - 1, 0), lastDim / 2);
                        dy.CopyTo(dy.Idx(batchId, lastDim / 2), dyIfReturnSequences, dyIfReturnSequences.Idx(batchId, 0, lastDim / 2), lastDim / 2);
                    }
                    else
                    {
                        dy.CopyTo(dy.Idx(batchId, 0), dyIfReturnSequences, dyIfReturnSequences.Idx(batchId, timeSteps - 1, 0), lastDim);
                    }
                }
            }
            #endregion

            const cudnnDataType_t dataType = cudnnDataType_t.CUDNN_DATA_FLOAT;
            var xDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, InputSize);
            var yIfReturnSequencesDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, _yIfReturnSequences.Shape.Last());
            var devSeqLengths = Network.GpuWrapper.GetDevSeqLengths(batchSize, timeSteps);

            Debug.Assert(_reserveSpace != null);
            var workSpaceBuffer = GetBuffer(_workSpaceBufferSizeInBytes);
            var dx = dxList[0] ?? GetFloatTensor(x.Shape);

            var res = CudnnWrapper.cudnnRNNBackwardData_v8(
                Network.GpuWrapper.CudnnHandle,
                _cudnnRNNDescriptor_t,
                devSeqLengths,
                yIfReturnSequencesDesc,
                _yIfReturnSequences,
                dyIfReturnSequences,
                xDesc,
                dx,
                new cudnnTensorDescriptor_t(), IntPtr.Zero, IntPtr.Zero, IntPtr.Zero,
                new cudnnTensorDescriptor_t(), IntPtr.Zero, IntPtr.Zero, IntPtr.Zero,
                _weightsAndBiases.CapacityInBytes,
                _weightsAndBiases,
                _workSpaceBufferSizeInBytes,
                workSpaceBuffer,
                _reserveSpace.CapacityInBytes,
                _reserveSpace);
            GPUWrapper.CheckStatus(res);

            _weightsAndBiasesGradients.ZeroMemory();
            res = CudnnWrapper.cudnnRNNBackwardWeights_v8(
                Network.GpuWrapper.CudnnHandle,
                _cudnnRNNDescriptor_t,
                cudnnWgradMode_t.CUDNN_WGRAD_MODE_ADD,
                devSeqLengths,
                xDesc,
                x,
                new cudnnTensorDescriptor_t(), IntPtr.Zero,
                yIfReturnSequencesDesc,
                _yIfReturnSequences,
                _weightsAndBiasesGradients.CapacityInBytes,
                _weightsAndBiasesGradients,
                _workSpaceBufferSizeInBytes,
                workSpaceBuffer,
                _reserveSpace.CapacityInBytes,
                _reserveSpace);
            GPUWrapper.CheckStatus(res);

            FreeFloatTensor(workSpaceBuffer);
            if (dxList[0] == null)
            {
                FreeFloatTensor(dx);
            }
            FreeFloatTensor(ref _reserveSpace);
            if (!_returnSequences)
            {
                FreeFloatTensor(dyIfReturnSequences);
                FreeFloatTensor(ref _yIfReturnSequences);
            }
            _yIfReturnSequences = null;
            Debug.Assert(_yIfReturnSequences == null);
            Debug.Assert(_reserveSpace == null);
        }
        #region forward and backward propagation for CPU
        /// <summary>
        /// This is a simple implementation on CPU.
        /// It only supports unidirectional SimpleRNN with _returnSequences == false
        /// </summary>
        /// <param name="allX"></param>
        /// <param name="y"></param>
        private void ForwardPropagationCPU(List<Tensor> allX, Tensor y)
        {
            if (_returnSequences)
            {
                throw new NotSupportedException("_returnSequences == true");
            }
            if (_rnnDescriptor.dirMode != cudnnDirectionMode_t.CUDNN_UNIDIRECTIONAL)
            {
                throw new NotSupportedException(_rnnDescriptor.dirMode + " != cudnnDirectionMode_t.CUDNN_UNIDIRECTIONAL");
            }
            if (_rnnDescriptor.cellMode != cudnnRNNMode_t.CUDNN_RNN_TANH)
            {
                throw new NotSupportedException(_rnnDescriptor.cellMode + " !=  cudnnRNNMode_t.CUDNN_RNN_TANH");
            }

            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            var batchSize = x.Shape[0];                 //x.Shape[0] : batch size : number of sentences 
            int timeSteps = x.Shape[1];                 //x.Shape[1] : number of words in each sentence
            Debug.Assert(x.Shape[2] == InputSize);      //x.Shape[2] : number of distinct words (_inputSize)
            var aShape = new[] { batchSize, HiddenSize };
            var xShape = new[] { batchSize, InputSize };
            var a_init = GetFloatTensor(aShape);
            var a_buffer1 = GetFloatTensor(aShape);
            var a_buffer2 = GetFloatTensor(aShape);

            var x_at_t_buffer = GetFloatTensor(xShape);
            a_init.ZeroMemory();

            GetFloatTensor(ref _yIfReturnSequences, new[] { timeSteps, batchSize, HiddenSize });


            var a_tShape = new[] { batchSize, HiddenSize };
            for (int t = 0; t < timeSteps; ++t)
            {
                var y_t = _yIfReturnSequences.ElementSlice(t, a_tShape);
                // y_t = hidden state at time step 't'          (batchSize, hiddenSize)
                //     = tanh( x_t[t]*Weights_ax + y[t-1]*Weights_aa)

                x.From_NCH_to_NH(x_at_t_buffer, t);

                a_buffer1.Dot(x_at_t_buffer, Weights_ax);
                var a_prev = (t == 0) ? a_init : _yIfReturnSequences.ElementSlice(t - 1, a_tShape);
                a_buffer2.Dot(a_prev, Weights_aa);
                a_buffer1.AddTensor(1, a_buffer2, 1);
                Bias.BroadcastAddVectorToOutput(a_buffer1);
                a_buffer1.ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_TANH, null, y_t);
            }
            _yIfReturnSequences.ElementSlice(timeSteps - 1, y.Shape).CopyTo(y);
            FreeFloatTensor(a_init);
            FreeFloatTensor(a_buffer1);
            FreeFloatTensor(a_buffer2);
            FreeFloatTensor(x_at_t_buffer);
        }
        private void BackwardPropagationCPU(List<Tensor> allX, Tensor dy)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];                // x shape  :     (batchSize, timeSteps, _inputSize)
            var batchSize = x.Shape[0];
            var timeSteps = x.Shape[1];
            Debug.Assert(InputSize == x.Shape[2]);
            var aShape = dy.Shape;          // a shape  :     (batchSize, _hiddenSize )
            Debug.Assert(_yIfReturnSequences.Shape[0] == timeSteps);

            _weightsAndBiasesGradients.ZeroMemory();

            var a_buffer1 = GetFloatTensor(aShape);
            var a_buffer2 = GetFloatTensor(aShape);
            var da_t = GetFloatTensor(new[] { 1, HiddenSize });
            var _1_minus_aSquare = GetFloatTensor(da_t.Shape);
            var _1_vector = GetFloatTensor(da_t.Shape);
            _1_vector.SetValue(1f);
            var tmpBuffer = GetFloatTensor(da_t.Shape);
            var tmpWeight_aa = GetFloatTensor(Weights_aaGradients.Shape);

            var a_tShape = new[] { batchSize, HiddenSize };

            for (int batchId = 0; batchId < batchSize; ++batchId)
            {
                var x_batchId = x.ElementSlice(batchId);
                x_batchId.Reshape(x.Shape.Skip(1).ToArray());
                for (int t = timeSteps - 1; t >= 0; --t)
                {
                    var at = _yIfReturnSequences.ElementSlice(t, a_tShape).ElementSlice(batchId);
                    var xt = x_batchId.ElementSlice(t);

                    //we compute _1_minus_aSquare = 1-a[t]^2
                    at.CopyTo(_1_minus_aSquare);
                    _1_minus_aSquare.Update_Multiply_By_x(at);
                    _1_minus_aSquare.AddTensor(1f, _1_vector, -1f);

                    //We compute da[t]
                    if (t == timeSteps - 1)
                    {
                        //da[t] = dy*(1-a[t]^2)
                        _1_minus_aSquare.CopyTo(da_t);
                        da_t.Update_Multiply_By_x(dy.ElementSlice(batchId));
                    }
                    else
                    {
                        //da[t] = (Weights_aa dot da[t+1]) * (1-a[t]^2)
                        da_t.CopyTo(tmpBuffer); //da_t+1
                        da_t.Dot(Weights_aa, false, tmpBuffer, true, 1f, 0f);
                        da_t.Update_Multiply_By_x(_1_minus_aSquare);
                    }

                    //we update the Bias Gradient
                    // _biasGradients += da[t]
                    BiasGradients?.Update_Adding_Alpha_X(1, da_t);

                    //we update _weights_ax_gradient
                    //_weights_ax_gradient += da[t] * x[t]
                    da_t.CopyTo(tmpBuffer);
                    tmpBuffer.Update_Multiply_By_x(xt);
                    Weights_axGradients.Update_Adding_Alpha_X(1, tmpBuffer);

                    if (t >= 1)
                    {
                        //we update _weights_aa_gradient
                        var a_t_1 = _yIfReturnSequences.ElementSlice(t - 1, a_tShape).ElementSlice(batchId);
                        tmpWeight_aa.Dot(a_t_1, true, da_t, false, 1f, 0f);
                        //_weights_aa_gradient += a[t-1].T * da[t].T
                        Weights_aaGradients.Update_Adding_Alpha_X(1, tmpWeight_aa);
                    }
                }
            }

            FreeFloatTensor(a_buffer1);
            FreeFloatTensor(a_buffer2);
            FreeFloatTensor(da_t);
            FreeFloatTensor(_1_minus_aSquare);
            FreeFloatTensor(tmpBuffer);
            FreeFloatTensor(tmpWeight_aa);
            FreeFloatTensor(_1_vector);
        }
        #endregion
        #endregion

        #region parameters and gradients
        public override Tensor Weights => _weightsAndBiases;
        private Tensor Weights_ax => _weightsAndBiases.Slice(0, _rnnDescriptor.Weight_ax_Shape);
        private Tensor Weights_aa => _weightsAndBiases.Slice(_rnnDescriptor.Weight_ax_count, _rnnDescriptor.Weight_recurrent_Shape);
        public override Tensor Bias
        {
            get
            {
                if (_rnnDescriptor.Bias_count == 0)
                {
                    return null;
                }
                return _weightsAndBiases.Slice(_rnnDescriptor.Weight_ax_count + _rnnDescriptor.Weight_recurrent_count, _rnnDescriptor.BiasShape);
            }
        }
        public override Tensor WeightGradients => _weightsAndBiasesGradients;
        private Tensor Weights_axGradients => _weightsAndBiasesGradients.Slice(0, _rnnDescriptor.Weight_ax_Shape);
        private Tensor Weights_aaGradients => _weightsAndBiasesGradients.Slice(_rnnDescriptor.Weight_ax_count, _rnnDescriptor.Weight_recurrent_Shape);
        public override Tensor BiasGradients
        {
            get
            {
                if (_rnnDescriptor.Bias_count == 0)
                {
                    return null;
                }
                return _weightsAndBiasesGradients.Slice(_rnnDescriptor.Weight_ax_count + _rnnDescriptor.Weight_recurrent_count, _rnnDescriptor.BiasShape);
            }
        }
        public override int DisableBias()
        {
            //TODO : be able to disable bias
            return 0;
        }
        public override List<Tuple<Tensor, string>> Parameters
        {
            get
            {
                var result = new List<Tuple<Tensor, string>>
                {
                    Tuple.Create(_weightsAndBiases, WeightDatasetPath),
                };
                result.RemoveAll(t => t.Item1 == null);
                return result;
            }
        }
        public override List<Tensor> ParameterGradients
        {
            get
            {
                var result = new List<Tensor>
                {
                    _weightsAndBiasesGradients,
                };
                result.RemoveAll(t => t == null);
                return result;
            }
        }
        protected override Optimizer Optimizer => _optimizer;

        protected override List<Tensor> EmbeddedTensors(bool includeOptimizeTensors)
        {
            var result = base.EmbeddedTensors(includeOptimizeTensors);
            result.AddRange(new[] { _reserveSpace, _yIfReturnSequences });
            result.RemoveAll(t => t == null);
            return result;
        }

        #region Multi GPU Support
        public override void ReplaceParameters(List<Tensor> newParameters)
        {
            Debug.Assert(newParameters.Count == 1);
            FreeFloatTensor(ref _weightsAndBiases);
            _weightsAndBiases = newParameters[0];
        }
        public override void ReplaceGradients(List<Tensor> newGradients)
        {
            Debug.Assert(newGradients.Count == 1);
            FreeFloatTensor(ref _weightsAndBiasesGradients);
            _weightsAndBiasesGradients = newGradients[0];
        }
        #endregion

        public override void UpdateWeights(int batchSize, double learningRate)
        {
            Debug.Assert(Network.IsMaster);
            Debug.Assert(_weightsAndBiases.SameShape(_weightsAndBiasesGradients));
            if (Trainable)
            {
                _optimizer.UpdateWeights(learningRate, batchSize, _weightsAndBiases, _weightsAndBiasesGradients, null, null);
            }
        }
        #endregion

        private string WeightDatasetPath => DatasetNameToDatasetPath("kernel:0");

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(HiddenSize), HiddenSize)
                .Add(nameof(CellMode), (int)CellMode)
                .Add(nameof(BiasMode), (int)BiasMode)
                .Add(nameof(_returnSequences), _returnSequences)
                .Add(nameof(IsBidirectional), IsBidirectional)
                .ToString();
        }
        public static RecurrentLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new RecurrentLayer(
                (int) serialized[nameof(HiddenSize)],
                (cudnnRNNMode_t) serialized[nameof(CellMode)],
                (cudnnRNNBiasMode_t) serialized[nameof(BiasMode)],
                (bool) serialized[nameof(_returnSequences)],
                (bool) serialized[nameof(IsBidirectional)],
                (bool) serialized[nameof(Trainable)],
                network,
                (string) serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        public override int[] OutputShape(int batchSize)
        {
            int timeSteps = PrevLayer.OutputShape(1)[1];
            return _rnnDescriptor.Y_Shape(_returnSequences, timeSteps, batchSize);
        }
    }
}
