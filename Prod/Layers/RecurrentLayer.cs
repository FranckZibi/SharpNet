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
    ///     (batchSize, timeSteps, features (= number of distinct words) )
    /// output 'y' shape :
    ///     if _returnSequences == true
    ///         (batchSize, timeSteps, Units)
    ///     else
    ///         (batchSize, _units)
    /// </summary>
    public abstract unsafe class RecurrentLayer : Layer
    {
        #region Fields
        #region trainable parameters
        [NotNull] private Tensor _weightsAndBiases;
        #endregion
        #region gradients
        [NotNull] private Tensor _weightsAndBiasesGradients;
        #endregion
        protected readonly bool _returnSequences;
        private readonly RNNDescriptor _rnnDescriptor;
        private readonly cudnnRNNDescriptor_t _cudnnRNNDescriptor_t;
        /// <summary>
        /// needed only for training, null for inference
        /// </summary>
        private Tensor _reserveSpace;
        /// <summary>
        /// when returnSequences is false and we are in training mode (not inference) :
        ///     we'll need to keep the original 'y' computed by the 'cudnnRNNForward' method
        ///     to use it during backward propagation
        /// will be null for inference or when returnSequences == true
        /// </summary>
        private Tensor _yRnnData;
        /// <summary>
        /// size of the temporary buffer used internally by the cuDNN API
        /// </summary>
        private size_t _workSpaceBufferSizeInBytes;
        [NotNull] private readonly Optimizer _optimizer;
        /// <summary>
        /// true if we are in time major mode
        ///     the tensor shape expected by the cuDNN API is : (timeSteps, batchSize, InputSize  or HiddenSize) 
        /// false if we are in batch major mode
        ///     the tensor shape expected by the cuDNN API is : (batchSize, timeSteps, InputSize  or HiddenSize) 
        /// </summary>
        private readonly bool time_major;
        #endregion

        private int InputSize => _rnnDescriptor.inputSize;       //number of distinct words in the dictionary  = Features
        protected int HiddenSize => _rnnDescriptor.hiddenSize;    //dimensionality of the output space = Units

        #region constructor
        protected RecurrentLayer(int hiddenSize, cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode, bool returnSequences, bool trainable, Network network, string layerName) : base(network, layerName)
        {
            int inputSize = PrevLayer.OutputShape(1)[2]; // InputSize = Features
            uint auxFlags = 0;
            auxFlags |= CudnnWrapper.CUDNN_RNN_PADDED_IO_ENABLED;
            _rnnDescriptor = new RNNDescriptor(
                cudnnRNNAlgo_t.CUDNN_RNN_ALGO_STANDARD,
                cellMode,
                biasMode,
                cudnnDirectionMode_t.CUDNN_UNIDIRECTIONAL,
                cudnnRNNInputMode_t.CUDNN_LINEAR_INPUT,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
                cudnnMathType_t.CUDNN_DEFAULT_MATH,
                inputSize,  /* = features */
                hiddenSize,      /* hiddenSize */
                hiddenSize,      /* projSize*/
                1,          /* numLayers */
                0.0,
                auxFlags);

            _returnSequences = returnSequences;
            Trainable = trainable;

            // Seems there is an issue when using batch major mode, so we are always using time major mode
            // See: https://forums.developer.nvidia.com/t/memory-corruption-after-calling-cudnnrnnforward-with-cudnn-rnn-data-layout-batch-major-unpacked-layout-cuda-11-0-cudnn-8-0-5/160640
            time_major = true;
         
            _cudnnRNNDescriptor_t = Network.GpuWrapper.RNNDesc(_rnnDescriptor, Network.GetRandomNumberGeneratorStatesBuffer());
            CudnnWrapper.cudnnGetRNNWeightSpaceSize(Network.GpuWrapper.CudnnHandle, _cudnnRNNDescriptor_t, out var weightSpaceSize);

            _weightsAndBiases = GetBuffer(weightSpaceSize);
            _weightsAndBiasesGradients = GetBuffer(weightSpaceSize);
            _optimizer = GetOptimizer(_weightsAndBiases.Shape, null);

            int expectedWeightAndBiasCount = _rnnDescriptor.WeightAndBiasCount;
            if (_weightsAndBiases.Count != expectedWeightAndBiasCount)
            {
                throw new ArgumentException("expecting " + expectedWeightAndBiasCount + " weights and bias but got " + _weightsAndBiases.Count);
            }
        }

        public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
        {
            var prevLayerNX = PrevLayer.n_x;
            Weights_ax.RandomMatrixNormalDistribution(Rand, 0.0 /* mean */, Math.Sqrt(2.0 / prevLayerNX) /*stdDev*/);
            Weights_aa.RandomMatrixNormalDistribution(Rand, 0.0 /* mean */, Math.Sqrt(2.0 / prevLayerNX) /*stdDev*/);
            Bias?.ZeroMemory();
        }

        #endregion




        #region forward and backward propagation
        /// <summary>
        /// </summary>
        /// <param name="allX"></param>
        /// shape:
        ///     (batchSize, timeSteps, InputSize = Features)
        /// <param name="y">
        /// if _returnSequences == true
        ///     shape = (batchSize, timeSteps, HiddenSize = Units)
        /// else
        ///     shape = (batchSize, HiddenSize = Units)
        /// </param>
        /// <param name="isTraining"></param>
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(_reserveSpace == null);
            Debug.Assert(_yRnnData == null);
            var x = allX[0];
            var batchSize = x.Shape[0];
            int timeSteps = x.Shape[1];
            Debug.Assert(x.Shape[2] == InputSize);
            Debug.Assert(y.Shape.Last() == HiddenSize);

            var xRnnData = GetFloatTensor(time_major ? new[] { timeSteps, batchSize, InputSize } : new[] { batchSize, timeSteps, InputSize });
            if (time_major)
            {
                x.Switch_First_2_axis(xRnnData);
            }
            else
            {
                x.CopyTo(xRnnData);
            }

            _yRnnData = GetFloatTensor(time_major ? new[] { timeSteps, batchSize, HiddenSize } : new[] { batchSize, timeSteps, HiddenSize });
            var dataType = cudnnDataType_t.CUDNN_DATA_FLOAT;
            var xRnnDataDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, InputSize, time_major);
            var yRnnDataDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, HiddenSize, time_major);
            //var hDesc = Network.GpuWrapper.TensorDesc(dataType, new[] { _rnnDescriptor.numLayers, batchSize, HiddenSize });

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

            var devSeqLengths = stackalloc int[batchSize];
            GPUWrapper.FillWithSameValue(devSeqLengths, batchSize, timeSteps);

            res = CudnnWrapper.cudnnRNNForward(
                Network.GpuWrapper.CudnnHandle,
                _cudnnRNNDescriptor_t,
                fMode,
                devSeqLengths,
                xRnnDataDesc,
                xRnnData,
                yRnnDataDesc,
                _yRnnData,
                new cudnnTensorDescriptor_t(),
                IntPtr.Zero,
                IntPtr.Zero,
                new cudnnTensorDescriptor_t(),
                IntPtr.Zero,
                IntPtr.Zero,
                _weightsAndBiases.CapacityInBytes,
                _weightsAndBiases,
                _workSpaceBufferSizeInBytes,
                workSpaceBuffer,
                reserveSpaceSize,       //needed only during training (data will be kept for backward propagation)
                _reserveSpace);        // null for inference
            GPUWrapper.CheckStatus(res);

            //We need to convert 'yRnnData' tensor to output 'y' tensor
            // yRnnData shape:
            //      (timeSteps, batchSize, hiddenSize = Units)
            // y shape:
            //      (batchSize, timeSteps, hiddenSize = Units)      when _returnSequences == true
            //      (batchSize, hiddenSize = Units)                 when _returnSequences == false
            if (_returnSequences)
            {
                if (time_major)
                {
                    _yRnnData.Switch_First_2_axis(y);
                }
                else
                {
                    _yRnnData.CopyTo(y);
                }
            }
            else
            {
                if (time_major)
                {
                    _yRnnData.ElementSlice(timeSteps - 1).CopyTo(y);
                }
                else
                {
                    throw new NotImplementedException();
                }
            }

            FreeFloatTensor(ref xRnnData);
            FreeFloatTensor(workSpaceBuffer);
            if (isTraining && !_returnSequences)
            {
                //we need to keep the _yRnnData tensor for backward propagation
            }
            else
            {
                FreeFloatTensor(ref _yRnnData);
            }
        }

        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dxList)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(_reserveSpace != null);
            Debug.Assert((_returnSequences && _yRnnData == null) || (!_returnSequences && _yRnnData != null));
            var x = allX[0];                // x shape  :     (batchSize, timeSteps, features)
            var batchSize = x.Shape[0];
            var timeSteps = x.Shape[1];
            Debug.Assert(x.Shape[2] == InputSize);
            Debug.Assert(dy.Shape.Last() == HiddenSize);

            #region we need to convert 'x' tensor shape to 'xRnnData' shape
            // x shape:
            //      (batchSize, timeSteps, features)
            // xRnnData shape:
            //      (timeSteps, batchSize, features)
            var xRnnData = GetFloatTensor(new[] { timeSteps, batchSize, InputSize });
            x.Switch_First_2_axis(xRnnData);
            #endregion

            #region we need to convert 'y' (& 'dy') tensor shape to 'yRnnData' (& 'dyRnnData') shape
            // y (& dy) shape:
            //      (batchSize, timeSteps, hiddenSize = Units)      when _returnSequences == true
            //      (batchSize, hiddenSize = Units)                 when _returnSequences == false
            // yRnnData (& dyRnnData) shape:
            //      (timeSteps, batchSize, hiddenSize = Units)
            if (_returnSequences)
            {
                Debug.Assert(_yRnnData == null);
                _yRnnData = GetFloatTensor(time_major ? new[] { timeSteps, batchSize, HiddenSize } : new[] { batchSize, timeSteps, HiddenSize });
            }
            else
            {
                Debug.Assert(_yRnnData != null); //the yRnnData must have been kept from forward propagation
            }
            var dyRnnData = GetFloatTensor(_yRnnData.Shape);
            if (_returnSequences)
            {
                if (time_major)
                {
                    y.Switch_First_2_axis(_yRnnData);
                    dy.Switch_First_2_axis(dyRnnData);
                }
                else
                {
                    y.CopyTo(_yRnnData);
                    dy.CopyTo(dyRnnData);
                }
            }
            else
            {
                //_yRnnData.ZeroMemory();
                dyRnnData.ZeroMemory();
                if (time_major)
                {
                    //y.CopyTo(_yRnnData.ElementSlice(timeSteps - 1));
                    dy.CopyTo(dyRnnData.ElementSlice(timeSteps - 1));
                }
                else
                {
                    throw new NotImplementedException();
                }
            }
            #endregion

            var dataType = cudnnDataType_t.CUDNN_DATA_FLOAT;
            var xDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, InputSize, time_major);
            var yRnnDataDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, HiddenSize, time_major);
            var devSeqLengths = stackalloc int[batchSize];
            GPUWrapper.FillWithSameValue(devSeqLengths, batchSize, timeSteps);

            Debug.Assert(_reserveSpace != null);
            var workSpaceBuffer = GetBuffer(_workSpaceBufferSizeInBytes);

            var dx = dxList[0] ?? GetFloatTensor(x.Shape);

            var res = CudnnWrapper.cudnnRNNBackwardData_v8(
                Network.GpuWrapper.CudnnHandle,
                _cudnnRNNDescriptor_t,
                devSeqLengths,
                yRnnDataDesc,
                _yRnnData,
                dyRnnData,
                xDesc,
                dx,
                new cudnnTensorDescriptor_t(),
                IntPtr.Zero,
                IntPtr.Zero,
                IntPtr.Zero,
                new cudnnTensorDescriptor_t(),
                IntPtr.Zero,
                IntPtr.Zero,
                IntPtr.Zero,
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
                xRnnData,
                new cudnnTensorDescriptor_t(),
                IntPtr.Zero,
                yRnnDataDesc,
                _yRnnData,
                _weightsAndBiasesGradients.CapacityInBytes,
                _weightsAndBiasesGradients,
                _workSpaceBufferSizeInBytes,
                workSpaceBuffer,
                _reserveSpace.CapacityInBytes,
                _reserveSpace);
            GPUWrapper.CheckStatus(res);

            FreeFloatTensor(workSpaceBuffer);
            FreeFloatTensor(xRnnData);
            FreeFloatTensor(dyRnnData);
            if (dxList[0] == null)
            {
                FreeFloatTensor(dx);
            }
            FreeFloatTensor(ref _reserveSpace);
            FreeFloatTensor(ref _yRnnData);
            Debug.Assert(_yRnnData == null);
            Debug.Assert(_reserveSpace == null);
        }
        #endregion

        public override List<Tuple<Tensor, string>> Parameters
        {
            get
            {
                var result = new List<Tuple<Tensor, string>>
                {
                    Tuple.Create(_weightsAndBiases, "Weights"),
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

        #region parameters and gradients
        public override Tensor Weights => _weightsAndBiases;
        public Tensor Weights_ax => _weightsAndBiases.Slice(0, _rnnDescriptor.Weight_ax_Shape);
        public Tensor Weights_aa => _weightsAndBiases.Slice(_rnnDescriptor.Weight_ax_count, _rnnDescriptor.Weight_recurrent_Shape);
        public override Tensor Bias 
        {
            get
            {
                if (_rnnDescriptor.Bias_count == 0)
                {
                    return null;
                }
                return _weightsAndBiases.Slice(_rnnDescriptor.Weight_ax_count+ _rnnDescriptor.Weight_recurrent_count, _rnnDescriptor.BiasShape);
            }
        }

        public override Tensor WeightGradients => _weightsAndBiasesGradients;
        public override Tensor BiasGradients => null;
        public override int DisableBias()
        {
            return 0; //?D
        }

        protected override Optimizer Optimizer => _optimizer;

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
      
        public override int[] OutputShape(int batchSize)
        {
            if (_returnSequences)
            {
                int timeSteps = PrevLayer.OutputShape(1)[1];
                return new[] { batchSize, timeSteps, HiddenSize };
            }
            //only the last output
            return new[] { batchSize, HiddenSize };
        }
    }
}