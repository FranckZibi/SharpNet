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
        [NotNull] private readonly List<Tensor> _weights_ax = new List<Tensor>();
        [NotNull] private readonly List<Tensor> _weights_aa = new List<Tensor>();
        [NotNull] private readonly List<Tensor> _weights_ax_bias = new List<Tensor>();
        [NotNull] private readonly List<Tensor> _weights_aa_bias = new List<Tensor>();
        #endregion
        #region gradients
        [NotNull] private Tensor _weightsAndBiasesGradients;
        [NotNull] private readonly List<Tensor> _weights_ax_gradients = new List<Tensor>();
        [NotNull] private readonly List<Tensor> _weights_aa_gradients = new List<Tensor>();
        [NotNull] private readonly List<Tensor> _weights_ax_bias_gradients = new List<Tensor>();
        [NotNull] private readonly List<Tensor> _weights_aa_bias_gradients = new List<Tensor>();
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
        private int NumLayers => _rnnDescriptor.numLayers;
        private double DropoutRate => _rnnDescriptor.dropoutRate;
        private int InputSize => _rnnDescriptor.inputSize;      // == features
        private int HiddenSize => _rnnDescriptor.hiddenSize;    // == units
        private bool IsBidirectional => _rnnDescriptor.dirMode == cudnnDirectionMode_t.CUDNN_BIDIRECTIONAL;

        #region constructor
        public RecurrentLayer(int hiddenSize, cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode, bool returnSequences, bool isBidirectional, int numLayers, double dropoutRate, bool trainable, Network network, string layerName) : base(network, layerName)
        {
            if (!Network.UseGPU)
            {
                throw new NotImplementedException();
            }
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
                numLayers,  /* numLayers */
                dropoutRate, /* dropout rate */
                auxFlags);

            _returnSequences = returnSequences;
            Trainable = trainable;


            _cudnnRNNDescriptor_t = Network.GpuWrapper.RNNDesc(_rnnDescriptor);
            CudnnWrapper.cudnnGetRNNWeightSpaceSize(Network.GpuWrapper.CudnnHandle, _cudnnRNNDescriptor_t, out var weightSpaceSize);

            _weightsAndBiases = GetFloatTensor(new[] { (int)(weightSpaceSize / 4) });
            _weightsAndBiasesGradients = GetFloatTensor(_weightsAndBiases.Shape);
            

            InitializeWeightAndBiasTensorList(_weightsAndBiases, false);
            InitializeWeightAndBiasTensorList(_weightsAndBiasesGradients, true);
            _optimizer = GetOptimizer(_weightsAndBiases.Shape, null);
          
            // ReSharper disable once VirtualMemberCallInConstructor
            ResetParameters(false);
        }



        #region extract of all tensors embedded in '_weightsAndBiases' and '_weightsAndBiasesGradients' GPU memory space

        /// <summary>
        /// Initialize the 8 following lists:
        ///     _weights_ax
        ///     _weights_aa
        ///     _weights_ax_bias  
        ///     _weights_aa_bias
        ///     _weights_ax_gradients
        ///     _weights_aa_gradients
        ///     _weights_ax_bias_gradients
        ///     _weights_aa_bias_gradients 
        /// </summary>
        private void InitializeWeightAndBiasTensorList(Tensor t, bool isGradient)
        {
            var res = CudnnWrapper.cudnnCreateTensorDescriptor(out var weightDesc);
            GPUWrapper.CheckStatus(res);
            res = CudnnWrapper.cudnnCreateTensorDescriptor(out var biasDesc);
            GPUWrapper.CheckStatus(res);

            for (int pseudoLayer = 0; pseudoLayer < _rnnDescriptor.PseudoLayersCount; ++pseudoLayer)
            {
                for (int linLayerID = 0; linLayerID < _rnnDescriptor.LinLayerIDCount; ++linLayerID)
                {
                    res = CudnnWrapper.cudnnGetRNNWeightParams(Network.GpuWrapper.CudnnHandle, _cudnnRNNDescriptor_t,
                        pseudoLayer, t.ReallyNeededMemoryInBytes, t.Pointer, linLayerID, weightDesc,
                        out IntPtr weightAddress, biasDesc, out IntPtr biasAddress);
                    GPUWrapper.CheckStatus(res);
                    var weight = GetWeightTensor(weightDesc, weightAddress, false);
                    var bias = GetWeightTensor(biasDesc, biasAddress, true);
                    bool isAx = linLayerID < (_rnnDescriptor.LinLayerIDCount / 2);
                    GetTensorListToUpdate(false, isGradient, isAx).Add(weight);
                    GetTensorListToUpdate(true, isGradient, isAx).Add(bias);
                }
            }
            res = CudnnWrapper.cudnnDestroyTensorDescriptor(weightDesc);
            GPUWrapper.CheckStatus(res);
            res = CudnnWrapper.cudnnDestroyTensorDescriptor(biasDesc);
            GPUWrapper.CheckStatus(res);
        }

        private List<Tensor> GetTensorListToUpdate(bool isBiasList, bool isGradientList, bool isAxList)
        {
            if (isBiasList)
            {
                if (isAxList)
                {
                    return isGradientList ? _weights_ax_bias_gradients : _weights_ax_bias;
                }
                return isGradientList ? _weights_aa_bias_gradients : _weights_aa_bias;
            }
            if (isAxList)
            {
                return isGradientList ? _weights_ax_gradients : _weights_ax;
            }
            return isGradientList ? _weights_aa_gradients : _weights_aa;
        }


        private GPUTensor<float> GetWeightTensor(cudnnTensorDescriptor_t tensorDesc, IntPtr tensorAddress, bool isBias)
        {
            if (tensorAddress == IntPtr.Zero)
            {
                return null;
            }
            var tensorShape = GPUWrapper.GetTensorShape(tensorDesc);
            Debug.Assert(tensorShape.Length >= 3);
            Debug.Assert(tensorShape[0] == 1);
            Debug.Assert(!isBias || tensorShape[2] == 1);
            var actualShape = isBias
                ? new[] {1, tensorShape[1] }                //for bias, tensorShape will be: {1, rows, 1}
                : new[] {tensorShape[1], tensorShape[2]};   //for weights, tensorShape will be: {1, rows, cols}
            return new GPUTensor<float>(actualShape, tensorAddress, Network.GpuWrapper);
        }
        #endregion

        public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
        {
            _weightsAndBiases.ZeroMemory();
            _weights_aa.ForEach(t=>t.Orthogonal(Rand));
            _weights_ax.ForEach(t => t.GlorotUniform(Rand));
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
            Debug.Assert(Network.UseGPU);
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
            Debug.Assert(Network.UseGPU);
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
                // ReSharper disable once PossibleNullReferenceException
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
        #endregion

        #region parameters and gradients
        public override Tensor Weights => throw new Exception("should never be called");
        public override Tensor Bias => throw new Exception("should never be called");
        public override Tensor WeightGradients => throw new Exception("should never be called");
        public override Tensor BiasGradients => throw new Exception("should never be called");
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

        public override void UpdateWeights(int batchSize, double learningRate, double maxLearningRate)
        {
            Debug.Assert(Network.IsMaster);
            Debug.Assert(_weightsAndBiases.SameShape(_weightsAndBiasesGradients));
            if (Trainable)
            {
                _optimizer.UpdateWeights(learningRate, maxLearningRate, batchSize, _weightsAndBiases, _weightsAndBiasesGradients, null, null);
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
                .Add(nameof(NumLayers), NumLayers)
                .Add(nameof(DropoutRate), DropoutRate)
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
                (int) serialized[nameof(NumLayers)],
                (double) serialized[nameof(DropoutRate)],
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
