﻿using System;
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
        [NotNull] public readonly List<Tensor> _weight_ih = new();
        [NotNull] public readonly List<Tensor> _weight_hh = new();
        [NotNull] public readonly List<Tensor> _bias_ih = new();
        [NotNull] public readonly List<Tensor> _bias_hh = new();
        #endregion
        #region gradients
        [NotNull] private Tensor _weightsAndBiasesGradients;
        [NotNull] private readonly List<Tensor> _weight_ih_gradients = new();
        [NotNull] private readonly List<Tensor> _weight_hh_gradients = new();
        [NotNull] private readonly List<Tensor> _bias_ih_gradients = new();
        [NotNull] private readonly List<Tensor> _bias_hh_gradients = new ();
        #endregion

        private readonly bool _returnSequences;
        private readonly bool IsEncoderThatWillBeFollowedByDecoder;
        private readonly RNNDescriptor _rnnDescriptor;
        /// <summary>
        /// when returnSequences is false and we are in training mode (not inference) :
        ///     we'll keep the original 'y' computed by the 'cudnnRNNForward' method to use it during backward propagation
        /// will be null for inference or when returnSequences == true
        /// </summary>
        private Tensor _yIfReturnSequences;
        [NotNull] private readonly Optimizer _optimizer;
        #endregion

        public cudnnRNNMode_t CellMode => _rnnDescriptor.cellMode;
        public cudnnRNNBiasMode_t BiasMode => _rnnDescriptor.biasMode;
        private int num_layers => _rnnDescriptor.numLayers;
        private double dropout => _rnnDescriptor.dropoutRate;
        private int input_size => _rnnDescriptor.inputSize;      // == features
        public int hidden_size => _rnnDescriptor.hiddenSize;    // == units
        public bool bidirectional => _rnnDescriptor.dirMode == cudnnDirectionMode_t.CUDNN_BIDIRECTIONAL;

        #region constructor
        /// <summary>
        /// 
        /// </summary>
        /// <param name="hiddenSize"></param>
        /// <param name="cellMode"></param>
        /// <param name="biasMode"></param>
        /// <param name="returnSequences"></param>
        /// <param name="isBidirectional"></param>
        /// <param name="numLayers"></param>
        /// <param name="dropoutRate"></param>
        /// <param name="isEncoderThatWillBeFollowedByDecoder">
        /// if True:
        ///      the Layer is an Encoder that will be followed by a Decoder
        /// </param>
        /// <param name="encoderLayerIndexIfLayerIsDecoder">
        /// If >= 0:
        ///     the layer is a Decoder, and the index of the associated encoder layer is at 'encoderLayerIndexIfLayerIsDecoder'
        /// </param>
        /// <param name="trainable"></param>
        /// <param name="network"></param>
        /// <param name="layerName"></param>
        /// <exception cref="NotImplementedException"></exception>
        /// <exception cref="ArgumentException"></exception>
        public RecurrentLayer(int hiddenSize, cudnnRNNMode_t cellMode, cudnnRNNBiasMode_t biasMode, bool returnSequences, bool isBidirectional, int numLayers, double dropoutRate, bool isEncoderThatWillBeFollowedByDecoder, int encoderLayerIndexIfLayerIsDecoder, bool trainable, Network network, string layerName) : 
            base(network,
                encoderLayerIndexIfLayerIsDecoder >= 0?new []{ network.Layers.Count-1, encoderLayerIndexIfLayerIsDecoder } :new[]{ network.Layers.Count - 1 },
                layerName)
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
            IsEncoderThatWillBeFollowedByDecoder = isEncoderThatWillBeFollowedByDecoder;  
            Trainable = trainable;
            if (IsEncoderThatWillBeFollowedByDecoder && _returnSequences)
            {
                throw new ArgumentException("an Encoder can not return a sequence");
            }
            if (IsEncoderThatWillBeFollowedByDecoder && encoderLayerIndexIfLayerIsDecoder != -1)
            {
                throw new ArgumentException("an Encoder do not need the index that will be used by the decoder: it does not need it");
            }

            _cudnnRNNDescriptor_t = Network.GpuWrapper.RNNDesc(_rnnDescriptor);
            CudnnWrapper.cudnnGetRNNWeightSpaceSize(Network.GpuWrapper.CudnnHandle, _cudnnRNNDescriptor_t, out var weightSpaceSize);

            _weightsAndBiases = GetFloatTensor(new[] { (int)(weightSpaceSize / 4) });
            _weightsAndBiasesGradients = GetFloatTensor(_weightsAndBiases.Shape);
            
            InitializeWeightAndBiasTensorList(_weightsAndBiases, false);
            InitializeWeightAndBiasTensorList(_weightsAndBiasesGradients, true);
            _optimizer = Sample.GetOptimizer(_weightsAndBiases.Shape, null, MemoryPool);
          
            // ReSharper disable once VirtualMemberCallInConstructor
            ResetParameters(false);
        }



        #region extract of all tensors embedded in '_weightsAndBiases' and '_weightsAndBiasesGradients' GPU memory space

        /// <summary>
        /// Initialize the 8 following lists:
        ///     _weight_ih
        ///     _weight_hh
        ///     _bias_ih  
        ///     _bias_hh
        ///     _weight_ih_gradients
        ///     _weight_hh_gradients
        ///     _bias_ih_gradients
        ///     _bias_hh_gradients 
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
                    return isGradientList ? _bias_ih_gradients : _bias_ih;
                }
                return isGradientList ? _bias_hh_gradients : _bias_hh;
            }
            if (isAxList)
            {
                return isGradientList ? _weight_ih_gradients : _weight_ih;
            }
            return isGradientList ? _weight_hh_gradients : _weight_hh;
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
                ? new[] {tensorShape[1] }                   //for bias, tensorShape will be: {rows}
                : new[] {tensorShape[1], tensorShape[2]};   //for weights, tensorShape will be: {1, rows, cols}
            return new GPUTensor<float>(actualShape, tensorAddress, Network.GpuWrapper);
        }
        #endregion

        public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
        {
            _weightsAndBiases.ZeroMemory();
            _weight_hh.ForEach(t=>t.Orthogonal(Rand));
            _weight_ih.ForEach(t => t.GlorotUniform(Rand));
        }
        #endregion

        private bool IsDecoder => PreviousLayerIndexes.Count == 2;
        
        public override string LayerType()
        {
            if (IsDecoder)
            {
                return "Decoder" + (bidirectional ? " Bidirectional" : "");
            }
            if (IsEncoderThatWillBeFollowedByDecoder)
            {
                return "Encoder" + (bidirectional ? " Bidirectional" : "");
            }
            if (bidirectional)
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
            int countBefore = bidirectional
                ? Layers.Count(l => l.LayerIndex < LayerIndex && l is RecurrentLayer layer && layer.bidirectional)
                : Layers.Count(l => l.LayerIndex < LayerIndex && l is RecurrentLayer layer && layer.CellMode == CellMode);
            if (countBefore != 0)
            {
                result += "_" + countBefore;
            }
            if (_returnSequences)
            {
                result += "_return_sequences";
            }
            return result;
        }

        #region forward and backward propagation

        /// <summary>
        /// </summary>
        /// <param name="allX"></param>
        /// input 'x' shape:
        ///     (batchSize, timeSteps, inputSize)
        /// <param name="yFull">
        /// output 'y' shape:
        ///     (batchSize, timeSteps, K*hiddenSize)      if _returnSequences == true
        ///     (batchSize, K*hiddenSize)                 if _returnSequences == false
        /// with
        ///      K == 2     if IsBidirectional == true 
        ///      K == 1     if IsBidirectional == false
        /// </param>
        /// <param name="isTraining"></param>
        public override void ForwardPropagation(List<Tensor> allX, Tensor yFull, bool isTraining)
        {
            Debug.Assert(Network.UseGPU);
            Debug.Assert(_reserveSpace == null);
            Debug.Assert(_yIfReturnSequences == null);
            var x = allX[0];
            var batchSize = x.Shape[0];
            int timeSteps = x.Shape[1];
            Debug.Assert(x.Shape[2] == input_size);
            Debug.Assert(x.Shape.Length == 3);

            var (y, cy) = IsEncoderThatWillBeFollowedByDecoder? ExtractEncoderContextVector(yFull):(yFull,null);
            var (hx, cx) = IsDecoder ? ExtractEncoderContextVector(allX[1]) : (null, null);

            var xRnnDataDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, input_size);
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
                HiddenDesc(batchSize),
                hx,
                _returnSequences ? null : y, /* hy */
                CellDesc(batchSize),
                cx,
                cy,
                _weightsAndBiases.CapacityInBytes,
                _weightsAndBiases,
                _workSpaceBufferSizeInBytes,
                workSpaceBuffer,
                reserveSpaceSize,       //needed only during training (data will be kept for backward propagation)
                _reserveSpace);        // null for inference
            GPUWrapper.CheckStatus(res);

            if (bidirectional & !_returnSequences)
            {
                //We need to convert initial y ('hy') tensor to output 'y' tensor
                // initial y (= 'hy') shape:
                //      (K=2, batchSize, hiddenSize)
                // target y shape:
                //      (batchSize, K*hiddenSize)
                Debug.Assert(y.Shape.Length == 2);
                var yCopy = GetFloatTensor(y.Shape);
                var yShape = y.Shape;
                const int K = 2;
                y.CopyTo(yCopy);
                yCopy.ReshapeInPlace(new[] { K, batchSize, hidden_size });
                yCopy.Switch_First_2_axis(y);
                y.ReshapeInPlace(yShape);
                FreeFloatTensor(yCopy);
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

        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dyFull, List<Tensor> dxList)
        {
            Debug.Assert(Network.UseGPU);
            Debug.Assert(_reserveSpace != null);
            Debug.Assert((_returnSequences && _yIfReturnSequences == null) || (!_returnSequences && _yIfReturnSequences != null));
            var x = allX[0];                // x shape  :     (batchSize, timeSteps, features)
            var batchSize = x.Shape[0];
            var timeSteps = x.Shape[1];
            Debug.Assert(x.Shape[2] == input_size);

            var (dy, dcy) = IsEncoderThatWillBeFollowedByDecoder ? ExtractEncoderContextVector(dyFull) : (dyFull, null);
            var (hx, cx) = IsDecoder ? ExtractEncoderContextVector(allX[1]) : (null, null);
            var (dhx, dcx) = IsDecoder ? ExtractEncoderContextVector(dxList[1]) : (null, null);

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
                    if (bidirectional)
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

            var xDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, input_size);
            var hDesc = HiddenDesc(batchSize);
            var cDesc = CellDesc(batchSize);
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
                hDesc,
                hx,
                IntPtr.Zero, /* dhy */
                dhx,
                cDesc, 
                cx,
                dcy,
                dcx,
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
                hDesc, 
                hx,
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

        /// <summary>
        /// Extract the Context Vector associated with the Encoder output
        /// </summary>
        /// <param name="encoderContextVector">The encoder context vector</param>
        /// <returns>2 tensors:  the hidden state of the encoder and the cell state of the encoder</returns>
        private (Tensor, Tensor) ExtractEncoderContextVector(Tensor encoderContextVector)
        {
            var stateShape = (int[])encoderContextVector.Shape.Clone();
            stateShape[^1] = encoderContextVector.Shape[^1] / 2;
            var hiddenStateTensor = new GPUTensor<float>(stateShape, encoderContextVector.Pointer, Network.GpuWrapper);
            var cellStateTensor = new GPUTensor<float>(stateShape, encoderContextVector.Pointer + (encoderContextVector.Count / 2) * encoderContextVector.TypeSize, Network.GpuWrapper);
            return (hiddenStateTensor, cellStateTensor);
        }

        const cudnnDataType_t dataType = cudnnDataType_t.CUDNN_DATA_FLOAT;

        private cudnnTensorDescriptor_t HiddenDesc(int batchSize)
        {
            int K = (_rnnDescriptor.dirMode == cudnnDirectionMode_t.CUDNN_BIDIRECTIONAL) ? 2 : 1;
            return Network.GpuWrapper.TensorDesc(dataType, new[] {K * num_layers, batchSize, hidden_size});
        }
        private cudnnTensorDescriptor_t CellDesc(int batchSize)
        {
            return HiddenDesc(batchSize);
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



        private string ParameterExtension(int layerIndex)
        {
            if (bidirectional)
            {
                if (layerIndex % 2 == 0)
                {
                    return "_l" + (layerIndex/2);
                }
                return "_l" + (layerIndex / 2)+"_reverse";
            }
            else
            {
                return "_l" + layerIndex; 
            }
        }
        public override List<Tuple<Tensor, string>> Parameters
        {
            get
            {
                var result = new List<Tuple<Tensor, string>>();
                for (int i = 0; i < _weight_ih.Count; ++i)
                {
                    result.Add(Tuple.Create(_weight_ih[i], DatasetNameToDatasetPath("weight_ih" + ParameterExtension(i) )));
                }
                for (int i = 0; i < _weight_hh.Count; ++i)
                {
                    result.Add(Tuple.Create(_weight_hh[i], DatasetNameToDatasetPath("weight_hh" + ParameterExtension(i))));
                }
                for (int i = 0; i < _bias_ih.Count; ++i)
                {
                    result.Add(Tuple.Create(_bias_ih[i], DatasetNameToDatasetPath("bias_ih" + ParameterExtension(i))));
                }
                for (int i = 0; i < _bias_hh.Count; ++i)
                {
                    result.Add(Tuple.Create(_bias_hh[i], DatasetNameToDatasetPath("bias_hh" + ParameterExtension(i))));
                }
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
#region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(hidden_size), hidden_size)
                .Add(nameof(CellMode), CellMode.ToString())
                .Add(nameof(BiasMode), BiasMode.ToString())
                .Add(nameof(_returnSequences), _returnSequences)
                .Add(nameof(bidirectional), bidirectional)
                .Add(nameof(num_layers), num_layers)
                .Add(nameof(dropout), dropout)
                .Add(nameof(IsEncoderThatWillBeFollowedByDecoder), IsEncoderThatWillBeFollowedByDecoder)
                .ToString();
        }
        public static RecurrentLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            var previousLayerIndexes = (int[]) serialized[nameof(PreviousLayerIndexes)];
            int encoderLayerIndexIfLayerIsDecoder = previousLayerIndexes.Length >= 2 ? previousLayerIndexes[1] : -1;
            return new RecurrentLayer(
                (int) serialized[nameof(hidden_size)],
                (cudnnRNNMode_t)Enum.Parse(typeof(cudnnRNNMode_t), (string)serialized[nameof(CellMode)]),
                (cudnnRNNBiasMode_t)Enum.Parse(typeof(cudnnRNNBiasMode_t), (string)serialized[nameof(BiasMode)]),
                (bool) serialized[nameof(_returnSequences)],
                (bool) serialized[nameof(bidirectional)],
                (int) serialized[nameof(num_layers)],
                (double) serialized[nameof(dropout)],
                (bool) serialized[nameof(IsEncoderThatWillBeFollowedByDecoder)],
                encoderLayerIndexIfLayerIsDecoder,
                (bool) serialized[nameof(Trainable)],
                network,
                (string) serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion


        #region PyTorch support
        public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
        {
            string extraInfo = "";
            string functionName = nameof(RecurrentLayer);
            switch (CellMode)
            {
                case cudnnRNNMode_t.CUDNN_RNN_RELU:
                case cudnnRNNMode_t.CUDNN_RNN_TANH:
                    //see: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
                    //torch.nn.RNN(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)
                    extraInfo += CellMode == cudnnRNNMode_t.CUDNN_RNN_RELU ? ", nonlinearity='relu'" : ", nonlinearity='tanh'";
                    functionName = "RNN";
                    break;
                case cudnnRNNMode_t.CUDNN_GRU:
                    functionName = "GRU";
                    break;
                case cudnnRNNMode_t.CUDNN_LSTM:
                    functionName = "LSTM";
                    break;
            }
            var input_shape = PreviousLayers.Count == 0 ? new[] { -1, -1 } : PreviousLayers[0].OutputShape(666);
            int input_size = input_shape[^1];
            const bool bias = true;
            constructorLines.Add("self." + LayerName + " = torch.nn."+ functionName+"(input_size=" + input_size + ", hidden_size=" + hidden_size 
                                 + ", num_layers=" + num_layers + extraInfo    
                                 + ", bias=" + bias 
                                 + ", batch_first=True, dropout=" + dropout + ", bidirectional=" + bidirectional +
                                 ")");
            foreach (var biasParam in GetBiasPyTorchParameters(false, true))
            {
                constructorLines.Add("torch.nn.init.zeros_(self." + LayerName + "." + biasParam + ")");
            }

            //string variableNamExtension = "_return_sequences_"+ Utils.ToPython(_returnSequences);
            string variableNamExtension = "";

            forwardLines.Add(GetPyTorchOutputVariableName()+ variableNamExtension + ", "+ GetPyTorchOutputVariableName() + "_hidden"+variableNamExtension +" = self." + LayerName + "(" + GetInputVariableName() + ")");
            if (!_returnSequences)
            {
                //we only keep the last index of the 2nd dimension 
                forwardLines.Add(GetPyTorchOutputVariableName() + " = " + GetPyTorchOutputVariableName() + variableNamExtension + "[:,-1,:]");
            }
        }

        private List<string> GetBiasPyTorchParameters(bool weights, bool bias)
        {
            var res = new List<string>();
            if (weights)
            {
                for (int i = 0; i < num_layers; ++i)
                {
                    res.Add("weight_ih_l" + i);
                    res.Add("weight_hh_l" + i);
                }
            }
            if (bias)
            {
                for (int i = 0; i < num_layers; ++i)
                {
                    res.Add("bias_ih_l" + i);
                    res.Add("bias_hh_l" + i);
                }
            }

            if (bidirectional)
            {
                for (int i = res.Count - 1; i >= 0; --i)
                {
                    res.Add(res[i]+ "_reverse");
                }
            }
            return res;
        }

        #endregion

        public override int[] OutputShape(int batchSize)
        {
            int timeSteps = PrevLayer.OutputShape(1)[1];
            return _rnnDescriptor.Y_Shape(_returnSequences, timeSteps, batchSize, IsEncoderThatWillBeFollowedByDecoder);
        }
    }
}
