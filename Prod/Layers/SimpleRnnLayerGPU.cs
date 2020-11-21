using System;
using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    /// <summary>
    /// input shape :
    ///     (batchSize, timeSteps, features (= number of distinct words) )
    /// output shape :
    ///     if _returnSequences == true
    ///         (batchSize, timeSteps, Units)
    ///     else
    ///         (batchSize, _units)
    /// </summary>
    public sealed unsafe class SimpleRnnLayerGPU : Layer
    {
        #region Fields
        #region trainable parameters
        [NotNull] private Tensor _weightsAndBiases;
        #endregion
        #region gradients
        [NotNull] private Tensor _weightsAndBiasesGradients;
        #endregion
        private readonly bool _returnSequences;
        private readonly RNNDescriptor _rnnDescriptor;
        private readonly cudnnRNNDescriptor_t _cudnnRNNDescriptor_t;
        private Tensor _workSpace;      //needed both for training and inference
        private Tensor _reserveSpace;   //needed only for training
        [NotNull] private readonly Optimizer _optimizer;
        #endregion

        private int Features => _rnnDescriptor.inputSize;   //number of distinct words in the dictionary 
        private int Units => _rnnDescriptor.hiddenSize;      //dimensionality of the output space

        #region constructor
        public SimpleRnnLayerGPU(int features, int units, bool returnSequences, bool trainable, Network network, string layerName) : base(network, layerName)
        {
            _returnSequences = returnSequences;
            Trainable = trainable;

            _rnnDescriptor = new RNNDescriptor(
                cudnnRNNAlgo_t.CUDNN_RNN_ALGO_STANDARD,
                cudnnRNNMode_t.CUDNN_RNN_TANH,
                cudnnRNNBiasMode_t.CUDNN_RNN_SINGLE_INP_BIAS,
                cudnnDirectionMode_t.CUDNN_UNIDIRECTIONAL,
                cudnnRNNInputMode_t.CUDNN_LINEAR_INPUT,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
                cudnnMathType_t.CUDNN_DEFAULT_MATH,
                features,
                units,  /* hiddenSize */
                units,  /* projSize*/
                1,      /* numLayers */
                0.0,
                0);

            _cudnnRNNDescriptor_t = Network.GpuWrapper.RNNDesc(_rnnDescriptor, Network.GetRandomNumberGeneratorStatesBuffer());
            CudnnWrapper.cudnnGetRNNWeightSpaceSize(Network.GpuWrapper.CudnnHandle, _cudnnRNNDescriptor_t, out var weightSpaceSize);
            _weightsAndBiases = GetBuffer(weightSpaceSize);
            _weightsAndBiasesGradients = GetBuffer(weightSpaceSize);

            int expectedWeightAndBiasCount = _rnnDescriptor.Weight_ax_count + _rnnDescriptor.Weight_aa_count + _rnnDescriptor.Bias_count;
            int observedWeighAndBiasCount = _weightsAndBiases.Count;
            if (observedWeighAndBiasCount != expectedWeightAndBiasCount)
            {
                throw new ArgumentException("expecting "+expectedWeightAndBiasCount+" weights and bias but got "+ observedWeighAndBiasCount);
            }

            _optimizer = GetOptimizer(_weightsAndBiases.Shape, null);
        }
        #endregion

        #region forward and backward propagation

        /// <summary>
        /// 
        /// </summary>
        /// <param name="allX"></param>
        /// <param name="yTakingIntoAccountReturnSequenceFlag">
        /// if _returnSequences == true
        ///     Shape = (batchSize, timeSteps, Units)
        /// else
        ///     Shape = (batchSize, Units)
        /// </param>
        /// <param name="isTraining"></param>
        public override void ForwardPropagation(List<Tensor> allX, Tensor yTakingIntoAccountReturnSequenceFlag, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];
            var batchSize = x.Shape[0];                 //x.Shape[0] : batch size : number of sentences 
            int timeSteps = x.Shape[1];                 //x.Shape[1] : number of time steps
            Debug.Assert(x.Shape[2] == Features);       //x.Shape[2] : number of distinct words (_features)


            var y = _returnSequences ? yTakingIntoAccountReturnSequenceFlag : GetFloatTensor(new[] { batchSize, timeSteps, Units });

            var dataType = cudnnDataType_t.CUDNN_DATA_FLOAT;
            var xDesc = Network.GpuWrapper.RNNDataDesc(dataType, x.Shape);
            var yDesc = Network.GpuWrapper.RNNDataDesc(dataType, y.Shape);
            var hDesc = Network.GpuWrapper.TensorDesc(dataType, new[] { _rnnDescriptor.numLayers, batchSize, _rnnDescriptor.hiddenSize });

            var fMode = isTraining
                ? cudnnForwardMode_t.CUDNN_FWD_MODE_TRAINING
                : cudnnForwardMode_t.CUDNN_FWD_MODE_INFERENCE;
            var res = CudnnWrapper.cudnnGetRNNTempSpaceSizes(
                Network.GpuWrapper.CudnnHandle,
                _cudnnRNNDescriptor_t,
                fMode,
                xDesc,
                out var workSpaceSize,
                out var reserveSpaceSize);
            GPUWrapper.CheckStatus(res);
            GetBuffer(ref _workSpace, workSpaceSize);
            GetBuffer(ref _reserveSpace, reserveSpaceSize);

            var devSeqLengths = stackalloc int[batchSize];
            GPUWrapper.FillWithSameValue(devSeqLengths, batchSize, timeSteps);

            res = CudnnWrapper.cudnnRNNForward(
                Network.GpuWrapper.CudnnHandle,
                _cudnnRNNDescriptor_t,
                fMode,
                devSeqLengths,
                xDesc,
                x,
                yDesc,
                y,
                hDesc,
                IntPtr.Zero,
                IntPtr.Zero,
                new cudnnTensorDescriptor_t(),
                IntPtr.Zero,
                IntPtr.Zero,
                _weightsAndBiases.CapacityInBytes,
                _weightsAndBiases,
                workSpaceSize,
                _workSpace,                     //needed both for training and inference
                reserveSpaceSize,
                _reserveSpace);                 //needed only for training
            GPUWrapper.CheckStatus(res);

            //TODO : always free '_workSpace' at the end of forward propagation 
            //FreeFloatTensor(ref _workSpace);
            if (!isTraining)
            {
                FreeFloatTensor(ref _workSpace);
                FreeFloatTensor(ref _reserveSpace);
            }

            if (!_returnSequences)
            {
                y.From_NCH_to_NH(yTakingIntoAccountReturnSequenceFlag, timeSteps - 1);
                FreeFloatTensor(ref y);
            }
        }

        public override void BackwardPropagation(List<Tensor> allX, Tensor yTakingIntoAccountReturnSequenceFlag, Tensor dyTakingIntoAccountReturnSequenceFlag, List<Tensor> dxList)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];                // x shape  :     (batchSize, timeSteps, features)
            var batchSize = x.Shape[0];
            var timeSteps = x.Shape[1];

            Tensor y;
            Tensor dy;
            if (!_returnSequences)
            {
                Debug.Assert(yTakingIntoAccountReturnSequenceFlag.Shape.Length == 2);
                Debug.Assert(yTakingIntoAccountReturnSequenceFlag.Shape[0] == batchSize);
                Debug.Assert(yTakingIntoAccountReturnSequenceFlag.Shape[1] == Units);
                y = GetFloatTensor(new[] { batchSize, timeSteps, Units });
                yTakingIntoAccountReturnSequenceFlag.From_NH_to_NCH(y, timeSteps - 1);
                dy = GetFloatTensor(y.Shape);
                dyTakingIntoAccountReturnSequenceFlag.From_NH_to_NCH(dy, timeSteps - 1);
            }
            else
            {
                Debug.Assert(yTakingIntoAccountReturnSequenceFlag.Shape.Length == 3);
                Debug.Assert(yTakingIntoAccountReturnSequenceFlag.Shape[0] == batchSize);
                Debug.Assert(yTakingIntoAccountReturnSequenceFlag.Shape[1] == timeSteps);
                Debug.Assert(yTakingIntoAccountReturnSequenceFlag.Shape[2] == Units);
                y = yTakingIntoAccountReturnSequenceFlag;
                dy = dyTakingIntoAccountReturnSequenceFlag;
            }



            var dataType = cudnnDataType_t.CUDNN_DATA_FLOAT;
            var xDesc = Network.GpuWrapper.RNNDataDesc(dataType, x.Shape);
            var yDesc = Network.GpuWrapper.RNNDataDesc(dataType, dy.Shape);
            var hDesc = Network.GpuWrapper.TensorDesc(dataType, new[] { _rnnDescriptor.numLayers, batchSize, _rnnDescriptor.hiddenSize });

            var devSeqLengths = stackalloc int[batchSize];
            GPUWrapper.FillWithSameValue(devSeqLengths, batchSize, timeSteps);

            Debug.Assert(_reserveSpace != null);
            Debug.Assert(_workSpace != null);

            var dx = dxList[0];
            if (dx == null)
            {
                dx = GetFloatTensor(x.Shape);
            }

            var res = CudnnWrapper.cudnnRNNBackwardData_v8(
                Network.GpuWrapper.CudnnHandle,
                _cudnnRNNDescriptor_t,
                devSeqLengths,
                yDesc,
                y,
                dy,
                xDesc,
                dx,
                hDesc,
                IntPtr.Zero,
                IntPtr.Zero,
                IntPtr.Zero,
                new cudnnTensorDescriptor_t(),
                IntPtr.Zero,
                IntPtr.Zero,
                IntPtr.Zero,
                _weightsAndBiases.CapacityInBytes,
                _weightsAndBiases,
                _workSpace.CapacityInBytes,
                _workSpace,
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
                IntPtr.Zero,
                yDesc,
                y,
                _weightsAndBiasesGradients.CapacityInBytes,
                _weightsAndBiasesGradients,
                _workSpace.CapacityInBytes,
                _workSpace,
                _reserveSpace.CapacityInBytes,
                _reserveSpace);
            GPUWrapper.CheckStatus(res);

            FreeFloatTensor(ref _workSpace);
            FreeFloatTensor(ref _reserveSpace);
            if (dxList[0] == null)
            {
                FreeFloatTensor(ref dx);
            }
            if (!_returnSequences)
            {
                FreeFloatTensor(ref y);
                FreeFloatTensor(ref dy);
            }
        }
        public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
        {
            var prevLayerNX = PrevLayer.n_x;
            Weights_ax.RandomMatrixNormalDistribution(Rand, 0.0 /* mean */, Math.Sqrt(2.0 / prevLayerNX) /*stdDev*/);
            Weights_aa.RandomMatrixNormalDistribution(Rand, 0.0 /* mean */, Math.Sqrt(2.0 / prevLayerNX) /*stdDev*/);

            if (_rnnDescriptor.biasMode != cudnnRNNBiasMode_t.CUDNN_RNN_NO_BIAS)
            {
                var biasShape = new[] { (_rnnDescriptor.biasMode == cudnnRNNBiasMode_t.CUDNN_RNN_DOUBLE_BIAS) ? 2 : 1, Units };
                var bias = _weightsAndBiases.Slice(Features * Units + Units * Units, biasShape);
                bias.ZeroMemory();
            }
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
        public override Tensor Bias => null;
        public override Tensor WeightGradients => _weightsAndBiasesGradients;
        public override Tensor BiasGradients => null;
        //public override Tensor Weights => _weightsAndBiases.Slice(0, new[] { Features + Units, Units });
        public Tensor Weights_ax => _weightsAndBiases.Slice(0, new[] { Features, Units });
        public Tensor Weights_aa => _weightsAndBiases.Slice(Features * Units, new [] { Units, Units });
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
        public override int DisableBias()
        {
            return 0; //?D
        }
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(Features), Features)
                .Add(nameof(Units), Units)
                .Add(nameof(_returnSequences), _returnSequences)
                .ToString();
        }
        public static SimpleRnnLayerGPU Deserialize(IDictionary<string, object> serialized, Network network)
        {
            Debug.Assert(network.UseGPU);
            return new SimpleRnnLayerGPU(
                (int)serialized[nameof(Features)],
                (int)serialized[nameof(Units)],
                (bool)serialized[nameof(_returnSequences)],
                (bool)serialized[nameof(Trainable)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion
        public override int[] OutputShape(int batchSize)
        {
            if (_returnSequences)
            {
                int timeSteps = PrevLayer.OutputShape(1)[1];
                return new[] { batchSize, timeSteps, Units };
            }
            //only the last output
            return new[] { batchSize, Units };
        }

    }
}