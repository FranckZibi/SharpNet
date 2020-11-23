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

        private int InputSize => _rnnDescriptor.inputSize;       //number of distinct words in the dictionary  = Features
        private int HiddenSize => _rnnDescriptor.hiddenSize;    //dimensionality of the output space = Units

        #region constructor
        public SimpleRnnLayerGPU(int units, bool returnSequences, bool trainable, Network network, string layerName) : base(network, layerName)
        {
            _returnSequences = returnSequences;
            Trainable = trainable;
            int inputSize = PrevLayer.OutputShape(1)[2]; // InputSize = Features

            uint auxFlags = 0;
            //auxFlags |= CudnnWrapper.CUDNN_RNN_PADDED_IO_ENABLED;

            _rnnDescriptor = new RNNDescriptor(
                cudnnRNNAlgo_t.CUDNN_RNN_ALGO_STANDARD,
                cudnnRNNMode_t.CUDNN_RNN_TANH,
                cudnnRNNBiasMode_t.CUDNN_RNN_SINGLE_INP_BIAS,
                cudnnDirectionMode_t.CUDNN_UNIDIRECTIONAL,
                cudnnRNNInputMode_t.CUDNN_LINEAR_INPUT,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
                cudnnDataType_t.CUDNN_DATA_FLOAT,
                cudnnMathType_t.CUDNN_DEFAULT_MATH,
                inputSize,  /* = features */
                units,      /* hiddenSize */
                units,      /* projSize*/
                1,          /* numLayers */
                0.0,
                auxFlags);

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
            var x = allX[0];
            var batchSize = x.Shape[0];                 
            int timeSteps = x.Shape[1];                 
            Debug.Assert(x.Shape[2] == InputSize);       
            Debug.Assert(y.Shape.Last() == HiddenSize);       

            var xRnnData = GetFloatTensor(new[] { timeSteps, batchSize, InputSize });
            x.Switch_First_2_axis(xRnnData);
            var yRnnData = GetFloatTensor(new[] { timeSteps, batchSize, HiddenSize });
            var dataType = cudnnDataType_t.CUDNN_DATA_FLOAT;
            var xRnnDataDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, InputSize);
            var yRnnDataDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, HiddenSize);
            var hDesc = Network.GpuWrapper.TensorDesc(dataType, new[] { _rnnDescriptor.numLayers, batchSize, HiddenSize });
            //var hDesc = new cudnnTensorDescriptor_t();


            var fMode = isTraining
                ? cudnnForwardMode_t.CUDNN_FWD_MODE_TRAINING
                : cudnnForwardMode_t.CUDNN_FWD_MODE_INFERENCE;

            var res = CudnnWrapper.cudnnGetRNNTempSpaceSizes(
                Network.GpuWrapper.CudnnHandle,
                _cudnnRNNDescriptor_t,
                fMode,
                xRnnDataDesc,
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
                xRnnDataDesc,
                xRnnData,
                yRnnDataDesc,
                yRnnData,
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

            //We need to convert 'yRnnData' tensor to output 'y' tensor
            // yRnnData shape:
            //      (timeSteps, batchSize, hiddenSize = Units)
            // y shape:
            //      (batchSize, timeSteps, hiddenSize = Units)      when _returnSequences == true
            //      (batchSize, hiddenSize = Units)                 when _returnSequences == false
            if (_returnSequences)
            {
                yRnnData.Switch_First_2_axis(y);
            }
            else
            {
                yRnnData.ElementSlice(timeSteps - 1).CopyTo(y);
            }
            FreeFloatTensor(ref xRnnData);
            FreeFloatTensor(ref yRnnData);
        }

        public override void BackwardPropagation(List<Tensor> allX, Tensor y, Tensor dy, List<Tensor> dxList)
        {
            Debug.Assert(allX.Count == 1);
            var x = allX[0];                // x shape  :     (batchSize, timeSteps, features)
            var batchSize = x.Shape[0];
            var timeSteps = x.Shape[1];
            Debug.Assert(x.Shape[2] == InputSize);
            Debug.Assert(dy.Shape.Last() == HiddenSize);


            //we need to convert 'x' tensor shape to 'xRnnData' shape
            // x shape:
            //      (batchSize, timeSteps, features)
            // xRnnData shape:
            //      (timeSteps, batchSize, features)
            var xRnnData = GetFloatTensor(new[] { timeSteps, batchSize, InputSize });
            x.Switch_First_2_axis(xRnnData);

            //we need to convert 'y' (& 'dy') tensor shape to 'yRnnData' (& 'dyRnnData') shape
            // y/dy shape:
            //      (batchSize, timeSteps, hiddenSize = Units)      when _returnSequences == true
            //      (batchSize, hiddenSize = Units)                 when _returnSequences == false
            // yRnnData/dyRnnData shape:
            //      (timeSteps, batchSize, hiddenSize = Units)
            var yRnnData = GetFloatTensor(new[] { timeSteps, batchSize, HiddenSize });
            var dyRnnData = GetFloatTensor(yRnnData.Shape);
            if (_returnSequences)
            {
                y.Switch_First_2_axis(yRnnData);
                dy.Switch_First_2_axis(dyRnnData);
            }
            else
            {
                yRnnData.ZeroMemory();
                y.CopyTo(yRnnData.ElementSlice(timeSteps - 1));
                dyRnnData.ZeroMemory();
                dy.CopyTo(dyRnnData.ElementSlice(timeSteps - 1));
            }

            var dataType = cudnnDataType_t.CUDNN_DATA_FLOAT;
            var xDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, InputSize);
            var yRnnDataDesc = Network.GpuWrapper.RNNDataDesc(dataType, timeSteps, batchSize, HiddenSize);
            var hDesc = Network.GpuWrapper.TensorDesc(dataType, new[] { _rnnDescriptor.numLayers, batchSize, HiddenSize });

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
                yRnnDataDesc,
                yRnnData,
                dyRnnData,
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
                xRnnData,
                hDesc,
                IntPtr.Zero,
                yRnnDataDesc,
                yRnnData,
                _weightsAndBiasesGradients.CapacityInBytes,
                _weightsAndBiasesGradients,
                _workSpace.CapacityInBytes,
                _workSpace,
                _reserveSpace.CapacityInBytes,
                _reserveSpace);
            GPUWrapper.CheckStatus(res);

            FreeFloatTensor(ref _workSpace);
            FreeFloatTensor(ref _reserveSpace);
            FreeFloatTensor(ref xRnnData);
            FreeFloatTensor(ref yRnnData);
            FreeFloatTensor(ref dyRnnData);
            if (dxList[0] == null)
            {
                FreeFloatTensor(ref dx);
            }
        }
        public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
        {
            var prevLayerNX = PrevLayer.n_x;
            Weights_ax.RandomMatrixNormalDistribution(Rand, 0.0 /* mean */, Math.Sqrt(2.0 / prevLayerNX) /*stdDev*/);
            Weights_aa.RandomMatrixNormalDistribution(Rand, 0.0 /* mean */, Math.Sqrt(2.0 / prevLayerNX) /*stdDev*/);

            if (_rnnDescriptor.biasMode != cudnnRNNBiasMode_t.CUDNN_RNN_NO_BIAS)
            {
                var biasShape = new[] { (_rnnDescriptor.biasMode == cudnnRNNBiasMode_t.CUDNN_RNN_DOUBLE_BIAS) ? 2 : 1, HiddenSize };
                var bias = _weightsAndBiases.Slice(InputSize * HiddenSize + HiddenSize * HiddenSize, biasShape);
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
        public Tensor Weights_ax => _weightsAndBiases.Slice(0, new[] { InputSize, HiddenSize });
        public Tensor Weights_aa => _weightsAndBiases.Slice(InputSize * HiddenSize, new [] { HiddenSize, HiddenSize });

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
                .Add(nameof(HiddenSize), HiddenSize)
                .Add(nameof(_returnSequences), _returnSequences)
                .ToString();
        }
        public static SimpleRnnLayerGPU Deserialize(IDictionary<string, object> serialized, Network network)
        {
            Debug.Assert(network.UseGPU);
            return new SimpleRnnLayerGPU((int)serialized[nameof(HiddenSize)],
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
                return new[] { batchSize, timeSteps, HiddenSize };
            }
            //only the last output
            return new[] { batchSize, HiddenSize };
        }

    }
}