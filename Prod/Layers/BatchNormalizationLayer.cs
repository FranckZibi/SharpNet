using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNet.Layers
{
    public sealed class BatchNormalizationLayer : Layer
    {
        #region Private fields
        /// <summary>
        /// the momentum used to compute the running mean and running variance:
        ///     runningMean[t] = (1-momentum) * currentMean  + momentum * runningMean[t-1]
        /// it is equal to:
        ///     1-exponentialAverageSmoothingFactor
        ///     (see https://en.wikipedia.org/wiki/Exponential_smoothing)
        /// The PyTorch momentum is (1 - momentum)
        /// </summary>
        private readonly double momentum;
        private readonly double eps;
        #region trainable parameters
        /// <summary>
        /// Scale (= gammas) Tensor
        /// </summary>
        private Tensor _scale;             // (1, C, H, W) or (C) : depending on previous layer
        /// <summary>
        /// Bias (= betas = offset) Tensor
        /// </summary>
        private Tensor _bias;                     // same shape as '_scale'
        #endregion
        #region non trainable parameters
        /// <summary>
        /// weighted average of all inputs (=x) mean
        /// used for inference only (updated during training)
        /// </summary>
        [NotNull] private Tensor _resultRunningMean;        // same shape as 'Scale"
        /// <summary>
        /// weighted average of all inputs (=x) variance
        /// used for inference only (updated during training)
        /// </summary>
        [NotNull] private Tensor _resultRunningVariance;    // same shape as 'Scale"
        #endregion
        #region gradients
        [NotNull] private Tensor _scaleGradients;         // same shape as 'Scale"
        [CanBeNull] private Tensor _biasGradients;        // same shape as 'Scale"
        #endregion
        /// <summary>
        /// Adam or SGD optimizer or Vanilla SGD
        /// </summary>
        [NotNull] private readonly Optimizer _optimizer;
        #region temporary buffer used for training only
        /// <summary>
        /// temporary buffer used to compute the current input (=x) mean
        /// </summary>
        [NotNull] private readonly Tensor _meanBuffer;                // same shape as '_scale'
        /// <summary>
        /// temporary buffer used to compute the current input (=x) variance
        /// </summary>
        [NotNull] private readonly Tensor _invertOfUnbiasedVolatilityBuffer;            // same shape as '_scale'
        #endregion
        #endregion

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public BatchNormalizationLayer(double momentum, double eps, bool trainable, Network network, string layerName) : base(network, layerName)
        {
            this.momentum = momentum;
            this.eps = eps;
            Trainable = trainable;

            var scaleAndBiasShape = ScaleAndBiasShape();

            //trainable parameters 
            _scale = GetFloatTensor(scaleAndBiasShape);
            _bias = GetFloatTensor(scaleAndBiasShape);
            //non trainable parameters 
            _resultRunningMean = GetFloatTensor(scaleAndBiasShape);
            _resultRunningVariance = GetFloatTensor(scaleAndBiasShape);

            //gradients
            _scaleGradients = GetFloatTensor(scaleAndBiasShape);
            _biasGradients = GetFloatTensor(scaleAndBiasShape);

            _optimizer = Sample.GetOptimizer(_scale.Shape, _bias.Shape, MemoryPool);

            //no need to reset optimizer weights: it has just been done above
            ResetParameters(false);

            //temporary buffers
            _meanBuffer = GetFloatTensor(scaleAndBiasShape);
            _invertOfUnbiasedVolatilityBuffer = GetFloatTensor(scaleAndBiasShape);

            //We disable bias for the previous layer(s)
            var nbDisabledWeights = PreviousLayers.Select(l=>l.DisableBias()).Sum();
            if (nbDisabledWeights != 0)
            {
                Log(nbDisabledWeights + " weights (bias) disabled thanks to batchNorm layer " + LayerName);
            }
        }

        #region forward and backward propagation
        public override void ForwardPropagation(List<Tensor> allX, Tensor y, bool isTraining)
        {
            Debug.Assert(allX.Count == 1);
            var exponentialAverageSmoothingFactor = 1 - momentum;
            allX[0].BatchNormalization(y, _scale, _bias, exponentialAverageSmoothingFactor, _resultRunningMean, _resultRunningVariance, LayerBatchNormalizationMode(), eps, _meanBuffer, _invertOfUnbiasedVolatilityBuffer, isTraining);
        }
        public override void BackwardPropagation(List<Tensor> allX, Tensor y_NotUsed, Tensor dy, List<Tensor> dx)
        {
            Debug.Assert(allX.Count == 1);
            Debug.Assert(y_NotUsed == null);
            var x = allX[0];
            x.BatchNormalizationBackward(dy, dx[0], _scale, _scaleGradients, _biasGradients, LayerBatchNormalizationMode(), eps, _meanBuffer, _invertOfUnbiasedVolatilityBuffer);
        }
        public override bool OutputNeededForBackwardPropagation => false;
        #endregion

        #region parameters and gradients
        public override Tensor Weights => _scale;
        public override Tensor Bias => _bias;
        public override Tensor WeightGradients => _scaleGradients;
        public override Tensor BiasGradients => _biasGradients;
        protected override Optimizer Optimizer => _optimizer;
        public override List<Tuple<Tensor, string>> Parameters
        {
            get
            {
                var result = new List<Tuple<Tensor, string>>
                             {
                                 Tuple.Create(_scale, ScaleDatasetPath),
                                 Tuple.Create(_bias, BiasDatasetPath),
                                 Tuple.Create(_resultRunningMean, RunningMeanDatasetPath),
                                 Tuple.Create(_resultRunningVariance, RunningVarianceDatasetPath)
                             };
                return result;
            }
        }
        public override int NonTrainableParams => _resultRunningMean.Count + _resultRunningVariance.Count;
        public override void ReplaceParameters(List<Tensor> newParameters)
        {
            Debug.Assert(newParameters.Count == 4);
            FreeFloatTensor(ref _scale);
            _scale = newParameters[0];
            FreeFloatTensor(ref _bias);
            _bias = newParameters[1];
            FreeFloatTensor(ref _resultRunningMean);
            _resultRunningMean = newParameters[2];
            FreeFloatTensor(ref _resultRunningVariance);
            _resultRunningVariance = newParameters[3];
        }
        public override void ResetParameters(bool resetAlsoOptimizerWeights = true)
        {
            //trainable params
            _scale.SetValue(1);
            _bias.ZeroMemory();
            //non trainable params
            _resultRunningMean.ZeroMemory();
            _resultRunningVariance.SetValue(1);
            if (resetAlsoOptimizerWeights)
            {
                _optimizer.ZeroMemory();
            }
        }
        public override void ReplaceGradients(List<Tensor> newGradients)
        {
            FreeFloatTensor(ref _scaleGradients);
            _scaleGradients = newGradients[0];
            if (_biasGradients != null)
            {
                Debug.Assert(newGradients.Count == 2);
                FreeFloatTensor(ref _biasGradients);
                _biasGradients = newGradients[1];
            }
            else
            {
                Debug.Assert(newGradients.Count == 1);
            }
        }
        public override void LoadParameters(IDictionary<string, Tensor> h5FileDataset, NetworkSample.CompatibilityModeEnum originFramework)
        {
            foreach (var layerParameters in Parameters)
            {
                var parameterId = layerParameters.Item2;
                if (h5FileDataset.ContainsKey(parameterId))
                {
                    h5FileDataset[parameterId].CopyTo(layerParameters.Item1);
                }
            }
        }
        public override IDictionary<string, CpuTensor<float>> GetParametersAsCpuFloatTensors(NetworkSample.CompatibilityModeEnum originFramework)
        {
            var result = new Dictionary<string, CpuTensor<float>>();
            result[ScaleDatasetPath] = _scale.ToCpuFloat();
            result[BiasDatasetPath] = _bias.ToCpuFloat();
            result[RunningMeanDatasetPath] = _resultRunningMean.ToCpuFloat();
            result[RunningVarianceDatasetPath] = _resultRunningVariance.ToCpuFloat();

            if (LayerBatchNormalizationMode() != cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION
                && originFramework == NetworkSample.CompatibilityModeEnum.TensorFlow
            )
            {
                Debug.Assert(_scale.Count == _scale.Shape[1]);
                var tensorFlowShape = new[] { _scale.Shape[1]};
                result[ScaleDatasetPath].ReshapeInPlace(tensorFlowShape);
                result[BiasDatasetPath].ReshapeInPlace(tensorFlowShape);
                result[RunningMeanDatasetPath].ReshapeInPlace(tensorFlowShape);
                result[RunningVarianceDatasetPath].ReshapeInPlace(tensorFlowShape);
            }

            return result;
        }


        private string ScaleDatasetPath => DatasetNameToDatasetPath("weight");
        private string BiasDatasetPath => DatasetNameToDatasetPath("bias");
        private string RunningMeanDatasetPath => DatasetNameToDatasetPath("running_mean");
        private string RunningVarianceDatasetPath => DatasetNameToDatasetPath("running_var");
        #endregion

        #region serialization
        public override string Serialize()
        {
            return RootSerializer().Add(nameof(eps), eps).Add(nameof(momentum), momentum).ToString();
        }
        public static BatchNormalizationLayer Deserialize(IDictionary<string, object> serialized, Network network)
        {
            return new BatchNormalizationLayer(
                (double)serialized[nameof(momentum)],
                (double)serialized[nameof(eps)],
                (bool)serialized[nameof(Trainable)],
                network,
                (string)serialized[nameof(LayerName)]);
        }
        public override void AddToOtherNetwork(Network otherNetwork) { AddToOtherNetwork(otherNetwork, Deserialize); }
        #endregion

        #region PyTorch support
        //see: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        public override void ToPytorchModule(List<string> constructorLines, List<string> forwardLines)
        {
            var input_shape = PreviousLayers.Count == 0 ? new[] { -1, -1, -1, -1 } : PreviousLayers[0].OutputShape(666);
            int num_features = input_shape[1];
            constructorLines.Add("self." + LayerName + " = torch.nn.BatchNorm2d(num_features=" + num_features + ", eps=" + eps.ToString(CultureInfo.InvariantCulture) + ", momentum=" + PyTorch_momentum().ToString(CultureInfo.InvariantCulture) + ")");
            UpdateForwardLines(forwardLines);
        }
        //see: https://stackoverflow.com/questions/48345857/batchnorm-momentum-convention-pytorch
        private double PyTorch_momentum() => 1-momentum;
        #endregion
        #region Tensorflow support
        // ReSharper disable once UnusedMember.Global
        public double Tensorflow_momentum() => momentum;
        #endregion

        public override string ToString()
        {
            var result = LayerName + ": " + ShapeChangeDescription();
            result += " (" + TotalParams + " neurons)";
            return result;
        }
       
        protected override List<Tensor> EmbeddedTensors(bool includeOptimizeTensors)
        {
            var result = base.EmbeddedTensors(includeOptimizeTensors);
            result.AddRange(new[] { _meanBuffer, _invertOfUnbiasedVolatilityBuffer });
            result.RemoveAll(t => t == null);
            return result;
        }

        protected override string ComputeLayerName()
        {
            return base.ComputeLayerName().Replace("batchnormalization", "batch_normalization");
        }

        //https://docs.nvidia.com/deeplearning/sdk/cudnn-archived/cudnn_701/cudnn-user-guide/index.html#cudnnBatchNormMode_t
        private cudnnBatchNormMode_t LayerBatchNormalizationMode()
        {
            //if previous layer is a dense layer
            if (PrevLayer != null && (PrevLayer.OutputShape(1).Length == 2 || PrevLayer.IsInputLayer))
            {
                //Normalization is performed per-activation. 
                //This mode is intended to be used after non-convolutional network layers. 
                //In this mode bnBias and bnScale tensor dimensions are (1, C, H, W)
                return cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION;
            }
            //Normalization is performed over N + spatial dimensions.
            //This mode is intended for use after convolutional layers(where spatial invariance is desired).
            //In this mode bnBias, bnScale tensor dimensions are (1, C, 1, 1)
            return cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL;
        }
        private int[] ScaleAndBiasShape()
        {
            var res = OutputShape(1);
            if (LayerBatchNormalizationMode() == cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION)
            {
                return res; //shape is (1, C, H, W) or (1, C)
            }
            return new[] { res[1] }; //shape is (C)
        }
    }
}