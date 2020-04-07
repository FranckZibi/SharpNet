using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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
        /// </summary>
        private readonly double _momentum;
        private readonly double _epsilon;
        /// <summary>
        /// Scale (= gammas) Tensor
        /// </summary>
        public Tensor Scale { get; }                      // (1, C, H, W) or (1, C, 1, 1) : depending on previous layer
        public Tensor ScaleGradients { get; }             // same as 'Scale"
        /// <summary>
        /// Bias (= betas = offset) Tensor
        /// </summary>
        public Tensor Bias { get; }                        // same as 'Scale"
        //BiasGradients
        public Tensor BiasGradients { get; }               // same as 'Scale"
        /// <summary>
        /// weighted average of all inputs (=x) mean 
        /// </summary>
        private readonly Tensor _resultRunningMean;        // same as 'Scale"
        /// <summary>
        /// weighted average of all inputs (=x) variance
        /// </summary>
        private readonly Tensor _resultRunningVariance;    // same as 'Scale"
        private readonly Optimizer _optimizer;             //Adam or SGD optimizer or Vanilla SGF

        /// <summary>
        /// temporary buffer used to compute the current input (=x) mean
        /// </summary>
        private readonly Tensor _meanBuffer;                // same as 'Scale"
        /// <summary>
        /// temporary buffer used to compute the current input (=x) variance
        /// </summary>
        private readonly Tensor _invertOfUnbiasedVolatilityBuffer;            // same as 'Scale"
        #endregion
        public override Tensor y { get; protected set; }        // (batchSize, C, H, W)

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public BatchNormalizationLayer(double momentum, double epsilon, Network network, string layerName) : base(network, layerName)
        {
            _momentum = momentum;
            _epsilon = epsilon;
            var scaleAndBiasShape = ScaleAndBiasShape();

            //trainable tensors 
            Scale = Network.NewNotInitializedFloatTensor(scaleAndBiasShape, nameof(Scale));
            Bias = Network.NewNotInitializedFloatTensor(scaleAndBiasShape, nameof(Bias));
            _resultRunningMean = Network.NewNotInitializedFloatTensor(scaleAndBiasShape, nameof(_resultRunningMean));
            _resultRunningVariance = Network.NewNotInitializedFloatTensor(scaleAndBiasShape, nameof(_resultRunningVariance));

            _optimizer = Network.GetOptimizer(Scale.Shape, Bias.Shape);

            //no need to reset optimizer weights: it has just been done above
            ResetWeights(false);

            //non trainable params
            ScaleGradients = Network.NewNotInitializedFloatTensor(scaleAndBiasShape, nameof(ScaleGradients));
            BiasGradients = Network.NewNotInitializedFloatTensor(scaleAndBiasShape, nameof(BiasGradients));
            _meanBuffer = Network.NewNotInitializedFloatTensor(scaleAndBiasShape, nameof(_meanBuffer));
            _invertOfUnbiasedVolatilityBuffer = Network.NewNotInitializedFloatTensor(scaleAndBiasShape, nameof(_invertOfUnbiasedVolatilityBuffer));

            //We disable bias for the previous layers
            var nbDisabledWeights = PreviousLayers.Select(l=>l.DisableBias()).Sum();
            if (nbDisabledWeights != 0)
            {
                Network.LogDebug(nbDisabledWeights + " weights (bias) disabled thanks to batchNorm layer " + LayerName);
            }
        }

        public override Layer Clone(Network newNetwork) { return new BatchNormalizationLayer(this, newNetwork); }
        private BatchNormalizationLayer(BatchNormalizationLayer other, Network newNetwork) : base(other, newNetwork)
        {
            _momentum = other._momentum;
            _epsilon = other._epsilon;

            //trainable params
            Scale = other.Scale.Clone(newNetwork.GpuWrapper);
            Bias = other.Bias.Clone(newNetwork.GpuWrapper);
            _resultRunningMean = other._resultRunningMean.Clone(newNetwork.GpuWrapper);
            _resultRunningVariance = other._resultRunningVariance.Clone(newNetwork.GpuWrapper);
            _optimizer = other._optimizer?.Clone(newNetwork);

            // non trainable params
            ScaleGradients = other.ScaleGradients?.Clone(newNetwork.GpuWrapper);
            BiasGradients = other.BiasGradients?.Clone(newNetwork.GpuWrapper);
            _meanBuffer = Network.NewNotInitializedFloatTensor(Scale.Shape, nameof(_meanBuffer));
            _invertOfUnbiasedVolatilityBuffer = Network.NewNotInitializedFloatTensor(Scale.Shape, nameof(_invertOfUnbiasedVolatilityBuffer));
        }


        #region serialization
        public override string Serialize()
        {
            return RootSerializer() // 'RootSerializer()' will also serialize layer trainable params
                .Add(nameof(_epsilon), _epsilon).Add(nameof(_momentum), _momentum)
                .Add(_optimizer?.Serialize())
                .ToString();
        }
        public BatchNormalizationLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _epsilon = (double)serialized[nameof(_epsilon)];
            _momentum = (double)serialized[nameof(_momentum)];

            //trainable params
            Scale = (Tensor)serialized[nameof(Scale)];
            Bias = (Tensor)serialized[nameof(Bias)];
            _resultRunningMean = (Tensor)serialized[nameof(_resultRunningMean)];
            _resultRunningVariance = (Tensor)serialized[nameof(_resultRunningVariance)];
            _optimizer = Optimizer.ValueOf(network.Config, serialized);

            //non trainable params
            ScaleGradients = Network.NewNotInitializedFloatTensor(Scale.Shape, nameof(ScaleGradients));
            BiasGradients = Network.NewNotInitializedFloatTensor(Scale.Shape, nameof(BiasGradients));
            _meanBuffer = Network.NewNotInitializedFloatTensor(Scale.Shape, nameof(_meanBuffer));
            _invertOfUnbiasedVolatilityBuffer = Network.NewNotInitializedFloatTensor(Scale.Shape, nameof(_invertOfUnbiasedVolatilityBuffer));
        }
        #endregion


        private int[] ScaleAndBiasShape()
        {
            var res = OutputShape(1);
            if (LayerBatchNormalizationMode() == cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION)
            {
                return res; //shape is (1, C, H, W) or (1, C)
            }
            for (int i = 2; i < res.Length; ++i)
            {
                res[i] = 1;
            }
            return res; //shape is (1, C, 1, 1)
        }
        public override bool Equals(Layer b, double epsilon, string id, ref string errors)
        {
            if (!base.Equals(b, epsilon, id, ref errors))
            {
                return false;
            }
            var other = (BatchNormalizationLayer)b;
            var equals = true;
            equals &= Utils.Equals(_momentum, other._momentum, epsilon, id, ref errors);
            equals &= Utils.Equals(_epsilon, other._epsilon, epsilon, id, ref errors);
            equals &= _optimizer.Equals(other._optimizer, epsilon, id + ":Optimizer", ref errors);
            return equals;
        }
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_if_necessary();
            var x = PrevLayer.y;
            var exponentialAverageSmoothingFactor = 1 - _momentum;
            x.BatchNormalization(y, Scale, Bias, exponentialAverageSmoothingFactor, _resultRunningMean, _resultRunningVariance, LayerBatchNormalizationMode(), _epsilon, _meanBuffer, _invertOfUnbiasedVolatilityBuffer, isTraining);
        }
        public override void BackwardPropagation(Tensor dy, List<Tensor> dx)
        {
            var x = PrevLayer.y;
            x.BatchNormalizationBackward(dy, dx[0], Scale, ScaleGradients, BiasGradients, LayerBatchNormalizationMode(), _epsilon, _meanBuffer, _invertOfUnbiasedVolatilityBuffer);
        }
        public override void UpdateWeights(double learningRate)
        {
            Debug.Assert(Scale.SameShape(ScaleGradients));
            Debug.Assert(Bias.SameShape(BiasGradients));
            if (!Trainable)
            {
                return;
            }
            var batchSize = y.Shape[0];
            _optimizer.UpdateWeights(learningRate, batchSize, Scale, ScaleGradients, Bias, BiasGradients);
        }
        public override void ResetWeights(bool resetAlsoOptimizerWeights = true)
        {
            //trainable params
            Scale.NewSameValueTensor(1.0);
            Bias.ZeroMemory();
            _resultRunningMean.ZeroMemory();
            _resultRunningVariance.NewSameValueTensor(1.0);
            
            if (resetAlsoOptimizerWeights)
            {
                _optimizer.ZeroMemory();
            }

            //no need to reset non trainable tensors
            //_meanBuffer.ZeroMemory();
            //_varianceBuffer.NewSameValueTensor(1.0);
            //ScaleGradients.ZeroMemory();
            //BiasGradients.ZeroMemory();
}
        public override void LoadFromH5Dataset(Dictionary<string, Tensor> h5FileDataset, NetworkConfig.CompatibilityModeEnum originFramework)
        {
            h5FileDataset[DatasetNameToDatasetPath("beta:0")].CopyTo(Bias);
            h5FileDataset[DatasetNameToDatasetPath("moving_mean:0")].CopyTo(_resultRunningMean);
            h5FileDataset[DatasetNameToDatasetPath("gamma:0")].CopyTo(Scale);
            h5FileDataset[DatasetNameToDatasetPath("moving_variance:0")].CopyTo(_resultRunningVariance);
        }
     
        protected override string DefaultLayerName() { return "batch_normalization_" + (1 + NbLayerOfSameTypeBefore()); }

        protected override List<Tensor> TrainableTensorsIndependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor> { Scale, Bias, _resultRunningMean, _resultRunningVariance};
                result.RemoveAll(t => t == null);
                return result;
            }
        }

        protected override List<Tensor> NonTrainableTensorsIndependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor> { ScaleGradients, BiasGradients, _meanBuffer, _invertOfUnbiasedVolatilityBuffer };
                result.RemoveAll(t => t == null);
                return result;
            }
        }
    }
}