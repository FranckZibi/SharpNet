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
        private readonly double _momentum;
        private readonly double _epsilon;
        //Weights (gammas, scale)
        private readonly Tensor _bnScale;                       // (1, C, H, W) or (1, C, 1, 1) : depending on previous layer
        //WeightGradients
        private readonly Tensor _resultBnScaleDiff;             // same as '_bnScale"
        //Bias (betas, offset)
        private readonly Tensor _bnBias;                        // same as '_bnScale"
        //BiasGradients
        private readonly Tensor _resultBnBiasDiff;              // same as '_bnScale"
        private readonly Tensor _resultRunningMean;             // same as '_bnScale"
        private readonly Tensor _resultRunningVariance;         // same as '_bnScale"
        private readonly Tensor _resultSaveMean;                // same as '_bnScale"
        private readonly Tensor _resultSaveVariance;            // same as '_bnScale"
        private readonly Optimizer _optimizer;                  //Adam or SGD optimizer or Vanilla SGF
        #endregion
        public override Tensor y { get; protected set; }        // (batchSize, C, H, W)

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public BatchNormalizationLayer(double momentum, double epsilon, Network network, string layerName = "") : base(network, layerName)
        {
            _momentum = momentum;
            _epsilon = epsilon;
            var scaleAndBiasShape = ScaleAndBiasShape();
            _bnScale = Network.NewNotInitializedTensor(scaleAndBiasShape, nameof(_bnScale));
            _resultBnScaleDiff = Network.NewNotInitializedTensor(scaleAndBiasShape, nameof(_resultBnScaleDiff));
            _bnBias = Network.NewNotInitializedTensor(scaleAndBiasShape, nameof(_bnBias));
            _resultBnBiasDiff = Network.NewNotInitializedTensor(scaleAndBiasShape, nameof(_resultBnBiasDiff));
            _resultRunningMean = Network.NewNotInitializedTensor(scaleAndBiasShape, nameof(_resultRunningMean));
            _resultRunningVariance = Network.NewNotInitializedTensor(scaleAndBiasShape, nameof(_resultRunningVariance));
            _resultSaveMean = Network.NewNotInitializedTensor(scaleAndBiasShape, nameof(_resultSaveMean));
            _resultSaveVariance = Network.NewNotInitializedTensor(scaleAndBiasShape, nameof(_resultSaveVariance));
            _optimizer = Network.GetOptimizer(_bnScale.Shape, _bnBias.Shape);
            //no need to reset optimizer weights: it has just been done above
            ResetWeights(false);

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
            _bnScale = other._bnScale?.Clone(newNetwork.GpuWrapper);
            _resultBnScaleDiff = other._resultBnScaleDiff?.Clone(newNetwork.GpuWrapper);
            _bnBias = other._bnBias?.Clone(newNetwork.GpuWrapper);
            _resultBnBiasDiff = other._resultBnBiasDiff?.Clone(newNetwork.GpuWrapper);
            _resultRunningMean = other._resultRunningMean?.Clone(newNetwork.GpuWrapper);
            _resultRunningVariance = other._resultRunningVariance?.Clone(newNetwork.GpuWrapper);
            _resultSaveMean = other._resultSaveMean?.Clone(newNetwork.GpuWrapper);
            _resultSaveVariance = other._resultSaveVariance?.Clone(newNetwork.GpuWrapper);
            _optimizer = other._optimizer?.Clone(newNetwork);
        }

        public Tensor Weights => _bnScale;
        public Tensor WeightGradients => _resultBnScaleDiff;
        public Tensor Bias => _bnBias;
        public Tensor BiasGradients => _resultBnBiasDiff;

        #region serialization
        public override string Serialize()
        {
            return RootSerializer()
                .Add(nameof(_epsilon), _epsilon).Add(nameof(_momentum), _momentum)
                .Add(_bnScale).Add(_resultBnScaleDiff).Add(_bnBias).Add(_resultBnBiasDiff).Add(_resultRunningMean).Add(_resultRunningVariance).Add(_resultSaveMean)
                .Add(_resultSaveVariance)
                .Add(_optimizer?.Serialize())
                .ToString();
        }
        public BatchNormalizationLayer(IDictionary<string, object> serialized, Network network) : base(serialized, network)
        {
            _epsilon = (double)serialized[nameof(_epsilon)];
            _momentum = (double)serialized[nameof(_momentum)];
            _bnScale = (Tensor)serialized[nameof(_bnScale)];
            _resultBnScaleDiff = (Tensor)serialized[nameof(_resultBnScaleDiff)];
            _bnBias = (Tensor)serialized[nameof(_bnBias)];
            _resultBnBiasDiff = (Tensor)serialized[nameof(_resultBnBiasDiff)];
            _resultRunningMean = (Tensor)serialized[nameof(_resultRunningMean)];
            _resultRunningVariance = (Tensor)serialized[nameof(_resultRunningVariance)];
            _resultSaveMean = (Tensor)serialized[nameof(_resultSaveMean)];
            _resultSaveVariance = (Tensor)serialized[nameof(_resultSaveVariance)];
            _optimizer = Optimizer.ValueOf(network.Config, serialized);
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
            x.BatchNormalization(y, _bnScale, _bnBias, _momentum, _resultRunningMean, _resultRunningVariance, LayerBatchNormalizationMode(), _epsilon, _resultSaveMean, _resultSaveVariance, isTraining);
        }
        public override void BackwardPropagation(Tensor dy, List<Tensor> dx)
        {
            var x = PrevLayer.y;
            x.BatchNormalizationBackward(dy, dx[0], _bnScale, _resultBnScaleDiff, _resultBnBiasDiff, LayerBatchNormalizationMode(), _epsilon, _resultSaveMean, _resultSaveVariance);
        }
        public override void UpdateWeights(double learningRate)
        {
            Debug.Assert(_bnScale.SameShape(_resultBnScaleDiff));
            Debug.Assert(_bnBias.SameShape(_resultBnBiasDiff));
            if (!Trainable)
            {
                return;
            }
            var batchSize = y.Shape[0];
            _optimizer.UpdateWeights(learningRate, batchSize, _bnScale, _resultBnScaleDiff, _bnBias, _resultBnBiasDiff);
        }
        public override void ResetWeights(bool resetAlsoOptimizerWeights = true)
        {
            _bnScale.NewSameValueTensor(1.0);
            _resultBnScaleDiff.ZeroMemory();
            _bnBias.ZeroMemory();
            _resultBnBiasDiff.ZeroMemory();
            _resultRunningMean.ZeroMemory();
            _resultRunningVariance.NewSameValueTensor(1.0);
            _resultSaveMean.ZeroMemory();
            _resultSaveVariance.NewSameValueTensor(1.0);
            if (resetAlsoOptimizerWeights)
            {
                _optimizer.ZeroMemory();
            }
        }
        public override void LoadFromH5Dataset(Dictionary<string, Tensor> h5FileDataset)
        {
            LoadFromH5Dataset(h5FileDataset, "beta:0", _bnBias);
            LoadFromH5Dataset(h5FileDataset, "moving_mean:0", _resultRunningMean);
            LoadFromH5Dataset(h5FileDataset, "gamma:0", _bnScale);
            LoadFromH5Dataset(h5FileDataset, "moving_variance:0", _resultRunningVariance);
        }

     
        public override int TotalParams => _bnScale.Count + _resultBnScaleDiff.Count + _bnBias.Count + _resultBnBiasDiff.Count;
        protected override string DefaultLayerName() { return "batch_normalization_" + (1 + NbLayerOfSameTypeBefore()); }
        public override string Type() {return "BatchNormalization"; }
        public override List<Tensor> TensorsIndependentOfBatchSize
        {
            get
            {
                var result = new List<Tensor> { _bnScale, _resultBnScaleDiff, _bnBias, _resultBnBiasDiff, _resultRunningMean, _resultRunningVariance, _resultSaveMean, _resultSaveVariance};
                result.RemoveAll(t => t == null);
                return result;
            }
        }
    }
}