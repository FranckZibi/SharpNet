using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Optimizers;

namespace SharpNet
{
    public sealed class BatchNormalizationLayer : Layer
    {
        #region Private fields

        private readonly double _momentum;
        private readonly double _epsilon;
        private readonly Tensor _bnScale;                       // (1, C, H, W) or (1, C, 1, 1) : depending on previous layer
        private readonly Tensor _resultBnScaleDiff;             // same as '_bnScale"
        private readonly Tensor _bnBias;                        // same as '_bnScale"
        private readonly Tensor _resultBnBiasDiff;              // same as '_bnScale"
        private readonly Tensor _resultRunningMean;             // same as '_bnScale"
        private readonly Tensor _resultRunningVariance;         // same as '_bnScale"
        private readonly Tensor _resultSaveMean;                // same as '_bnScale"
        private readonly Tensor _resultSaveVariance;            // same as '_bnScale"
        private readonly Optimizer _optimizer;                  //Adam or SGD optimizer or Vanilla SGF
        #endregion
        public override Tensor y { get; protected set; }        // (batchSize, C, H, W)
        public override Tensor dy { get; protected set; }       // same as 'y'

        //No need to configure the number of channels by filter: it is always the same as in previous layer
        public BatchNormalizationLayer(double momentum, double epsilon, Network network) : base(network)
        {
            _momentum = momentum;
            _epsilon = epsilon;
            var scaleAndBiasShape = ScaleAndBiasShape();
            _bnScale = Network.NewNotInitializedTensor(scaleAndBiasShape, _bnScale, nameof(_bnScale));
            _resultBnScaleDiff = Network.NewNotInitializedTensor(scaleAndBiasShape, _resultBnScaleDiff, nameof(_resultBnScaleDiff));
            _bnBias = Network.NewNotInitializedTensor(scaleAndBiasShape, _bnBias, nameof(_bnBias));
            _resultBnBiasDiff = Network.NewNotInitializedTensor(scaleAndBiasShape, _resultBnBiasDiff, nameof(_resultBnBiasDiff));
            _resultRunningMean = Network.NewNotInitializedTensor(scaleAndBiasShape, _resultRunningMean, nameof(_resultRunningMean));
            _resultRunningVariance = Network.NewNotInitializedTensor(scaleAndBiasShape, _resultRunningVariance, nameof(_resultRunningVariance));
            _resultSaveMean = Network.NewNotInitializedTensor(scaleAndBiasShape, _resultSaveMean, nameof(_resultSaveMean));
            _resultSaveVariance = Network.NewNotInitializedTensor(scaleAndBiasShape, _resultSaveVariance, nameof(_resultSaveVariance));
            _optimizer = Network.GetOptimizer(_bnScale.Shape, _bnBias.Shape);
            ResetWeights(false);

            //We disable bias for the previous layers
            var nbDisabledWeights = PreviousLayers.Select(l=>l.DisableBias()).Sum();
            if (nbDisabledWeights != 0)
            {
                Network.LogDebug(nbDisabledWeights + " weights (bias) disabled thanks to batchNorm layer " + SummaryName());
            }
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
                .Add(_bnScale).Add(_resultBnScaleDiff).Add(_bnBias).Add(_resultBnBiasDiff).Add(_resultRunningMean).Add(_resultRunningVariance).Add(_resultSaveMean).Add(_resultSaveVariance)
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
            var allAreOk = true;
            allAreOk &= Utils.Equals(_momentum, other._momentum, epsilon, id, ref errors);
            allAreOk &= Utils.Equals(_epsilon, other._epsilon, epsilon, id, ref errors);
            allAreOk &= _optimizer.Equals(other._optimizer, epsilon, id + ":Optimizer", ref errors);
            return allAreOk;
        }
        public override void ForwardPropagation(bool isTraining)
        {
            Allocate_y_dy_if_necessary();
            var x = PrevLayer.y;
            x.BatchNormalization(y, _bnScale, _bnBias, _momentum, _resultRunningMean, _resultRunningVariance, LayerBatchNormalizationMode(), _epsilon, _resultSaveMean, _resultSaveVariance, isTraining);
        }
        public override void BackwardPropagation()
        {
            //At this stage, we already know dy
            var x = PrevLayer.y;
            var dx = PrevLayer.dy;
            x.BatchNormalizationBackward(dy, dx, _bnScale, _resultBnScaleDiff, _resultBnBiasDiff, LayerBatchNormalizationMode(), _epsilon, _resultSaveMean, _resultSaveVariance);
        }
        public override void UpdateWeights(double learningRate)
        {
            Debug.Assert(_bnScale.SameShape(_resultBnScaleDiff));
            Debug.Assert(_bnBias.SameShape(_resultBnBiasDiff));
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
        public override int TotalParams => _bnScale.Count + _resultBnScaleDiff.Count + _bnBias.Count + _resultBnBiasDiff.Count;
        public override string SummaryName() { return "batch_normalization_" + (1 + NbLayerOfSameTypeBefore()); }
        public override string Type() {return "BatchNorm";}
        public override List<Tensor> TensorsIndependantOfBatchSize
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