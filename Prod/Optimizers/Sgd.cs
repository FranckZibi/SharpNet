using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public class Sgd : Optimizer
    {
        #region private fields
        private readonly Tensor _velocityWeight;    // same as 'Weights'
        private readonly Tensor _velocityBias;      // same as 'Bias'
        private int _iterations;
        #endregion

        public Sgd(Network network, int[] weightShape, int[] biasShapeIfAny): base(network.Config)
        {
            _velocityWeight = network.NewNotInitializedTensor(weightShape, _velocityWeight, nameof(_velocityWeight));
            _velocityBias = (biasShapeIfAny==null)?null:network.NewNotInitializedTensor(biasShapeIfAny, _velocityBias, nameof(_velocityBias));
            ZeroMemory();
            _iterations = 0;
        }
        public override bool Equals(Optimizer other, double epsilon, string id, ref string errors)
        {
            if (!Utils.Equals(GetType(), other.GetType(), id + ":GetType", ref errors))
            {
                return false;
            }
            var b = (Sgd)other;
            return Utils.Equals(_iterations, b._iterations, id + ":_iterations", ref errors)
                   && _velocityWeight.Equals(b._velocityWeight, epsilon, id + ":_velocityWeight", ref errors)
                   && _velocityBias.Equals(b._velocityBias, epsilon, id + ":_velocityBias", ref errors);
        }
        public override List<Tensor> EmbeddedTensors => new List<Tensor> { _velocityWeight, _velocityBias};
        public override void UpdateWeights(double learningRate, int batchSize, Tensor weights, Tensor weightGradients, Tensor bias, Tensor biasGradient)
        {
            Debug.Assert(weights.SameShape(weightGradients));
            Debug.Assert(bias == null || bias.SameShape(biasGradient));
            var momentum = _networkConfig.SGD_momentum;
            var useNesterov = _networkConfig.SGD_usenesterov;
            ++_iterations;
            var ponderedLearningRate = PonderedLearning(learningRate, batchSize);
            weights.UpdateSGDOptimizer(ponderedLearningRate, momentum, useNesterov, weightGradients, _velocityWeight);
            bias?.UpdateSGDOptimizer(ponderedLearningRate, momentum, useNesterov, biasGradient, _velocityBias);
        }
        public static Optimizer DeserializeSGD(NetworkConfig networkConfig, IDictionary<string, object> serialized)
        {
            return serialized.ContainsKey(nameof(_velocityWeight)) ? new Sgd(networkConfig, serialized) : null;
        }
        public override string Serialize()
        {
            return new Serializer()
                .Add(nameof(_iterations), _iterations)
                .Add(_velocityWeight).Add(_velocityBias)
                .ToString();
        }
        private Sgd(NetworkConfig networkConfig, IDictionary<string, object> serialized) : base(networkConfig)
        {
            serialized.TryGet(nameof(_iterations), out _iterations);
            serialized.TryGet(nameof(_velocityWeight), out _velocityWeight);
            serialized.TryGet(nameof(_velocityBias), out _velocityBias);
        }
    }
}