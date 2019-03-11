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

        public Sgd(Network network, int[] weightShape, int[] biasShape): base(network.Config)
        {
            _velocityWeight = network.NewTensor(weightShape, nameof(_velocityWeight));
            _velocityBias = network.NewTensor(biasShape, nameof(_velocityBias));
            _iterations = 0;
        }
        public override List<Tensor> EmbeddedTensors => new List<Tensor> { _velocityWeight, _velocityBias};
        public override void UpdateWeights(double learningRate, int batchSize, Tensor weights, Tensor weightGradients, Tensor bias, Tensor biasGradient)
        {
            Debug.Assert(weights.SameShape(weightGradients));
            Debug.Assert(bias == null || bias.SameShape(biasGradient));
            var momentum = _networkConfig.SGD_momentum;
            var decay = _networkConfig.SGD_decay;
            var useNesterov = _networkConfig.SGD_usenesterov;
            ++_iterations;
            if (decay > 0)
            {
                learningRate *= 1 / (1 + decay * _iterations);
            }
            var ponderedLearningRate = PonderedLearning(learningRate, batchSize);
            weights.UpdateSGDOptimizer(ponderedLearningRate, momentum, decay, useNesterov, weightGradients, _velocityWeight);
            bias?.UpdateSGDOptimizer(ponderedLearningRate, momentum, decay, useNesterov, biasGradient, _velocityBias);
        }
        public static Optimizer DeserializeSGD(NetworkConfig networkConfig, IDictionary<string, object> serialized)
        {
            return serialized.ContainsKey(nameof(_velocityWeight)) ? new Sgd(networkConfig, serialized) : null;
        }
        public override string Serialize()
        {
            return new Serializer()
                .Add(_velocityWeight).Add(_velocityBias)
                .ToString();
        }
        private Sgd(NetworkConfig networkConfig, IDictionary<string, object> serialized) : base(networkConfig)
        {
            _velocityWeight = (Tensor)serialized[nameof(_velocityWeight)];
            _velocityBias = (Tensor)serialized[nameof(_velocityBias)];
        }
    }
}