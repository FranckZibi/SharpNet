using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Optimizers
{
    public class Sgd : Optimizer
    {
        #region private fields
        private int _iterations;
        private readonly double _SGD_momentum;
        private readonly bool _SGD_usenesterov;
        private readonly Tensor _velocityWeight;    // same as 'Weights'
        private readonly Tensor _velocityBias;      // same as 'Bias'
        #endregion

        public Sgd(Network network, double SGD_momentum, bool SGD_usenesterov, int[] weightShape, int[] biasShapeIfAny)
        {
            _iterations = 0;
            _SGD_momentum = SGD_momentum;
            _SGD_usenesterov = SGD_usenesterov;
            _velocityWeight = network.NewNotInitializedTensor(weightShape, _velocityWeight, nameof(_velocityWeight));
            _velocityBias = (biasShapeIfAny==null)?null:network.NewNotInitializedTensor(biasShapeIfAny, _velocityBias, nameof(_velocityBias));
            ZeroMemory();
        }
        

        public override bool Equals(Optimizer other, double epsilon, string id, ref string errors)
        {
            if (!Utils.Equals(GetType(), other.GetType(), id + ":GetType", ref errors))
            {
                return false;
            }
            var b = (Sgd)other;
            return 
                      Utils.Equals(_iterations, b._iterations, id + ":_iterations", ref errors)
                   && Utils.Equals(_SGD_momentum, b._SGD_momentum, epsilon, id + ":_SGD_momentum", ref errors)
                   && Utils.Equals(_SGD_usenesterov, b._SGD_usenesterov, id + ":_SGD_usenesterov", ref errors)
                   && _velocityWeight.Equals(b._velocityWeight, epsilon, id + ":_velocityWeight", ref errors)
                   && _velocityBias.Equals(b._velocityBias, epsilon, id + ":_velocityBias", ref errors);
        }
        public override List<Tensor> EmbeddedTensors => new List<Tensor> { _velocityWeight, _velocityBias};
        public override void UpdateWeights(double learningRate, int batchSize, Tensor weights, Tensor weightGradients, Tensor bias, Tensor biasGradient)
        {
            Debug.Assert(weights.SameShape(weightGradients));
            Debug.Assert(bias == null || bias.SameShape(biasGradient));
            ++_iterations;
            var ponderedLearningRate = PonderedLearning(learningRate, batchSize);
            weights.UpdateSGDOptimizer(ponderedLearningRate, _SGD_momentum, _SGD_usenesterov, weightGradients, _velocityWeight);
            bias?.UpdateSGDOptimizer(ponderedLearningRate, _SGD_momentum, _SGD_usenesterov, biasGradient, _velocityBias);
        }
        public static Optimizer DeserializeSGD(IDictionary<string, object> serialized)
        {
            return serialized.ContainsKey(nameof(_velocityWeight)) ? new Sgd(serialized) : null;
        }
        public override string Serialize()
        {
            return new Serializer()
                .Add(nameof(_iterations), _iterations)
                .Add(nameof(_SGD_momentum), _SGD_momentum)
                .Add(nameof(_SGD_usenesterov), _SGD_usenesterov)
                .Add(_velocityWeight).Add(_velocityBias)
                .ToString();
        }

        public override Optimizer Clone(Network newNetwork) { return new Sgd(this, newNetwork); }
        private Sgd(Sgd toClone, Network newNetwork)
        {
            _iterations = toClone._iterations;
            _SGD_momentum = toClone._SGD_momentum;
            _SGD_usenesterov = toClone._SGD_usenesterov;
            _velocityWeight = toClone._velocityWeight?.Clone(newNetwork.GpuWrapper);
            _velocityBias = toClone._velocityBias?.Clone(newNetwork.GpuWrapper);

        }

        private Sgd(IDictionary<string, object> serialized)
        {
            serialized.TryGet(nameof(_iterations), out _iterations);
            serialized.TryGet(nameof(_SGD_momentum), out _SGD_momentum);
            serialized.TryGet(nameof(_SGD_usenesterov), out _SGD_usenesterov);
            serialized.TryGet(nameof(_velocityWeight), out _velocityWeight);
            serialized.TryGet(nameof(_velocityBias), out _velocityBias);
        }
    }
}