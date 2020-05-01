using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public class Sgd : Optimizer
    {
        #region private fields
        private int _iterations;
        private readonly TensorMemoryPool _memoryPool;
        private readonly double _SGD_momentum;
        private readonly bool _SGD_usenesterov;
        private readonly Tensor _velocityWeight;  // same dimension as 'Weights'
        [CanBeNull] private readonly Tensor _velocityBias;  // same dimension as 'Bias'
        #endregion

        public Sgd(TensorMemoryPool memoryPool, double SGD_momentum, bool SGD_usenesterov, int[] weightShape, int[] biasShapeIfAny)
        {
            _iterations = 0;
            _memoryPool = memoryPool;
            _SGD_momentum = SGD_momentum;
            _SGD_usenesterov = SGD_usenesterov;
            _memoryPool.GetFloatTensor(ref _velocityWeight, weightShape);
            if (biasShapeIfAny != null)
            {
                _memoryPool.GetFloatTensor(ref _velocityBias, biasShapeIfAny);
            }
            ZeroMemory();
        }
        
        public override bool Equals(Optimizer other, double epsilon, string id, ref string errors)
        {
            if (!Utils.Equals(GetType(), other.GetType(), id + nameof(GetType), ref errors))
            {
                return false;
            }
            var b = (Sgd)other;
            return 
                      Utils.Equals(_iterations, b._iterations, id + nameof(_iterations), ref errors)
                   && Utils.Equals(_SGD_momentum, b._SGD_momentum, epsilon, id + nameof(_SGD_momentum), ref errors)
                   && Utils.Equals(_SGD_usenesterov, b._SGD_usenesterov, id + nameof(_SGD_usenesterov), ref errors)
                   && _velocityWeight.Equals(b._velocityWeight, epsilon, id + nameof(_velocityWeight), ref errors)
                   && _velocityBias.Equals(b._velocityBias, epsilon, id + nameof(_velocityBias), ref errors);
        }
        public override List<Tensor> EmbeddedTensors
        {
            get
            {
                var result = new List<Tensor> {_velocityWeight, _velocityBias};
                result.RemoveAll(t => t == null);
                return result;
            }
        }

        public override void UpdateWeights(double learningRate, int batchSize, Tensor weights, Tensor weightGradients, Tensor bias, Tensor biasGradient)
        {
            Debug.Assert(weights.SameShape(weightGradients));
            Debug.Assert(bias == null || bias.SameShape(biasGradient));
            ++_iterations;
            var ponderedLearningRate = PonderedLearning(learningRate, batchSize);
            weights.UpdateSGDOptimizer(ponderedLearningRate, _SGD_momentum, _SGD_usenesterov, weightGradients, _velocityWeight);
            bias?.UpdateSGDOptimizer(ponderedLearningRate, _SGD_momentum, _SGD_usenesterov, biasGradient, _velocityBias);
        }

        public override void Dispose()
        {
            if (_isDisposed)
            {
                return;
            }
            _isDisposed = true;
            base.Dispose();
            EmbeddedTensors.ForEach(t=>_memoryPool.FreeFloatTensor(t));
        }

        #region serialization
        public override string Serialize()
        {
            return new Serializer()
                .Add(nameof(_iterations), _iterations)
                .Add(nameof(_SGD_momentum), _SGD_momentum)
                .Add(nameof(_SGD_usenesterov), _SGD_usenesterov)
                .Add(nameof(_velocityWeight), _velocityWeight)
                .Add(nameof(_velocityBias), _velocityBias)
                .ToString();
        }
        public static Optimizer DeserializeSGD(IDictionary<string, object> serialized)
        {
            return serialized.ContainsKey(nameof(_velocityWeight)) ? new Sgd(serialized) : null;
        }
        private Sgd(IDictionary<string, object> serialized)
        {
            serialized.TryGet(nameof(_iterations), out _iterations);
            serialized.TryGet(nameof(_SGD_momentum), out _SGD_momentum);
            serialized.TryGet(nameof(_SGD_usenesterov), out _SGD_usenesterov);
            serialized.TryGet(nameof(_velocityWeight), out _velocityWeight);
            serialized.TryGet(nameof(_velocityBias), out _velocityBias);
        }
        #endregion
    }
}
