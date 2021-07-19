using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public class Sgd : Optimizer
    {
        #region private fields
        private readonly TensorMemoryPool _memoryPool;
        private readonly double _SGD_momentum;
        private readonly bool _SGD_usenesterov;
        private readonly Tensor _velocityWeight;  // same dimension as 'Weights'
        [CanBeNull] private readonly Tensor _velocityBias;  // same dimension as 'Bias'
        #endregion

        public Sgd(TensorMemoryPool memoryPool, double SGD_momentum, bool SGD_usenesterov, int[] weightShape, int[] biasShapeIfAny)
        {
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

        public override List<Tensor> EmbeddedTensors
        {
            get
            {
                var result = new List<Tensor> {_velocityWeight, _velocityBias};
                result.RemoveAll(t => t == null);
                return result;
            }
        }

        public override void UpdateWeights(double learningRate, double maxLearningRate, int batchSize, Tensor weights,
            Tensor weightGradients,
            Tensor bias, Tensor biasGradient)
        {
            Debug.Assert(weights.SameShape(weightGradients));
            Debug.Assert(bias == null || bias.SameShape(biasGradient));
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
            EmbeddedTensors.ForEach(t=>_memoryPool?.FreeFloatTensor(t));
        }

        #region serialization
        public override string Serialize()
        {
            return new Serializer()
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
            serialized.TryGet(nameof(_SGD_momentum), out _SGD_momentum);
            serialized.TryGet(nameof(_SGD_usenesterov), out _SGD_usenesterov);
            serialized.TryGet(nameof(_velocityWeight), out _velocityWeight);
            serialized.TryGet(nameof(_velocityBias), out _velocityBias);
        }
        #endregion
    }
}
