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
        private readonly bool nesterov;
        private readonly Tensor _velocityWeight;  // same dimension as 'Weights'
        [CanBeNull] private readonly Tensor _velocityBias;  // same dimension as 'Bias'
        #endregion

        public Sgd(TensorMemoryPool memoryPool, double SGD_momentum, bool nesterov, int[] weightShape, int[] biasShapeIfAny)
        {
            _memoryPool = memoryPool;
            _SGD_momentum = SGD_momentum;
            this.nesterov = nesterov;
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
            weights.UpdateSGDOptimizer(ponderedLearningRate, _SGD_momentum, nesterov, weightGradients, _velocityWeight);
            bias?.UpdateSGDOptimizer(ponderedLearningRate, _SGD_momentum, nesterov, biasGradient, _velocityBias);
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
    }
}
