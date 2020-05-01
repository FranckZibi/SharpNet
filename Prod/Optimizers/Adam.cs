using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    //beta1 = 0., beta2 = 0.: "vanilla" SGD
    //beta1 = 0.9, beta2 = 0.: "Classic momentum"
    //beta1 = 0.0, beta2 = 0.2: RMS prop
    //beta1 = 0.999, beta2 = 0.9: Classic Adam
    // Adam with beta1=1 is equivalent to RMSProp with momentum=0. 
    //The argument beta2 of Adam and the argument decay of RMSProp are the same
    public class Adam : Optimizer
    {
        #region private fields
        private int _timestep = 1;
        private readonly double _adam_beta1;
        private readonly double _adam_beta2;
        private readonly double _adam_epsilon;
        private readonly Tensor _adam_VW;                      // same as 'Weights'
        private readonly Tensor _adam_SW;                      // same as 'Weights'
        [CanBeNull] private readonly Tensor _adam_VB;          // same as 'Bias'
        [CanBeNull] private readonly Tensor _adam_SB;          // same as 'Bias'
        private readonly TensorMemoryPool _memoryPool;

        #endregion

        public Adam(TensorMemoryPool memoryPool, double adam_beta1, double adam_beta2, double adam_epsilon, int[] weightShape, int[] biasShapeIfAny)
        {
            _memoryPool = memoryPool;
            _adam_beta1 = adam_beta1;
            _adam_beta2 = adam_beta2;
            _adam_epsilon = adam_epsilon;
            _memoryPool.GetFloatTensor(ref _adam_VW, weightShape);
            _memoryPool.GetFloatTensor(ref _adam_SW , weightShape);
            if (biasShapeIfAny != null)
            {
                _memoryPool.GetFloatTensor(ref _adam_VB, biasShapeIfAny);
                _memoryPool.GetFloatTensor(ref _adam_SB, biasShapeIfAny);
            }
            ZeroMemory();
        }

        public override bool Equals(Optimizer other, double epsilon, string id, ref string errors)
        {
            if (!Utils.Equals(GetType(), other.GetType(), id + nameof(GetType), ref errors))
            {
                return false;
            }
            var b = (Adam)other;
            return  
                    Utils.Equals(_timestep, b._timestep, id+ "_timestep", ref errors)
                && Utils.Equals(_adam_beta1, b._adam_beta1, epsilon, id+ "_adam_beta1", ref errors)
                && Utils.Equals(_adam_beta2, b._adam_beta2, epsilon, id+ "_adam_beta2", ref errors)
                && Utils.Equals(_adam_epsilon, b._adam_epsilon, epsilon, id+ "_adam_epsilon", ref errors)
                && _adam_VW.Equals(b._adam_VW, epsilon, id + "_adam_VW", ref errors)
                && _adam_SW.Equals(b._adam_SW, epsilon, id + "_adam_SW", ref errors)
                && _adam_VB.Equals(b._adam_VB, epsilon, id + "_adam_VB", ref errors)
                && _adam_SB.Equals(b._adam_SB, epsilon, id + "_adam_SB", ref errors);
        }

        public override List<Tensor> EmbeddedTensors
        {
            get
            {
                var result = new List<Tensor> {_adam_VW, _adam_SW, _adam_SB, _adam_VB};
                result.RemoveAll(t => t == null);
                return result;
            }
        }

        public override void UpdateWeights(double learningRate, int batchSize, Tensor weights, Tensor weightGradients, Tensor bias, Tensor biasGradient)
        {
            Debug.Assert(weights.SameShape(weightGradients));
            Debug.Assert(bias == null || bias.SameShape(biasGradient));
            ++_timestep;
            var ponderedLearningRate = PonderedLearning(learningRate, batchSize);
            weights.UpdateAdamOptimizer(ponderedLearningRate, _adam_beta1, _adam_beta2, _adam_epsilon, weightGradients, _adam_VW, _adam_SW, _timestep);
            bias?.UpdateAdamOptimizer(ponderedLearningRate, _adam_beta1, _adam_beta2, _adam_epsilon, biasGradient, _adam_VB, _adam_SB, _timestep);
        }
        public override void Dispose()
        {
            if (_isDisposed)
            {
                return;
            }
            _isDisposed = true;
            base.Dispose();
            EmbeddedTensors.ForEach(t => _memoryPool.FreeFloatTensor(t));
        }

        #region serialization
        public override string Serialize()
        {
            return new Serializer()
                .Add(nameof(_timestep), _timestep)
                .Add(nameof(_adam_beta1), _adam_beta1)
                .Add(nameof(_adam_beta2), _adam_beta2)
                .Add(nameof(_adam_epsilon), _adam_epsilon)
                .Add(nameof(_adam_VW), _adam_VW)
                .Add(nameof(_adam_SW), _adam_SW)
                .Add(nameof(_adam_VB), _adam_VB)
                .Add(nameof(_adam_SB), _adam_SB)
                .ToString();
        }
        public static Optimizer DeserializeAdam(IDictionary<string, object> serialized)
        {
            return serialized.ContainsKey(nameof(_adam_VW)) ? new Adam(serialized) : null;
        }
        private Adam(IDictionary<string, object> serialized)
        {
            serialized.TryGet(nameof(_timestep), out _timestep);
            serialized.TryGet(nameof(_adam_beta1), out _adam_beta1);
            serialized.TryGet(nameof(_adam_beta2), out _adam_beta2);
            serialized.TryGet(nameof(_adam_epsilon), out _adam_epsilon);
            serialized.TryGet(nameof(_adam_VW), out _adam_VW);
            serialized.TryGet(nameof(_adam_SW), out _adam_SW);
            serialized.TryGet(nameof(_adam_VB), out _adam_VB);
            serialized.TryGet(nameof(_adam_SB), out _adam_SB);
        }
        #endregion
    }
}
