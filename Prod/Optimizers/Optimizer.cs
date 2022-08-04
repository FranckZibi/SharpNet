using System;
using System.Collections.Generic;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public abstract class Optimizer : IDisposable
    {
        //for AdamW, see: https://www.fast.ai/2018/07/02/adam-weight-decay/
        public enum OptimizationEnum { VanillaSGD, Adam, SGD, AdamW, VanillaSGDOrtho}
        protected bool _isDisposed;

        public static Optimizer ValueOf(IDictionary<string, object> serialized)
        {
            var sgd = Sgd.DeserializeSGD(serialized);
            if (sgd != null)
            {
                return sgd;
            }
            var adam = Adam.DeserializeAdam(serialized);
            if (adam != null)
            {
                return adam;
            }
            var vanillaSgdOrtho = VanillaSgdOrtho.DeserializeVanillaSgdOrtho(serialized);
            if (vanillaSgdOrtho != null)
            {
                return vanillaSgdOrtho;
            }
            return VanillaSgd.Instance;
        }
        public abstract List<Tensor> EmbeddedTensors { get; }
        public abstract void UpdateWeights(double learningRate, double maxLearningRate, int batchSize, Tensor weights,
            Tensor weightGradients, Tensor bias, Tensor biasGradient);
        /// <summary>
        /// reset( by setting to 0) all embedded tensors in the current optimizer
        /// </summary>
        public void ZeroMemory()
        {
            EmbeddedTensors.ForEach(t => t.ZeroMemory());
        }
        public abstract string Serialize();

        public virtual bool IsOrthogonal => false;

        protected static float PonderedLearning(double learningRate, int batchSize)
        {
            return (float)learningRate / batchSize;
        }
        public virtual void Dispose()
        {
        }
    }
}