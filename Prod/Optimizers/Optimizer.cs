using System;
using System.Collections.Generic;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Optimizers
{
    public abstract class Optimizer : IDisposable
    {
        public enum OptimizationEnum { VanillaSGD, Adam, SGD }
        protected bool _isDisposed;

        public static Optimizer ValueOf(NetworkConfig networkConfig, IDictionary<string, object> serialized)
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
            return VanillaSgd.Instance;
        }
        public abstract bool Equals(Optimizer other, double epsilon, string id, ref string foundError);
        public abstract List<Tensor> EmbeddedTensors { get; }
        public abstract void UpdateWeights(double learningRate, int batchSize, Tensor weights, Tensor weightGradients, Tensor bias, Tensor biasGradient);
        /// <summary>
        /// reset( by setting to 0) all embedded tensors in the current optimizer
        /// </summary>
        public void ZeroMemory()
        {
            EmbeddedTensors.ForEach(t => t.ZeroMemory());
        }
        public abstract string Serialize();

        protected static float PonderedLearning(double learningRate, int batchSize)
        {
            return (float)learningRate / batchSize;
        }
        public virtual void Dispose()
        {
        }
    }
}