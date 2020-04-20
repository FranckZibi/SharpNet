using System;
using System.Collections.Generic;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Optimizers
{
    public abstract class Optimizer : IDisposable
    {
        public enum OptimizationEnum { VanillaSGD, Adam, SGD }

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
            EmbeddedTensors.ForEach(t => t?.ZeroMemory());
        }
        public abstract string Serialize();
        public void Dispose()
        {
            EmbeddedTensors.ForEach(x => x?.Dispose());
        }

        protected float PonderedLearning(double learningRate, int batchSize)
        {
            return (float)learningRate / batchSize;
        }

        public abstract Optimizer Clone(Network newNetwork);
    }
}