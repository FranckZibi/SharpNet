﻿using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Optimizers
{
    public class VanillaSgd : Optimizer
    {
        private VanillaSgd() {}
        public static readonly VanillaSgd Instance = new VanillaSgd();

        public override void UpdateWeights(double learningRate, int batchSize, Tensor weights, Tensor weightGradients, Tensor bias, Tensor biasGradients)
        {
            Debug.Assert(weights.SameShape(weightGradients));
            Debug.Assert(bias == null || bias.SameShape(biasGradients));
            var ponderedLearningRate = PonderedLearning(learningRate, batchSize);
            weights.Update_Adding_Alpha_X(-ponderedLearningRate, weightGradients);
            bias?.Update_Adding_Alpha_X(-ponderedLearningRate, biasGradients);
        }
        public override List<Tensor> EmbeddedTensors => new List<Tensor>();
        public override bool Equals(Optimizer other, double epsilon, string id, ref string errors)
        {
            return Utils.Equals(GetType(), other.GetType(), id + nameof(GetType), ref errors);
        }

        public override Optimizer CloneForSlaveNetwork(Network newSlaveNetwork) { return Instance; }

        #region serialization
        public override string Serialize() {return "";}
        #endregion
    }
}
