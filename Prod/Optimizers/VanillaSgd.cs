﻿using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;

namespace SharpNet.Optimizers
{
    public class VanillaSgd : Optimizer
    {
        private VanillaSgd() {}
        public static readonly VanillaSgd Instance = new ();

        public override void UpdateWeights(double learningRate, double maxLearningRate, int batchSize, Tensor weights,
            Tensor weightGradients,
            Tensor bias, Tensor biasGradients)
        {
            Debug.Assert(weights.SameShape(weightGradients));
            Debug.Assert(bias == null || bias.SameShape(biasGradients));
            var ponderedLearningRate = PonderedLearning(learningRate, batchSize);
            weights.Update_Adding_Alpha_X(-ponderedLearningRate, weightGradients);
            bias?.Update_Adding_Alpha_X(-ponderedLearningRate, biasGradients);
        }
        public override List<Tensor> EmbeddedTensors => new ();
    }
}
