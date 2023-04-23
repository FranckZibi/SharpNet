using System;
using SharpNet.MathTools;

namespace SharpNet.HPO
{
    public class SingleHyperParameterValueStatistics
    {
        public readonly DoubleAccumulator CostToDecrease = new();
        private readonly DoubleAccumulator ElapsedTimeInSeconds = new();

        public void RegisterScore(IScore score, double elapsedTimeInSeconds)
        {
            if (float.IsNaN(score.Value) || float.IsInfinity(score.Value))
            {
                return;
            }
            //the cost is something we want to minimize
            var costToDecrease = score.HigherIsBetter ? -score.Value : score.Value;
            CostToDecrease.Add(costToDecrease);
            ElapsedTimeInSeconds.Add(elapsedTimeInSeconds);
        }

        public override string ToString()
        {
            if (CostToDecrease.Count == 0)
            {
                return "empty";
            }
            return CostToDecrease.Average + " +/- " + CostToDecrease.Volatility + " (" + CostToDecrease.Count + " evals at "+Math.Round(ElapsedTimeInSeconds.Average,1) + "s/eval)";
        }
    }
}
