using System;
using SharpNet.MathTools;

namespace SharpNet.HPO
{
    public class SingleHyperParameterValueStatistics
    {
        public readonly DoubleAccumulator Cost = new DoubleAccumulator();
        private readonly DoubleAccumulator ElapsedTimeInSeconds = new DoubleAccumulator();

        public void RegisterCost(double cost, double elapsedTimeInSeconds)
        {
            Cost.Add(cost,1);
            ElapsedTimeInSeconds.Add(elapsedTimeInSeconds, 1);
        }

        public override string ToString()
        {
            if (Cost.Count == 0)
            {
                return "empty";
            }
            return Cost.Average + " +/- " + Cost.Volatility + " (" + Cost.Count + " evals at "+Math.Round(ElapsedTimeInSeconds.Average,1) + "s/eval)";
        }
    }
}
