using System;
using SharpNet.MathTools;

namespace SharpNet.HPO
{
    public class SingleHyperParameterValueStatistics
    {
        public readonly DoubleAccumulator Errors = new DoubleAccumulator();
        private readonly DoubleAccumulator ElapsedTimeInSeconds = new DoubleAccumulator();

        public void RegisterError(double error, double elapsedTimeInSeconds)
        {
            Errors.Add(error,1);
            ElapsedTimeInSeconds.Add(elapsedTimeInSeconds, 1);
        }

        public override string ToString()
        {
            if (Errors.Count == 0)
            {
                return "empty";
            }
            return Errors.Average + " +/- " + Errors.Volatility + " (" + Errors.Count + " evals at "+Math.Round(ElapsedTimeInSeconds.Average,1) + "s/eval)";
        }
    }
}
