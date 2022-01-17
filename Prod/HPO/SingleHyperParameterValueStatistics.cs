using SharpNet.MathTools;

namespace SharpNet.HPO
{
    public class SingleHyperParameterValueStatistics
    {
        public readonly DoubleAccumulator Errors = new DoubleAccumulator();

        public void RegisterError(double error)
        {
            Errors.Add(error,1);
        }

        public override string ToString()
        {
            if (Errors.Count == 0)
            {
                return "empty";
            }
            return Errors.Average + " +/- " + Errors.Volatility + " (" + Errors.Count + " evals)";
        }
    }
}