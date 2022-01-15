using SharpNet.MathTools;

namespace SharpNet.HPO
{
    public class HyperParameterStatistics
    {
        public DoubleAccumulator Errors = new DoubleAccumulator();

        public void RegisterResult(double error)
        {
            Errors.Add(error,1);
        }
    }
}