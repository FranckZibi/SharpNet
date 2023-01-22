namespace SharpNet.DataAugmentation.Operations
{
    public class ShearY : Operation
    {
        private readonly double _verticalMultiplier;

        public ShearY(double verticalMultiplier)
        {
            _verticalMultiplier = verticalMultiplier;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row/_verticalMultiplier, col );
        }
    }
}