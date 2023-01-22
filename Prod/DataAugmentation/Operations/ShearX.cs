namespace SharpNet.DataAugmentation.Operations
{
    public class ShearX : Operation
    {
        private readonly double _horizontalMultiplier;

        public ShearX(double horizontalMultiplier)
        {
            _horizontalMultiplier = horizontalMultiplier;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row, col/_horizontalMultiplier );
        }
    }
}