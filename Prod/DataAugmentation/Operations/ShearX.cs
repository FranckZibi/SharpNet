namespace SharpNet.DataAugmentation.Operations
{
    public class ShearX : Operation
    {
        private readonly double _level;

        public ShearX(double level)
        {
            _level = level;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row, col - row * _level );
        }
    }
}