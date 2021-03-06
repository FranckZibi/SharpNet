namespace SharpNet.DataAugmentation.Operations
{
    public class ShearY : Operation
    {
        private readonly double _level;

        public ShearY(double level)
        {
            _level = level;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row - col*_level, col );
        }
    }
}