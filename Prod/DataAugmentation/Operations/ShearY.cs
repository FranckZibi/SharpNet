using System.Diagnostics;

namespace SharpNet.DataAugmentation.Operations
{
    public class ShearY : Operation
    {
        private readonly double _heightMultiplier;

        public ShearY(double heightMultiplier)
        {
            Debug.Assert(heightMultiplier > 0);
            _heightMultiplier = heightMultiplier;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row/ _heightMultiplier, col);
        }
        public override bool ChangeCoordinates()
        {
            return true;
        }

    }
}