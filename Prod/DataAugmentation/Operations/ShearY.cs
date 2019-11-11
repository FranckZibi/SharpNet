using System.Diagnostics;

namespace SharpNet.DataAugmentation.Operations
{
    public class ShearY : Operation
    {
        private readonly double _heightMultiplier;

        public ShearY(double heightMultiplier, int[] miniBatchShape) : base(miniBatchShape)
        {
            Debug.Assert(heightMultiplier > 0);
            _heightMultiplier = heightMultiplier;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row/ _heightMultiplier, col);
        }
    }
}