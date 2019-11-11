using System.Diagnostics;

namespace SharpNet.DataAugmentation.Operations
{
    public class ShearX : Operation
    {
        private readonly double _widthMultiplier;

        public ShearX(double widthMultiplier, int[] miniBatchShape) : base(miniBatchShape)
        {
            Debug.Assert(widthMultiplier > 0);
            _widthMultiplier = widthMultiplier;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row, col/ _widthMultiplier );
        }
    }
}