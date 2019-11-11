using System;

namespace SharpNet.DataAugmentation.Operations
{
    public class Rotate : Operation
    {
        private readonly double _rotationInDegrees;

        private Rotate(double rotationInDegrees, int[] miniBatchShape) : base(miniBatchShape)
        {
            _rotationInDegrees = rotationInDegrees;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return XYCoordinateRotater.RotateInTopLeftReferential(row, col, NbRows, NbCols, NbRows, NbCols,
                -_rotationInDegrees);
        }

        public static Rotate ValueOf(double rotationRangeInDegrees, Random rand, int[] miniBatchShape)
        {
            if (rotationRangeInDegrees <= 0)
            {
                return null;
            }

            var rotationInDegrees = 2 * rotationRangeInDegrees * rand.NextDouble() - rotationRangeInDegrees;
            return new Rotate(rotationInDegrees, miniBatchShape);
        }

    }
}