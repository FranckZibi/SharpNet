using System;

namespace SharpNet.DataAugmentation.Operations
{
    public class Rotate : Operation
    {
        private readonly double _rotationInDegrees;
        private readonly int _nbRows;
        private readonly int _nbCols;

        public Rotate(double rotationInDegrees, int nbRows, int nbCols)
        {
            _rotationInDegrees = rotationInDegrees;
            _nbRows = nbRows;
            _nbCols = nbCols;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return XYCoordinateRotator.RotateInTopLeftReferential(row, col, _nbRows, _nbCols, _nbRows, _nbCols,
                -_rotationInDegrees);
        }
        public static Rotate ValueOf(double rotationRangeInDegrees, Random rand, int nbRows, int nbCols)
        {
            if (rotationRangeInDegrees <= 0)
            {
                return null;
            }

            var rotationInDegrees = 2 * rotationRangeInDegrees * rand.NextDouble() - rotationRangeInDegrees;
            return new Rotate(rotationInDegrees, nbRows, nbCols);
        }

    }
}