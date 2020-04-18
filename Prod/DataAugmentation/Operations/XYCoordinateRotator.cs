using System;

namespace SharpNet.DataAugmentation.Operations
{
    public static class XYCoordinateRotator
    {
        public static (double, double) RotateInTopLeftReferential(double rowFromTop, double colFromLeft,
            int nbRowsBefore, int nbColsBefore, int nbRowsAfter, int nbColsAfter, double rotationRangeInDegrees)
        {
            if (Math.Abs(rotationRangeInDegrees) < 1e-6)
            {
                return (rowFromTop, colFromLeft);
            }

            var centered = FromTopLeftToCenteredReferential(rowFromTop, colFromLeft, nbRowsBefore, nbColsBefore);
            var rotateInCentered = Convert(centered.row, centered.col, rotationRangeInDegrees);
            return FromCenteredToTopLeftReferential(rotateInCentered.row, rotateInCentered.col, nbRowsAfter,
                nbColsAfter);
        }

        private static (double row, double col) FromTopLeftToCenteredReferential(double row, double col, int nbRows,
            int nbCols)
        {
            return (row - (nbRows - 1.0) / 2.0, col - (nbCols - 1.0) / 2.0);
        }

        private static (double row, double col) FromCenteredToTopLeftReferential(double row, double col, int nbRows,
            int nbCols)
        {
            return (row + (nbRows - 1.0) / 2.0, col + (nbCols - 1.0) / 2.0);
        }

        private static (double row, double col) Convert(double row, double col, double rotationRangeInDegrees)
        {
            var rotationRangeInRadians = (2 * Math.PI * rotationRangeInDegrees) / 360.0;
            var cosAngleInRadians = Math.Cos(rotationRangeInRadians);
            var sinAngleInRadians = Math.Sin(rotationRangeInRadians);
            return (row * cosAngleInRadians - col * sinAngleInRadians,
                col * cosAngleInRadians + row * sinAngleInRadians);
        }
    }
}
