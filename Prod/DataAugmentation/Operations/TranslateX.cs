using System;

namespace SharpNet.DataAugmentation.Operations
{
    public class TranslateX : Operation
    {
        private readonly int _horizontalShift;

        public TranslateX(int horizontalShift)
        {
            _horizontalShift = horizontalShift;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row,col- _horizontalShift);
        }
        public static TranslateX ValueOf(double widthShiftRangeInPercentage, Random rand, int nbCols)
        {
            if (widthShiftRangeInPercentage <= 0)
            {
                return null;
            }
            int horizontalShiftRangeInPixels = GetShiftInPixel(nbCols, widthShiftRangeInPercentage);
            var horizontalShift = rand.Next(2 * horizontalShiftRangeInPixels + 1) - horizontalShiftRangeInPixels;
            return new TranslateX(horizontalShift);
        }

        public static int GetShiftInPixel(int pictureWidth, double widthShiftRangeInPercentage)
        {
            if (widthShiftRangeInPercentage <= 0)
            {
                return 0;
            }
            return (int)Math.Ceiling(pictureWidth * widthShiftRangeInPercentage);
        }
    }
}