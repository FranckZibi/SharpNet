using System;

namespace SharpNet.DataAugmentation.Operations
{
    public class TranslateX : Operation
    {
        private readonly int _horizontalShift;

        public TranslateX(int horizontalShift, int[] miniBatchShape) : base(miniBatchShape)
        {
            _horizontalShift = horizontalShift;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row,col- _horizontalShift);
        }
        public static TranslateX ValueOf(double widthShiftRangeInPercentage, Random rand, int[] miniBatchShape)
        {
            if (widthShiftRangeInPercentage <= 0)
            {
                return null;
            }
            int horizontalShiftRangeInPixels = GetShiftInPixel(miniBatchShape[3], widthShiftRangeInPercentage);
            var horizontalShift = rand.Next(2 * horizontalShiftRangeInPixels + 1) - horizontalShiftRangeInPixels;
            return new TranslateX(horizontalShift, miniBatchShape);
        }

        public static int GetShiftInPixel(int pictureWidth, double widthShiftRange)
        {
            if (widthShiftRange <= 0)
            {
                return 0;
            }
            return (int)Math.Ceiling(pictureWidth * widthShiftRange);
        }
    }
}