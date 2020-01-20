using System;

namespace SharpNet.DataAugmentation.Operations
{
    public class TranslateY : Operation
    {
        private readonly int _verticalShift;

        public TranslateY(int verticalShift)
        {
            _verticalShift = verticalShift;
        }

        public override (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row- _verticalShift, col);
        }
        public static TranslateY ValueOf(double heightShiftRangeInPercentage, Random rand, int nbRows)
        {
            if (heightShiftRangeInPercentage <= 0)
            {
                return null;
            }
            int verticalShiftRangeInPixels = TranslateX.GetShiftInPixel(nbRows, heightShiftRangeInPercentage);
            var verticalShift = rand.Next(2 * verticalShiftRangeInPixels + 1) - verticalShiftRangeInPixels;
            return new TranslateY(verticalShift);
        }
    }
}
