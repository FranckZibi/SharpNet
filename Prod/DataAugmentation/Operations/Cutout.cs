using System;

namespace SharpNet.DataAugmentation.Operations
{
    public class Cutout : Operation
    {
        private readonly int _rowStart;
        private readonly int _rowEnd;
        private readonly int _colStart;
        private readonly int _colEnd;

        private Cutout(int rowStart, int rowEnd, int colStart, int colEnd, int[] miniBatchShape) : base(miniBatchShape)
        {
            _rowStart = rowStart;
            _rowEnd = rowEnd;
            _colStart = colStart;
            _colEnd = colEnd;
        }


        public override float AugmentedValue(float originalValue, int channelOutput, int rowOutput, int colOutput,
            out bool isFinalAugmentedValue)
        {
            //we check if we should apply Cutout to the pixel
            //this Cutout check must be performed *after* the CutMix check (above)
            if (rowOutput >= _rowStart && rowOutput <= _rowEnd && colOutput >= _colStart && colOutput <= _colEnd)
            {
                isFinalAugmentedValue = true;
                return 0;
            }
            isFinalAugmentedValue = false;
            return originalValue;
        }


        public static Cutout ValueOf(double cutoutPatchPercentage, Random rand, int[] miniBatchShape)
        {
            if (cutoutPatchPercentage <= 0)
            {
                return null;
            }
            if (cutoutPatchPercentage > 1.0)
            {
                throw new ArgumentException("invalid _cutoutPatchPercentage:" + cutoutPatchPercentage);
            }
            var nbRows = miniBatchShape[2];
            var nbCols = miniBatchShape[3];

            int cutoutPatchLength = (int)Math.Round(cutoutPatchPercentage * Math.Max(nbRows, nbCols), 0.0);

            //the cutout patch will be centered at (rowMiddle,colMiddle)
            //its size will be between '1x1' (minimum patch size if the center is a corner) to 'cutoutPatchLength x cutoutPatchLength' (maximum size)
            var rowMiddle = rand.Next(nbRows);
            var colMiddle = rand.Next(nbCols);

            //test on 12-aug-2019 : -46bps
            //Cutout of always the max dimension
            //rowMiddle = (cutoutPatchLength/2)+rand.Next(nbRows- cutoutPatchLength);
            //colMiddle = (cutoutPatchLength/2) + rand.Next(nbCols - cutoutPatchLength);

            //Tested on 10-aug-2019: -10bps
            //rowStart = Math.Max(0, rowMiddle - cutoutPatchLength / 2);
            //rowEnd = Math.Min(nbRows - 1, rowMiddle + cutoutPatchLength/2 - 1);
            //colStart = Math.Max(0, colMiddle - cutoutPatchLength / 2);
            //colEnd = Math.Min(nbCols - 1, colMiddle + cutoutPatchLength/2 - 1);

            var rowStart = Math.Max(0, rowMiddle - cutoutPatchLength / 2);
            var rowEnd = Math.Min(nbRows - 1, rowStart + cutoutPatchLength - 1);
            var colStart = Math.Max(0, colMiddle - cutoutPatchLength / 2);
            var colEnd = Math.Min(nbCols - 1, colStart + cutoutPatchLength - 1);

            return new Cutout(rowStart, rowEnd, colStart, colEnd, miniBatchShape);
        }
    }
}
