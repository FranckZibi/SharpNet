using System;
using System.Diagnostics;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Cutout : Operation
    {
        private readonly int _rowStart;
        private readonly int _rowEnd;
        private readonly int _colStart;
        private readonly int _colEnd;

        public Cutout(int rowStart, int rowEnd, int colStart, int colEnd)
        {
            Debug.Assert(rowStart >= 0);
            Debug.Assert(rowEnd>=0);
            Debug.Assert(colStart >= 0);
            Debug.Assert(colEnd >= 0);
            _rowStart = rowStart;
            _rowEnd = rowEnd;
            _colStart = colStart;
            _colEnd = colEnd;
        }

        public override float AugmentedValue(int indexInMiniBatch,
            int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            //we check if we should apply Cutout to the pixel
            //this Cutout must be the last performed operation (so must be after the CutMix operation if any)
            if (rowOutput >= _rowStart && rowOutput <= _rowEnd && colOutput >= _colStart && colOutput <= _colEnd)
            {
                //TODO check if we should return the mean instead
                return 0;
            }
            return xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
        }

        public static Cutout ValueOf(double cutoutPatchPercentage, Random rand, int nbRows, int nbCols)
        {
            if (cutoutPatchPercentage <= 0)
            {
                return null;
            }
            if (cutoutPatchPercentage > 1.0)
            {
                throw new ArgumentException("invalid _cutoutPatchPercentage:" + cutoutPatchPercentage);
            }

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

            return new Cutout(rowStart, rowEnd, colStart, colEnd);
        }

        //cutout entire columns
        public static Cutout ValueOfColumnsCutout(double columnsCutoutPatchPercentage, Random rand, int nbRows, int nbCols)
        {
            if (columnsCutoutPatchPercentage <= 0)
            {
                return null;
            }
            if (columnsCutoutPatchPercentage > 1.0)
            {
                throw new ArgumentException($"invalid {nameof(columnsCutoutPatchPercentage)}: {columnsCutoutPatchPercentage}");
            }

            int numberOfColumnsToCutout = (int)Math.Round(columnsCutoutPatchPercentage * nbCols, 0.0);

            //the cutout columns will be centered at 'colMiddle'
            //its size will be between 'nbRows x 1' (only 1 column will be entirely cutout)
            //to 'nbRows x cutoutPatchLengthInNumberOfColumns' (maximum size)
            var colMiddle = rand.Next(nbCols);

            var colStart = Math.Max(0, colMiddle - numberOfColumnsToCutout / 2);
            var colEnd = Math.Min(nbCols - 1, colStart + numberOfColumnsToCutout - 1);
            return new Cutout(0, nbRows, colStart, colEnd);
        }


        //cutout entire rows
        public static Cutout ValueORowsCutout(double rowsCutoutPatchPercentage, Random rand, int nbRows, int nbCols)
        {
            if (rowsCutoutPatchPercentage <= 0)
            {
                return null;
            }
            if (rowsCutoutPatchPercentage > 1.0)
            {
                throw new ArgumentException($"invalid {nameof(rowsCutoutPatchPercentage)}: {rowsCutoutPatchPercentage}");
            }

            int numberOfRowsToCutout = (int)Math.Round(rowsCutoutPatchPercentage * nbRows, 0.0);

            //the cutout rows will be centered at 'rowMiddle'
            //its size will be between '1 x nbCols' (only 1 row will be entirely cutout)
            //to 'cutoutPatchLengthInNumberOfRows x nbCols' (maximum size)
            var rowMiddle = rand.Next(nbRows);

            var rowStart = Math.Max(0, rowMiddle - numberOfRowsToCutout / 2);
            var rowEnd = Math.Min(nbRows - 1, rowStart + numberOfRowsToCutout - 1);
            return new Cutout(rowStart, rowEnd, 0, nbCols);
        }
    }
}
