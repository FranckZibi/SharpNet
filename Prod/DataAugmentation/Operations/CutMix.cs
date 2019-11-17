using System;
using System.Diagnostics;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class CutMix : Operation
    {
        private readonly int _rowStart;
        private readonly int _rowEnd;
        private readonly int _colStart;
        private readonly int _colEnd;
        private readonly int _indexInMiniBatchForCutMix;
        private readonly CpuTensor<float> _xOriginalMiniBatch;

        public CutMix(int rowStart, int rowEnd, int colStart, int colEnd, int indexInMiniBatchForCutMix, CpuTensor<float> xOriginalMiniBatch)
        {
            Debug.Assert(indexInMiniBatchForCutMix < xOriginalMiniBatch.Shape[0]);
            _rowStart = rowStart;
            _rowEnd = rowEnd;
            _colStart = colStart;
            _colEnd = colEnd;
            _indexInMiniBatchForCutMix = indexInMiniBatchForCutMix;
            _xOriginalMiniBatch = xOriginalMiniBatch;
        }

        public static CutMix ValueOf(double alphaCutMix, int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch, Random rand)
        {
            if (alphaCutMix <= 0.0)
            {
                return null;
            }

            //CutMix V2 : we ensure that we keep at least 50% of the original image when mixing with another one
            //validated on 18-aug-2019
            var lambda = 0.5 + 0.5 * (float)Utils.BetaDistribution(alphaCutMix, alphaCutMix, rand);
            //var lambda = (float)Utils.BetaDistribution(_alphaCutMix, _alphaCutMix, rand);

            var miniBatchShape = xOriginalMiniBatch.Shape;
            var miniBatchSize = miniBatchShape[0];
            var nbRows = miniBatchShape[2];
            var nbCols = miniBatchShape[3];
            var cutMixHeight = (int)(nbRows * Math.Sqrt(1.0 - lambda));
            var cutMixWidth = (int)(nbCols * Math.Sqrt(1.0 - lambda));

            //the cutout patch will be centered at (rowMiddle,colMiddle)
            //its size will be between '1x1' (minimum patch size if the center is a corner) to 'cutoutPatchLength x cutoutPatchLength' (maximum size)
            var rowMiddle = rand.Next(nbRows);
            var colMiddle = rand.Next(nbCols);
            var rowStart = Math.Max(0, rowMiddle - cutMixHeight / 2);
            var rowEnd = Math.Min(nbRows - 1, rowMiddle + cutMixHeight / 2 - 1);
            var colStart = Math.Max(0, colMiddle - cutMixWidth / 2 / 2);
            var colEnd = Math.Min(nbCols - 1, colMiddle + cutMixWidth / 2 - 1);

            int indexInMiniBatchForCutMix = (indexInMiniBatch + 1) % miniBatchSize;
            return new CutMix(rowStart, rowEnd, colStart, colEnd, indexInMiniBatchForCutMix, xOriginalMiniBatch);
        }

        public override void UpdateY(CpuTensor<float> yMiniBatch, int indexInMiniBatch, Func<int, int> indexInMiniBatchToCategoryId)
        {
            // if CutMix has been used, wee need to update the expected output ('y' tensor)
            var originalCategoryId = indexInMiniBatchToCategoryId(indexInMiniBatch);
            var cutMixCategoryId = indexInMiniBatchToCategoryId(_indexInMiniBatchForCutMix);
            if (originalCategoryId != cutMixCategoryId)
            {
                float cutMixLambda = 1f - ((float)((_rowEnd - _rowStart + 1) * (_colEnd - _colStart + 1))) / (NbCols * NbRows);
                // We need to update the expected y using CutMix lambda
                // the associated y is:
                //        'cutMixLambda' % of the category of the element at 'indexInMiniBatch'
                //      '1-cutMixLambda' % of the category of the element at 'indexInMiniBatchForCutMix'
                yMiniBatch.Set(indexInMiniBatch, originalCategoryId, cutMixLambda);
                yMiniBatch.Set(indexInMiniBatch, cutMixCategoryId, 1f - cutMixLambda);
            }
        }

        public override float AugmentedValue(float originalValue, int channelOutput, int rowOutput, int colOutput, out bool isFinalAugmentedValue)
        {
            //we check if we should apply cutMix to the pixel
            //this CutMix check must be performed *before* the Cutout check
            if (rowOutput >= _rowStart && rowOutput <= _rowEnd && colOutput >= _colStart && colOutput <= _colEnd)
            {
                isFinalAugmentedValue = true;
                return _xOriginalMiniBatch.Get(_indexInMiniBatchForCutMix, channelOutput, rowOutput, colOutput);
            }
            isFinalAugmentedValue = false;
            return originalValue;
        }

        private int NbRows => _xOriginalMiniBatch.Shape[2];
        private int NbCols => _xOriginalMiniBatch.Shape[3];

    }
}