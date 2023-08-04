using System;
using System.Diagnostics;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class CutMix : Operation
    {
        #region private fields
        private readonly int _rowStart;
        private readonly int _rowEnd;
        private readonly int _colStart;
        private readonly int _colEnd;
        private readonly int _indexInMiniBatchForCutMix;
        private readonly CpuTensor<float> _xOriginalMiniBatch;
        #endregion

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

            int indexToMixWithForCutMix = (indexInMiniBatch + 1) % miniBatchSize;
            return new CutMix(rowStart, rowEnd, colStart, colEnd, indexToMixWithForCutMix, xOriginalMiniBatch);
        }
        public override void UpdateY(CpuTensor<float> yOriginalMiniBatch, CpuTensor<float> yDataAugmentedMiniBatch, int indexInMiniBatch, Func<int, int> indexInMiniBatchToCategoryIndex)
        {
            // if CutMix has been used, wee need to update the expected output ('y' tensor)
            

            //special case:  when the y tensor is of shape (batchSizae, 1)
            if (yOriginalMiniBatch.Shape.Length == 2 && yOriginalMiniBatch.Shape[1] == 1)
            {
                var originalValue = yOriginalMiniBatch.Get(indexInMiniBatch, 0);
                var cutMixValue = yOriginalMiniBatch.Get(_indexInMiniBatchForCutMix, 0);
                if (originalValue != cutMixValue)
                {
                    var cutMixLambda = CutMixLambda;

                    // We need to update the expected y value at 'indexInMiniBatch':
                    // the udpated y value is:
                    //      'cutMixLambda' * y value of the element at 'indexInMiniBatch'
                    //      +'1-cutMixLambda' * y value of the element at 'indexInMiniBatchForCutMix'
                    yDataAugmentedMiniBatch.Set(indexInMiniBatch, 0, cutMixLambda * originalValue + (1 - cutMixLambda) * cutMixValue);
                }
                return;
            }

            var originalCategoryIndex = indexInMiniBatchToCategoryIndex(indexInMiniBatch);
            var cutMixCategoryIndex = indexInMiniBatchToCategoryIndex(_indexInMiniBatchForCutMix);
            if (originalCategoryIndex != cutMixCategoryIndex)
            {
                // We need to update the expected y using CutMix lambda
                // the associated y is:
                //        'cutMixLambda' % of the category of the element at 'indexInMiniBatch'
                //      '1-cutMixLambda' % of the category of the element at 'indexInMiniBatchForCutMix'
                var cutMixLambda = CutMixLambda;
                yDataAugmentedMiniBatch.Set(indexInMiniBatch, originalCategoryIndex, cutMixLambda);
                yDataAugmentedMiniBatch.Set(indexInMiniBatch, cutMixCategoryIndex, 1f - cutMixLambda);
            }
        }
        public override float AugmentedValue(int indexInMiniBatch, int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            //we check if we should apply cutMix to the pixel
            //this CutMix check must be performed *before* the Cutout check
            if (rowOutput >= _rowStart && rowOutput <= _rowEnd && colOutput >= _colStart && colOutput <= _colEnd)
            {
                return _xOriginalMiniBatch.Get(_indexInMiniBatchForCutMix, channel, rowInput, colInput);
            }
            return xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
        }

        private float CutMixLambda
        {
            get
            {
                int nbRows = _xOriginalMiniBatch.Shape[2];
                int nbCols = _xOriginalMiniBatch.Shape[3];
                float cutMixLambda = 1f - ((float)((_rowEnd - _rowStart + 1) * (_colEnd - _colStart + 1))) / (nbCols * nbRows);
                return cutMixLambda;
            }
        }
    }
}