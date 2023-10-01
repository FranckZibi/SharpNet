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
            //var lambda = (float)Utils.BetaDistribution(alphaCutMix, alphaCutMix, rand);

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

        public static CutMix ValueOfRowsCutMix(double alphaRowsCutMix, int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch, Random rand)
        {
            if (alphaRowsCutMix <= 0.0)
            {
                return null;
            }
            var miniBatchShape = xOriginalMiniBatch.Shape;
            var miniBatchSize = miniBatchShape[0];
            var nbRows = miniBatchShape[2];
            var nbCols = miniBatchShape[3];

            var percentageFromOriginalElement = (float)Utils.BetaDistribution(alphaRowsCutMix, alphaRowsCutMix, rand);
            int numberOfRowsToCutMix = (int)Math.Round((1-percentageFromOriginalElement) * nbRows, 0.0);
            if (numberOfRowsToCutMix <= 0)
            {
                return null;
            }

            //the CutMix rows will be centered at 'rowMiddle'
            //its size will be between '1 x nbCols' (only 1 row will be entirely CutMix) to 'numberOfRowsToCutMix x nbCols' (maximum size)
            var rowMiddle = rand.Next(nbRows);
            var rowStart = Math.Max(0, rowMiddle - numberOfRowsToCutMix / 2);
            var rowEnd = Math.Min(nbRows - 1, rowStart + numberOfRowsToCutMix - 1);
            int indexToMixWithForCutMix = (indexInMiniBatch + 1) % miniBatchSize;
            return new CutMix(rowStart, rowEnd, 0, nbCols-1, indexToMixWithForCutMix, xOriginalMiniBatch);
        }


        public static CutMix ValueOfColumnsCutMix(double alphaColumnsCutMix, int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch, Random rand)
        {
            if (alphaColumnsCutMix <= 0.0)
            {
                return null;
            }
            var miniBatchShape = xOriginalMiniBatch.Shape;
            var miniBatchSize = miniBatchShape[0];
            var nbRows = miniBatchShape[2];
            var nbCols = miniBatchShape[3];

            var percentageFromOriginalElement = (float)Utils.BetaDistribution(alphaColumnsCutMix, alphaColumnsCutMix, rand);
            int numberOfColumnsToCutMix = (int)Math.Round((1 - percentageFromOriginalElement) * nbCols, 0.0);
            if (numberOfColumnsToCutMix <= 0)
            {
                return null;
            }

            //the cutout columns will be centered at 'colMiddle'
            //its size will be between 'nbRows x 1' (only 1 column will be entirely CutMix) to 'nbRows x numberOfColumnsToCutMix' (maximum size)
            var colMiddle = rand.Next(nbCols);

            var colStart = Math.Max(0, colMiddle - numberOfColumnsToCutMix / 2);
            var colEnd = Math.Min(nbCols - 1, colStart + numberOfColumnsToCutMix - 1);
            int indexToMixWithForCutMix = (indexInMiniBatch + 1) % miniBatchSize;
            return new CutMix(0, nbRows-1, colStart, colEnd, indexToMixWithForCutMix, xOriginalMiniBatch);
        }

        public override void UpdateY(CpuTensor<float> yOriginalMiniBatch, CpuTensor<float> yDataAugmentedMiniBatch, int indexInMiniBatch, Func<int, int> indexInMiniBatchToCategoryIndex)
        {
            // if CutMix has been used, wee need to update the expected output ('y' tensor)
            

            //special case:  when the y tensor is of shape (batchSize, 1)
            if (yOriginalMiniBatch.Shape.Length == 2 && yOriginalMiniBatch.Shape[1] == 1)
            {
                var originalValue = yOriginalMiniBatch.Get(indexInMiniBatch, 0);
                var otherValue = yOriginalMiniBatch.Get(_indexInMiniBatchForCutMix, 0);
                if (originalValue != otherValue)
                {
                    var percentageFromOriginalElement = PercentageFromOriginalElement;

                    // We need to update the expected y value at 'indexInMiniBatch':
                    // the updated y value is:
                    //      'percentageTakenFromOtherElement' * y value of the element at 'indexInMiniBatch' (original element)
                    //      +'1-percentageTakenFromOtherElement' * y value of the element at 'indexInMiniBatchForCutMix' (other element)
                    yDataAugmentedMiniBatch.Set(indexInMiniBatch, 0, percentageFromOriginalElement * originalValue + (1-percentageFromOriginalElement) * otherValue);
                }
                return;
            }

            var originalCategoryIndex = indexInMiniBatchToCategoryIndex(indexInMiniBatch);
            var otherCategoryIndex = indexInMiniBatchToCategoryIndex(_indexInMiniBatchForCutMix);
            if (originalCategoryIndex != otherCategoryIndex)
            {
                // We need to update the expected y using CutMix lambda
                // the associated y is:
                //      'percentageTakenFromOtherElement' % of the category of the element at 'indexInMiniBatch' (original element)
                //      '1-percentageTakenFromOtherElement' % of the category of the element at 'indexInMiniBatchForCutMix' (other element)
                var percentageFromOriginalElement = PercentageFromOriginalElement;
                yDataAugmentedMiniBatch.Set(indexInMiniBatch, originalCategoryIndex, percentageFromOriginalElement);
                yDataAugmentedMiniBatch.Set(indexInMiniBatch, otherCategoryIndex, 1-percentageFromOriginalElement);
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
                //we take the value from the other element
                return _xOriginalMiniBatch.Get(_indexInMiniBatchForCutMix, channel, rowInput, colInput);
            }
            //we take the value from the original element
            return xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
        }

        private float PercentageFromOriginalElement
        {
            get
            {
                int nbRows = _xOriginalMiniBatch.Shape[2];
                int nbCols = _xOriginalMiniBatch.Shape[3];
                float percentageFromOriginalElement = 1f - ((float)((_rowEnd - _rowStart + 1) * (_colEnd - _colStart + 1))) / (nbCols * nbRows);
                return percentageFromOriginalElement;
            }
        }
    }
}