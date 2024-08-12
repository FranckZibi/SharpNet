using System;
using SharpNet.CPU;
using SharpNet.MathTools;

namespace SharpNet.DataAugmentation.Operations
{
    public class MixUp : Operation
    {
        private readonly float _percentageFromOriginalElement;
        private readonly int _indexInMiniBatchForMixUp;
        private readonly CpuTensor<float> _xOriginalMiniBatch;

        public MixUp(float percentageFromOriginalElement, int indexInMiniBatchForMixUp, CpuTensor<float> xOriginalMiniBatch)
        {
            _percentageFromOriginalElement = percentageFromOriginalElement;
            _indexInMiniBatchForMixUp = indexInMiniBatchForMixUp;
            _xOriginalMiniBatch = xOriginalMiniBatch;
        }

        public static MixUp ValueOf(double alpha, int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch, Random rand)
        {
            if (alpha <= 0.0)
            {
                return null;
            }
            var percentageFromOriginalElement = (float)Utils.BetaDistribution(alpha, alpha, rand);
            var miniBatchShape = xOriginalMiniBatch.Shape;
            var miniBatchSize = miniBatchShape[0];
            int indexInMiniBatchForMixUp = (indexInMiniBatch + 2) % miniBatchSize;
            return new MixUp(percentageFromOriginalElement, indexInMiniBatchForMixUp, xOriginalMiniBatch);
        }

        // ReSharper disable once UnusedMember.Global
        public static void DisplayStatsForAlphaMixUp()
        {
            var rand = new Random();
            for (int i = 1; i <= 100; ++i)
            {
                var alphaMixUp = i / 10.0;
                var acc = new DoubleAccumulator();
                for (int t = 0; t < 10000; ++t)
                {
                    var MixUpLambda = Utils.BetaDistribution(alphaMixUp, alphaMixUp, rand);
                    acc.Add(MixUpLambda);
                }
                Console.WriteLine($"for alphaMixUp={alphaMixUp}, MixUpLambda={acc}");
            }
        }


        public override void UpdateY(CpuTensor<float> yOriginalMiniBatch, CpuTensor<float> yDataAugmentedMiniBatch, int indexInMiniBatch, Func<int, int> indexInMiniBatchToCategoryIndex)
        {
            // We need to update the expected y using MixUp lambda
            //special case: when the y tensor is of shape (batchSize, 1)
            if (yOriginalMiniBatch.Shape.Length == 2 && yOriginalMiniBatch.Shape[1] == 1)
            {
                var originalValue = yOriginalMiniBatch.Get(indexInMiniBatch, 0);
                var otherValue = yOriginalMiniBatch.Get(_indexInMiniBatchForMixUp, 0);
                if (originalValue != otherValue)
                {
                    // We need to update the expected y value at 'indexInMiniBatch':
                    // the updated y value is:
                    //      '_percentageFromOriginalElement' * y value of the element at 'indexInMiniBatch' (original element)
                    //      +'1-_percentageFromOriginalElement' * y value of the element at '_indexInMiniBatchForMixUp' (other element)
                    yDataAugmentedMiniBatch.Set(indexInMiniBatch, 0, _percentageFromOriginalElement * originalValue + (1 - _percentageFromOriginalElement) * otherValue);
                }
                return;
            }

            // the associated y is:
            //      '_percentageFromOriginalElement' % of the category of the element at 'indexInMiniBatch' (original element)
            //      '1-_percentageFromOriginalElement' % of the category of the element at 'indexInMiniBatchForMixUp' (other element)
            var originalCategoryIndex = indexInMiniBatchToCategoryIndex(indexInMiniBatch);
            var otherCategoryIndex = indexInMiniBatchToCategoryIndex(_indexInMiniBatchForMixUp);
            yDataAugmentedMiniBatch.Set(indexInMiniBatch, originalCategoryIndex, _percentageFromOriginalElement);
            yDataAugmentedMiniBatch.Set(indexInMiniBatch, otherCategoryIndex, 1f - _percentageFromOriginalElement);
        }

        public override float AugmentedValue(int indexInMiniBatch, int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            var originalValue = xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
            var otherValue = _xOriginalMiniBatch.Get(_indexInMiniBatchForMixUp, channel, rowInput, colInput);

            //!D To test: return Math.Max(initialValue, otherValue);

            return _percentageFromOriginalElement * originalValue + (1 - _percentageFromOriginalElement) * otherValue;
        }
    }
}
