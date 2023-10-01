using System;
using SharpNet.CPU;
using SharpNet.MathTools;

namespace SharpNet.DataAugmentation.Operations
{
    public class Mixup : Operation
    {
        private readonly float _percentageFromOriginalElement;
        private readonly int _indexInMiniBatchForMixup;
        private readonly CpuTensor<float> _xOriginalMiniBatch;

        public Mixup(float percentageFromOriginalElement, int indexInMiniBatchForMixup, CpuTensor<float> xOriginalMiniBatch)
        {
            _percentageFromOriginalElement = percentageFromOriginalElement;
            _indexInMiniBatchForMixup = indexInMiniBatchForMixup;
            _xOriginalMiniBatch = xOriginalMiniBatch;
        }

        public static Mixup ValueOf(double alphaMixup, int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch, Random rand)
        {
            if (alphaMixup <= 0.0)
            {
                return null;
            }
            var percentageFromOriginalElement = (float)Utils.BetaDistribution(alphaMixup, alphaMixup, rand);
            var miniBatchShape = xOriginalMiniBatch.Shape;
            var miniBatchSize = miniBatchShape[0];
            int indexInMiniBatchForMixup = (indexInMiniBatch + 2) % miniBatchSize;
            return new Mixup(percentageFromOriginalElement, indexInMiniBatchForMixup, xOriginalMiniBatch);
        }

        // ReSharper disable once UnusedMember.Global
        public static void DisplayStatsForAlphaMixup()
        {
            var rand = new Random();
            for (int i = 1; i <= 100; ++i)
            {
                var alphaMixup = i / 10.0;
                var acc = new DoubleAccumulator();
                for (int t = 0; t < 10000; ++t)
                {
                    var mixupLambda = Utils.BetaDistribution(alphaMixup, alphaMixup, rand);
                    acc.Add(mixupLambda);
                }
                Console.WriteLine($"for alphaMixup={alphaMixup}, mixupLambda={acc}");
            }
        }


        public override void UpdateY(CpuTensor<float> yOriginalMiniBatch, CpuTensor<float> yDataAugmentedMiniBatch, int indexInMiniBatch, Func<int, int> indexInMiniBatchToCategoryIndex)
        {
            // We need to update the expected y using Mixup lambda


            //special case: when the y tensor is of shape (batchSize, 1)
            if (yOriginalMiniBatch.Shape.Length == 2 && yOriginalMiniBatch.Shape[1] == 1)
            {
                var originalValue = yOriginalMiniBatch.Get(indexInMiniBatch, 0);
                var otherValue = yOriginalMiniBatch.Get(_indexInMiniBatchForMixup, 0);
                if (originalValue != otherValue)
                {
                    // We need to update the expected y value at 'indexInMiniBatch':
                    // the updated y value is:
                    //      '_percentageFromOriginalElement' * y value of the element at 'indexInMiniBatch' (original element)
                    //      +'1-_percentageFromOriginalElement' * y value of the element at '_indexInMiniBatchForMixup' (other element)
                    yDataAugmentedMiniBatch.Set(indexInMiniBatch, 0, _percentageFromOriginalElement * originalValue + (1 - _percentageFromOriginalElement) * otherValue);
                }
                return;
            }

            // the associated y is:
            //      '_percentageFromOriginalElement' % of the category of the element at 'indexInMiniBatch' (original element)
            //      '1-_percentageFromOriginalElement' % of the category of the element at 'indexInMiniBatchForMixup' (other element)
            var originalCategoryIndex = indexInMiniBatchToCategoryIndex(indexInMiniBatch);
            var otherCategoryIndex = indexInMiniBatchToCategoryIndex(_indexInMiniBatchForMixup);
            yDataAugmentedMiniBatch.Set(indexInMiniBatch, originalCategoryIndex, _percentageFromOriginalElement);
            yDataAugmentedMiniBatch.Set(indexInMiniBatch, otherCategoryIndex, 1f - _percentageFromOriginalElement);
        }

        public override float AugmentedValue(int indexInMiniBatch, int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            var originalValue = xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
            var otherValue = _xOriginalMiniBatch.Get(_indexInMiniBatchForMixup, channel, rowInput, colInput);

            //!D To test: return Math.Max(initialValue, otherValue);

            return _percentageFromOriginalElement * originalValue + (1 - _percentageFromOriginalElement) * otherValue;
        }
    }
}
