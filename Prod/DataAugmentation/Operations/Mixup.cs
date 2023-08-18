using System;
using SharpNet.CPU;
using SharpNet.MathTools;

namespace SharpNet.DataAugmentation.Operations
{
    public class Mixup : Operation
    {
        private readonly float _mixupLambda;
        private readonly int _indexInMiniBatchForMixup;
        private readonly CpuTensor<float> _xOriginalMiniBatch;

        public Mixup(float mixupLambda, int indexInMiniBatchForMixup, CpuTensor<float> xOriginalMiniBatch)
        {
            _mixupLambda = mixupLambda;
            _indexInMiniBatchForMixup = indexInMiniBatchForMixup;
            _xOriginalMiniBatch = xOriginalMiniBatch;
        }

        public static Mixup ValueOf(double alphaMixup, int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch, Random rand)
        {
            if (alphaMixup <= 0.0)
            {
                return null;
            }
            var mixupLambda = (float)Utils.BetaDistribution(alphaMixup, alphaMixup, rand);
            var miniBatchShape = xOriginalMiniBatch.Shape;
            var miniBatchSize = miniBatchShape[0];
            int indexInMiniBatchForMixup = (indexInMiniBatch + 2) % miniBatchSize;
            return new Mixup(mixupLambda, indexInMiniBatchForMixup, xOriginalMiniBatch);
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
                var mixupValue = yOriginalMiniBatch.Get(_indexInMiniBatchForMixup, 0);
                if (originalValue != mixupValue)
                {
                    // We need to update the expected y value at 'indexInMiniBatch':
                    // the udpated y value is:
                    //      '_mixupLambda' * y value of the element at 'indexInMiniBatch'
                    //      +'1-_mixupLambda' * y value of the element at '_indexInMiniBatchForMixup'
                    yDataAugmentedMiniBatch.Set(indexInMiniBatch, 0, _mixupLambda * originalValue + (1 - _mixupLambda) * mixupValue);
                }
                return;
            }

            // the associated y is:
            //        'mixupLambda' % of the category of the element at 'indexInMiniBatch'
            //      '1-mixupLambda' % of the category of the element at 'indexInMiniBatchForMixup'
            var originalCategoryIndex = indexInMiniBatchToCategoryIndex(indexInMiniBatch);
            var mixupCategoryIndex = indexInMiniBatchToCategoryIndex(_indexInMiniBatchForMixup);
            yDataAugmentedMiniBatch.Set(indexInMiniBatch, originalCategoryIndex, _mixupLambda);
            yDataAugmentedMiniBatch.Set(indexInMiniBatch, mixupCategoryIndex, 1f - _mixupLambda);
        }

        public override float AugmentedValue(int indexInMiniBatch, int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            var initialValue = xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
            var otherValue = _xOriginalMiniBatch.Get(_indexInMiniBatchForMixup, channel, rowInput, colInput);

            //!D To test: return Math.Max(initialValue, otherValue);

            return _mixupLambda * initialValue + (1 - _mixupLambda) * otherValue;
        }
    }
}
