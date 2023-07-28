using System;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Mixup : Operation
    {
        private readonly float _mixupLambda;
        private readonly int _indexInMiniBatchForMixup;
        private readonly CpuTensor<float> _xOriginalMiniBatch;
        private readonly bool _useMax;

        public Mixup(float mixupLambda, int indexInMiniBatchForMixup, CpuTensor<float> xOriginalMiniBatch, bool useMax = false)
        {
            if (useMax)
            {
                mixupLambda = 0.5f; //!D TO TEST
            }

            _mixupLambda = mixupLambda;
            _indexInMiniBatchForMixup = indexInMiniBatchForMixup;
            _xOriginalMiniBatch = xOriginalMiniBatch;
            _useMax = useMax;
        }

        public static Mixup ValueOf(double alphaMixup, bool useMax, bool mixOnlySameCategory, int indexInMiniBatch, CpuTensor<float> xOriginalMiniBatch, CpuTensor<float> yOriginalMiniBatch, Random rand)
        {
            if (alphaMixup <= 0.0)
            {
                return null;
            }
            var mixupLambda = (float)Utils.BetaDistribution(alphaMixup, alphaMixup, rand);
            var miniBatchShape = xOriginalMiniBatch.Shape;
            var miniBatchSize = miniBatchShape[0];
            int indexToMixWith = GetIndexToMixWith(mixOnlySameCategory, indexInMiniBatch, miniBatchSize, yOriginalMiniBatch, rand);
            if (indexToMixWith < 0)
            {
                return null;
            }
            return new Mixup(mixupLambda, indexToMixWith, xOriginalMiniBatch, useMax);
        }


        public override void UpdateY(CpuTensor<float> yOriginalMiniBatch, CpuTensor<float> yDataAugmentedMiniBatch, int indexInMiniBatch, Func<int, int> indexInMiniBatchToCategoryIndex)
        {
            // We need to update the expected y using Mixup lambda


            //special case: when the y tensor is of shape (batchSizae, 1)
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
                    yDataAugmentedMiniBatch.Set(indexInMiniBatch, 0,
                                                _useMax
                        ? Math.Max(originalValue, mixupValue)
                        :_mixupLambda * originalValue + (1 - _mixupLambda) * mixupValue   
                        );
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
            return _mixupLambda * initialValue + (1 - _mixupLambda) * _xOriginalMiniBatch.Get(_indexInMiniBatchForMixup, channel, rowInput, colInput);
        }
    }
}
