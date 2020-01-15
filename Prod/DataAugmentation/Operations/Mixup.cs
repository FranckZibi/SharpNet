using System;
using SharpNet.CPU;

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

        public override void UpdateY(CpuTensor<float> yMiniBatch, int indexInMiniBatch, Func<int, int> indexInMiniBatchToCategoryIndex)
        {
            // We need to update the expected y using Mixup lambda
            // the associated y is:
            //        'mixupLambda' % of the category of the element at 'indexInMiniBatch'
            //      '1-mixupLambda' % of the category of the element at 'indexInMiniBatchForMixup'
            var originalCategoryIndex = indexInMiniBatchToCategoryIndex(indexInMiniBatch);
            var mixupCategoryIndex = indexInMiniBatchToCategoryIndex(_indexInMiniBatchForMixup);
            yMiniBatch.Set(indexInMiniBatch, originalCategoryIndex, _mixupLambda);
            yMiniBatch.Set(indexInMiniBatch, mixupCategoryIndex, 1f - _mixupLambda);
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
