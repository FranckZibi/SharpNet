using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Invert : Operation
    {
        private readonly List<Tuple<float, float>> _meanAndVolatilityForEachChannel;

        public Invert(List<Tuple<float, float>> meanAndVolatilityForEachChannel)
        {
            _meanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
        }

        public override float AugmentedValue(float initialValue, int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch, CpuTensor<float> xDataAugmentedMiniBatch, int channel, int rowOutput,
            int colOutput, out bool isFinalAugmentedValue)
        {
            isFinalAugmentedValue = false;
            var unnormalizedInitialValue = UnnormalizedValue(initialValue, channel, _meanAndVolatilityForEachChannel);
            var unnormalizedAugmentedValue = 255.0f - unnormalizedInitialValue;
            return NormalizedValue(unnormalizedAugmentedValue, channel, _meanAndVolatilityForEachChannel);
        }
    }
}