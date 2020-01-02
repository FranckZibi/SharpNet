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

        public override float AugmentedValue(int indexInMiniBatch, int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            var initialValue = xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
            var unnormalizedInitialValue = UnnormalizedValue(initialValue, channel, _meanAndVolatilityForEachChannel);
            var unnormalizedAugmentedValue = 255.0f - unnormalizedInitialValue;
            return NormalizedValue(unnormalizedAugmentedValue, channel, _meanAndVolatilityForEachChannel);
        }
    }
}