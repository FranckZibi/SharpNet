using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Posterize : Operation
    {
        private readonly int _bitsPerPixel;
        private readonly List<Tuple<float, float>> _meanAndVolatilityForEachChannel;

        public Posterize(int bitsPerPixel, List<Tuple<float, float>> meanAndVolatilityForEachChannel)
        {
            _bitsPerPixel = bitsPerPixel;
            _meanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
        }

        public override float AugmentedValue(int indexInMiniBatch, int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            var initialValue = xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
            var unnormalizedInitialValue = (int)Math.Round(UnnormalizedValue(initialValue, channel, _meanAndVolatilityForEachChannel),0);
            var unnormalizedAugmentedValue = (unnormalizedInitialValue >>(8-_bitsPerPixel))<<(8-_bitsPerPixel);
            return NormalizedValue(unnormalizedAugmentedValue, channel, _meanAndVolatilityForEachChannel);
        }
    }
}