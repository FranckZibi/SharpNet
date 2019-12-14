using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class AutoContrast : Operation
    {
        private readonly List<Tuple<int, int>> _pixelThresholdByChannel;
        private readonly List<Tuple<float, float>> _meanAndVolatilityForEachChannel;

        /// <summary>
        /// </summary>
        /// <param name="pixelThresholdByChannel"></param>
        /// <param name="meanAndVolatilityForEachChannel"></param>
        public AutoContrast(List<Tuple<int, int>> pixelThresholdByChannel, List<Tuple<float, float>> meanAndVolatilityForEachChannel)
        {
            _pixelThresholdByChannel = pixelThresholdByChannel;
            _meanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
        }

        public override float AugmentedValue(float initialValue, int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch, CpuTensor<float> xDataAugmentedMiniBatch, int channel, int rowOutput,
            int colOutput)
        {
            var bounds = _pixelThresholdByChannel[channel];
            if (bounds.Item1 >= bounds.Item2)
            {
                return initialValue;
            }
            var b = UnnormalizedValue(initialValue, channel, _meanAndVolatilityForEachChannel);
            var scale = 255f/ (bounds.Item2 - bounds.Item1);
            var targetValue = scale*(b - bounds.Item1 );
            targetValue = Math.Max(Math.Min(targetValue, 255), 0);
            var bNormalized = NormalizedValue(targetValue, channel, _meanAndVolatilityForEachChannel);
            return bNormalized;
        }
    }
}