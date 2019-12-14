using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Equalize : Operation
    {
        private readonly List<int[]> _originalPixelToEqualizedPixelByChannel;
        private readonly List<Tuple<float, float>> _meanAndVolatilityForEachChannel;

        /// <summary>
        /// </summary>
        public Equalize(List<int[]> originalPixelToEqualizedPixelByChannel, List<Tuple<float, float>> meanAndVolatilityForEachChannel)
        {
            _originalPixelToEqualizedPixelByChannel = originalPixelToEqualizedPixelByChannel;
            _meanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
        }

        public override float AugmentedValue(float initialValue, int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch, CpuTensor<float> xDataAugmentedMiniBatch, int channel, int rowOutput,
            int colOutput)
        {
            var unnormalizedValue = (int)(UnnormalizedValue(initialValue, channel, _meanAndVolatilityForEachChannel)+0.5f);
            unnormalizedValue = Math.Min(unnormalizedValue, 255);
            unnormalizedValue = Math.Max(unnormalizedValue, 0);
            var unnormalizedEqualizedValue = _originalPixelToEqualizedPixelByChannel[channel][unnormalizedValue];
            return NormalizedValue(unnormalizedEqualizedValue, channel, _meanAndVolatilityForEachChannel);
        }


        public static List<int[]> GetOriginalPixelToEqualizedPixelByChannel(ImageStatistic stats)
        {
            var originalPixelToEqualizedPixelByChannel = new List<int[]>();
            var pixelCount = stats.Shape[1] * stats.Shape[2];
            foreach (var v in stats.PixelCountByChannel)
            {
                var originalPixelToEqualizedPixel = new int[v.Length];
                var cdf_min = 0;
                {
                    int cdf_i = 0;
                    for (int i = 0; i < v.Length; ++i)
                    {
                        if (v[i] == 0)
                        {
                            continue;
                        }

                        cdf_i += v[i];
                        if (cdf_min == 0)
                        {
                            cdf_min = cdf_i;
                        }

                        originalPixelToEqualizedPixel[i] = (int)(0.5f + (((float)cdf_i - cdf_min) * (v.Length - 1)) / (pixelCount - cdf_min));
                    }
                }
                originalPixelToEqualizedPixelByChannel.Add(originalPixelToEqualizedPixel);
            }
            return originalPixelToEqualizedPixelByChannel;
        }


    }
}