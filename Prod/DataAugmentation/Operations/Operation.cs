using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public abstract class Operation
    {
        public virtual (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row, col);
        }

        public virtual float AugmentedValue(float initialValue, int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch, CpuTensor<float> xDataAugmentedMiniBatch, int channel, int rowOutput,
            int colOutput)
        {
            return initialValue;
        }

        public virtual void UpdateY(CpuTensor<float> yMiniBatch, int indexInMiniBatch, Func<int, int> indexInMiniBatchToCategoryId)
        {
        }

        public static float UnnormalizedValue(float normalizedValue, int channel, List<Tuple<float, float>> meanAndVolatilityForEachChannel)
        {
            if (meanAndVolatilityForEachChannel == null)
            {
                return normalizedValue;
            }
            var meanAndVolatility = meanAndVolatilityForEachChannel[channel];
            return normalizedValue * meanAndVolatility.Item2 + meanAndVolatility.Item1;
        }
        public static float NormalizedValue(float unnormalizedValue, int channel, List<Tuple<float, float>> meanAndVolatilityForEachChannel)
        {
            if (meanAndVolatilityForEachChannel == null)
            {
                return unnormalizedValue;
            }
            var meanAndVolatility = meanAndVolatilityForEachChannel[channel];
            return (unnormalizedValue - meanAndVolatility.Item1) / meanAndVolatility.Item2;
        }

        public static float GetGreyScale(float r, float g, float b)
        {
            return r * 0.299f + g * 0.587f + b * 0.114f;
        }
    }
}