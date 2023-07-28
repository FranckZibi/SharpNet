using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public abstract class Operation
    {
        public virtual (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row, col);
        }

        /// <summary>
        /// true if the operation may change the coordinate of the pixel in the input (ex: rotation)
        /// false if it doesn't change the position of pixels (for example it if will only change pixel value)
        /// </summary>
        /// <returns></returns>
        public bool ChangeCoordinates()
        {
            foreach (var outputRow in new[] {-10.0, 0, 10.0})
            {
                foreach (var outputCol in new[] {-10.0, 0, 10.0})
                {
                    var (inputRow, inputCol) = Unconvert_Slow(outputRow, outputCol);
                    if (Math.Abs(inputRow - outputRow) > 1e-6 || Math.Abs(inputCol - outputCol) > 1e-6)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        public virtual float AugmentedValue(int indexInMiniBatch, int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            return xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
        }

        public virtual void UpdateY(CpuTensor<float> yOriginalMiniBatch, CpuTensor<float> yDataAugmentedMiniBatch, int indexInMiniBatch, Func<int, int> indexInMiniBatchToCategoryIndex)
        {
        }

        protected static float UnnormalizedValue(float normalizedValue, int channel, List<Tuple<float, float>> meanAndVolatilityForEachChannel)
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

        public static int GetIndexToMixWith(bool mixOnlySameCategory, int indexInMiniBatch, int miniBatchSize, CpuTensor<float> yOriginalMiniBatch, Random rand)
        {
            int indexToMixWith = (indexInMiniBatch + 2) % miniBatchSize;
            if (!mixOnlySameCategory || yOriginalMiniBatch == null)
            {
                return indexToMixWith;
            }
            var indexInMiniBatchYContent = yOriginalMiniBatch.RowSpanSlice(indexInMiniBatch, 0);
            for (int nbTries = 0; nbTries < 10; ++nbTries)
            {
                var randomIndex = rand.Next(miniBatchSize);
                if (randomIndex != indexInMiniBatch)
                {
                    var randomIndexYContent = yOriginalMiniBatch.RowSpanSlice(randomIndex, 0);
                    bool sameYContent = true;
                    for (int j=0;j< randomIndexYContent.Length;++j)
                    {
                        if (Math.Abs(indexInMiniBatchYContent[j]- randomIndexYContent[j])>1e-6)
                        {
                            sameYContent = false;
                            break;
                        }
                    }
                    if (sameYContent)
                    {
                        return randomIndex;
                    }
                }

            }
            return -1; //fail to find another entry with the same content
        }
    }
}