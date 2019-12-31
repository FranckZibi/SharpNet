using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Sharpness : Operation
    {
        private readonly List<Tuple<float, float>> _meanAndVolatilityForEachChannel;
        private readonly float _enhancementFactor;

        /// <summary>
        /// </summary>
        /// <param name="enhancementFactor"></param>
        public Sharpness(List<Tuple<float, float>> meanAndVolatilityForEachChannel, float enhancementFactor)
        {
            _meanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
            _enhancementFactor = enhancementFactor;
        }

        public override float AugmentedValue(float initialValue, int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch, CpuTensor<float> xDataAugmentedMiniBatch, int channel, int rowOutput,
            int colOutput)
        {
            var nbRows = xOriginalMiniBatch.Shape[2];
            var nbCols = xOriginalMiniBatch.Shape[3];

            int count = 0;
            var unnormalized = 0f;
            for (int row = Math.Max(rowOutput - 1,0); row <= Math.Min(rowOutput + 1, nbRows-1); ++row)
            {
                for (int col = Math.Max(colOutput - 1, 0); col <= Math.Min(colOutput + 1, nbCols - 1); ++col)
                {
                    int weight = (row == rowOutput && col == colOutput) ? 5 : 1;
                    unnormalized += weight * UnnormalizedValue(xOriginalMiniBatch.Get(indexInMiniBatch, channel, row, col), channel, _meanAndVolatilityForEachChannel);
                    count += weight;
                }
            }
            unnormalized /= count;

            //!D
            unnormalized = Math.Max(Math.Min(255, unnormalized), 0);

            var smoothValueNormalized = NormalizedValue(unnormalized, channel, _meanAndVolatilityForEachChannel);
            smoothValueNormalized = (1 - _enhancementFactor) * smoothValueNormalized + _enhancementFactor * initialValue;             
            return smoothValueNormalized;
        }
    }
}