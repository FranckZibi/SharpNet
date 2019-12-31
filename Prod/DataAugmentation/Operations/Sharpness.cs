using System;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Sharpness : Operation
    {
        private readonly float _enhancementFactor;

        public Sharpness(float enhancementFactor)
        {
            _enhancementFactor = enhancementFactor;
        }

        public override float AugmentedValue(float initialValue, int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch, CpuTensor<float> xDataAugmentedMiniBatch, int channel, int rowOutput,
            int colOutput)
        {
            var nbRows = xDataAugmentedMiniBatch.Shape[2];
            var nbCols = xDataAugmentedMiniBatch.Shape[3];
            int count = 0;
            var smoothValue = 0f;
            /*
             * We use the following weight value centered on the pixel
             *      [ [ 1 1 1 ]
             *        [ 1 5 1 ]
             *        [ 1 1 1 ] ]
             */
            for (int row = Math.Max(rowOutput - 1, 0); row <= Math.Min(rowOutput + 1, nbRows - 1); ++row)
            {
                for (int col = Math.Max(colOutput - 1, 0); col <= Math.Min(colOutput + 1, nbCols - 1); ++col)
                {
                    int weight = (row == rowOutput && col == colOutput) ? 5 : 1;
                    smoothValue += weight * xDataAugmentedMiniBatch.Get(indexInMiniBatch, channel, row, col);
                    count += weight;
                }
            }
            smoothValue /= count;
            smoothValue = (1 - _enhancementFactor) * smoothValue + _enhancementFactor * initialValue;
            return smoothValue;
        }
    }
}
