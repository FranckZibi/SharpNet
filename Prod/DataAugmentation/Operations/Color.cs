using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Color : Operation
    {
        private readonly float _enhancementFactor;

        /// <summary>
        /// </summary>
        /// <param name="enhancementFactor"></param>
        public Color(float enhancementFactor)
        {
            _enhancementFactor = enhancementFactor;
        }

        public override float AugmentedValue(float initialValue, int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch, CpuTensor<float> xDataAugmentedMiniBatch, int channel, int rowOutput,
            int colOutput)
        {
            if (channel != 2)
            {
                return initialValue;
            }
            var r = xDataAugmentedMiniBatch.Get(indexInMiniBatch, 0, rowOutput, colOutput);
            var g = xDataAugmentedMiniBatch.Get(indexInMiniBatch, 1, rowOutput, colOutput);
            var b = initialValue;
            var greyScale = GetGreyScale(r, g, b);
            xDataAugmentedMiniBatch.Set(indexInMiniBatch, 0, rowOutput, colOutput, (1 - _enhancementFactor) * greyScale+ _enhancementFactor*r);
            xDataAugmentedMiniBatch.Set(indexInMiniBatch, 1, rowOutput, colOutput, (1 - _enhancementFactor) * greyScale+_enhancementFactor*g);
            return (1 - _enhancementFactor) * greyScale + _enhancementFactor* b;
        }
    }
}