using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Contrast : Operation
    {
        private readonly float _enhancementFactor;
        private readonly float _greyMean;

        /// <summary>
        /// </summary>
        /// <param name="enhancementFactor"></param>
        /// <param name="greyMean"></param>
        public Contrast(float enhancementFactor, float greyMean)
        {
            _enhancementFactor = enhancementFactor;
            _greyMean = greyMean;
        }

        public override float AugmentedValue(float initialValue, int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch, CpuTensor<float> xDataAugmentedMiniBatch, int channel, int rowOutput,
            int colOutput)
        {
            return (1 - _enhancementFactor) * _greyMean + _enhancementFactor * initialValue;
        }
    }
}