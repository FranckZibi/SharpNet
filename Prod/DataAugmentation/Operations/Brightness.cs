using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Brightness : Operation
    {
        private readonly float _blackMean;
        private readonly float _enhancementFactor;

        /// <summary>
        /// </summary>
        /// <param name="enhancementFactor"></param>
        /// <param name="blackMean"></param>
        public Brightness(float enhancementFactor, float blackMean)
        {
            _enhancementFactor = enhancementFactor;
            _blackMean = blackMean;
        }

        public override float AugmentedValue(float initialValue, int indexInMiniBatch,
            CpuTensor<float> xOriginalMiniBatch, CpuTensor<float> xDataAugmentedMiniBatch, int channel, int rowOutput,
            int colOutput, out bool isFinalAugmentedValue)
        {
            isFinalAugmentedValue = false;
            return (1 - _enhancementFactor) * _blackMean + _enhancementFactor * initialValue;
        }
    }
}