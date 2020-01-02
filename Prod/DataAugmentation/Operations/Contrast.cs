using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Contrast : Operation
    {
        private readonly float _enhancementFactor;
        private readonly float _greyMean;

        public Contrast(float enhancementFactor, float greyMean)
        {
            _enhancementFactor = enhancementFactor;
            _greyMean = greyMean;
        }

        public override float AugmentedValue(int indexInMiniBatch, int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            var initialValue = xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
            return (1 - _enhancementFactor) * _greyMean + _enhancementFactor * initialValue;
        }
    }
}