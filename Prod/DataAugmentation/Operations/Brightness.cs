using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Brightness : Operation
    {
        private readonly float _blackMean;
        private readonly float _enhancementFactor;

        public Brightness(float enhancementFactor, float blackMean)
        {
            _enhancementFactor = enhancementFactor;
            _blackMean = blackMean;
        }

        public override float AugmentedValue(int indexInMiniBatch, int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            var initialValue = xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
            return (1 - _enhancementFactor) * _blackMean + _enhancementFactor * initialValue;
        }
    }
}