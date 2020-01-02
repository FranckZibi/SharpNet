using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public class Color : Operation
    {
        private readonly float _enhancementFactor;

        public Color(float enhancementFactor)
        {
            _enhancementFactor = enhancementFactor;
        }

        public override float AugmentedValue(int indexInMiniBatch, int channel,
            CpuTensor<float> xInputMiniBatch, int rowInput, int colInput, 
            CpuTensor<float> xOutputMiniBatch, int rowOutput, int colOutput)
        {
            var initialValue = xInputMiniBatch.Get(indexInMiniBatch, channel, rowInput, colInput);
            if (channel != 2)
            {
                return initialValue;
            }
            var r = xInputMiniBatch.Get(indexInMiniBatch, 0, rowInput, colInput);
            var g = xInputMiniBatch.Get(indexInMiniBatch, 1, rowInput, colInput);
            var b = initialValue;
            var greyScale = GetGreyScale(r, g, b);
            xOutputMiniBatch.Set(indexInMiniBatch, 0, rowOutput, colOutput, (1 - _enhancementFactor) * greyScale+ _enhancementFactor*r);
            xOutputMiniBatch.Set(indexInMiniBatch, 1, rowOutput, colOutput, (1 - _enhancementFactor) * greyScale+_enhancementFactor*g);
            return (1 - _enhancementFactor) * greyScale + _enhancementFactor* b;
        }
    }
}