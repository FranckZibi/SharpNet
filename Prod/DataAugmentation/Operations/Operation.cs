using System;
using SharpNet.CPU;

namespace SharpNet.DataAugmentation.Operations
{
    public abstract class Operation
    {
        public virtual (double row, double col) Unconvert_Slow(double row, double col)
        {
            return (row, col);
        }

        public virtual float AugmentedValue(float originalValue, int channelOutput, int rowOutput, int colOutput, out bool isFinalAugmentedValue)
        {
            isFinalAugmentedValue = false;
            return originalValue;
        }

        public virtual void UpdateY(CpuTensor<float> yMiniBatch, int indexInMiniBatch, Func<int, int> indexInMiniBatchToCategoryId)
        {
        }
    }
}