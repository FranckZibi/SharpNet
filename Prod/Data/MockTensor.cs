using System;
using System.Runtime.InteropServices;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Data
{
    public class MockTensor<T> : Tensor
    {
        public MockTensor(int[] shape, string description) : base(shape, Marshal.SizeOf(typeof(T)), false, description)
        {
            CapacityInBytes = (ulong)(Utils.Product(Shape) * TypeSize);
        }
        public override void Reshape(int[] newShape)
        {
            if (Shape == null)
            {
                throw new ArgumentException("MockTensor has been disposed");
            }
            CapacityInBytes = Math.Max(CapacityInBytes, (ulong)(Utils.Product(newShape) * TypeSize));
            Shape = newShape;
            RecomputeMultDim();
        }
        public override void Dispose()
        {
            Shape = null;
            CapacityInBytes = 0;
        }
        public override string ToString()
        {
            return Description;
        }

        #region not implemented methods
        public override void ZeroMemory() {throw new NotImplementedException();}
        public override void Dot(Tensor a, bool transposeA, Tensor b, bool transposeB, float alpha, float beta) {throw new NotImplementedException();}
        public override void MultiplyTensor(Tensor a, Tensor x){throw new NotImplementedException();}
        public override void ZeroPadding(Tensor unpaddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight) {throw new NotImplementedException();}
        public override void ZeroUnpadding(Tensor paddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight) {throw new NotImplementedException();}
        public override void MultiplyEachRowIntoSingleValue(Tensor a, Tensor b){throw new NotImplementedException();}
        public override void BroadcastAddVectorToOutput(Tensor y){throw new NotImplementedException();}
        public override void From_NCH_to_NH(Tensor tensor_NH, int channel){throw new NotImplementedException();}
        public override void Update_Adding_Alpha_X(float alpha, Tensor x){throw new NotImplementedException();}
        public override void AddTensor(float alpha, Tensor x, float beta){throw new NotImplementedException();}
        public override void Concatenate(Tensor a, Tensor b){throw new NotImplementedException();}
        public override Tensor Clone(GPUWrapper gpuWrapper){throw new NotImplementedException();}
        public override void Split(Tensor a, Tensor b){throw new NotImplementedException();}
        public override void Update_Multiplying_By_Alpha(float alpha){throw new NotImplementedException();}
        public override void ActivationForward(cudnnActivationMode_t activationType, Tensor y){throw new NotImplementedException();}
        public override void Convolution(Tensor convolution, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int stride,
            Tensor y, bool isDepthwiseConvolution, GPUWrapper.ConvolutionAlgoPreference forwardAlgoPreference)
        {
            throw new NotImplementedException();
        }
        public override void BroadcastConvolutionBiasToOutput(Tensor y){throw new NotImplementedException();}
        public override void ConvolutionBackwardBias(Tensor bias){throw new NotImplementedException();}
        public override void ConvolutionGradient(Tensor convolution, Tensor dy, int paddingTop, int paddingBottom, int paddingLeft,
            int paddingRight, int stride, Tensor dx, Tensor convGradient, bool isDepthwiseConvolution,
            GPUWrapper.ConvolutionAlgoPreference backwardAlgoPreference)
        {
            throw new NotImplementedException();
        }
        public override void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride){throw new NotImplementedException();}
        public override void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth,int poolingStride){throw new NotImplementedException();}
        public override void CopyTo(Tensor b){throw new NotImplementedException();}
        public override void CopyTo(int startElement, Tensor other, int otherStartElement, int elementCount){throw new NotImplementedException();}
        public override void ActivationBackward(Tensor dy, Tensor x, cudnnActivationMode_t activationType, Tensor dx){throw new NotImplementedException();}
        public override void Compute_BiasGradient_from_dy(Tensor biasGradient){throw new NotImplementedException();}
        public override void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon, Tensor dW, Tensor adam_vW,Tensor adam_sW, int timestep){throw new NotImplementedException();}
        public override void UpdateSGDOptimizer(double learningRate, double momentum, bool usenesterov, Tensor dW, Tensor velocity){throw new NotImplementedException();}
        public override Tensor ExtractSubTensor(int startRowIndex, int nbRows){throw new NotImplementedException();}
        public override void AssertIsNotDisposed(){}
        public override Tensor Transpose(){throw new NotImplementedException();}
        public override void BatchNormalization(Tensor y, Tensor scale, Tensor bias, double exponentialAverageSmoothingFactor,
            Tensor runningInputMean, Tensor runningInputVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer,
            Tensor invertOfUnbiasedVolatilityBuffer, bool isTraining)
        {
            throw new NotImplementedException();
        }
        public override void BatchNormalizationBackward(Tensor dy, Tensor dx, Tensor scale, Tensor scaleGradient, Tensor biasGradient,cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer){throw new NotImplementedException();}

        public override void DropoutForward(Tensor y, double dropProbability, bool isTraining, Random dropoutRandom, Tensor dropoutMaskBufferForCpu, ref DeviceMemory randomNumberGeneratorStatesBufferForGPU,ref DeviceMemory dropoutReserveSpaceForGPU, ref IntPtr dropoutDescriptorForGPU) 
        {
            throw new NotImplementedException();
        }
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropProbability, Tensor dropoutMaskBufferForCpu, DeviceMemory randomNumberGeneratorStatesBufferForGPU, DeviceMemory dropoutReserveSpaceForGPU,IntPtr dropoutDescriptorForGPU)
        {
            throw new NotImplementedException();
        }
        public override double ComputeAccuracy(Tensor yPredicted, Tensor notUsedBuffer){throw new NotImplementedException();}
        public override double ComputeAccuracyFromCategoryIndexes(Tensor yPredicted, Tensor notUsedBuffer){throw new NotImplementedException();}
        protected override IntPtr DevicePointer => throw new NotImplementedException();
        public override double ComputeLoss(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer){throw new NotImplementedException();}
        public override double ComputeLossFromCategoryIndexes(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer){throw new NotImplementedException();}
        public override void RandomMatrixNormalDistribution(Random rand, double mean, double stdDev){throw new NotImplementedException();}
        public override void NewSameValueTensor(double sameValue){throw new NotImplementedException();}
        public override float[] ContentAsFloatArray(){throw new NotImplementedException();}
        #endregion
    }
}