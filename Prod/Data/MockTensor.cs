using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;

namespace SharpNet.Data
{
    public class MockTensor<T> : Tensor
    {
        public MockTensor(int[] shape) : base(shape, Marshal.SizeOf(typeof(T)), false)
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
        public override Tensor WithNewShape(int[] newShape)
        {
            throw new NotImplementedException();
        }

        public override void Dispose()
        {
            Shape = null;
            CapacityInBytes = 0;
        }
        public override string ToString()
        {
            return "Mock";
        }

        #region not implemented methods
        public override void ZeroMemory() {throw new NotImplementedException();}
        public override void Dot(Tensor a, bool transposeA, Tensor b, bool transposeB, float alpha, float beta) {throw new NotImplementedException();}
        public override void MultiplyTensor(Tensor a, Tensor diagonalMatrix){throw new NotImplementedException();}
        public override void UpSampling2D(Tensor tensorBeforeUpSampling, int rowFactor, int colFactor, UpSampling2DLayer.InterpolationEnum interpolation) {throw new NotImplementedException();}
        public override void DownSampling2D(Tensor tensorBeforeDownSampling, int rowFactor, int colFactor) {throw new NotImplementedException();}
        public override void ZeroPadding(Tensor unpaddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight) {throw new NotImplementedException();}
        public override void ZeroUnpadding(Tensor paddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight) {throw new NotImplementedException();}
        public override void MultiplyEachRowIntoSingleValue(Tensor a, Tensor b){throw new NotImplementedException();}
        public override void BroadcastAddVectorToOutput(Tensor y){throw new NotImplementedException();}
        public override void Switch_First_2_axis(Tensor target) {throw new NotImplementedException();}
        public override void Update_Adding_Alpha_X(float alpha, Tensor x){throw new NotImplementedException();}
        public override void AddTensor(float alpha, Tensor x, float beta){throw new NotImplementedException();}
        public override void Concatenate(IList<Tensor> tensors){throw new NotImplementedException();}
        public override void Split(IList<Tensor> tensors){throw new NotImplementedException();}
        public override void Update_Multiplying_By_Alpha(float alpha){throw new NotImplementedException();}
        public override void ActivationForward(cudnnActivationMode_t activationType, Tensor activationParameter,
            Tensor y){throw new NotImplementedException();}
        public override void Convolution(Tensor convolution, int paddingTop, int paddingBottom, int paddingLeft,
            int paddingRight, int stride,
            Tensor y, bool isDepthwiseConvolution, GPUWrapper.ConvolutionAlgoPreference forwardAlgoPreference,
            TensorMemoryPool memoryPool)
        {
            throw new NotImplementedException();
        }
        public override void BroadcastConvolutionBiasToOutput(Tensor y){throw new NotImplementedException();}
        public override void ConvolutionBackwardBias(Tensor bias){throw new NotImplementedException();}
        public override void ConvolutionGradient(Tensor convolution, Tensor dy, int paddingTop, int paddingBottom,
            int paddingLeft,
            int paddingRight, int stride, Tensor dx, Tensor convGradient, bool isDepthwiseConvolution,
            GPUWrapper.ConvolutionAlgoPreference backwardAlgoPreference, TensorMemoryPool memoryPool)
        {
            throw new NotImplementedException();
        }
        public override void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride){throw new NotImplementedException();}
        public override void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth,int poolingStride){throw new NotImplementedException();}
        public override void CopyTo(Tensor b){throw new NotImplementedException();}
        public override void CopyTo(int startElement, Tensor other, int otherStartElement, int elementCount){throw new NotImplementedException();}
        public override void ActivationBackward(cudnnActivationMode_t activationType, Tensor activationParameter,
            Tensor dy,
            Tensor x, Tensor y){throw new NotImplementedException();}
        public override void Compute_BiasGradient_from_dy(Tensor biasGradient){throw new NotImplementedException();}
        public override void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon, Tensor dW, Tensor adam_vW,Tensor adam_sW, int timestep){throw new NotImplementedException();}
        public override void UpdateSGDOptimizer(double learningRate, double momentum, bool usenesterov, Tensor dW, Tensor velocity){throw new NotImplementedException();}
        public override void YOLOV3Forward(Tensor x, int inputImageHeight, int inputImageWidth, int[] anchors) {throw new NotImplementedException();}
        public override Tensor Slice(int startIndex, int[] sliceShape) { throw new NotImplementedException(); }
        public override bool IsOwnerOfMemory => true;

        public override void AssertIsNotDisposed(){}
        public override void BatchNormalization(Tensor y, Tensor scale, Tensor bias, double exponentialAverageSmoothingFactor,
            Tensor runningInputMean, Tensor runningInputVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer,
            Tensor invertOfUnbiasedVolatilityBuffer, bool isTraining)
        {
            throw new NotImplementedException();
        }
        public override void BatchNormalizationBackward(Tensor dy, Tensor dx, Tensor scale, Tensor scaleGradient, Tensor biasGradient,cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer){throw new NotImplementedException();}
        public override void DropoutForward(Tensor y, double dropProbability, bool isTraining, Random dropoutRandom, Tensor dropoutReservedSpaceForTraining, Tensor randomNumberGeneratorStatesBufferForGPU) {throw new NotImplementedException();}
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropProbability, Tensor dropoutReserveSpace) {throw new NotImplementedException();}
        public override double ComputeAccuracy(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer){throw new NotImplementedException();}
        public override void CategoricalCrossentropyWithHierarchyGradient(Tensor yExpected, Tensor yPredicted) {throw new NotImplementedException();}
        public override void HuberGradient(Tensor yExpected, Tensor yPredicted, float huberDelta) {throw new NotImplementedException();}
        public override void HuberLoss(Tensor yExpected, Tensor yPredicted, float huberDelta) { throw new NotImplementedException(); }
        public override IntPtr Pointer => throw new NotImplementedException();
        public override double ComputeLoss(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer){throw new NotImplementedException();}
        public override void RandomizeUniformDistribution(Random rand, double minValue, double maxValue) {throw new NotImplementedException();}
        public override void RandomMatrixNormalDistribution(Random rand, double mean, double stdDev){throw new NotImplementedException();}
        public override void SetValue(float sameValue){throw new NotImplementedException();}
        public override float[] ContentAsFloatArray(){throw new NotImplementedException();}
        public override Tensor Clone() {throw new NotImplementedException();}
        public override void WordEmbeddingForwardPropagation(Tensor x, Tensor wordEmbedding) { throw new NotImplementedException(); }
        public override void WordEmbeddingBackwardPropagation(Tensor tensor, Tensor dy) {throw new NotImplementedException();}
        #endregion
    }
}