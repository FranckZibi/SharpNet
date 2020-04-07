
// When enabled:
//      will only use native cuDNN functions to compute Swish Activation (forward&backward)
//      (under testing)
// Else:
//      will use cuda hand coded function 
//#define USE_NATIVE_CUDNN_SWISH

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.GPU
{
    public sealed unsafe class GPUTensor<T> : Tensor
    {
        #region Private fields
        private DeviceMemory _deviceMemory;
        #endregion

        public GPUTensor(GPUTensor<T> memoryOwner, int[] shape, int offsetInBytes, string description) : base(shape, Marshal.SizeOf(typeof(T)), true, description)
        {
            Debug.Assert(memoryOwner.DevicePointer != IntPtr.Zero);
            Wrapper = memoryOwner.Wrapper;
            CapacityInBytes = ReallyNeededMemoryInBytes;
            _deviceMemory = Wrapper.NewDeviceMemory(memoryOwner.DevicePointer+offsetInBytes, CapacityInBytes);
        }

        public GPUTensor(int[] shape, string description, GPUWrapper wrapper) : this(shape, IntPtr.Zero, description, wrapper)
        {
        }

        public GPUTensor(int[] shape, IntPtr hostMemoryPointer, string description, GPUWrapper wrapper) : base(shape, Marshal.SizeOf(typeof(T)), true, description)
        {
            Wrapper = wrapper;
            Wrapper.CheckThreadId();

            CapacityInBytes = ReallyNeededMemoryInBytes;
            _deviceMemory = Wrapper.NewDeviceMemory(CapacityInBytes);
            if (hostMemoryPointer != IntPtr.Zero)
            {
                CopyToDevice(hostMemoryPointer, false);
            }
        }

        private GPUWrapper Wrapper { get; }

        /// <summary>
        /// copy from CPU (Host) to GPU (Device) memory
        /// </summary>
        /// <param name="hostPinnedPointer">point to host (pinned) memory (in CPU) </param>
        /// <param name="useSynchronousCall">true if we want to make a synchronous copy from host to device
        /// false for asynchronous copy</param>
        public void CopyToDevice(IntPtr hostPinnedPointer, bool useSynchronousCall)
        {
            AssertIsNotDisposed();
            Debug.Assert(hostPinnedPointer != IntPtr.Zero);
            Wrapper.SwCopyToDevice.Start();
            Wrapper.LogCopyToDeviceCall(ReallyNeededMemoryInBytes);

            var res =useSynchronousCall
                    ?NVCudaWrapper.cuMemcpyHtoD_v2(DevicePointer, hostPinnedPointer, ReallyNeededMemoryInBytes)
                    :NVCudaWrapper.cuMemcpyHtoDAsync_v2(DevicePointer, hostPinnedPointer, ReallyNeededMemoryInBytes, Wrapper.DefaultStream.StreamHandle);

            GPUWrapper.CheckStatus(res, ToString);
            Wrapper.SwCopyToDevice.Stop();
        }

        /// <summary>
        /// copy the first 'Count' element of 'buffer into the 'this' Tensor
        /// </summary>
        /// <param name="buffer">a buffer to read from
        /// It must contains at least 'Count' elements
        /// </param>
        public void CopyToDevice(T[] buffer)
        {
            Debug.Assert(buffer.Length >= Count);
            using (var m = new HostPinnedMemory<T>(buffer))
            {
                CopyToDevice(m.Pointer, false);
            }
        }

        private T[] DeviceContent()
        {
            Debug.Assert(!_disposed);
            Wrapper.SwCopyToHost.Start();
            Wrapper.LogCopyToHostCall(ReallyNeededMemoryInBytes);

            var _hostMemory = new T[Count];
            var handle = GCHandle.Alloc(_hostMemory, GCHandleType.Pinned);
            var _hostMemoryPointer = handle.AddrOfPinnedObject();
            var res = NVCudaWrapper.cuMemcpyDtoH_v2(_hostMemoryPointer, DevicePointer, ReallyNeededMemoryInBytes);
            GPUWrapper.CheckStatus(res, ToString);
            handle.Free();
            Wrapper.SwCopyToHost.Stop();
            return _hostMemory;
        }

        #region Tensor implementation
        public override void BatchNormalization(Tensor y, Tensor scale, Tensor bias, double exponentialAverageSmoothingFactor, Tensor runningInputMean, Tensor runningInputVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer, bool isTraining)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y, scale, bias, runningInputMean, runningInputVariance, meanBuffer, invertOfUnbiasedVolatilityBuffer }));
            var xDesc = TensorDesc(x);
            var yDesc = xDesc;
            var bnScaleBiasMeanVarDesc = TensorDesc(scale);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            if (isTraining)
            {
               var res = CudnnWrapper.cudnnBatchNormalizationForwardTraining(CudnnHandle, mode, one, zero,
                        xDesc, x, yDesc, y,
                        bnScaleBiasMeanVarDesc, scale, bias, exponentialAverageSmoothingFactor,
                        runningInputMean,
                        runningInputVariance, epsilon, meanBuffer,
                        invertOfUnbiasedVolatilityBuffer);
                CheckStatus(res);
            }
            else
            {
                var res = CudnnWrapper.cudnnBatchNormalizationForwardInference(CudnnHandle, mode, one, zero,
                        xDesc, x, yDesc, y,
                        bnScaleBiasMeanVarDesc, scale, bias, 
                        runningInputMean,
                        runningInputVariance, epsilon);
                CheckStatus(res);
            }
        }
        public override void BatchNormalizationBackward(Tensor dy, Tensor dx, Tensor scale, Tensor scaleGradient, Tensor biasGradient, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, dy, dx, scale, scaleGradient, biasGradient, meanBuffer, invertOfUnbiasedVolatilityBuffer }));
            var xDesc = TensorDesc(x);
            var bnScaleBiasDiffDesc = TensorDesc(scale);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnBatchNormalizationBackward(CudnnHandle, mode, 
                one, zero, one, zero,
                xDesc, x,
                xDesc, dy,
                xDesc, dx,
                bnScaleBiasDiffDesc, scale, scaleGradient, biasGradient,
                epsilon, meanBuffer,
                invertOfUnbiasedVolatilityBuffer);
            CheckStatus(res);
        }


#if USE_NATIVE_CUDNN_SWISH
        //under testing: new way of computing swish using only native cuDNN functions
        private Tensor tmpTensorForActivation;
#endif
        public override void ActivationForward(cudnnActivationMode_t activationType, Tensor y)
        {
            AssertIsNotDisposed();
            y.AssertIsNotDisposed();

            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> {x, y}));
            Debug.Assert(x.SameShape(y));
            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            cudnnStatus_t res;
            if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)
            {
                res = CudnnWrapper.cudnnSoftmaxForward(CudnnHandle, cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_INSTANCE, one, xDesc, x, zero, yDesc, y);
            }
            else if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH)
            {
                // y = x * sigmoid(x) 
#if USE_NATIVE_CUDNN_SWISH
                //under testing: new way of computing swish using only native cuDNN functions
                //we'll store in 'tmpTensorForActivation' sigmoid(x)
                tmpTensorForActivation = NewNotInitializedFloatTensor(y.Shape, tmpTensorForActivation, nameof(tmpTensorForActivation), Wrapper);
                x.ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID, tmpTensorForActivation);
                y.MultiplyTensor(x, tmpTensorForActivation);
#else
                ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID, y);
                y.Update_Multiply_By_x(this);
#endif
                res = cudnnStatus_t.CUDNN_STATUS_SUCCESS;
            }
            else
            {
                var activationDescriptor = ActivationDesc(activationType);
                res = CudnnWrapper.cudnnActivationForward(CudnnHandle, activationDescriptor, one, xDesc, x, zero, yDesc, y);
            }
            CheckStatus(res);
        }
        public override void ActivationBackward(Tensor dy, Tensor x, cudnnActivationMode_t activationType, Tensor dx)
        {
            var y = this;
            Debug.Assert(AreCompatible(new List<Tensor> {y, dy, x, dx}));
            Debug.Assert(y.SameShape(dy, x, dx));
            var yDesc = TensorDesc(y);
            var xDesc = TensorDesc(x);
            var dyDesc = TensorDesc(dy);
            var dxDesc = TensorDesc(dx);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            cudnnStatus_t res;
            if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)
            {
                res = CudnnWrapper.cudnnSoftmaxBackward(CudnnHandle, cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_INSTANCE, one, yDesc, y, dyDesc, dy, zero, dxDesc, dx);
            }
            else if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH)
            {
#if USE_NATIVE_CUDNN_SWISH
                //under testing: new way of computing swish using only native cuDNN functions
                //we know that 'x.tmpTensorForActivation' contains sigmoid(x)
                var sigmoidX = ((GPUTensor<float>)x).tmpTensorForActivation;
                var sigmoidActivationDesc = ActivationDesc(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);
                // dx = x* (sigmoid_x * (1 - sigmoid_x)
                res = CudnnWrapper.cudnnActivationBackward(CudnnHandle, sigmoidActivationDesc, one, yDesc, sigmoidX, dyDesc, x /*dy*/, xDesc, x, zero, dxDesc, dx);
                CheckStatus(res);
                // dx = sigmoid_x + x* (sigmoid_x * (1 - sigmoid_x)
                dx.AddTensor(1, sigmoidX, 1);
                dx.Update_Multiply_By_x(dy);
#else
                Wrapper.RunKernel("SwishGradient", dx.Count, new object[] { y, dy, x, dx });
                res = cudnnStatus_t.CUDNN_STATUS_SUCCESS;
#endif
            }
            else
            {
                var activationDesc = ActivationDesc(activationType);
                res = CudnnWrapper.cudnnActivationBackward(CudnnHandle, activationDesc, one, yDesc, y, dyDesc, dy, xDesc, x, zero, dxDesc, dx);
            }
            CheckStatus(res);
        }

        public override void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            var poolingDesc = PoolingDesc(poolingMode, poolingHeight, poolingWidth, poolingStride);
            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnPoolingForward(CudnnHandle, poolingDesc, one, xDesc, x, zero, yDesc, y);
            CheckStatus(res);
        }
        public override void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, y, x, dx }));
            var poolingDesc = PoolingDesc(poolingMode, poolingHeight, poolingWidth, poolingStride);
            var xDesc = TensorDesc(x);
            var dxDesc = TensorDesc(dx);
            var yDesc = TensorDesc(y);
            var dyDesc = TensorDesc(dy);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnPoolingBackward(CudnnHandle, poolingDesc, one, yDesc, y, dyDesc, dy, xDesc, x, zero, dxDesc, dx);
            CheckStatus(res);
        }
        public override void From_NCH_to_NH(Tensor tensor_NH, int channel)
        {
            throw new NotImplementedException(); //TODO
        }


        //TODO
        public override Tensor Transpose()
        {
            throw new NotImplementedException();
        }

        public override void Concatenate(Tensor a, Tensor b)
        {
#if DEBUG
            CheckConcatenate(a, b);
#endif
            var concat = this;
            Wrapper.RunKernel("Concatenate", Count, new object[] { Shape[0], concat, concat.MultDim0, a, a.MultDim0, b, b.MultDim0});
        }

        public override Tensor Clone(GPUWrapper gpuWrapper)
        {
            var result = new GPUTensor<T>((int[]) Shape.Clone(), Description, gpuWrapper??Wrapper);
            result.CopyToDevice(DeviceContent());
            return result;
        }

        public override void Split(Tensor a, Tensor b)
        {
#if DEBUG
            CheckConcatenate(a, b);
#endif
            var concat = this;
            Wrapper.RunKernel("Split", Count, new object[] { Shape[0], concat, concat.MultDim0, a, a.MultDim0, b, b.MultDim0 });
        }
        /// <summary>
        /// resize the current GPU tensor to a different shape
        /// </summary>
        /// <param name="newShape"></param>
        public override void Reshape(int[] newShape)
        {
            AssertIsNotDisposed();
            if (ReallyNeededMemoryInBytesForShape(newShape) <= CapacityInBytes)
            {
                //smaller shape
                Shape = newShape;
                RecomputeMultDim();
            }
            else
            {
                //bigger shape : we do not have enough space to store it
                Shape = newShape;
                RecomputeMultDim();
                CapacityInBytes = ReallyNeededMemoryInBytes;
                _deviceMemory?.Dispose();
                _deviceMemory = Wrapper.NewDeviceMemory(CapacityInBytes);
            }
        }

        /// <summary>
        /// compute: this = alpha * this 
        /// </summary>
        /// <param name="alphaFloat"></param>
        public override void Update_Multiplying_By_Alpha(float alphaFloat)
        {
            var y = this;
            var yDesc = TensorDesc(y);
            var res = CudnnWrapper.cudnnScaleTensor(CudnnHandle, yDesc, y, &alphaFloat);
            CheckStatus(res);
        }

        public override void BroadcastAddVectorToOutput(Tensor y)
        {
            var bias = this;
            Debug.Assert(AreCompatible(new List<Tensor> { bias, y }));
            Debug.Assert(y.Dimension >= 2);
            Debug.Assert(y.MultDim0 == Count);
            y.Update_Adding_Alpha_X(1, bias);
        }
        public override void Compute_BiasGradient_from_dy(Tensor biasGradient)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, biasGradient}));
            Debug.Assert(Dimension >= 2);
            var dyDesc = TensorDesc(dy);
            var dbDesc = TensorDesc(biasGradient);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnConvolutionBackwardBias(CudnnHandle, one, dyDesc, dy, zero, dbDesc, biasGradient);
            CheckStatus(res);
        }

#region Convolution
        public override void Convolution(Tensor filters, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int stride, Tensor y, bool isDepthwiseConvolution)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, filters, y }));
            int inputChannelCount = x.Shape[1];
            Debug.Assert(inputChannelCount == filters.Shape[1]);

            if (isDepthwiseConvolution)
            {
                //only depthMultiplier=1 is supported
                int depthMultiplier = filters.Shape[0];
                if (depthMultiplier != 1)
                {
                    throw new NotImplementedException("only depthMultiplier=1 is supported");
                }
                Debug.Assert(inputChannelCount == y.Shape[1]);
            }

            int groupCount = isDepthwiseConvolution ? inputChannelCount : 1;
            var convDesc = ConvDesc(paddingTop, paddingBottom, paddingLeft, paddingRight, stride, groupCount);
            var filterDesc = FilterDesc(filters, isDepthwiseConvolution);
            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);


            var res = CudnnWrapper.cudnnGetConvolutionForwardAlgorithm(CudnnHandle, xDesc, filterDesc, convDesc, yDesc, cudnnConvolutionFwdPreference_t.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, out cudnnConvolutionFwdAlgo_t algo);
            CheckStatus(res);
            res = CudnnWrapper.cudnnGetConvolutionForwardWorkspaceSize(CudnnHandle, xDesc, filterDesc, convDesc, yDesc, algo, out size_t workspaceSize); 
            CheckStatus(res);
            var storageBuffer = Wrapper.StorageBuffer(workspaceSize);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            res = CudnnWrapper.cudnnConvolutionForward(CudnnHandle, one, xDesc, x, filterDesc, filters, convDesc, algo, storageBuffer.Pointer, storageBuffer.SizeInBytes, zero, yDesc, y);
            CheckStatus(res);
        }
        public override void BroadcastConvolutionBiasToOutput(Tensor y)
        {
            var convolutionBias = this;
            Debug.Assert(AreCompatible(new List<Tensor> { convolutionBias, y }));
            y.Update_Adding_Alpha_X(1, convolutionBias);
        }
        public override void ConvolutionBackwardBias(Tensor bias)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, bias }));
            var dyDesc = TensorDesc(dy);
            var dbDesc = TensorDesc(bias);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnConvolutionBackwardBias(CudnnHandle, one, dyDesc, dy, zero, dbDesc, bias);
            CheckStatus(res);
        }
        public override void ConvolutionGradient(Tensor convolution, Tensor dy, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int stride, Tensor dx, Tensor convGradient, bool isDepthwiseConvolution)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, convolution, dy, dx, convGradient }));
            var xDesc = TensorDesc(x);
            var dyDesc = TensorDesc(dy);
            var dwDesc = FilterDesc(convGradient, isDepthwiseConvolution);
            int inputChannelCount = x.Shape[1];
            int groupCount = isDepthwiseConvolution ? inputChannelCount : 1;
            var convDesc = ConvDesc(paddingTop, paddingBottom, paddingLeft, paddingRight, stride, groupCount);
            var res = CudnnWrapper.cudnnGetConvolutionBackwardFilterAlgorithm(CudnnHandle, xDesc, dyDesc, convDesc, dwDesc, cudnnConvolutionBwdFilterPreference_t.CUDNN_CONVOLUTION_BWD_FILTER_​PREFER_FASTEST, 0, out cudnnConvolutionBwdFilterAlgo_t filterAlgo);
            CheckStatus(res);
            res = CudnnWrapper.cudnnGetConvolutionBackwardFilterWorkspaceSize(CudnnHandle, xDesc, dyDesc, convDesc, dwDesc, filterAlgo, out size_t filterWorkspaceSize);
            CheckStatus(res);
            var storageBuffer = Wrapper.StorageBuffer(Math.Max(1, filterWorkspaceSize));

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            res = CudnnWrapper.cudnnConvolutionBackwardFilter(CudnnHandle, one, xDesc, x, dyDesc, dy, convDesc, filterAlgo, storageBuffer.Pointer, storageBuffer.SizeInBytes, zero, dwDesc, convGradient);
            CheckStatus(res);

            if (dx == null)
            {
                return;
            }
            var dxDesc = TensorDesc(dx);
            var wDesc = FilterDesc(convolution, isDepthwiseConvolution);
            res = CudnnWrapper.cudnnGetConvolutionBackwardDataAlgorithm(CudnnHandle, wDesc, dyDesc, convDesc, dxDesc, cudnnConvolutionBwdDataPreference_t.CUDNN_CONVOLUTION_BWD_DATA_​PREFER_FASTEST, 0, out cudnnConvolutionBwdDataAlgo_t dataAlgo);
            CheckStatus(res);

            res = CudnnWrapper.cudnnGetConvolutionBackwardDataWorkspaceSize(CudnnHandle, dwDesc, dyDesc, convDesc, dxDesc, dataAlgo, out size_t dataWorkspaceSize);
            CheckStatus(res);

            storageBuffer = Wrapper.StorageBuffer(dataWorkspaceSize);
            res = CudnnWrapper.cudnnConvolutionBackwardData(CudnnHandle, one, wDesc, convolution, dyDesc, dy, convDesc, dataAlgo, storageBuffer.Pointer, storageBuffer.SizeInBytes, zero, dxDesc, dx);
            CheckStatus(res);
        }
#endregion
        public override void RandomMatrixNormalDistribution(Random rand, double mean, double stdDev)
        {
            var array = new float[Count];
            Utils.RandomizeNormalDistribution(array, rand, mean, stdDev);
            CopyToDevice(array as T[]);
        }

        public override void NewSameValueTensor(double sameValue)
        {
            var array = new float[Count];
            var sameValueAsFloat = (float) sameValue;
            for (int i = 0; i < array.Length; ++i)
            {
                array[i] = sameValueAsFloat;
            }
            CopyToDevice(array as T[]);
        }
        public override float[] ContentAsFloatArray()
        {
            return (DeviceContent() as float[]);
        }

        /// <summary>
        /// this = yExpectedOneHot
        /// </summary>
        /// <param name="yPredicted"></param>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public override double ComputeAccuracy(Tensor yPredicted, Tensor buffer)
        {
            var yExpectedOneHot = this;
            Debug.Assert(AreCompatible(new List<Tensor> {yExpectedOneHot, yPredicted}));
            Debug.Assert(yPredicted != null);
            Debug.Assert(buffer != null);
            Debug.Assert(yExpectedOneHot.SameShape(yPredicted));
            Debug.Assert(yExpectedOneHot.Dimension == 2);
            int nbRows = yExpectedOneHot.Shape[0];
            var categoryCount = yExpectedOneHot.MultDim0;
            Wrapper.RunKernel("ComputeAccuracy", nbRows, new object[] { categoryCount, buffer, yExpectedOneHot, yPredicted });
            var countOk = (int) buffer.ContentAsFloatArray().Sum();
            return ((double)countOk) / Shape[0];
        }

        public override double ComputeAccuracyFromCategoryIndexes(Tensor yPredicted, Tensor buffer)
        {
            var categoryIndexes = this;
            Debug.Assert(AreCompatible(new List<Tensor> { categoryIndexes, yPredicted }));
            Debug.Assert(yPredicted != null);
            Debug.Assert(buffer != null);
            Debug.Assert(categoryIndexes.Dimension == 1);
            int nbRows = yPredicted.Shape[0];
            var categoryCount = yPredicted.Shape[1];
            Debug.Assert(categoryIndexes.Shape[0] == nbRows);
            Wrapper.RunKernel("ComputeAccuracyFromCategoryIndexes", nbRows, new object[] { categoryCount, buffer, categoryIndexes, yPredicted });
            var countOk = (int)buffer.ContentAsFloatArray().Sum();
            return ((double)countOk) / Shape[0];
        }

        /// <summary>
        /// this = yExpectedOneHot
        /// </summary>
        /// <param name="yPredicted"></param>
        /// <param name="lossFunction"></param>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public override double ComputeLoss(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer)
        {
            var yExpectedOneHot = this;
            Debug.Assert(AreCompatible(new List<Tensor> { yExpectedOneHot, yPredicted }));
            Debug.Assert(yPredicted != null);
            Debug.Assert(buffer != null);
            Debug.Assert(yPredicted.SameShape(yExpectedOneHot));
            Debug.Assert(yExpectedOneHot.Dimension == 2);
            var kernelName = (lossFunction == NetworkConfig.LossFunctionEnum.BinaryCrossentropy)
                ? "ComputeBinaryCrossentropyLoss"
                : "ComputeCategoricalCrossentropyLoss";
            int nbRows = yExpectedOneHot.Shape[0];
            var categoryCount = yExpectedOneHot.Shape[1];
            Wrapper.RunKernel(kernelName, nbRows, new object[] { categoryCount, buffer, yExpectedOneHot, yPredicted });
            return ((double)buffer.ContentAsFloatArray().Sum() / nbRows);
        }


        /// <summary>
        /// this = expected Category Indexes
        /// </summary>
        /// <param name="yPredicted"></param>
        /// <param name="lossFunction"></param>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public override double ComputeLossFromCategoryIndexes(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer)
        {
            var categoryIndexes = this;
            int nbRows = yPredicted.Shape[0];
            Debug.Assert(AreCompatible(new List<Tensor> { categoryIndexes, yPredicted }));
            Debug.Assert(yPredicted != null);
            Debug.Assert(buffer != null);
            Debug.Assert(categoryIndexes.Dimension == 1);
            Debug.Assert(nbRows == categoryIndexes.Shape[0]);
            var kernelName = (lossFunction == NetworkConfig.LossFunctionEnum.BinaryCrossentropy)
                ? "ComputeBinaryCrossentropyLossFromCategoryIndexes"
                : "ComputeCategoricalCrossentropyLossFromCategoryIndexes";
            var categoryCount = yPredicted.Shape[1];
            Wrapper.RunKernel(kernelName, nbRows, new object[] { categoryCount, buffer, categoryIndexes, yPredicted });
            return ((double)buffer.ContentAsFloatArray().Sum() / nbRows);
        }

        private DeviceMemory _randomNumberGeneratorStatesBuffer;
        private DeviceMemory _dropoutReserveSpace;
        private IntPtr _dropoutDescriptor;
        public override void DropoutForward(Tensor y, double dropProbability, bool isTraining, Random dropoutRandom, Tensor dropoutMaskBuffer)
        {
            var x = this;
            Debug.Assert(dropoutMaskBuffer == null); //no need of dropout mask for GPU
            if (!isTraining)
            {
                x.CopyTo(y);
                return;
            }

            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);
            cudnnStatus_t res;
            if (_randomNumberGeneratorStatesBuffer == null)
            {
                res = CudnnWrapper.cudnnDropoutGetStatesSize(CudnnHandle, out var dropoutStateSize);
                CheckStatus(res);
                _randomNumberGeneratorStatesBuffer = Wrapper.NewDeviceMemory(Math.Max(dropoutStateSize, 1));
                res = CudnnWrapper.cudnnDropoutGetReserveSpaceSize(xDesc, out var dropoutReserveSpaceSize);
                CheckStatus(res);
                _dropoutReserveSpace = Wrapper.NewDeviceMemory(Math.Max(dropoutReserveSpaceSize, 1));
                _dropoutDescriptor = Wrapper.DropoutDesc(dropProbability, _randomNumberGeneratorStatesBuffer.Pointer);
            }
            res = CudnnWrapper.cudnnDropoutForward(CudnnHandle, _dropoutDescriptor, xDesc, x, yDesc, y, _dropoutReserveSpace.Pointer, _dropoutReserveSpace.SizeInBytes);
            CheckStatus(res);
        }

        /// <summary>
        /// this = x
        /// </summary>
        /// <param name="dy"></param>
        /// <param name="dx"></param>
        /// <param name="dropProbability"></param>
        /// <param name="usedDropoutMask"></param>
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropProbability, Tensor usedDropoutMask)
        {
            var dxDesc = TensorDesc(dx);
            var dyDesc = TensorDesc(dy);
            Debug.Assert(usedDropoutMask == null);
            var res = CudnnWrapper.cudnnDropoutBackward(CudnnHandle, _dropoutDescriptor, dyDesc, dy, dxDesc, dx, _dropoutReserveSpace.Pointer, _dropoutReserveSpace.SizeInBytes);
            CheckStatus(res);
        }
        public override void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon, Tensor dW, Tensor adam_vW, Tensor adam_sW, int timestep)
        {
            var W = this;
            var beta1_power = Math.Pow(beta1, timestep);
            var beta2_power = Math.Pow(beta2, timestep);
            var multiplicative_factor = learningRate * (Math.Sqrt(1.0 - beta2_power) / (1.0 - beta1_power));
            Wrapper.RunKernel("UpdateAdamOptimizer", Count, new object[] { beta1, beta2, epsilon, multiplicative_factor, dW, W, adam_vW, adam_sW });
        }
        public override void UpdateSGDOptimizer(double learningRate, double momentum, bool usenesterov, Tensor dW,
            Tensor velocity)
        {
            var W = this;
            //velocity[i] = (momentum * velocity[i]) - (dW[i] * learningRate);
            velocity.AddTensor((float)-learningRate, dW, (float)momentum);
            if (usenesterov)
            {
                //W[i] += momentum * velocity[i] - (dW[i] * learningRate);
                W.Update_Adding_Alpha_X((float)momentum, velocity);
                W.Update_Adding_Alpha_X((float)-learningRate, dW);
            }
            else
            {
                //W[i] += velocity[i];
                W.Update_Adding_Alpha_X(1, velocity);
            }
        }
        public override void Dot(Tensor a, bool transposeA, Tensor b, bool transposeB, float alpha, float beta)
        {
            AssertIsNotDisposed();
            a.AssertIsNotDisposed();
            b.AssertIsNotDisposed();
            Debug.Assert(AreCompatible(new List<Tensor> { this, a, b }));
            Debug.Assert(b.Dimension >= 2);
            Debug.Assert(a.Dimension >= 2);
            Debug.Assert(Dimension >= 2);
            var bH = b.Shape[0];
            var bW = b.MultDim0;
            var aH = a.Shape[0];
            var aW = a.MultDim0;
            var transLeft = transposeB ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N;
            var transRight = transposeA ? cublasOperation_t.CUBLAS_OP_T : cublasOperation_t.CUBLAS_OP_N;
            int N = transposeB ? bH : bW; //number of rows of the matrix op(Left) (= number of rows of matrix C)
            int M = transposeA ? aW : aH; //number of columns of the matrix op(Right) (= number of columns of the matrix C)
            int K = transposeB ? bW : bH; //number of columns of the matrix op(Left) (= number of rows of the matrix op(B))
            int ldb = bW; //number of rows of the matrix B (because order = ColumnMajor)
            int lda = aW; //number of rows of the matrix y (because order = ColumnMajor)
            int ldc = N; //number of rows of the matrix C (because order = ColumnMajor)
            //Cuda is column major : we have to compute B*y instead of y*B
            var res = CublasWrapper.cublasSgemm_v2(CublasHandle, transLeft, transRight, N, M, K, ref alpha, b, ldb, a, lda, ref beta, this, ldc);

            //The lib may return CUBLAS_STATUS_INTERNAL_ERROR (in version 10.0) in some cases => ignored (see Non Reg test that reproduces the issue)
            if (res == cublasStatus_t.CUBLAS_STATUS_INTERNAL_ERROR)
            {
                return;
            }

            GPUWrapper.CheckStatus(res, ToString);
        }

        // compute: this += alpha * bias
        public override void Update_Adding_Alpha_X(float alpha, Tensor x)
        {
            AddTensor(alpha, x, 1);
        }

        /// <summary>
        /// compute: this = alpha * x + beta * this
        /// Each dimension of the 'x' tensor must match the corresponding dimension of the destination tensor 'this' or must be equal to 1.
        /// In the latter case, the same value from 'x' for those dimensions will be used to blend into the 'this' tensor.
        /// </summary>
        /// <param name="alpha"></param>
        /// <param name="x"></param>
        /// <param name="beta"></param>
        public override void AddTensor(float alpha, Tensor x, float beta)
        {
            var c = this;
            Debug.Assert(AreCompatible(new List<Tensor> { c, x }));
            var cDesc = TensorDesc(c);
            var xDesc = TensorDesc(x);
            var res = CudnnWrapper.cudnnAddTensor(CudnnHandle, &alpha, xDesc, x, &beta, cDesc, c);
            CheckStatus(res);
        }

        public override void MultiplyTensor(Tensor a, Tensor x)
        {
            var c = this;
            var mode = cublasSideMode_t.CUBLAS_SIDE_RIGHT;
            Debug.Assert(Count%x.Count == 0);
            int m = Count/x.Count; //number of rows of matrix A and C.
            int n = x.Count; //number of columns of matrix A and C.
            //IntPtr A; //input<type> array of dimensions lda x n with lda >= max(1, m)
            int lda = m; //leading dimension of two-dimensional array used to store the matrix A.
            //IntPtr x; //input one-dimensional < type > array of size | i n c | × m if mode == CUBLAS_SIDE_LEFT and | i n c | × n if mode == CUBLAS_SIDE_RIGHT
            int incx = 1; //stride of one - dimensional array x.
            //IntPtr C; //in/out	< type > array of dimensions ldc x n with ldc >= max(1, m).
            int ldc = lda; ///leading dimension of a two - dimensional array used to store the matrix C.
            var res = CublasWrapper.cublasSdgmm(CublasHandle, mode, m, n, a, lda, x, incx, c, ldc);
            GPUWrapper.CheckStatus(res, ToString);
        }

        public override void MultiplyEachRowIntoSingleValue(Tensor a, Tensor b)
        {
            Debug.Assert(a.SameShape(b));
            int nbRows = Count;
            Debug.Assert(nbRows <= a.Count);
            Debug.Assert(a.Count % nbRows == 0);
            int nbColumns_in_a_and_b = a.Count / nbRows;
            Wrapper.RunKernel("MultiplyEachRowIntoSingleValue", nbRows, new object[] { nbColumns_in_a_and_b, this, a, b });
        }

        public override void ZeroPadding(Tensor unpaddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            var paddedTensor = this;
            Debug.Assert(AreCompatible(new List<Tensor> { this, unpaddedTensor }));
            Debug.Assert(paddedTensor.Dimension == 4);
            Debug.Assert(paddedTensor.Dimension == unpaddedTensor.Dimension);
            Debug.Assert(paddedTensor.Shape[0] == unpaddedTensor.Shape[0]); //same batch size
            Debug.Assert(paddedTensor.Shape[1] == unpaddedTensor.Shape[1]); //same number of channels
            Debug.Assert(paddedTensor.Shape[2] == (paddingTop + unpaddedTensor.Shape[2] + paddingBottom)); //valid height for destination
            Debug.Assert(paddedTensor.Shape[3] == (paddingLeft + unpaddedTensor.Shape[3] + paddingRight)); //valid width destination
            ZeroMemory();
            int h_src = unpaddedTensor.Shape[2];
            int w_src = unpaddedTensor.Shape[3];
            // number of distinct rows in tensor 'src' (n, c, h_src, w_src)
            int srcNbRowId = unpaddedTensor.Shape[0] * unpaddedTensor.Shape[1] * h_src;
            Wrapper.RunKernel("ApplyZeroPaddingForRowId", srcNbRowId, new object[] { h_src, w_src, paddingTop, paddingBottom, paddingLeft, paddingRight, paddedTensor, unpaddedTensor, false});
        }

        public override void ZeroUnpadding(Tensor paddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            //            ((CpuTensor<T>)paddedTensor).ZeroPadding_and_Unpadding(this, paddingTop, paddingBottom, paddingLeft, paddingRight, true);
            var unpaddedTensor = this;
            int h_src = unpaddedTensor.Shape[2];
            int w_src = unpaddedTensor.Shape[3];
            // number of distinct rows in tensor 'src' (n, c, h_src, w_src)
            int srcNbRowId = unpaddedTensor.Shape[0] * unpaddedTensor.Shape[1] * h_src;
            Wrapper.RunKernel("ApplyZeroPaddingForRowId", srcNbRowId, new object[] { h_src, w_src, paddingTop, paddingBottom, paddingLeft, paddingRight, paddedTensor, unpaddedTensor, true });
        }

        public override void CopyTo(Tensor b)
        {
            CopyTo(0, b, 0, Count);
        }
        public override void CopyTo(int startElement, Tensor other, int otherStartElement, int elementCount)
        {
            AssertIsNotDisposed();
            other.AssertIsNotDisposed();

            Debug.Assert(AreCompatible(new List<Tensor> { this, other }));
            var thisPointer = (IntPtr)this + (TypeSize * startElement);
            var otherPointer = (IntPtr)other + (TypeSize * otherStartElement);
            var res = CublasWrapper.cublasScopy_v2(CublasHandle, elementCount, thisPointer, 1, otherPointer, 1);
            GPUWrapper.CheckStatus(res, ToString);

        }
        public override Tensor ExtractSubTensor(int startRowIndex, int nbRows)
        {
            var shape = (int[])Shape.Clone();
            shape[0] = startRowIndex;
            var offset = ReallyNeededMemoryInBytesForShape(shape);
            shape[0] = nbRows;
            return new GPUTensor<T>(this, shape, (int)offset, Description);
        }
        public override void ZeroMemory()
        {
            _deviceMemory.ZeroMemory();
        }
#endregion


        public override void AssertIsNotDisposed()
        {
            if (_disposed)
            {
                throw new Exception("Tensor is disposed " + this);
            }
            Wrapper.CheckThreadId();
            _deviceMemory.AssertIsNotDisposed();
        }

#region Dispose pattern
        public override void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        private void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            if (disposing)
            {
                //managed memory
                _randomNumberGeneratorStatesBuffer?.Dispose();
                _dropoutReserveSpace?.Dispose();
                _deviceMemory?.Dispose();
            }

            //unmanaged memory
            var res = CudnnWrapper.cudnnDestroyDropoutDescriptor(_dropoutDescriptor);
            CheckStatus(res);

            _randomNumberGeneratorStatesBuffer = null;
            _dropoutReserveSpace = null;
            _dropoutDescriptor = IntPtr.Zero;
        }
        ~GPUTensor()
        {
            Dispose(false);
        }
#endregion

        protected override IntPtr DevicePointer
        {
            get
            {
                AssertIsNotDisposed();
                return _deviceMemory.Pointer;
            }
        }

        private IntPtr TensorDesc(Tensor a) { return Wrapper.TensorDesc(CudaType, a.Shape); }
        private IntPtr FilterDesc(Tensor a, bool isDepthwiseConvolution) { return Wrapper.FilterDesc(CudaType, a.Shape, isDepthwiseConvolution); }
        private IntPtr ActivationDesc(cudnnActivationMode_t activationFunctionType)
        {
            return Wrapper.ActivationDesc(activationFunctionType);
        }
        private IntPtr PoolingDesc(cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride)
        {
            return Wrapper.PoolingDesc(poolingMode, poolingHeight, poolingWidth, poolingStride);
        }
        private IntPtr ConvDesc(int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int stride, int groupCount) { return Wrapper.ConvDesc(CudaType, paddingTop, paddingBottom, paddingLeft, paddingRight, stride, groupCount); }
        private cudnnDataType_t CudaType => cudnnDataType_t.CUDNN_DATA_FLOAT;
        private IntPtr CudnnHandle => Wrapper.CudnnHandle;
        private IntPtr CublasHandle => Wrapper.CudaBlasHandle;
        private void CheckStatus(cudnnStatus_t _status)
        {
            GPUWrapper.CheckStatus(_status, ToString);
        }
    }
}
