using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Layers;

namespace SharpNet.GPU
{
    public sealed unsafe class GPUTensor<T> : Tensor
    {
        #region Private fields
        private readonly GPUWrapper _wrapper;
        /// <summary>
        /// pointer to the start of the tensor in the device (GPU) memory 
        /// </summary>
        private readonly IntPtr _pointerToStartOfTensor;
        #endregion
        #region public properties
        /// <summary>
        /// true if the 'this' tensor is the owner of the associate memory
        ///     => in this case the memory associated with the 'this' tensor should be freed by the tensor when it is disposed
        /// false if the 'this' tensor is a span to an already allocated memory area in the device
        ///     => in this case this memory associated with the 'this' tensor should not be de allocated by the tensor when it is disposed
        /// </summary>
        public override bool IsOwnerOfMemory {get;}
        #endregion

        #region constructors
        public GPUTensor(int[] shape, Memory<T>? unpinnedHostMemory, GPUWrapper wrapper) : base(shape, Marshal.SizeOf(typeof(T)), true)
        {
            _wrapper = wrapper;
            _wrapper.CheckThreadId();
            CapacityInBytes = ReallyNeededMemoryInBytes;
            _pointerToStartOfTensor = IntPtr.Zero;

            var res = NVCudaWrapper.cuMemAlloc_v2(out _pointerToStartOfTensor, CapacityInBytes);
            GPUWrapper.CheckStatus(res);
            if (unpinnedHostMemory.HasValue)
            {
                InitializeFromHostMemory(unpinnedHostMemory.Value);
            }
            IsOwnerOfMemory = true;
        }
        /// <summary>
        /// construct a tensor that is a span to an already allocated memory
        /// this memory should not be de allocated 
        /// </summary>
        /// <param name="shape">shape of the tensor</param>
        /// <param name="pointerToMemoryOwner">the already allocated memory area that the tensor will use</param>
        /// <param name="wrapper"></param>
        public GPUTensor(int[] shape, IntPtr pointerToMemoryOwner, GPUWrapper wrapper) : base(shape, Marshal.SizeOf(typeof(T)), true)
        {
            _wrapper = wrapper;
            _wrapper.CheckThreadId();
            CapacityInBytes = ReallyNeededMemoryInBytes;
            _pointerToStartOfTensor = pointerToMemoryOwner;
            IsOwnerOfMemory = false;
        }
        #endregion

        /// <summary>
        /// copy from CPU (Host) pinned memory to GPU (Device) memory
        /// </summary>
        /// <param name="hostPinnedPointer">pointer to host (pinned) memory (in CPU) </param>
        public void InitializeFromHostPinnedMemory(IntPtr hostPinnedPointer)
        {
            AssertIsNotDisposed();
            Debug.Assert(hostPinnedPointer != IntPtr.Zero);
            _wrapper.SwCopyHostToDevice.Start();
            _wrapper.LogCopyHostToDeviceCall(ReallyNeededMemoryInBytes);
            //asynchronous copy
            var res = NVCudaWrapper.cuMemcpyHtoDAsync_v2(Pointer, hostPinnedPointer, ReallyNeededMemoryInBytes, _wrapper.DefaultStream.StreamHandle);
            //for synchronous copy: var res = NVCudaWrapper.cuMemcpyHtoD_v2(Pointer, hostPinnedPointer, ReallyNeededMemoryInBytes)
            GPUWrapper.CheckStatus(res);
            _wrapper.SwCopyHostToDevice.Stop();
        }
        /// <summary>
        /// copy the first 'Count' element of 'buffer into the 'this' Tensor
        /// </summary>
        /// <param name="buffer">a buffer to read from
        /// It must contains at least 'Count' elements
        /// </param>
        public void InitializeFromHostMemory(Memory<T> buffer)
        {
            Debug.Assert(buffer.Length >= Count);
            using var m = new HostPinnedMemory<T>(buffer);
            InitializeFromHostPinnedMemory(m.Pointer);
        }
        /// <summary>
        /// copy from 'src' GPU (Device) to 'this' GPU (Device) memory
        /// the source and target device (GPU) may be different
        /// </summary>
        /// <param name="src">the tensor stored in device memory that we should copy into the current tensor</param>
        public void InitializeFromDeviceMemory(GPUTensor<T> src)
        {
            AssertIsNotDisposed();
            Debug.Assert(Count == src.Count);
            var srcDeviceId = src._wrapper.DeviceId;
            if (_wrapper.DeviceId == srcDeviceId)
            {
                //copy in the same device (GPU)
                _wrapper.SwCopyDeviceToSameDevice.Start();
                _wrapper.LogCopyDeviceToSameDeviceCall(ReallyNeededMemoryInBytes);
                src.CopyTo(0, this, 0, Count);
                _wrapper.SwCopyDeviceToSameDevice.Stop();
            }
            else
            {
                //copy between 2 distinct devices (GPU)
                _wrapper.SwCopyDeviceToOtherDevice.Start();
                _wrapper.LogCopyDeviceToOtherDeviceCall(ReallyNeededMemoryInBytes);
                //asynchronous copy
                var res = CudartWrapper.cudaMemcpyPeerAsync(Pointer, _wrapper.DeviceId, src.Pointer, srcDeviceId, ReallyNeededMemoryInBytes, IntPtr.Zero);
                //for synchronous copy:
                //var res = CudartWrapper.cudaMemcpyPeer(Pointer, _wrapper.DeviceId, src.Pointer, srcDeviceId, ReallyNeededMemoryInBytes);
                GPUWrapper.CheckStatus(res);
                _wrapper.SwCopyDeviceToOtherDevice.Stop();
            }
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
        public override void ActivationForward(cudnnActivationMode_t activationType, Tensor activationParameter, Tensor y)
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
            else if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY)
            {
                Debug.Assert(activationParameter.UseGPU);
                x.CopyTo(y);
                _wrapper.RunKernel("ComputeSoftmaxWithHierarchy", y.Shape[0], new object[] { y.MultDim0, activationParameter, y});
                res = cudnnStatus_t.CUDNN_STATUS_SUCCESS;
            }
            else if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_LN)
            {
                Debug.Assert(activationParameter.UseGPU);
                _wrapper.RunKernel("ComputeLn", x.Count, new object[] { x, y });
                res = cudnnStatus_t.CUDNN_STATUS_SUCCESS;
            }
            else if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU)
            {
                Debug.Assert(activationParameter.UseGPU);
                var activationDes = ActivationDesc(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                res = CudnnWrapper.cudnnActivationForward(CudnnHandle, activationDes, one, xDesc, x, zero, yDesc, y);
                var alphaActivation = activationParameter.ContentAsFloatArray()[0];
                y.AddTensor(alphaActivation, x, 1- alphaActivation);
            }
            else if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH)
            {
                // y = x * sigmoid(x) 
                ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID, activationParameter, y);
                y.Update_Multiply_By_x(this);
                res = cudnnStatus_t.CUDNN_STATUS_SUCCESS;
            }
            else
            {
                var activationDescriptor = ActivationDesc(activationType);
                res = CudnnWrapper.cudnnActivationForward(CudnnHandle, activationDescriptor, one, xDesc, x, zero, yDesc, y);
            }
            CheckStatus(res);
        }
        public override void ActivationBackward(cudnnActivationMode_t activationType, Tensor activationParameter, Tensor dy, Tensor x, Tensor y)
        {
            var dx = this;
            Debug.Assert(AreCompatible(new List<Tensor> {dx, dy, x, y }));
            Debug.Assert(dx.SameShape(dy, x, y));
            var dxDesc = TensorDesc(dx);
            var dyDesc = dxDesc;
            var xDesc = dxDesc;
            var yDesc = dxDesc;

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            cudnnStatus_t res;
            if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)
            {
                res = CudnnWrapper.cudnnSoftmaxBackward(CudnnHandle, cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_INSTANCE, one, yDesc, y, dyDesc, dy, zero, dxDesc, dx);
            }
            else if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY)
            {
                Debug.Assert(activationParameter.UseGPU);
                Debug.Assert(dx.MultDim0 == activationParameter.Count);
                _wrapper.RunKernel("ComputeSoftmaxGradientWitHierarchy", dx.Count, new object[] { dx.MultDim0, activationParameter, y, dy, dx});
                res = cudnnStatus_t.CUDNN_STATUS_SUCCESS;
            }
            else if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU)
            {
                var activationDesc = ActivationDesc(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                res = CudnnWrapper.cudnnActivationBackward(CudnnHandle, activationDesc, one, yDesc, y, dyDesc, dy, xDesc, x, zero, dxDesc, dx);
                var alphaActivation = activationParameter.ContentAsFloatArray()[0];
                dx.AddTensor(alphaActivation, dy, 1 - alphaActivation);
            }
            else if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH)
            {
                _wrapper.RunKernel("SwishGradient", dx.Count, new object[] { y, dy, x, dx });
                res = cudnnStatus_t.CUDNN_STATUS_SUCCESS;
            }
            else if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_LN)
            {
                _wrapper.RunKernel("LnGradient", dx.Count, new object[] { dy, x, dx });
                res = cudnnStatus_t.CUDNN_STATUS_SUCCESS;
            }
            else
            {
                var activationDesc = ActivationDesc(activationType);
                res = CudnnWrapper.cudnnActivationBackward(CudnnHandle, activationDesc, one, yDesc, y, dyDesc, dy, xDesc, x, zero, dxDesc, dx);
            }
            CheckStatus(res);
        }
        public override void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride)
        {
            var x = this;

            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            var poolingDesc = PoolingDesc(poolingMode, poolingHeight, poolingWidth, verticalStride, horizontalStride);
            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnPoolingForward(CudnnHandle, poolingDesc, one, xDesc, x, zero, yDesc, y);
            CheckStatus(res);
        }
        public override void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, y, x, dx }));
            Debug.Assert(x.Shape.Length == 4);
            Debug.Assert(SameDimension(new List<Tensor> { dy, y, x, dx }));
            var poolingDesc = PoolingDesc(poolingMode, poolingHeight, poolingWidth, verticalStride, horizontalStride);
            var xDesc = TensorDesc(x);
            var dxDesc = xDesc;
            var yDesc = TensorDesc(y);
            var dyDesc = yDesc;

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnPoolingBackward(CudnnHandle, poolingDesc, one, yDesc, y, dyDesc, dy, xDesc, x, zero, dxDesc, dx);
            CheckStatus(res);
        }

        public override void LinearFunction(float slope, Tensor x, float intercept)
        {
            var y = this;
            Debug.Assert(y.Count == x.Count);
            if (Math.Abs(intercept) < 1e-8)
            {
                AddTensor(slope, x, 0f);
                return;
            }
            _wrapper.RunKernel("LinearFunction", Count, new object[] { y, slope, x, intercept });
        }
     
        public override void Concatenate(IList<Tensor> tensors)
        {
#if DEBUG
            CheckConcatenate(tensors);
            Debug.Assert(tensors.Count >= 2);
            Debug.Assert(tensors.Count <= 3);
#endif
            var concat = this;
            var a = tensors[0];
            var b = tensors[1];
            if (tensors.Count == 2)
            {
                _wrapper.RunKernel("Concatenate", Count, new object[] { Shape[0], concat, concat.MultDim0, a, a.MultDim0, b, b.MultDim0 });
            }
            else
            {
                var c = tensors[2];
                _wrapper.RunKernel("Concatenate3", Count, new object[] { Shape[0], concat, concat.MultDim0, a, a.MultDim0, b, b.MultDim0, c, c.MultDim0 });
            }

        }

        public override void Split(IList<Tensor> tensors)
        {
#if DEBUG
            CheckConcatenate(tensors);
            Debug.Assert(tensors.Count>=2);
            Debug.Assert(tensors.Count<=3);
#endif
            var concat = this;
            var a = tensors[0];
            var b = tensors[1];
            if (tensors.Count == 2)
            {
                _wrapper.RunKernel("Split", Count, new object[] {Shape[0], concat, concat.MultDim0, a, a.MultDim0, b, b.MultDim0});
            }
            else
            {
                var c = tensors[2];
                _wrapper.RunKernel("Split3", Count, new object[] { Shape[0], concat, concat.MultDim0, a, a.MultDim0, b, b.MultDim0, c, c.MultDim0 });
            }
        }
        /// <summary>
        /// resize the current GPU tensor to a different shape
        /// </summary>
        /// <param name="newShape"></param>
        public override void Reshape(int[] newShape)
        {
            AssertIsNotDisposed();
            if (SameShape(newShape))
            {
                //nothing to do
            }
            else if (ReallyNeededMemoryInBytesForShape(newShape) <= CapacityInBytes)
            {
                //smaller shape
                Shape = newShape;
                RecomputeMultDim();
            }
            else
            {
                //bigger shape : we do not have enough space to store it
                throw new ArgumentException("CapacityInBytes: " + CapacityInBytes + " but need memory  " + ReallyNeededMemoryInBytesForShape(newShape) + " for " + this);
            }
        }

        public override Tensor WithNewShape(int[] newShape)
        {
            AssertIsNotDisposed();
            if (SameShape(newShape))
            {
                return this;
            }
            if (ReallyNeededMemoryInBytesForShape(newShape) <= CapacityInBytes)
            {
                return new GPUTensor<T>(newShape, Pointer, _wrapper);
            }
            //bigger shape : we do not have enough space to store it
            throw new ArgumentException("CapacityInBytes: " + CapacityInBytes + " but need memory  " + ReallyNeededMemoryInBytesForShape(newShape) + " for " + this);
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

        public override void Switch_First_2_axis(Tensor target)
        {
            Debug.Assert(Shape.Length >= 2);
            int aLength = Shape[0];
            int bLength = Shape[1];
            int cLength = MultDim1;
            _wrapper.RunKernel("Switch_First_2_axis", Count, new object[] { aLength, bLength, cLength, this, target});
            var targetShape = (int[])Shape.Clone();
            targetShape[0] = bLength;
            targetShape[1] = aLength;
            target.Reshape(targetShape);
        }

        public override void SwitchSecondAndThirdDimension(Tensor target)
        {
            Debug.Assert(Shape.Length == 3 || (Shape.Length==4&&Shape[3]==1));
            Debug.Assert(target.Shape.Length == 3 || (target.Shape.Length==4&& target.Shape[3]==1));
            Debug.Assert(Shape[0] == target.Shape[0]);
            Debug.Assert(Shape[1] == target.Shape[2]);
            Debug.Assert(Shape[2] == target.Shape[1]);
            int n = Shape[0];
            int c = Shape[1];
            int h = Shape[2];
            _wrapper.RunKernel("SwitchSecondAndThirdDimension", n*c, new object[] { n, c, h, this, target });
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
        public override void Convolution(Tensor filters, int paddingTop, int paddingBottom, int paddingLeft,
            int paddingRight, int stride, Tensor y, bool isDepthwiseConvolution,
            GPUWrapper.ConvolutionAlgoPreference forwardAlgoPreference, TensorMemoryPool memoryPool)
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
            var forwardAlgo = _wrapper.ConvolutionForwardAlgorithm(xDesc, filterDesc, convDesc, yDesc, forwardAlgoPreference);
            var res = CudnnWrapper.cudnnGetConvolutionForwardWorkspaceSize(CudnnHandle, xDesc, filterDesc, convDesc, yDesc, forwardAlgo, out var workspaceSize); 
            CheckStatus(res);
            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;
            var storageBuffer = memoryPool.GetBuffer(workspaceSize);
            res = CudnnWrapper.cudnnConvolutionForward(CudnnHandle, one, xDesc, x, filterDesc, filters, convDesc, forwardAlgo, storageBuffer.Pointer, workspaceSize, zero, yDesc, y);
            memoryPool.FreeFloatTensor(ref storageBuffer);
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
        public override void ConvolutionGradient(Tensor convolution, Tensor dy, int paddingTop, int paddingBottom,
            int paddingLeft, int paddingRight, int stride, Tensor dx, Tensor convGradient, bool isDepthwiseConvolution,
            GPUWrapper.ConvolutionAlgoPreference backwardAlgoPreference, TensorMemoryPool memoryPool)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, convolution, dy, dx, convGradient }));
            Debug.Assert(dx==null || x.SameShape(dx));
            var xDesc = TensorDesc(x);
            var dyDesc = TensorDesc(dy);
            var dwDesc = FilterDesc(convGradient, isDepthwiseConvolution);
            int inputChannelCount = x.Shape[1];
            int groupCount = isDepthwiseConvolution ? inputChannelCount : 1;
            var convDesc = ConvDesc(paddingTop, paddingBottom, paddingLeft, paddingRight, stride, groupCount);
            var backwardFilterAlgo = _wrapper.ConvolutionBackwardFilterAlgorithm(xDesc, dyDesc, convDesc, dwDesc, backwardAlgoPreference);
            var res = CudnnWrapper.cudnnGetConvolutionBackwardFilterWorkspaceSize(CudnnHandle, xDesc, dyDesc, convDesc, dwDesc, backwardFilterAlgo, out var filterWorkspaceSize);
            CheckStatus(res);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            //we compute 'convGradient'
            var filterStorageBuffer = memoryPool.GetBuffer(filterWorkspaceSize);
            res = CudnnWrapper.cudnnConvolutionBackwardFilter(CudnnHandle, one, xDesc, x, dyDesc, dy, convDesc, backwardFilterAlgo, filterStorageBuffer.Pointer, filterWorkspaceSize, zero, dwDesc, convGradient);
            memoryPool.FreeFloatTensor(ref filterStorageBuffer);
            CheckStatus(res);

            //we compute 'dx'
            if (dx == null)
            {
                return;
            }
            var dxDesc = TensorDesc(dx);
            var wDesc = FilterDesc(convolution, isDepthwiseConvolution);
            var backwardDataAlgo = _wrapper.ConvolutionBackwardDataAlgorithm(dwDesc, dyDesc, convDesc, xDesc, backwardAlgoPreference);
            res = CudnnWrapper.cudnnGetConvolutionBackwardDataWorkspaceSize(CudnnHandle, dwDesc, dyDesc, convDesc, dxDesc, backwardDataAlgo, out var dataWorkspaceSize);
            CheckStatus(res);
            var dataStorageBuffer = memoryPool.GetBuffer(dataWorkspaceSize);
            res = CudnnWrapper.cudnnConvolutionBackwardData(CudnnHandle, one, wDesc, convolution, dyDesc, dy, convDesc, backwardDataAlgo, dataStorageBuffer.Pointer, dataWorkspaceSize, zero, dxDesc, dx);
            CheckStatus(res);
            memoryPool.FreeFloatTensor(ref dataStorageBuffer);
        }
        #endregion
        public override void UniformDistribution(Random rand, double minValue, double maxValue)
        {
            var array = new float[Count];
            Utils.UniformDistribution(array, rand, minValue, maxValue);
            InitializeFromHostMemory(array as T[]);
        }
        public override void NormalDistribution(Random rand, double mean, double stdDev)
        {
            var array = new float[Count];
            Utils.NormalDistribution(array, rand, mean, stdDev);
            InitializeFromHostMemory(array as T[]);
        }

        public override void SetValue(float sameValue)
        {
            var array = new float[Count];
            var sameValueAsFloat = sameValue;
            for (int i = 0; i < array.Length; ++i)
            {
                array[i] = sameValueAsFloat;
            }
            InitializeFromHostMemory(array as T[]);
        }
        public override float[] ContentAsFloatArray()
        {
            return (DeviceContent() as float[]);
        }

        public override Tensor Clone()
        {
            var cloned  = new GPUTensor<T>(Shape, null, _wrapper);
            CopyTo(cloned);
            return cloned;
        }

        #region Compute of Loss and Metrics

        /// <summary>
        /// this = yExpectedOneHot
        /// </summary>
        /// <param name="yPredicted"></param>
        /// <param name="evaluationMetric"></param>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public override double ComputeEvaluationMetric([NotNull] Tensor yPredicted, EvaluationMetricEnum evaluationMetric, [NotNull] Tensor buffer)
        {
            var yExpected = this;
            Debug.Assert(AreCompatible(new List<Tensor> { yExpected, yPredicted }));
            Debug.Assert(yPredicted.SameShape(yExpected));
            Debug.Assert(yExpected.Dimension >= 2);
            int nbRows = yExpected.Shape[0];
            var categoryCount = yExpected.MultDim0;

            switch (evaluationMetric)
            {
                case EvaluationMetricEnum.Accuracy:
                    return ComputeAccuracy(yPredicted, buffer);
                case EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy:
                    return ComputeAccuracyCategoricalCrossentropyWithHierarchy(yPredicted, buffer);
                case EvaluationMetricEnum.Huber:
                    const float huberDelta = 1.0f;
                    _wrapper.RunKernel("HuberLoss", nbRows, new object[] { categoryCount, huberDelta, buffer, yExpected, yPredicted });
                    return buffer.ContentAsFloatArray().Average();
                case EvaluationMetricEnum.CosineSimilarity504:
                    Debug.Assert(yPredicted.Count % CosineSimilarity504_TimeSeries_Length == 0);
                    buffer.CosineSimilarityLoss(yExpected, yPredicted, CosineSimilarity504_TimeSeries_Length);
                    return buffer.ContentAsFloatArray().Average();
                case EvaluationMetricEnum.Rmse:
                    return Math.Sqrt(ComputeEvaluationMetric(yPredicted, EvaluationMetricEnum.Mse, buffer));
                default:
                    var kernelName = evaluationMetric + "Loss";
                    _wrapper.RunKernel(kernelName, nbRows, new object[] { categoryCount, buffer, yExpected, yPredicted });
                    return buffer.ContentAsFloatArray().Average();
            }

        }

        /// <summary>
        /// this = yExpectedOneHot
        /// </summary>
        /// <param name="yPredicted"></param>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public override double ComputeAccuracy([NotNull] Tensor yPredicted, [NotNull] Tensor buffer)
        {
            var yExpected = this;
            Debug.Assert(AreCompatible(new List<Tensor> {yExpected, yPredicted}));
            Debug.Assert(buffer.Shape.Length == 1);
            Debug.Assert(buffer.Shape[0] == yPredicted.Shape[0]);
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(yExpected.Dimension >= 2);
            int nbRows = yExpected.Shape[0];
            var nbCols = yExpected.Shape[1];
            _wrapper.RunKernel("ComputeAccuracy", nbRows, new object[] { nbCols, buffer, yExpected, yPredicted });
            return buffer.ContentAsFloatArray().Average();
        }

        public override double ComputeAccuracyCategoricalCrossentropyWithHierarchy([NotNull] Tensor yPredicted, [NotNull] Tensor buffer)
        {
            var yExpected = this;
            Debug.Assert(AreCompatible(new List<Tensor> { yExpected, yPredicted }));
            Debug.Assert(buffer.Shape.Length == 1);
            Debug.Assert(buffer.Shape[0] == yPredicted.Shape[0]);
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(yExpected.Dimension >= 2);
            int nbRows = yExpected.Shape[0];
            var nbCols = yExpected.Shape[1];
            _wrapper.RunKernel("ComputeSingleAccuracyForCategoricalCrossentropyWithHierarchy", nbRows, new object[] { nbCols, buffer, yExpected, yPredicted });
            return buffer.ContentAsFloatArray().Average();
        }
        public override void MseOfLogLoss(Tensor yExpected, Tensor yPredicted, float epsilon)
        {
            var mseLoss = this;
            int batchSize = yExpected.Shape[0];
            Debug.Assert(mseLoss.SameShape(new[] { batchSize }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            _wrapper.RunKernel("MseOfLogLoss", batchSize, new object[] { yExpected.MultDim0, mseLoss, yExpected, yPredicted, epsilon });
        }

        public override (float f1, float precision, float recall) F1PrecisionRecallMicro(Tensor yExpected, Tensor yPredicted)
        {
            throw new NotImplementedException();
        }

        public override void CosineSimilarityLoss(Tensor yExpected, Tensor yPredicted, int timeSeriesLength)
        {
            var cosineSimilarityLoss = this;
            Debug.Assert(cosineSimilarityLoss.SameShape(new[] { timeSeriesLength }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            _wrapper.RunKernel("CosineSimilarityLoss", timeSeriesLength, new object[] { yExpected.Count, cosineSimilarityLoss, yExpected, yPredicted });
        }

        public override void HuberLoss(Tensor yExpected, Tensor yPredicted, float huberDelta)
        {
            var huberLoss = this;
            int batchSize = yExpected.Shape[0];
            Debug.Assert(huberLoss.SameShape(new[] { batchSize }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            _wrapper.RunKernel("HuberLoss", batchSize, new object[] { yExpected.MultDim0, huberDelta, huberLoss, yExpected, yPredicted });
        }

        #endregion

        #region Compute of Gradients (for backward propagation)
        public override void CategoricalCrossentropyWithHierarchyGradient(Tensor yExpected, Tensor yPredicted)
        {
            var loss = this;
            int nbRows = yExpected.Shape[0];
            var nbCols = yExpected.Shape[1];
            loss.ZeroMemory();
            _wrapper.RunKernel("CategoricalCrossentropyWithHierarchyGradient", nbRows, new object[] { nbCols, loss, yExpected, yPredicted });
        }

        public override void CosineSimilarityGradient(Tensor yExpected, Tensor yPredicted, int timeSeriesLength)
        {
            var cosineSimilarityGradient = this;
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(cosineSimilarityGradient.SameShape(yPredicted));
            _wrapper.RunKernel("CosineSimilarityGradient", timeSeriesLength, new object[] { yExpected.Count, cosineSimilarityGradient, yExpected, yPredicted });
        }

        public override void HuberGradient(Tensor yExpected, Tensor yPredicted, float huberDelta)
        {
            var huberGradient = this;
            int batchSize = yExpected.Shape[0];
            _wrapper.RunKernel("HuberGradient", batchSize, new object[] { yExpected.MultDim0, huberDelta, huberGradient, yExpected, yPredicted });
        }

        public override void MseGradient(Tensor yExpected, Tensor yPredicted)
        {
            var mseGradient = this;
            int batchSize = yExpected.Shape[0];
            _wrapper.RunKernel("MseGradient", batchSize, new object[] { yExpected.MultDim0, mseGradient, yExpected, yPredicted });
        }

        public override void MseOfLogGradient(Tensor yExpected, Tensor yPredicted, float epsilon)
        {
            var mseGradient = this;
            int batchSize = yExpected.Shape[0];
            _wrapper.RunKernel("MseOfLogGradient", batchSize, new object[] { yExpected.MultDim0, mseGradient, yExpected, yPredicted, epsilon });
        }

        public override void MaeGradient(Tensor yExpected, Tensor yPredicted)
        {
            var mseGradient = this;
            int batchSize = yExpected.Shape[0];
            _wrapper.RunKernel("MaeGradient", batchSize, new object[] { yExpected.MultDim0, mseGradient, yExpected, yPredicted });
        }
        #endregion


        public override void DropoutForward(Tensor y, double dropoutRate, bool isTraining, Random dropoutRandom, [NotNull] Tensor dropoutReservedSpaceForTraining) 
        {
            var x = this;
            if (!isTraining)
            {
                x.CopyTo(y);
                return;
            }
            Debug.Assert(dropoutReservedSpaceForTraining.UseGPU); 
            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);
            cudnnDropoutDescriptor_t dropoutDesc = _wrapper.DropoutDesc(dropoutRate);
            var res = CudnnWrapper.cudnnDropoutForward(CudnnHandle, dropoutDesc, xDesc, x, yDesc, y, dropoutReservedSpaceForTraining.Pointer, dropoutReservedSpaceForTraining.CapacityInBytes);
            CheckStatus(res);
        }

        /// <summary>
        /// this = x
        /// </summary>
        /// <param name="dy"></param>
        /// <param name="dx"></param>
        /// <param name="dropoutRate"></param>
        /// <param name="dropoutReserveSpace"></param>
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropoutRate, [NotNull] Tensor dropoutReserveSpace)
        {
            Debug.Assert(dropoutReserveSpace.UseGPU);
            var dxDesc = TensorDesc(dx);
            var dyDesc = TensorDesc(dy);
            //no need of memory pool : the descriptor has been already created on forward propagation
            var dropoutDesc = _wrapper.DropoutDesc(dropoutRate);
            var res = CudnnWrapper.cudnnDropoutBackward(CudnnHandle, dropoutDesc, dyDesc, dy, dxDesc, dx, dropoutReserveSpace, dropoutReserveSpace.CapacityInBytes);
            CheckStatus(res);
        }
        public override void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon,
            double adamW_l2Regularization, Tensor dW, Tensor adam_vW, Tensor adam_sW, int timeStep)
        {
            var W = this;
            var beta1_power = Math.Pow(beta1, timeStep);
            var beta2_power = Math.Pow(beta2, timeStep);
            var multiplicative_factor = learningRate * (Math.Sqrt(1.0 - beta2_power) / (1.0 - beta1_power));
            _wrapper.RunKernel("UpdateAdamOptimizer", Count, new object[] { beta1, beta2, epsilon, adamW_l2Regularization, multiplicative_factor, dW, W, adam_vW, adam_sW });
        }
        public override void UpdateSGDOptimizer(double learningRate, double momentum, bool usenesterov, Tensor dW, Tensor velocity)
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

            GPUWrapper.CheckStatus(res);
        }

        public override void SetIdentityMatrix()
        {
            Debug.Assert(Shape.Length == 2);
            Debug.Assert(Shape[0] == Shape[1]);
            int nbRows = Shape[0];
            ZeroMemory();
            _wrapper.RunKernel("Set1InMainDiagonal", nbRows, new object[] { nbRows, this });
        }


        public override void SetToZeroAllElementsBelowMainDiagonal()
        {
            Debug.Assert(Shape.Length == 2);
            int nbRows = Shape[0];
            int nbColumns = Shape[1];
            _wrapper.RunKernel("SetToZeroAllElementsBelowMainDiagonal", nbRows, new object[] { nbColumns , this });
        }

        public override void Transpose(Tensor transposed)
        {
            AssertIsNotDisposed();
            transposed.AssertIsNotDisposed();
            Debug.Assert(AreCompatible(new List<Tensor> { this, transposed }));
            Debug.Assert(Dimension == 2);
            Debug.Assert(transposed.Dimension == Dimension);
            Debug.Assert(transposed.Shape[0] == Shape[1]);
            Debug.Assert(transposed.Shape[1] == Shape[0]);

            // because cublas is column major:
            //  (*) the 'this' tensor (=A) is seen (from cublas) as a matrix of shape (Shape[1], Shape[0]) = (lda, M)
            //      =>  lda = Shape[1]
            //          M = Shape[0]
            //  (*) the transpose of the 'this' tensor (=C) is seen as a matrix of shape (Shape[0], Shape[1]) = (M, N) = (ldc, N)
            //          ldc = M = Shape[0]
            //          N = Shape[1]
            //  (*) B is a matrix of same dimension as transposed matrix = (ldb, N) = (M, N) = (ldc, N)
            //          ldb = M = ldc = Shape[0]

            int M = Shape[0];
            int N = Shape[1];
            int lda = N;
            int ldc = M;
            int ldb = ldc;
            float alpha = 1f;
            float beta = 0f;

            var res = CublasWrapper.cublasSgeam(CublasHandle, cublasOperation_t.CUBLAS_OP_T, cublasOperation_t.CUBLAS_OP_N, M, N, ref alpha, this, lda, ref beta, IntPtr.Zero, ldb, transposed, ldc);
            GPUWrapper.CheckStatus(res);
        }


        public override void Orthogonal(Random rand)
        {
            var array = new float[Count];
            Utils.NormalDistribution(array, rand, 0, 1);
            var Q = new CpuTensor<float>(new[] { Shape[0], MultDim0 }, array);
            Q.Q_Factorization();
            InitializeFromHostMemory(array as T[]);
        }
        public override void QRFactorization(Tensor Q, Tensor R, Tensor buffer)
        {
            var A = this;
            int m = A.Shape[0];
            int n = A.Shape[1];
            Debug.Assert(m >= n);
            int lda = m;
            int k = n;

            var A_columnMajor = new GPUTensor<float>(new[] { n, m }, buffer.Pointer, _wrapper);
            var A_columnMajorLength = n * m;
            var tauLength = k;
            const int devInfoIntLength = 1;
            var workSpaceLength = buffer.Count - A_columnMajorLength - tauLength - devInfoIntLength;
            var TAU = buffer.Pointer + A_columnMajorLength * sizeof(float);
            var devInfoInt = TAU + tauLength * sizeof(float);
            var workSpace = devInfoInt + devInfoIntLength * sizeof(float); ;

            // 1st step involving geqrf method
            A.Transpose(A_columnMajor);
            var res = CusolverWrapper.cusolverDnSgeqrf(CusolverDnHandle, m, n, A_columnMajor, lda, TAU, workSpace, workSpaceLength, devInfoInt);
            GPUWrapper.CheckStatus(res);

            // 2nd step involving orgqr method
            res = CusolverWrapper.cusolverDnSorgqr(CusolverDnHandle, m, n, k, A_columnMajor, lda, TAU, workSpace, workSpaceLength, devInfoInt);
            GPUWrapper.CheckStatus(res);
            // we compute 'Q' matrix (of shape (m, n))
            A_columnMajor.Transpose(Q);

            // we compute 'R' matrix (of shape (n, n))
            R.Dot(Q, true, this, false, 1.0f, 0.0f);
        }
        public override int QRFactorization_FloatBufferLength()
        {
            int m = Shape[0];
            int n = Shape[1];
            int lda = m;
            int k = n;

            // we compute the workSpace size needed for the computation
            var res = CusolverWrapper.cusolverDnSgetrf_bufferSize(_wrapper.CusolverDnHandle, m, n, this, lda, out var workSpaceLength_geqrf);
            GPUWrapper.CheckStatus(res);
            //res = CusolverWrapper.cusolverDnSormqr_bufferSize(_wrapper.CusolverDnHandle, cublasSideMode_t.CUBLAS_SIDE_LEFT, cublasOperation_t.CUBLAS_OP_T, m, n, k, this, lda, this /*TAU*/, this /*Q*/, m, out var workSpaceLength_ormqr);
            res = CusolverWrapper.cusolverDnSorgqr_bufferSize(_wrapper.CusolverDnHandle, m, n, k, this /* A */, lda, IntPtr.Zero, out var workSpaceLength_orgqr);
            GPUWrapper.CheckStatus(res);

            var A_columnMajorLength = n * m;
            var tauLength = k;
            const int devInfoIntLength = 1;
            var workSpaceLength = Math.Max(workSpaceLength_orgqr, workSpaceLength_geqrf);
            return A_columnMajorLength
                   + tauLength
                   + devInfoIntLength
                   + workSpaceLength;
        }


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
        public override void MultiplyTensor(Tensor a, Tensor diagonalMatrix)
        {
            var c = this;
            //'a' shape:                (lda, n) with lda = m
            //'c' shape:                (ldc, n) with ldc = m
            //'diagonalMatrix' shape:   (1, n) for mode == CUBLAS_SIDE_RIGHT (and (1, m) for mode == CUBLAS_SIDE_LEFT)
            const cublasSideMode_t mode = cublasSideMode_t.CUBLAS_SIDE_RIGHT;
            Debug.Assert(Count%diagonalMatrix.Count == 0);
            int m = Count/diagonalMatrix.Count; //number of rows of matrix 'a and 'c'
            int n = diagonalMatrix.Count; //number of columns of matrix 'a' and 'c'
            int lda = m; //leading dimension of two-dimensional array used to store the matrix 'a'
            const int incx = 1; //stride of one
            int ldc = lda; //leading dimension of a two-dimensional array used to store the matrix 'c'
            var res = CublasWrapper.cublasSdgmm(CublasHandle, mode, m, n, a, lda, diagonalMatrix, incx, c, ldc);
            GPUWrapper.CheckStatus(res);
        }

        public override void UpSampling2D(Tensor tensorBeforeUpSampling, int rowFactor, int colFactor, UpSampling2DLayer.InterpolationEnum interpolation)
        {
            Debug.Assert(rowFactor>=1);
            Debug.Assert(colFactor >= 1);
            Debug.Assert(rowFactor * tensorBeforeUpSampling.Shape[2] == Shape[2]);
            Debug.Assert(colFactor * tensorBeforeUpSampling.Shape[3] == Shape[3]);
            if (interpolation == UpSampling2DLayer.InterpolationEnum.Bilinear)
            {
                throw new NotImplementedException("only "+ UpSampling2DLayer.InterpolationEnum.Nearest+" interpolation is supported (not "+interpolation+")");
            }
            _wrapper.RunKernel("UpSampling2D", tensorBeforeUpSampling.Count, new object[] { tensorBeforeUpSampling.Shape[1], tensorBeforeUpSampling.Shape[2], tensorBeforeUpSampling.Shape[3], rowFactor, colFactor, tensorBeforeUpSampling , this, true});
        }
        public override void DownSampling2D(Tensor tensorBeforeDownSampling, int rowFactor, int colFactor)
        {
            Debug.Assert(rowFactor >= 1);
            Debug.Assert(colFactor >= 1);
            Debug.Assert(rowFactor * Shape[2] == tensorBeforeDownSampling.Shape[2]);
            Debug.Assert(colFactor * Shape[3] == tensorBeforeDownSampling.Shape[3]);
            _wrapper.RunKernel("UpSampling2D", Count, new object[] { Shape[1], Shape[2], Shape[3], rowFactor, colFactor, this, tensorBeforeDownSampling, false});
        }

        public override void MultiplyEachRowIntoSingleValue(Tensor a, Tensor b)
        {
            Debug.Assert(a.SameShape(b));
            int nbRows = Count;
            Debug.Assert(nbRows <= a.Count);
            Debug.Assert(a.Count % nbRows == 0);
            int nbColumns_in_a_and_b = a.Count / nbRows;
            _wrapper.RunKernel("MultiplyEachRowIntoSingleValue", nbRows, new object[] { nbColumns_in_a_and_b, this, a, b });
        }
        public override void Clip(float lower, float upper)
        {
            Debug.Assert(upper>=lower);
            int nbRows = Shape[0];
            Debug.Assert(Count % nbRows == 0);
            int nbColumns = Count / nbRows;
            if (nbRows < 100)
            {
                nbRows = Count;
                nbColumns = 1;
            }
            Debug.Assert(nbRows*nbColumns == Count);
            _wrapper.RunKernel("Clip", nbRows, new object[] { nbColumns, this, lower, upper });
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
            _wrapper.RunKernel("ApplyZeroPaddingForRowId", srcNbRowId, new object[] { h_src, w_src, paddingTop, paddingBottom, paddingLeft, paddingRight, paddedTensor, unpaddedTensor, false});
        }
        public override void ZeroUnpadding(Tensor paddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            //            ((CpuTensor<T>)paddedTensor).ZeroPadding_and_Unpadding(this, paddingTop, paddingBottom, paddingLeft, paddingRight, true);
            var unpaddedTensor = this;
            int h_src = unpaddedTensor.Shape[2];
            int w_src = unpaddedTensor.Shape[3];
            // number of distinct rows in tensor 'src' (n, c, h_src, w_src)
            int srcNbRowId = unpaddedTensor.Shape[0] * unpaddedTensor.Shape[1] * h_src;
            _wrapper.RunKernel("ApplyZeroPaddingForRowId", srcNbRowId, new object[] { h_src, w_src, paddingTop, paddingBottom, paddingLeft, paddingRight, paddedTensor, unpaddedTensor, true });
        }
        public override void CopyTo(Tensor b)
        {
            if (Count != b.Count)
            {
                throw new ArgumentException("can't copy " + this + " to " + b);
            }
            if (b.UseGPU)
            {
                //copy GPU => GPU 
                b.AsGPU<T>().InitializeFromDeviceMemory(this);
            }
            else
            {
                //copy GPU => Cpu
                Memory<T> data = DeviceContent();
                data.CopyTo(b.AsCpu<T>().Content);
            }
        }
        public override void CopyTo(int startElement, Tensor dest, int otherStartElement, int elementCount)
        {
            Debug.Assert(dest.UseGPU);
            Debug.Assert(AreCompatible(new List<Tensor> { this, dest }));
            AssertIsNotDisposed();
            dest.AssertIsNotDisposed();

            var srcPointer = (IntPtr)this + (TypeSize * startElement);
            var destPointer = (IntPtr)dest + (TypeSize * otherStartElement);

            //Asynchronous copy
            var res = NVCudaWrapper.cuMemcpyDtoDAsync_v2(destPointer, srcPointer, (ulong)(elementCount*TypeSize), IntPtr.Zero);
            //for synchronous copy: var res = NVCudaWrapper.cuMemcpyDtoD_v2(destPointer, srcPointer, (ulong)(elementCount*TypeSize));
            //var res = CublasWrapper.cublasScopy_v2(CublasHandle, elementCount, srcPointer, 1, destPointer, 1);
            GPUWrapper.CheckStatus(res);
        }
        public override void YOLOV3Forward(Tensor x, int inputImageHeight, int inputImageWidth, int[] anchors)
        {
            var y = this;
            Debug.Assert(anchors.Length == 6);
            int nbAnchors = anchors.Length / 2;
            Debug.Assert(inputImageHeight % x.Shape[2] == 0);
            Debug.Assert(inputImageWidth % x.Shape[3] == 0);
            Debug.Assert(y.Shape[0] == x.Shape[0]);
            Debug.Assert(x.Shape[1] % nbAnchors == 0);
            Debug.Assert(nbAnchors * y.Shape[2] == x.Shape[1]);
            Debug.Assert(y.Shape[1] == nbAnchors * x.Shape[2] * x.Shape[3]);
            int predictionLength = x.Shape[1] / nbAnchors;
            _wrapper.RunKernel("YOLOV3Forward", x.Count/predictionLength, new object[] { y, x, x.Shape[1], x.Shape[2], x.Shape[3], inputImageHeight, inputImageWidth, anchors[0], anchors[1], anchors[2], anchors[3], anchors[4], anchors[5] });
        }
        public override Tensor Slice(int startIndex, int[] sliceShape)
        {
            return new GPUTensor<T>((int[])sliceShape.Clone(), Pointer+startIndex*TypeSize, _wrapper);
        }

        //this (= 'y') shape :      (batchSize, maxWordCountBySentence, embeddingDim)
        //'x' shape:                (batchSize, maxWordCountBySentence)
        //'wordEmbedding' shape:    (vocabularySize, embeddingDim)
        public override void WordEmbeddingForwardPropagation( /*in*/ Tensor x, /*in*/ Tensor wordEmbedding, int xIndexInLastDimensionToUse, int yIndexInLastDimensionToUse, int copyCountBeforeIndex, int copyCountAfterIndex)
        {
            var y = this;
            Debug.Assert(wordEmbedding.Shape.Length == 2);
            Debug.Assert(x.Shape[0] == y.Shape[0]); //same batchSize
            Debug.Assert(x.Shape[1] == y.Shape[1]); //same timeSteps
            Debug.Assert(y.Shape.Length == 3);
            Debug.Assert(xIndexInLastDimensionToUse >= 0);
            Debug.Assert(yIndexInLastDimensionToUse >= 0);
            int inputSize = x.Shape.Length == 2 ? 1 : x.Shape[2];
            int batchSize = y.Shape[0];
            int timeSteps = y.Shape[1];
            int embeddingDim = wordEmbedding.Shape[1];
            _wrapper.RunKernel("WordEmbeddingForwardPropagation", batchSize* timeSteps, new object[] { inputSize, xIndexInLastDimensionToUse, yIndexInLastDimensionToUse, copyCountBeforeIndex, copyCountAfterIndex, embeddingDim, x, y, wordEmbedding});
        }

        public override void WordEmbeddingBackwardPropagation( /*in*/ Tensor x, /*out*/ Tensor dx, /*in*/ Tensor dy, int dxIndexInLastDimensionToUse, int dyIndexInLastDimensionToUse, int copyCountBeforeIndex, int copyCountAfterIndex)
        {
            var dW = this;

            Debug.Assert(dW.Shape.Length == 2);
            Debug.Assert(x.Shape[0] == dy.Shape[0]); //same batchSize
            Debug.Assert(x.Shape[1] == dy.Shape[1]); //same timeSteps
            Debug.Assert(dy.Shape.Length == 3);
            Debug.Assert(dxIndexInLastDimensionToUse >= 0);
            Debug.Assert(dyIndexInLastDimensionToUse >= 0);
            int inputSize = x.Shape.Length == 2 ? 1 : x.Shape[2];
            // 'x' shape:   (batchSize, timeSteps, inputSize)
            // 'dy' shape:  (batchSize, timeSteps, inputSize+embeddingDim-1)

            dW.ZeroMemory();
            int batchSize = dy.Shape[0];
            int timeSteps = dy.Shape[1];
            int embeddingDim = dW.Shape[1];
            _wrapper.RunKernel("WordEmbeddingBackwardPropagation", batchSize* timeSteps, new object[] { inputSize, dxIndexInLastDimensionToUse, dyIndexInLastDimensionToUse, copyCountBeforeIndex, copyCountAfterIndex, embeddingDim, x, dx, dy, dW });
        }

        protected override int DeviceId => _wrapper.DeviceId;
        public override void ZeroMemory()
        {
            AssertIsNotDisposed();
            CUresult res;
            if (ReallyNeededMemoryInBytes % 4 == 0)
            {
                res = NVCudaWrapper.cuMemsetD32_v2(Pointer, 0, ReallyNeededMemoryInBytes / 4);
            }
            else
            {
                res = NVCudaWrapper.cuMemsetD8_v2(Pointer, (char)0, ReallyNeededMemoryInBytes);
            }
            GPUWrapper.CheckStatus(res);
        }

        public override void AssertIsNotDisposed()
        {
            if (_disposed)
            {
                throw new Exception("Tensor is disposed " + this);
            }
        }
        /// <summary>
        /// pointer to device memory (in GPU)
        /// </summary>
        public override IntPtr Pointer
        {
            get
            {
                AssertIsNotDisposed();
                Debug.Assert(_pointerToStartOfTensor != IntPtr.Zero);
                return _pointerToStartOfTensor;
            }
        }
        #endregion

        #region Dispose pattern
        public override void Dispose()
        {
            FreeDeviceMemory();
            GC.SuppressFinalize(this);
        }
        ~GPUTensor()
        {
            FreeDeviceMemory();
        }
        // ReSharper disable once UnusedParameter.Local
        private void FreeDeviceMemory()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            //unmanaged memory
            if (IsOwnerOfMemory)
            {
                /*var res =*/ NVCudaWrapper.cuMemFree_v2(_pointerToStartOfTensor);
                //GPUWrapper.CheckStatus(res);
            }
        }
        #endregion

        private cudnnTensorDescriptor_t TensorDesc(Tensor a) { return _wrapper.TensorDesc(CudaType, a.Shape); }

        private cudnnFilterDescriptor_t FilterDesc(Tensor a, bool isDepthwiseConvolution) { return _wrapper.FilterDesc(CudaType, a.Shape, isDepthwiseConvolution); }
        private cudnnActivationDescriptor_t ActivationDesc(cudnnActivationMode_t activationFunctionType)
        {
            return _wrapper.ActivationDesc(activationFunctionType);
        }
        private cudnnPoolingDescriptor_t PoolingDesc(cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride)
        {
            return _wrapper.PoolingDesc(poolingMode, poolingHeight, poolingWidth, verticalStride, horizontalStride);
        }
        private cudnnConvolutionDescriptor_t ConvDesc(int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int stride, int groupCount) { return _wrapper.ConvDesc(CudaType, paddingTop, paddingBottom, paddingLeft, paddingRight, stride, groupCount); }
        private cudnnDataType_t CudaType { get; } = cudnnDataType_t.CUDNN_DATA_FLOAT;
        private cudnnHandle_t CudnnHandle => _wrapper.CudnnHandle;
        private IntPtr CublasHandle => _wrapper.CudaBlasHandle;
        private cusolverDnHandle_t CusolverDnHandle => _wrapper.CusolverDnHandle;
        private CudartWrapper CudartWrapper => _wrapper.CudartWrapper;
        //private CudnnWrapper CudnnWrapper => _wrapper.CudnnWrapper;
        private CublasWrapper CublasWrapper => _wrapper.CublasWrapper;
        private T[] DeviceContent()
        {
            Debug.Assert(!_disposed);
            _wrapper.SwCopyDeviceToHost.Start();
            _wrapper.LogCopyDeviceToHostCall(ReallyNeededMemoryInBytes);

            var _hostMemory = new T[Count];
            var handle = GCHandle.Alloc(_hostMemory, GCHandleType.Pinned);
            var _hostMemoryPointer = handle.AddrOfPinnedObject();
            var res = NVCudaWrapper.cuMemcpyDtoH_v2(_hostMemoryPointer, Pointer, ReallyNeededMemoryInBytes);
            GPUWrapper.CheckStatus(res);
            handle.Free();
            _wrapper.SwCopyDeviceToHost.Stop();
            return _hostMemory;
        }
        private static void CheckStatus(cudnnStatus_t status)
        {
            GPUWrapper.CheckStatus(status);
        }
    }
}
