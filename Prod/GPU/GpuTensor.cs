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
    public sealed unsafe class GPUTensor<T> : Tensor where T : struct
    {
        #region Private fields
        private GPUWrapper Wrapper { get; }
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
            CapacityInBytes = ReallyNeededMemoryInBytes;
            _deviceMemory = Wrapper.NewDeviceMemory(CapacityInBytes);
            if (hostMemoryPointer != IntPtr.Zero)
            {
                CopyToDevice(hostMemoryPointer);
            }
        }


        /// <summary>
        /// copy from CPU (Host) to GPU (Device) memory
        /// </summary>
        /// <param name="hostPinnedPointer">point to host (pinned) memory (in CPU) </param>
        public void CopyToDevice(IntPtr hostPinnedPointer)
        {
            Wrapper.SwCopyToDevice.Start();
            Wrapper.LogCopyToDeviceCall(ReallyNeededMemoryInBytes);
            var res = NVCudaWrapper.cuMemcpyHtoD_v2(DevicePointer, hostPinnedPointer, ReallyNeededMemoryInBytes);
            GPUWrapper.CheckStatus(res);
            Wrapper.SwCopyToDevice.Stop();
        }

        public void CopyToDevice(T[] data)
        {
            Debug.Assert(data.Length == Count);
            using (var m = new HostPinnedMemory<T>(data))
            {
                CopyToDevice(m.Pointer);
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
            GPUWrapper.CheckStatus(res);
            handle.Free();
            Wrapper.SwCopyToHost.Stop();
            return _hostMemory;
        }
        public IntPtr DevicePointer => _deviceMemory.Pointer;

        #region Tensor implementation
        public override void BatchNormalization(Tensor y, Tensor bnScale, Tensor bnBias, double exponentialAverageFactor, Tensor resultRunningMean, Tensor resultRunningVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor resultSaveMean, Tensor resultSaveVariance, bool isTraining)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y, bnScale, bnBias, resultRunningMean, resultRunningVariance, resultSaveMean, resultSaveVariance }));
            var xDesc = TensorDesc(x);
            var bnScaleBiasMeanVarDesc = TensorDesc(bnScale);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            if (isTraining)
            {
               var res = CudnnWrapper.cudnnBatchNormalizationForwardTraining(CudnnHandle, mode, one, zero,
                        xDesc, x, xDesc, y,
                        bnScaleBiasMeanVarDesc, bnScale, bnBias, exponentialAverageFactor,
                        resultRunningMean,
                        resultRunningVariance, epsilon, resultSaveMean,
                        resultSaveVariance);
                CheckStatus(res);
            }
            else
            {
                var res = CudnnWrapper.cudnnBatchNormalizationForwardInference(CudnnHandle, mode, one, zero,
                        xDesc, x, xDesc, y,
                        bnScaleBiasMeanVarDesc, bnScale, bnBias, resultRunningMean,
                        resultRunningVariance, epsilon);
                CheckStatus(res);
            }
        }
        public override void BatchNormalizationBackward(Tensor dy, Tensor dx, Tensor bnScale, Tensor resultBnScaleDiff, Tensor resultBnBiasDiff, cudnnBatchNormMode_t mode, double epsilon, Tensor resultSaveMean, Tensor resultSaveVariance)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, dy, dx, bnScale, resultBnScaleDiff, resultBnBiasDiff, resultSaveMean, resultSaveVariance }));
            var xDesc = TensorDesc(x);
            var bnScaleBiasDiffDesc = TensorDesc(bnScale);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnBatchNormalizationBackward(CudnnHandle, mode, 
                one, zero, one, zero,
                xDesc, x,
                xDesc, dy,
                xDesc, dx,
                bnScaleBiasDiffDesc, bnScale, resultBnScaleDiff, resultBnBiasDiff,
                epsilon, resultSaveMean,
                resultSaveVariance);
            CheckStatus(res);
        }
        public override void ActivationForward(cudnnActivationMode_t activationType, Tensor y)
        {
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
                res = CudnnWrapper.cudnnSoftmaxForward(CudnnHandle, cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE,
                    cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_INSTANCE, one, xDesc, x, zero, yDesc, y);
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
            var activationDesc = ActivationDesc(activationType);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            cudnnStatus_t res;
            if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)
            {
                res = CudnnWrapper.cudnnSoftmaxBackward(CudnnHandle, cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE,
                    cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_INSTANCE,
                    one, yDesc, y, dyDesc, dy, zero, dxDesc, dx);
            }
            else
            {
                res = CudnnWrapper.cudnnActivationBackward(CudnnHandle, activationDesc, one, yDesc, y, dyDesc, dy,
                    xDesc, x, zero, dxDesc, dx);
            }
            CheckStatus(res);
        }
        public override void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingSize, int poolingStride)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            var poolingDesc = PoolingDesc(poolingMode, poolingSize, poolingStride);
            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnPoolingForward(CudnnHandle, poolingDesc, one, xDesc, x, zero, yDesc, y);
            CheckStatus(res);
        }
        public override void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode, int poolingSize, int poolingStride)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, y, x, dx }));
            var poolingDesc = PoolingDesc(poolingMode, poolingSize, poolingStride);
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

        // compute: this += alpha * bias
        public override void Update_Adding_Alpha_X(float alpha, Tensor x)
        {
            AddTensor(alpha, x, 1);
        }

        // compute: this = alpha * x + beta * this
        public override void AddTensor(float alpha, Tensor x, float beta)
        {
            var c = this;
            Debug.Assert(AreCompatible(new List<Tensor> { c, x }));
            var cDesc = TensorDesc(c);
            var xDesc = TensorDesc(x);
            var res = CudnnWrapper.cudnnAddTensor(CudnnHandle, &alpha, xDesc, x, &beta, cDesc, c);
            CheckStatus(res);
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
            /*
             v1: 1.05ms
            for (int m = 0; m < Shape[0]; ++m)
            {
                a.CopyTo(a.Idx(m), this, Idx(m), a.MultDim0);
                b.CopyTo(b.Idx(m), this, Idx(m) + a.MultDim0, b.MultDim0);
            }
            */
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
            /*
             v1 : = 1.1ms
            for (int m = 0; m < Shape[0]; ++m)
            {
                CopyTo(Idx(m), a, a.Idx(m), a.MultDim0);
                CopyTo(Idx(m) + a.MultDim0, b, b.Idx(m), b.MultDim0);
            } */

            var concat = this;
            Wrapper.RunKernel("Split", Count, new object[] { Shape[0], concat, concat.MultDim0, a, a.MultDim0, b, b.MultDim0 });
        }
        /// <summary>
        /// resize the current GPU tensor to a different shape
        /// </summary>
        /// <param name="newShape"></param>
        public override void Reshape(int[] newShape)
        {
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

        // compute: this = alpha * this
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
        public override void Convolution(Tensor filters, int padding, int stride, Tensor y)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, filters, y }));
            var convDesc = ConvDesc(padding, stride);
            var filterDesc = FilterDesc(filters);
            var xDesc = TensorDesc(x);
            var yDesc = TensorDesc(y);

            var res = CudnnWrapper.cudnnGetConvolutionForwardAlgorithm(CudnnHandle, xDesc, filterDesc, convDesc, yDesc,
                cudnnConvolutionFwdPreference_t.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0,
                out cudnnConvolutionFwdAlgo_t algo);
            CheckStatus(res);
            res = CudnnWrapper.cudnnGetConvolutionForwardWorkspaceSize(CudnnHandle, xDesc, filterDesc, convDesc, yDesc,
                algo, out size_t workspaceSize);
            CheckStatus(res);
            var storageBuffer = Wrapper.StorageBuffer(workspaceSize);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            res = CudnnWrapper.cudnnConvolutionForward(CudnnHandle, one, xDesc, x, filterDesc, filters, convDesc, algo,
                storageBuffer.Pointer, storageBuffer.SizeInBytes, zero, yDesc, y);
            CheckStatus(res);
        }
        public override void ConvolutionBackwardBias(Tensor convolutionBackwardBias)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, convolutionBackwardBias }));
            var dyDesc = TensorDesc(dy);
            var dbDesc = TensorDesc(convolutionBackwardBias);

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            var res = CudnnWrapper.cudnnConvolutionBackwardBias(CudnnHandle, one, dyDesc, dy, zero, dbDesc, convolutionBackwardBias);
            CheckStatus(res);
        }
        public override void ConvolutionGradient(Tensor conv, Tensor dy, int padding, int stride, Tensor dx, Tensor convGradient)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> {x, conv, dy, dx, convGradient}));
            var xDesc = TensorDesc(x);
            var dyDesc = TensorDesc(dy);
            var dwDesc = FilterDesc(convGradient);
            var convDesc = ConvDesc(padding, stride);
            var res = CudnnWrapper.cudnnGetConvolutionBackwardFilterAlgorithm(CudnnHandle, xDesc, dyDesc, convDesc,
                dwDesc, cudnnConvolutionBwdFilterPreference_t.CUDNN_CONVOLUTION_BWD_FILTER_​PREFER_FASTEST, 0,
                out cudnnConvolutionBwdFilterAlgo_t filterAlgo);
            CheckStatus(res);
            res = CudnnWrapper.cudnnGetConvolutionBackwardFilterWorkspaceSize(CudnnHandle, xDesc, dyDesc, convDesc,
                dwDesc, filterAlgo, out size_t filterWorkspaceSize);
            CheckStatus(res);
            var storageBuffer = Wrapper.StorageBuffer(Math.Max(1, filterWorkspaceSize));

            float oneFloat = 1f, zeroFloat = 0f;
            var zero = &zeroFloat;
            var one = &oneFloat;

            res = CudnnWrapper.cudnnConvolutionBackwardFilter(CudnnHandle, one, xDesc, x, dyDesc, 
                dy, convDesc, filterAlgo, storageBuffer.Pointer, storageBuffer.SizeInBytes,
                zero, dwDesc, convGradient);
            CheckStatus(res);

            if (dx == null)
            {
                return;
            }
            var dxDesc = TensorDesc(dx);
            var wDesc = FilterDesc(conv);
            res = CudnnWrapper.cudnnGetConvolutionBackwardDataAlgorithm(CudnnHandle, wDesc, dyDesc, convDesc, dxDesc,
                cudnnConvolutionBwdDataPreference_t.CUDNN_CONVOLUTION_BWD_DATA_​PREFER_FASTEST, 0,
                out cudnnConvolutionBwdDataAlgo_t dataAlgo);
            CheckStatus(res);

            res = CudnnWrapper.cudnnGetConvolutionBackwardDataWorkspaceSize(CudnnHandle, dwDesc, dyDesc, convDesc,
                dxDesc, dataAlgo, out size_t dataWorkspaceSize);
            CheckStatus(res);

            storageBuffer = Wrapper.StorageBuffer(dataWorkspaceSize);
            res = CudnnWrapper.cudnnConvolutionBackwardData(CudnnHandle, one, wDesc, conv, dyDesc, dy, convDesc,
                dataAlgo, storageBuffer.Pointer, storageBuffer.SizeInBytes, zero, dxDesc, dx);
            CheckStatus(res);
        }
       
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
            var deviceContent = DeviceContent();
            return (deviceContent as float[]);
        }
        //this = yExpectedOneHot
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
        //this = yExpectedOneHot
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
            var categoryCount = yExpectedOneHot.MultDim0;
            Wrapper.RunKernel(kernelName, nbRows, new object[] { categoryCount, buffer, yExpectedOneHot, yPredicted });
            return ((double)buffer.ContentAsFloatArray().Sum() / nbRows);
        }


        private DeviceMemory _randomNumberGeneratorStatesBuffer;
        private DeviceMemory _dropoutReserveSpace;
        private IntPtr _dropoutDescriptor;
        public override void DropoutForward(Tensor y, double dropProbability, bool isTraining, Random dropoutRandom, Tensor dropoutMaskBuffer)
        {
            var x = this;
            Debug.Assert(dropoutMaskBuffer == null); //no need of dropout mask for CPU
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
        //this = x
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropProbability, Tensor usedDropoutMask)
        {
            var dxDesc = TensorDesc(dx);
            var dyDesc = TensorDesc(dy);
            Debug.Assert(usedDropoutMask == null);
            var res = CudnnWrapper.cudnnDropoutBackward(CudnnHandle, _dropoutDescriptor, dyDesc, dy, dxDesc, dx, _dropoutReserveSpace.Pointer, _dropoutReserveSpace.SizeInBytes);
            CheckStatus(res);
        }
        public override void BroadcastConvolutionBiasToOutput(Tensor y)
        {
            var convolutionBias = this;
            Debug.Assert(AreCompatible(new List<Tensor> { convolutionBias, y}));
            y.Update_Adding_Alpha_X(1, convolutionBias);
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
            CublasWrapper.cublasSgemm_v2(CublasHandle, transLeft, transRight, N, M, K, ref alpha, b, ldb, a, lda, ref beta, this, ldc);
        }
        public override void CopyTo(Tensor b)
        {
            CopyTo(0, b, 0, Count);
        }
        public override void CopyTo(int startElement, Tensor other, int otherStartElement, int elementCount)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, other }));
            var thisPointer = (IntPtr)this + (TypeSize * startElement);
            var otherPointer = (IntPtr)other + (TypeSize * otherStartElement);
            var _status = CublasWrapper.cublasScopy_v2(CublasHandle, elementCount, thisPointer, 1, otherPointer, 1);
            CheckStatus(_status);
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

#region Dispose pattern
        private bool _disposed;
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
            CudnnWrapper.cudnnDestroyDropoutDescriptor(_dropoutDescriptor);

            _randomNumberGeneratorStatesBuffer = null;
            _dropoutReserveSpace = null;
            _dropoutDescriptor = IntPtr.Zero;
        }
        ~GPUTensor()
        {
            Dispose(false);
        }
#endregion

        private IntPtr TensorDesc(Tensor a) { return Wrapper.TensorDesc(CudaType, a.Shape); }
        private IntPtr FilterDesc(Tensor a) { return Wrapper.FilterDesc(CudaType, a.Shape); }
        private IntPtr ActivationDesc(cudnnActivationMode_t activationFunctionType)
        {
            return Wrapper.ActivationDesc(activationFunctionType);
        }
        private IntPtr PoolingDesc(cudnnPoolingMode_t poolingMode, int poolingSize, int poolingStride)
        {
            return Wrapper.PoolingDesc(poolingMode, poolingSize, poolingStride);
        }
        private IntPtr ConvDesc(int padding, int stride) { return Wrapper.ConvDesc(CudaType, padding, stride); }
        private cudnnDataType_t CudaType => cudnnDataType_t.CUDNN_DATA_FLOAT;
        private IntPtr CudnnHandle => Wrapper.CudnnHandle;
        private IntPtr CublasHandle => Wrapper.CudaBlasHandle;
        private static void CheckStatus(cublasStatus_t _status)
        {
            GPUWrapper.CheckStatus(_status);
        }
        private static void CheckStatus(cudnnStatus_t _status)
        {
            GPUWrapper.CheckStatus(_status);
        }
    }
}
