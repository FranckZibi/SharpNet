using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.InteropServices;
using SharpNet.Data;

namespace SharpNet.GPU
{
    public sealed unsafe class GPUTensor<T> : Tensor where T : struct
    {
        #region Private fields
        public override ulong CapacityInBytes { get; }
        private readonly CudaDeviceMemory _deviceMemory;
        private GPUWrapper Wrapper { get; }
        private T[] _hostPointer;
        private readonly IntPtr _hostStart;
        private GCHandle _handle;
        #endregion

        public GPUTensor(GPUTensor<T> memoryOwner, int[] shape, int offsetInBytes, string description) : base(shape, Marshal.SizeOf(typeof(T)), true, description)
        {
            Wrapper = memoryOwner.Wrapper;
            CapacityInBytes = ReallyNeededMemoryInBytes;
            _deviceMemory = new CudaDeviceMemory(memoryOwner.DevicePointer+offsetInBytes, false);         
            _hostStart = memoryOwner._hostStart + offsetInBytes;
            _hostPointer = null;
        }
        public GPUTensor(int[] shape, T[] data, string description, GPUWrapper wrapper) : base(shape, Marshal.SizeOf(typeof(T)), true, description)
        {
            Wrapper = wrapper;
            CapacityInBytes = ReallyNeededMemoryInBytes;
            _deviceMemory = new CudaDeviceMemory(ReallyNeededMemoryInBytes);
            _hostPointer = data ?? new T[Count];
            _handle = GCHandle.Alloc(_hostPointer, GCHandleType.Pinned);
            _hostStart = _handle.AddrOfPinnedObject();
            if (data == null)
            {
                ZeroMemory();
            }
            else
            {
                CopyToDevice();
            }
        }
        public void CopyToDevice()
        {
            //copy from CPU (Host) to GPU (Device) memory
            Wrapper.LogCopyToDeviceCall(ReallyNeededMemoryInBytes);
            var res = NVCudaWrapper.cuMemcpyHtoD_v2(DevicePointer, _hostStart, ReallyNeededMemoryInBytes);
            GPUWrapper.CheckStatus(res);
        }
        [SuppressMessage("ReSharper", "AssignNullToNotNullAttribute")]
        public T[] DeviceContent()
        {
            Debug.Assert(!_disposed);
            Wrapper.LogCopyToHostCall(ReallyNeededMemoryInBytes);
            var res = NVCudaWrapper.cuMemcpyDtoH_v2(_hostStart, DevicePointer, ReallyNeededMemoryInBytes);
            GPUWrapper.CheckStatus(res);
            if (_hostPointer == null)
            {
                _hostPointer = new T[Count];
                if (UseDoublePrecision)
                {
                    Marshal.Copy(_hostStart, _hostPointer as double[], 0, Count);
                }
                else
                {
                    Marshal.Copy(_hostStart, _hostPointer as float[], 0, Count);
                }
            }
            return _hostPointer;
        }
        public IntPtr DevicePointer => _deviceMemory.DevicePointer;

        #region Tensor implementation
        public override void BatchNormalization(Tensor y, Tensor bnScale, Tensor bnBias, double exponentialAverageFactor, Tensor resultRunningMean, Tensor resultRunningVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor resultSaveMean, Tensor resultSaveVariance, bool isTraining)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y, bnScale, bnBias, resultRunningMean, resultRunningVariance, resultSaveMean, resultSaveVariance }));
            var xDesc = TensorDesc(x);
            var bnScaleBiasMeanVarDesc = TensorDesc(bnScale);

            float oneFloat = 1.0f, zeroFloat = 0.0f; double oneDouble = 1.0d, zeroDouble = 0.0d;
            var zero = (UseDoublePrecision) ? (void*)&zeroDouble : &zeroFloat;
            var one = (UseDoublePrecision) ? (void*)&oneDouble : &oneFloat;

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

            float oneFloat = 1.0f, zeroFloat = 0.0f;double oneDouble = 1.0d, zeroDouble = 0.0d;
            var zero = (UseDoublePrecision) ? (void*)&zeroDouble : &zeroFloat;
            var one = (UseDoublePrecision) ? (void*)&oneDouble : &oneFloat;

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
            var activationDescriptor = ActivationDesc(activationType);

            float oneFloat = 1.0f, zeroFloat = 0.0f;
            double oneDouble = 1.0d, zeroDouble = 0.0d;
            var zero = (UseDoublePrecision) ? (void*) &zeroDouble : &zeroFloat;
            var one = (UseDoublePrecision) ? (void*) &oneDouble : &oneFloat;

            cudnnStatus_t res;
            if (activationType == cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)
            {
                res = CudnnWrapper.cudnnSoftmaxForward(CudnnHandle, cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE,
                    cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_INSTANCE, one, xDesc, x, zero, yDesc, y);
            }
            else
            {
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

            float oneFloat = 1.0f, zeroFloat = 0.0f;
            double oneDouble = 1.0d, zeroDouble = 0.0d;
            var zero = (UseDoublePrecision) ? (void*) &zeroDouble : &zeroFloat;
            var one = (UseDoublePrecision) ? (void*) &oneDouble : &oneFloat;

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

            float oneFloat = 1.0f, zeroFloat = 0.0f; double oneDouble = 1.0d, zeroDouble = 0.0d;
            var zero = (UseDoublePrecision) ? (void*)&zeroDouble : &zeroFloat;
            var one = (UseDoublePrecision) ? (void*)&oneDouble : &oneFloat;

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

            float oneFloat = 1.0f, zeroFloat = 0.0f; double oneDouble = 1.0d, zeroDouble = 0.0d;
            var zero = (UseDoublePrecision) ? (void*)&zeroDouble : &zeroFloat;
            var one = (UseDoublePrecision) ? (void*)&oneDouble : &oneFloat;

            var res = CudnnWrapper.cudnnPoolingBackward(CudnnHandle, poolingDesc, one, yDesc, y, dyDesc, dy, xDesc, x, zero, dxDesc, dx);
            CheckStatus(res);
        }
        // compute: this += alpha * bias
        public override void Update_Adding_Alpha_X(double alphaDouble, Tensor bias)
        {
            var c = this;
            Debug.Assert(AreCompatible(new List<Tensor> { c, bias }));
            var cDesc = TensorDesc(c);
            var biasDesc = TensorDesc(bias);

            var alphaFloat = (float)alphaDouble;
            float oneFloat = 1.0f; double oneDouble = 1.0d;

            var alpha = (UseDoublePrecision) ? (void*)&alphaDouble : &alphaFloat;
            var one = (UseDoublePrecision) ? (void*)&oneDouble : &oneFloat;

            var res = CudnnWrapper.cudnnAddTensor(CudnnHandle, alpha, biasDesc, bias, one, cDesc, c);
            CheckStatus(res);
        }
        public override void Update_Multiplying_By_Alpha(double alphaDouble)
        {
            var y = this;
            var yDesc = TensorDesc(y);
            var alphaFloat = (float)alphaDouble;
            var alpha = (UseDoublePrecision) ? (void*)&alphaDouble : &alphaFloat;
            var res = CudnnWrapper.cudnnScaleTensor(CudnnHandle, yDesc, y, alpha);
            CheckStatus(res);
        }
        public override void BroadcastAddVectorToOutput(Tensor y)
        {
            var bias = this;
            Debug.Assert(AreCompatible(new List<Tensor> { bias, y }));
            Debug.Assert(y.Dimension >= 2);
            Debug.Assert(y.MultDim0 == Count);
            y.Update_Adding_Alpha_X(1.0, bias);
        }
        public override void Compute_BiasGradient_from_dy(Tensor biasGradient)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, biasGradient}));
            Debug.Assert(Dimension >= 2);
            var dyDesc = TensorDesc(dy);
            var dbDesc = TensorDesc(biasGradient);

            float oneFloat = 1.0f, zeroFloat = 0.0f; double oneDouble = 1.0d, zeroDouble = 0.0d;
            var zero = (UseDoublePrecision) ? (void*)&zeroDouble : &zeroFloat;
            var one = (UseDoublePrecision) ? (void*)&oneDouble : &oneFloat;

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

            float oneFloat = 1.0f, zeroFloat = 0.0f;
            double oneDouble = 1.0d, zeroDouble = 0.0d;
            var zero = (UseDoublePrecision) ? (void*) &zeroDouble : &zeroFloat;
            var one = (UseDoublePrecision) ? (void*) &oneDouble : &oneFloat;

            res = CudnnWrapper.cudnnConvolutionForward(CudnnHandle, one, xDesc, x, filterDesc, filters, convDesc, algo,
                storageBuffer.DevicePointer, storageBuffer.SizeInBytes, zero, yDesc, y);
            CheckStatus(res);
        }
        public override void ConvolutionBackwardBias(Tensor convolutionBackwardBias)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, convolutionBackwardBias }));
            var dyDesc = TensorDesc(dy);
            var dbDesc = TensorDesc(convolutionBackwardBias);

            float oneFloat = 1.0f, zeroFloat = 0.0f; double oneDouble = 1.0d, zeroDouble = 0.0d;
            var zero = (UseDoublePrecision) ? (void*)&zeroDouble : &zeroFloat;
            var one = (UseDoublePrecision) ? (void*)&oneDouble : &oneFloat;

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

            float oneFloat = 1.0f, zeroFloat = 0.0f;
            double oneDouble = 1.0d, zeroDouble = 0.0d;
            var zero = (UseDoublePrecision) ? (void*) &zeroDouble : &zeroFloat;
            var one = (UseDoublePrecision) ? (void*) &oneDouble : &oneFloat;

            res = CudnnWrapper.cudnnConvolutionBackwardFilter(CudnnHandle, one, xDesc, x, dyDesc, 
                dy, convDesc, filterAlgo, storageBuffer.DevicePointer, storageBuffer.SizeInBytes,
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
                dataAlgo, storageBuffer.DevicePointer, storageBuffer.SizeInBytes, zero, dxDesc, dx);
            CheckStatus(res);
        }
        //this = yExpectedOneHot
        //TODO use a CUDA implementation of Compute Loss
        public override double ComputeLoss(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction)
        {
            var yExpectedOneHot = this;
            Debug.Assert(yPredicted != null);
            Debug.Assert(yPredicted.SameShape(yExpectedOneHot));
            return UseDoublePrecision
                ? yExpectedOneHot.ToCpu<double>().ComputeLoss(yPredicted.ToCpu<double>(), lossFunction)
                : yExpectedOneHot.ToCpu<float>().ComputeLoss(yPredicted.ToCpu<float>(), lossFunction);
        }
        //this = yExpectedOneHot
        //TODO use a CUDA implementation of Compute Accuracy
        public override int ComputeAccuracy(Tensor yPredicted)
        {
            var yExpectedOneHot = this;
            Debug.Assert(AreCompatible(new List<Tensor> {yExpectedOneHot, yPredicted}));
            Debug.Assert(yExpectedOneHot.SameShape(yPredicted));
            Debug.Assert(yExpectedOneHot.UseGPU);
            Debug.Assert(yExpectedOneHot.Dimension == 2);

            return UseDoublePrecision
                ? yExpectedOneHot.ToCpu<double>().ComputeAccuracy(yPredicted.ToCpu<double>())
                : yExpectedOneHot.ToCpu<float>().ComputeAccuracy(yPredicted.ToCpu<float>());
        }
        private CudaDeviceMemory _randomNumberGeneratorStatesBuffer;
        private CudaDeviceMemory _dropoutReserveSpace;
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
                _randomNumberGeneratorStatesBuffer = new CudaDeviceMemory(Math.Max(dropoutStateSize, 1));
                res = CudnnWrapper.cudnnDropoutGetReserveSpaceSize(xDesc, out var dropoutReserveSpaceSize);
                CheckStatus(res);
                _dropoutReserveSpace = new CudaDeviceMemory(Math.Max(dropoutReserveSpaceSize, 1));
                _dropoutDescriptor = Wrapper.DropoutDesc(dropProbability, _randomNumberGeneratorStatesBuffer.DevicePointer);
            }
            res = CudnnWrapper.cudnnDropoutForward(CudnnHandle, _dropoutDescriptor, xDesc, x, yDesc, y, _dropoutReserveSpace.DevicePointer, _dropoutReserveSpace.SizeInBytes);
            CheckStatus(res);
        }
        //this = x
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropProbability, Tensor usedDropoutMask)
        {
            var dxDesc = TensorDesc(dx);
            var dyDesc = TensorDesc(dy);
            Debug.Assert(usedDropoutMask == null);
            var res = CudnnWrapper.cudnnDropoutBackward(CudnnHandle, _dropoutDescriptor, dyDesc, dy, dxDesc, dx, _dropoutReserveSpace.DevicePointer, _dropoutReserveSpace.SizeInBytes);
            CheckStatus(res);
        }
        public override void BroadcastConvolutionBiasToOutput(Tensor y)
        {
            var convolutionBias = this;
            Debug.Assert(AreCompatible(new List<Tensor> { convolutionBias, y}));
            y.Update_Adding_Alpha_X(1.0, convolutionBias);
        }
        public override void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon, Tensor dW, Tensor adam_vW, Tensor adam_sW, int timestep)
        {
            var W = this;
            var beta1_power = Math.Pow(beta1, timestep);
            var beta2_power = Math.Pow(beta2, timestep);
            var multiplicative_factor = learningRate * (Math.Sqrt(1.0 - beta2_power) / (1.0 - beta1_power));
            Wrapper.RunKernel("UpdateAdamOptimizer", Count, new object[] { beta1, beta2, epsilon, multiplicative_factor, dW, W, adam_vW, adam_sW });
        }
        public override void UpdateSGDOptimizer(double learningRate, double momentum, double decay, bool usenesterov, Tensor dW, Tensor velocity)
        {
            var W = this;
            Wrapper.RunKernel("UpdateSGDOptimizer", Count, new object[] { learningRate, momentum, decay, usenesterov, dW, W, velocity });
        }
        public override void Dot(Tensor a, bool transposeA, Tensor b, bool transposeB, double alpha, double beta)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, a, b }));
            Debug.Assert(b.Dimension >= 2);
            Debug.Assert(a.Dimension >= 2);
            Debug.Assert(Dimension >= 2);
            var bH = b.Height;
            var bW = b.MultDim0;
            var aH = a.Height;
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
            if (UseDoublePrecision)
            {
                CublasWrapper.cublasDgemm_v2(CublasHandle, transLeft, transRight, N, M, K, ref alpha, b, ldb, a, lda, ref beta, this, ldc);
            }
            else
            {
                var alphaFloat = (float)alpha;
                var betaFloat = (float)beta;
                CublasWrapper.cublasSgemm_v2(CublasHandle, transLeft, transRight, N, M, K, ref alphaFloat, b, ldb, a, lda, ref betaFloat, this, ldc);
            }
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
            var _status = UseDoublePrecision
                ? CublasWrapper.cublasDcopy_v2(CublasHandle, elementCount, thisPointer, 1, otherPointer, 1)
                : CublasWrapper.cublasScopy_v2(CublasHandle, elementCount, thisPointer, 1, otherPointer, 1);
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
        public override void Reshape(int[] newShape)
        {
            if (Shape.SequenceEqual(newShape))
            {
                return;
            }
            Debug.Assert(ReallyNeededMemoryInBytesForShape(newShape) <= CapacityInBytes);
            Shape = newShape;
            RecomputeMultDim();
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
                GC.SuppressFinalize(this);
                _randomNumberGeneratorStatesBuffer?.Dispose();
                _dropoutReserveSpace?.Dispose();
                if (_handle.IsAllocated)
                {
                    _handle.Free();
                }
                CudnnWrapper.cudnnDestroyDropoutDescriptor(_dropoutDescriptor);
            }
            if (_deviceMemory.IsOwner) //if the 'this' object is the real owner of memory
            {
                _deviceMemory.Dispose();
            }
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
        private cudnnDataType_t CudaType => UseDoublePrecision ? cudnnDataType_t.CUDNN_DATA_DOUBLE : cudnnDataType_t.CUDNN_DATA_FLOAT;
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
