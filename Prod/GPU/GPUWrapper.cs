using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;

namespace SharpNet.GPU
{
    public class GPUWrapper : IDisposable
    {
        #region Private fields
        // ReSharper disable once PrivateFieldCanBeConvertedToLocalVariable
        private readonly IntPtr _deviceHandle;
        private readonly string _deviceName;
        private readonly Version _driverVersion;
        private readonly KernelManager _kernelManager;
        private readonly IDictionary<Tuple<cudnnDataType_t, int, int, int, int>, IntPtr> cacheTensorDesc = new Dictionary<Tuple<cudnnDataType_t, int, int, int, int>, IntPtr>();
        private readonly IDictionary<Tuple<cudnnDataType_t, int, int, int, int>, IntPtr> cacheFilterDesc = new Dictionary<Tuple<cudnnDataType_t, int, int, int, int>, IntPtr>();
        private readonly IDictionary<Tuple<cudnnPoolingMode_t, int, int>, IntPtr> cachePoolingDesc = new Dictionary<Tuple<cudnnPoolingMode_t, int, int>, IntPtr>();
        private readonly IDictionary<Tuple<cudnnDataType_t, int, int>, IntPtr> cacheConvolutionDesc = new Dictionary<Tuple<cudnnDataType_t, int, int>, IntPtr>();
        private readonly IDictionary<cudnnActivationMode_t, IntPtr> cacheActivationDesc = new Dictionary<cudnnActivationMode_t, IntPtr>();
        private readonly IDictionary<CUdevice_attribute, int> properties = new Dictionary<CUdevice_attribute, int>();
        private IntPtr _cudaBlasHandle;
        private IntPtr _contextHandle;
        private StreamWrapper _defaultStream;
        private IntPtr _cudnnHandle;
        private int _copyToDeviceCalls;
        private ulong _bytesCopiedToDevice;
        private int _copyToHostCalls;
        private ulong _bytesCopiedToHost;
        private CudaDeviceMemory _lazyStorageBuffer;
        #endregion
        public static readonly GPUWrapper Default = new GPUWrapper(0);
        #region readonly properties
        public int MaxThreadsPerBlock { get; }
        public int MultiProcessorCount { get; }
        public int WarpSize { get; }
        #endregion

        private GPUWrapper(int deviceId)
        {
            var cublasRes = CublasWrapper.cublasCreate_v2(ref _cudaBlasHandle);
            CheckStatus(cublasRes);

            _deviceHandle = GetDeviceHandle(deviceId);

            var res = NVCudaWrapper.cuCtxCreate_v2(out _contextHandle, 0, _deviceHandle);
            CheckStatus(res);

            var devName = new byte[256];
            res = NVCudaWrapper.cuDeviceGetName(devName, devName.Length, _deviceHandle);
            CheckStatus(res);
            System.Text.ASCIIEncoding enc = new System.Text.ASCIIEncoding();
            _deviceName = enc.GetString(devName).Replace("\0", "");

            res = NVCudaWrapper.cuDriverGetVersion(out int driverVersion);
            CheckStatus(res);
            _driverVersion = new Version(driverVersion / 1000, driverVersion % 100);

            foreach (var e in Enum.GetValues(typeof(CUdevice_attribute)).Cast<CUdevice_attribute>())
            {
                var cuDeviceGetAttribute = NVCudaWrapper.cuDeviceGetAttribute(out int tmp, e, _deviceHandle);
                if (cuDeviceGetAttribute == CUresult.CUDA_SUCCESS)
                {
                    properties[e] = tmp;
                }
            }
            MaxThreadsPerBlock = properties[CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK];
            MultiProcessorCount = properties[CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT];
            WarpSize = properties[CUdevice_attribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE];
            DefaultStream = new StreamWrapper();
            var cudnnRes = CudnnWrapper.cudnnCreate(out _cudnnHandle);
            CheckStatus(cudnnRes);
            _kernelManager = new KernelManager(this);
        }
        public IntPtr CudnnHandle => _cudnnHandle;
        public void RunKernel(string kernelName, int count, object[] parameterLists)
        {
            _kernelManager.RunKernel(kernelName, count, parameterLists);
        }
        public IntPtr ActivationDesc(cudnnActivationMode_t activationFunctionType)
        {
            if (!cacheActivationDesc.TryGetValue(activationFunctionType, out var desc))
            {
                CudnnWrapper.cudnnCreateActivationDescriptor(out desc);
                CudnnWrapper.cudnnSetActivationDescriptor(desc, activationFunctionType, cudnnNanPropagation_t.CUDNN_NOT_PROPAGATE_NAN, 1.0);
                cacheActivationDesc[activationFunctionType] = desc;
            }
            return desc;
        }
        public IntPtr PoolingDesc(cudnnPoolingMode_t poolingMode, int poolingSize, int poolingStride)
        {
            var key = Tuple.Create(poolingMode, poolingSize, poolingStride);
            if (!cachePoolingDesc.TryGetValue(key, out var desc))
            {
                CudnnWrapper.cudnnCreatePoolingDescriptor(out desc);
                CudnnWrapper.cudnnSetPooling2dDescriptor(desc, poolingMode, cudnnNanPropagation_t.CUDNN_NOT_PROPAGATE_NAN, poolingSize, poolingSize, 0, 0, poolingStride, poolingStride);
                cachePoolingDesc[key] = desc;
            }
            return desc;
        }
        public IntPtr ConvDesc(cudnnDataType_t cudaType, int padding, int stride)
        {
            var key = Tuple.Create(cudaType, padding, stride);
            if (!cacheConvolutionDesc.TryGetValue(key, out var desc))
            {
                CudnnWrapper.cudnnCreateConvolutionDescriptor(out desc);
                CudnnWrapper.cudnnSetConvolution2dDescriptor(desc, padding, padding, stride, stride, 1, 1, cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION, cudaType);
                cacheConvolutionDesc[key] = desc;
            }
            return desc;
        }
        public IntPtr FilterDesc(cudnnDataType_t cudaType, int[] shape)
        {
            var n = shape[0];
            var c = shape[1];
            var h = shape[2];
            var w = shape[3];
            var key = Tuple.Create(cudaType, n, c, h, w);
            if (!cacheFilterDesc.TryGetValue(key, out var desc))
            {
                CudnnWrapper.cudnnCreateFilterDescriptor(out desc);
                CudnnWrapper.cudnnSetFilter4dDescriptor(desc, cudaType, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, n, c, h, w);
                cacheFilterDesc[key] = desc;
            }
            return desc;
        }
        public IntPtr DropoutDesc(double dropoutRate, IntPtr randomNumberGeneratorStatesBuffer)
        {
            CudnnWrapper.cudnnCreateDropoutDescriptor(out IntPtr desc);
            var res = CudnnWrapper.cudnnDropoutGetStatesSize(CudnnHandle, out var stateSize);
            CheckStatus(res);
            res = CudnnWrapper.cudnnSetDropoutDescriptor(desc, _cudnnHandle, (float)dropoutRate, randomNumberGeneratorStatesBuffer, stateSize, 0);
            CheckStatus(res);
            return desc;
        }
        public IntPtr TensorDesc(cudnnDataType_t cudaType, int[] shape)
        {
            var n = shape[0];
            var c = shape.Length >= 2 ? shape[1] : 1;
            var h = shape.Length >= 3 ? shape[2] : 1;
            var w = shape.Length >= 4 ? shape[3] : 1;
            if (c == 1 && h == 1 && w > 1)
            {
                c = w;
                w = 1;
            }
            var key = Tuple.Create(cudaType, n, c, h, w);
            if (!cacheTensorDesc.TryGetValue(key, out var desc))
            {
                CudnnWrapper.cudnnCreateTensorDescriptor(out desc);
                CudnnWrapper.cudnnSetTensor4dDescriptor(desc, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, cudaType, n, c, h, w);
                cacheTensorDesc[key] = desc;
            }
            return desc;
        }
        //returns a buffer storage (in Device memory) with at least 'sizeInBytes' size
        public CudaDeviceMemory StorageBuffer(size_t sizeInBytes)
        {
            if (_lazyStorageBuffer == null || _lazyStorageBuffer.SizeInBytes < sizeInBytes)
            {
                _lazyStorageBuffer?.Dispose();
                _lazyStorageBuffer = new CudaDeviceMemory(Math.Max(sizeInBytes,10*1024*1024));
            }
            return _lazyStorageBuffer;
        }
        public void ClearMemory()
        {
            cacheTensorDesc.Values.ToList().ForEach(x => CudnnWrapper.cudnnDestroyTensorDescriptor(x));
            cacheTensorDesc.Clear();
            cacheFilterDesc.Values.ToList().ForEach(x => CudnnWrapper.cudnnDestroyFilterDescriptor(x));
            cacheFilterDesc.Clear();
            //cacheDropoutDesc.Values.ToList().ForEach(x => CudnnWrapper.cudnnDestroyDropoutDescriptor(x));
            //cacheDropoutDesc.Clear();
            cachePoolingDesc.Values.ToList().ForEach(x => CudnnWrapper.cudnnDestroyPoolingDescriptor(x));
            cachePoolingDesc.Clear();
            cacheConvolutionDesc.Values.ToList().ForEach(x => CudnnWrapper.cudnnDestroyConvolutionDescriptor(x));
            cacheConvolutionDesc.Clear();
            cacheActivationDesc.Values.ToList().ForEach(x => CudnnWrapper.cudnnDestroyActivationDescriptor(x));
            cacheActivationDesc.Clear();
        }
        public void LogCopyToDeviceCall(ulong byteCopied)
        {
            Debug.Assert(byteCopied > 0);
            ++_copyToDeviceCalls;
            _bytesCopiedToDevice += byteCopied;
        }
        public void LogCopyToHostCall(ulong byteCopied)
        {
            Debug.Assert(byteCopied > 0);
            ++_copyToHostCalls;
            _bytesCopiedToHost += byteCopied;
        }
        public string DeviceName()
        {
            return _deviceName + " " + _driverVersion;
        }
        public override string ToString()
        {
            return _deviceName+" "+_driverVersion+" - "+MemoryInfo();
        }
        public string MemoryInfo()
        {
            var res = NVCudaWrapper.cuMemGetInfo_v2(out size_t freeMemoryInBytes, out size_t totalMemoryInBytes);
            CheckStatus(res);
            var result = "FreeMemory: " + Utils.MemoryBytesToString(freeMemoryInBytes)  + "/" + Utils.MemoryBytesToString(totalMemoryInBytes);
            result += " - " + Utils.MemoryBytesToString(_bytesCopiedToDevice)+" CopiedToDevice (" + _copyToDeviceCalls + "calls)";
            result += " - " + Utils.MemoryBytesToString(_bytesCopiedToHost) + " CopiedToHost (" + _copyToHostCalls + "calls)";
            return result;
        }

        public IntPtr CudaBlasHandle => _cudaBlasHandle;
        public StreamWrapper DefaultStream
        {
            get => _defaultStream;
            private set => _defaultStream = value;
        }
        public static void CheckStatus(cublasStatus_t _status)
        {
            if (_status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            {
                throw new Exception(_status.ToString());
            }

        }
        public static void CheckStatus(cudnnStatus_t _status)
        {
            if (_status != cudnnStatus_t.CUDNN_STATUS_SUCCESS)
            {
                throw new Exception(_status.ToString());
            }
        }
        public static void CheckStatus(CUresult _status)
        {
            if (_status != CUresult.CUDA_SUCCESS)
            {
                throw new Exception(_status.ToString());
            }
        }
        public static void CheckStatus(nvrtcResult _status)
        {
            if (_status != nvrtcResult.NVRTC_SUCCESS)
            {
                throw new Exception(_status.ToString());
            }
        }

        #region Dispose pattern
        public void Dispose()
        {
            Dispose(true);
        }
        // ReSharper disable once RedundantDefaultMemberInitializer
        private bool disposed = false;
        private void Dispose(bool disposing)
        {
            if (disposed)
            {
                return;
            }
            disposed = true;
            if (disposing)
            {
                GC.SuppressFinalize(this);
                Dispose(ref _defaultStream);
                CublasWrapper.cublasDestroy_v2(_cudaBlasHandle);
                _cudaBlasHandle = IntPtr.Zero;
                CudnnWrapper.cudnnDestroy(_cudnnHandle);
                _cudnnHandle = IntPtr.Zero;
                NVCudaWrapper.cuCtxDestroy_v2(_contextHandle);
                _contextHandle = IntPtr.Zero;
            }
        }
        private static void Dispose<T>(ref T field) where T : class, IDisposable
        {
            if (field != null)
            {
                try
                {
                    field.Dispose();
                }
                catch (Exception ex)
                {
                    Debug.WriteLine(ex.Message);
                }

                field = null;
            }
        }
        ~GPUWrapper()
        {
            Dispose(false);
        }
        #endregion

        private static int GetDeviceCount()
        {
            var res = NVCudaWrapper.cuDeviceGetCount(out int deviceCount);
            if (res == CUresult.CUDA_ERROR_NOT_INITIALIZED)
            {
                res = NVCudaWrapper.cuInit(0);
                CheckStatus(res);
                res = NVCudaWrapper.cuDeviceGetCount(out deviceCount);
                CheckStatus(res);
            }
            else
            {
                CheckStatus(res);
            }

            return deviceCount;
        }
        private static IntPtr GetDeviceHandle(int deviceId)
        {
            int deviceCount = GetDeviceCount();
            if (deviceCount == 0)
            {
                throw new Exception(CUresult.CUDA_ERROR_NO_DEVICE + " Cuda initialization error: There is no device supporting CUDA");
            }
            if (deviceId < 0 || deviceId > deviceCount - 1)
            {
                throw new ArgumentOutOfRangeException(nameof(deviceId), deviceId, "The device ID is not in the range [0.." + (deviceCount - 1) + "]");
            }
            var res = NVCudaWrapper.cuDeviceGet(out IntPtr deviceHandle, deviceId);
            CheckStatus(res);
            return deviceHandle;
        }
    }
}
