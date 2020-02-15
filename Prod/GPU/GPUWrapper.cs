using SharpNet.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpNet.GPU
{
    public enum CUDA_Versions { CUDA_10_0, CUDA_10_1 };

    [DebuggerDisplay("{DeviceName()}")]
    public class GPUWrapper : IDisposable
    {
        #region Private fields
        // ReSharper disable once PrivateFieldCanBeConvertedToLocalVariable
        private readonly IntPtr _deviceHandle;
        private readonly string _deviceName;
        private readonly Version _cublasVersion;
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
        private IntPtr _cudnnHandle;
        private int _copyToDeviceCalls;
        private ulong _bytesCopiedToDevice;
        private int _copyToHostCalls;
        private ulong _bytesCopiedToHost;
        private DeviceMemory _lazyStorageBuffer;
        private static readonly IDictionary<int, GPUWrapper> Cache = new Dictionary<int, GPUWrapper>();
        private readonly bool _preAllocateAllDeviceMemory;
        private readonly int _threadId;
        private IntPtr _pointToDeviceMemory;
        private size_t _sizeInBytesOfAllocatedMemory;
        private long _offsetNextSpaceInDeviceMemory;
        //private int _nbChunksInDeviceMemory = 0;
        #endregion

        #region readonly properties
        public int DeviceId { get; }
        public int MaxThreadsPerBlock { get; }
        public int MultiProcessorCount { get; }
        public int WarpSize { get; }
        public Stopwatch SwCopyToDevice { get; } = new Stopwatch();
        public Stopwatch SwCopyToHost { get; } = new Stopwatch();
        #endregion

        public static GPUWrapper FromDeviceId(int deviceId)
        {
            lock (Cache)
            {
                if (!Cache.ContainsKey(deviceId))
                {
                    Cache[deviceId] = new GPUWrapper(deviceId);
                }
                return Cache[deviceId];
            }
        }
        public IntPtr CudnnHandle => _cudnnHandle;
        public void RunKernel(string kernelName, int count, object[] parameterLists)
        {
            CheckThreadId();
            _kernelManager.RunKernel(kernelName, count, parameterLists);
        }
        public IntPtr ActivationDesc(cudnnActivationMode_t activationFunctionType)
        {
            CheckThreadId();
            if (!cacheActivationDesc.TryGetValue(activationFunctionType, out var desc))
            {
                var res = CudnnWrapper.cudnnCreateActivationDescriptor(out desc);
                CheckStatus(res);
                res = CudnnWrapper.cudnnSetActivationDescriptor(desc, activationFunctionType, cudnnNanPropagation_t.CUDNN_NOT_PROPAGATE_NAN, 1.0);
                CheckStatus(res);
                cacheActivationDesc[activationFunctionType] = desc;
            }
            return desc;
        }
        public IntPtr PoolingDesc(cudnnPoolingMode_t poolingMode, int poolingSize, int poolingStride)
        {
            var key = Tuple.Create(poolingMode, poolingSize, poolingStride);
            if (!cachePoolingDesc.TryGetValue(key, out var desc))
            {
                var res = CudnnWrapper.cudnnCreatePoolingDescriptor(out desc);
                CheckStatus(res);
                res = CudnnWrapper.cudnnSetPooling2dDescriptor(desc, poolingMode, cudnnNanPropagation_t.CUDNN_NOT_PROPAGATE_NAN, poolingSize, poolingSize, 0, 0, poolingStride, poolingStride);
                CheckStatus(res);
                cachePoolingDesc[key] = desc;
            }
            return desc;
        }
        public IntPtr ConvDesc(cudnnDataType_t cudaType, int padding, int stride)
        {
            CheckThreadId();
            var key = Tuple.Create(cudaType, padding, stride);
            if (!cacheConvolutionDesc.TryGetValue(key, out var desc))
            {
                var res = CudnnWrapper.cudnnCreateConvolutionDescriptor(out desc);
                CheckStatus(res);
                res = CudnnWrapper.cudnnSetConvolution2dDescriptor(desc, padding, padding, stride, stride, 1, 1, cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION, cudaType);
                CheckStatus(res);
                cacheConvolutionDesc[key] = desc;
            }
            return desc;
        }
        public IntPtr FilterDesc(cudnnDataType_t cudaType, int[] shape)
        {
            CheckThreadId();
            var n = shape[0];
            var c = shape[1];
            var h = shape[2];
            var w = shape[3];
            var key = Tuple.Create(cudaType, n, c, h, w);
            if (!cacheFilterDesc.TryGetValue(key, out var desc))
            {
                var res = CudnnWrapper.cudnnCreateFilterDescriptor(out desc);
                CheckStatus(res);
                res = CudnnWrapper.cudnnSetFilter4dDescriptor(desc, cudaType, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, n, c, h, w);
                CheckStatus(res);
                cacheFilterDesc[key] = desc;
            }
            return desc;
        }
        public IntPtr DropoutDesc(double dropoutRate, IntPtr randomNumberGeneratorStatesBuffer)
        {
            CheckThreadId();
            var res = CudnnWrapper.cudnnCreateDropoutDescriptor(out IntPtr desc);
            CheckStatus(res);
            res = CudnnWrapper.cudnnDropoutGetStatesSize(CudnnHandle, out var stateSize);
            CheckStatus(res);
            res = CudnnWrapper.cudnnSetDropoutDescriptor(desc, _cudnnHandle, (float)dropoutRate, randomNumberGeneratorStatesBuffer, stateSize, 0);
            CheckStatus(res);
            return desc;
        }
        public IntPtr TensorDesc(cudnnDataType_t cudaType, int[] shape)
        {
            CheckThreadId();
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
                var res = CudnnWrapper.cudnnCreateTensorDescriptor(out desc);
                CheckStatus(res);
                res = CudnnWrapper.cudnnSetTensor4dDescriptor(desc, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, cudaType, n, c, h, w);
                CheckStatus(res);
                cacheTensorDesc[key] = desc;
            }
            return desc;
        }
        //returns a buffer storage (in Device memory) with at least 'sizeInBytes' size
        public DeviceMemory StorageBuffer(size_t sizeInBytes)
        {
            CheckThreadId();
            if (_lazyStorageBuffer == null || _lazyStorageBuffer.SizeInBytes < sizeInBytes)
            {
                _lazyStorageBuffer?.Dispose();
                _lazyStorageBuffer = NewDeviceMemory(Math.Max(sizeInBytes, 10 * 1024 * 1024));
            }
            return _lazyStorageBuffer;
        }
        public DeviceMemory NewDeviceMemory(size_t sizeInBytes)
        {
            CheckThreadId();
            if (!_preAllocateAllDeviceMemory)
            {
                return new DeviceMemory(sizeInBytes);
            }
            if (sizeInBytes > FreePreAllocatedMemoryInBytes())
            {
                throw new OutOfMemoryException("Not enough memory to allocate " + sizeInBytes + " (available:" + FreePreAllocatedMemoryInBytes() + ") in " + this);
            }
            var result = NewDeviceMemory(new IntPtr(_pointToDeviceMemory.ToInt64() + (long)(ulong)_offsetNextSpaceInDeviceMemory), sizeInBytes);
            _offsetNextSpaceInDeviceMemory += (long)(ulong)sizeInBytes;
            const long alignment = 128;
            if (_offsetNextSpaceInDeviceMemory % alignment != 0)
            {
                _offsetNextSpaceInDeviceMemory += alignment - (_offsetNextSpaceInDeviceMemory % alignment);
            }
            //++_nbChunksInDeviceMemory;
            return result;
        }
        public DeviceMemory NewDeviceMemory(IntPtr pointer, size_t sizeInBytes)
        {
            CheckThreadId();
            var res = new DeviceMemory(pointer, sizeInBytes);
            return res;
        }

        public void Reset()
        {
            CheckThreadId();
            _copyToDeviceCalls = 0;
            _bytesCopiedToDevice = 0;
            _copyToHostCalls = 0;
            _bytesCopiedToHost = 0;
            SwCopyToDevice.Reset();
            SwCopyToHost.Reset();
            _offsetNextSpaceInDeviceMemory = 0;
            //_nbChunksInDeviceMemory = 0;
            cacheTensorDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyTensorDescriptor(x)));
            cacheTensorDesc.Clear();
            cacheFilterDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyFilterDescriptor(x)));
            cacheFilterDesc.Clear();
            //cacheDropoutDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyDropoutDescriptor(x)));
            //cacheDropoutDesc.Clear();
            cachePoolingDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyPoolingDescriptor(x)));
            cachePoolingDesc.Clear();
            cacheConvolutionDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyConvolutionDescriptor(x)));
            cacheConvolutionDesc.Clear();
            cacheActivationDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyActivationDescriptor(x)));
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
            return _deviceName + ", driver v" + _driverVersion + ", cublas v"+_cublasVersion+", deviceId:" + DeviceId + ", threadId:" + _threadId;
        }
        public override string ToString()
        {
            return DeviceName() + " - " + MemoryInfo();
        }
        public string MemoryInfo()
        {
            CheckThreadId();
            var result = "Free GPU Memory: " + Utils.MemoryBytesToString(FreeMemoryInBytes()) + "/" + Utils.MemoryBytesToString(TotalMemoryInBytes());
            if (_preAllocateAllDeviceMemory)
            {
                result += " - Free PreAllocated GPU Memory: " + Utils.MemoryBytesToString(FreePreAllocatedMemoryInBytes()) + "/" + Utils.MemoryBytesToString(_sizeInBytesOfAllocatedMemory);
            }
            var cudaVersion = CudaVersionFromCudaPath();
            result += string.IsNullOrEmpty(cudaVersion) ? " - no CUDA found" : (" - CUDA " + cudaVersion);
            result += " - Private Memory: " + Utils.MemoryBytesToString((ulong)Process.GetCurrentProcess().PrivateMemorySize64);
            result += " - Used in GC: " + Utils.MemoryBytesToString((ulong)GC.GetTotalMemory(false));
            result += " - Available RAM Memory: " + Utils.MemoryBytesToString(Utils.AvailableRamMemoryInBytes());
            result += " - " + Utils.MemoryBytesToString(_bytesCopiedToDevice) + " CopiedToDevice (" + _copyToDeviceCalls + "calls, " + SwCopyToDevice.ElapsedMilliseconds + "ms)";
            result += " - " + Utils.MemoryBytesToString(_bytesCopiedToHost) + " CopiedToHost (" + _copyToHostCalls + "calls, " + SwCopyToHost.ElapsedMilliseconds + "ms)";
            result += " - CurrentThreadId#" + System.Threading.Thread.CurrentThread.ManagedThreadId;
            return result;
        }
        public size_t AvailableMemoryInBytes()
        {
            CheckThreadId();
            return (_preAllocateAllDeviceMemory ? FreePreAllocatedMemoryInBytes() : (size_t)FreeMemoryInBytes());
        }
        public IntPtr CudaBlasHandle => _cudaBlasHandle;
        public StreamWrapper DefaultStream { get; }

        public static void CheckStatus(cublasStatus_t _status)
        {
            if (_status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            {
                throw new Exception(_status.ToString());
            }

        }
        public static void CheckStatus(cublasStatus_t _status, Func<string> getComment)
        {
            if (_status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            {
                throw new Exception(_status + " " + getComment());
            }

        }
        public static void CheckStatus(cudnnStatus_t _status)
        {
            if (_status != cudnnStatus_t.CUDNN_STATUS_SUCCESS)
            {
                throw new Exception(_status.ToString());
            }
        }
        public static void CheckStatus(cudnnStatus_t _status, Func<string> getComment)
        {
            if (_status != cudnnStatus_t.CUDNN_STATUS_SUCCESS)
            {
                throw new Exception(_status+" "+ getComment());
            }
        }
        
        public static void CheckStatus(CUresult _status)
        {
            if (_status != CUresult.CUDA_SUCCESS)
            {
                throw new Exception(_status.ToString());
            }
        }

        public static void LogDebug(string msg)
        {
            var res = CudartWrapper.cudaGetDevice(out int currentDevice);
            CheckStatus(res);
            var fileName = Path.Combine(@"c:\temp\SharpNet_" + System.Threading.Thread.CurrentThread.ManagedThreadId +".txt");
            File.AppendAllText(fileName, DateTime.Now.ToString("HH:mm:ss.ff") + " id:"+ currentDevice+" " +msg +Environment.NewLine);
        }


        public static void CheckStatus(cudaError_t _status)
        {
            if (_status != cudaError_t.cudaSuccess)
            {
                throw new Exception(_status.ToString());
            }
        }
        public static void CheckStatus(CUresult _status, Func<string> getComment)
        {
            if (_status != CUresult.CUDA_SUCCESS)
            {
                throw new Exception(_status+" "+ getComment());
            }
        }
        public static void CheckStatus(nvrtcResult _status)
        {
            if (_status != nvrtcResult.NVRTC_SUCCESS)
            {
                throw new Exception(_status.ToString());
            }
        }
        public static CUDA_Versions GetInstalledCudaVersion()
        {
            var cudaPath = (Environment.GetEnvironmentVariable("CUDA_PATH") ?? "").ToLowerInvariant();
            if (cudaPath.ToLowerInvariant().Contains("v10.0"))
            {
                return CUDA_Versions.CUDA_10_0;
            }
            if (cudaPath.ToLowerInvariant().Contains("v10.1"))
            {
                return CUDA_Versions.CUDA_10_1;
            }
            return CUDA_Versions.CUDA_10_0;
        }
        public static int GetDeviceCount()
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

#region Dispose pattern
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
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
                //managed memory
                DefaultStream?.Dispose();
            }

            //unmanaged memory
            if (_preAllocateAllDeviceMemory)
            {
                var cuResult = NVCudaWrapper.cuMemFree_v2(_pointToDeviceMemory);
                CheckStatus(cuResult);
                _pointToDeviceMemory = IntPtr.Zero;
                _sizeInBytesOfAllocatedMemory = 0;
                _offsetNextSpaceInDeviceMemory = 0;
                //_nbChunksInDeviceMemory = 0;
            }
            var cublasRes = CublasWrapper.cublasDestroy_v2(_cudaBlasHandle);
            CheckStatus(cublasRes);
            _cudaBlasHandle = IntPtr.Zero;
            var cudnnRes = CudnnWrapper.cudnnDestroy(_cudnnHandle);
            CheckStatus(cudnnRes);
            _cudnnHandle = IntPtr.Zero;
            var cuRes = NVCudaWrapper.cuCtxDestroy_v2(_contextHandle);
            CheckStatus(cuRes);
            _contextHandle = IntPtr.Zero;
        }
        ~GPUWrapper()
        {
            Dispose(false);
        }
#endregion


        private GPUWrapper(int deviceId)
        {
            CudartWrapper.cudaDeviceReset();
            CudartWrapper.cudaSetDevice(deviceId);
            DeviceId = deviceId;
            _threadId = System.Threading.Thread.CurrentThread.ManagedThreadId;
            _preAllocateAllDeviceMemory = false;
            var cublasRes = CublasWrapper.cublasCreate_v2(ref _cudaBlasHandle);
            CheckStatus(cublasRes);

            //We retrieve the cublas version
            cublasRes = CublasWrapper.cublasGetVersion_v2(CudaBlasHandle, out var cublasVersion);
            CheckStatus(cublasRes);
            _cublasVersion = NewVersion(cublasVersion);

            _deviceHandle = GetDeviceHandle(deviceId);

            var cuRes = NVCudaWrapper.cuCtxCreate_v2(out _contextHandle, 0, _deviceHandle);
            CheckStatus(cuRes);

            var devName = new byte[256];
            cuRes = NVCudaWrapper.cuDeviceGetName(devName, devName.Length, _deviceHandle);
            CheckStatus(cuRes);
            System.Text.ASCIIEncoding enc = new System.Text.ASCIIEncoding();
            _deviceName = enc.GetString(devName).Replace("\0", "");

            cuRes = NVCudaWrapper.cuDriverGetVersion(out int driverVersion);
            CheckStatus(cuRes);
            _driverVersion = NewVersion(driverVersion);

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
            if (_preAllocateAllDeviceMemory)
            {
                _sizeInBytesOfAllocatedMemory = (75 * FreeMemoryInBytes()) / 100;
                while (CUresult.CUDA_SUCCESS != NVCudaWrapper.cuMemAlloc_v2(out _pointToDeviceMemory, _sizeInBytesOfAllocatedMemory))
                {
                    _sizeInBytesOfAllocatedMemory = (90 * _sizeInBytesOfAllocatedMemory) / 100;
                    if (_sizeInBytesOfAllocatedMemory < 0.1 * FreeMemoryInBytes())
                    {
                        throw new Exception("Fail to allocate enough Device memory in " + this);
                    }
                }
            }
        }

        private static Version NewVersion(int driverVersion)
        {
            var major = driverVersion / 1000;
            var minor = driverVersion % 100;
            var build = (driverVersion % 1000)/100;
            return (build==0)?new Version(major, minor): new Version(major, minor, build);
        }

        private static void CuMemGetInfoV2(out size_t freeMemoryInBytes, out size_t totalMemoryInBytes)
        {
            var res = NVCudaWrapper.cuMemGetInfo_v2(out freeMemoryInBytes, out totalMemoryInBytes);
            CheckStatus(res);
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
        private static ulong TotalMemoryInBytes()
        {
            CuMemGetInfoV2(out size_t _, out size_t totalMemoryInBytes);
            return totalMemoryInBytes;

        }
        private static string CudaVersionFromCudaPath()
        {
            try
            {
                var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
                return string.IsNullOrEmpty(cudaPath) ? "" : cudaPath.Split(new[] { '/', '\\' }, StringSplitOptions.RemoveEmptyEntries).LastOrDefault();
            }
            catch (Exception)
            {
                return "";
            }
        }
        private ulong FreeMemoryInBytes()
        {
            CuMemGetInfoV2(out size_t freeMemoryInBytes, out size_t _);
            return freeMemoryInBytes;
        }
        private size_t FreePreAllocatedMemoryInBytes()
        {
            return (_sizeInBytesOfAllocatedMemory - (ulong)_offsetNextSpaceInDeviceMemory);
        }
        /// <summary>
        /// Ensure that the current ThreadId is the same used when creating the 'this' object
        /// </summary>
        public void CheckThreadId()
        {
            if (_threadId != System.Threading.Thread.CurrentThread.ManagedThreadId)
            {
                throw new Exception("invalid Thread Id " + this);
            }
        }
    }
}
