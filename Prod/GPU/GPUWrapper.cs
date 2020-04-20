using SharpNet.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;

namespace SharpNet.GPU
{
    public enum CUDA_Versions { CUDA_10_0, CUDA_10_1, CUDA_10_2 };

    [DebuggerDisplay("{"+nameof(DeviceName)+"()}")]
    public unsafe class GPUWrapper : IDisposable
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
        private readonly IDictionary<Tuple<cudnnPoolingMode_t, int, int, int>, IntPtr> cachePoolingDesc = new Dictionary<Tuple<cudnnPoolingMode_t, int, int, int>, IntPtr>();
        private readonly IDictionary<Tuple<cudnnDataType_t, int, int, int, int, int, int>, IntPtr> cacheConvolutionDesc = new Dictionary<Tuple<cudnnDataType_t, int, int, int, int, int, int>, IntPtr>();
        private readonly IDictionary<cudnnActivationMode_t, IntPtr> cacheActivationDesc = new Dictionary<cudnnActivationMode_t, IntPtr>();
        private readonly IDictionary<Tuple<IntPtr, IntPtr, IntPtr, IntPtr, ConvolutionAlgoPreference>, cudnnConvolutionBwdFilterAlgo_t> cacheConvolutionBackwardFilterAlgorithm = new Dictionary<Tuple<IntPtr, IntPtr, IntPtr, IntPtr, ConvolutionAlgoPreference>, cudnnConvolutionBwdFilterAlgo_t>();
        private readonly IDictionary<Tuple<IntPtr, IntPtr, IntPtr, IntPtr, ConvolutionAlgoPreference>, cudnnConvolutionBwdDataAlgo_t> cacheFindConvolutionBackwardDataAlgorithm = new Dictionary<Tuple<IntPtr, IntPtr, IntPtr, IntPtr, ConvolutionAlgoPreference>, cudnnConvolutionBwdDataAlgo_t>();
        private readonly IDictionary<Tuple<IntPtr, IntPtr, IntPtr, IntPtr, ConvolutionAlgoPreference>, cudnnConvolutionFwdAlgo_t> cacheConvolutionForwardAlgorithm = new Dictionary<Tuple<IntPtr, IntPtr, IntPtr, IntPtr, ConvolutionAlgoPreference>, cudnnConvolutionFwdAlgo_t>();
        private readonly IDictionary<CUdevice_attribute, int> properties = new Dictionary<CUdevice_attribute, int>();
        private IntPtr _cudaBlasHandle;
        private IntPtr _contextHandle;
        private IntPtr _cudnnHandle;
        private int _copyToDeviceCalls;
        private ulong _bytesCopiedToDevice;
        private int _copyToHostCalls;
        private ulong _bytesCopiedToHost;
        private readonly int _threadId;
        private readonly int _deviceId;
        private static readonly IDictionary<int, GPUWrapper> Cache = new Dictionary<int, GPUWrapper>();
        #endregion

        #region readonly properties
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

        public int DeviceId => _deviceId;

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
        public IntPtr PoolingDesc(cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride)
        {
            var key = Tuple.Create(poolingMode, poolingHeight, poolingWidth, poolingStride);
            if (!cachePoolingDesc.TryGetValue(key, out var desc))
            {
                var res = CudnnWrapper.cudnnCreatePoolingDescriptor(out desc);
                CheckStatus(res);
                res = CudnnWrapper.cudnnSetPooling2dDescriptor(desc, poolingMode, cudnnNanPropagation_t.CUDNN_NOT_PROPAGATE_NAN, poolingHeight, poolingWidth, 0, 0, poolingStride, poolingStride);
                CheckStatus(res);
                cachePoolingDesc[key] = desc;
            }
            return desc;
        }
        private static List<T1> ExtractFromStackalloc<T1>(T1* stackAllocated, int length) where T1 : unmanaged
        {
            var result = new List<T1>();
            for (int i = 0; i < length; ++i)
            {
                result.Add(stackAllocated[i]);
            }
            return result;
        }

        #region Convolution


        /// <summary>
        /// Here is benchmark performed on a GTX 1080 with cuDNN 7.6
        /// for WRN-16-10:
        ///     FASTEST:                                          97.6s/epoch
        ///     FASTEST_DETERMINIST:                              same speed (98.8s/epoch)
        ///     10_USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS:   1.4x slower then FASTEST (137.3s/epoch)
        ///     FASTEST_DETERMINIST_NO_TRANSFORM:                 1.8x slower then FASTEST (178.2s/epoch)
        /// for WRN-28-10:
        ///     FASTEST:                                          185.3s/epoch
        ///     FASTEST_DETERMINIST:                              same speed (185.9s/epoch)
        ///     10_USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS:   1.6x slower then FASTEST (291.1s/epoch)
        ///     FASTEST_DETERMINIST_NO_TRANSFORM:                 2x slower then FASTEST (376.2s/epoch)
        /// </summary>
        public enum ConvolutionAlgoPreference
        {
            //fastest *determinist* algorithm : RECOMMENDED METHOD
            //achieves nearly same speed as FASTEST (<1% slower) but is deterministic which is easier for debugging
            FASTEST_DETERMINIST,

            //fastest algorithm, even if it is not determinist
            //it is only slightly faster (<1 %) then 'FASTEST_DETERMINIST' but more difficult to investigate / debug
            // ReSharper disable once UnusedMember.Global
            FASTEST,

            //fastest determinist algorithm not based on Fast-Fourier or Winograd Transform
            //this is the only supported mode on CPU
            //it is mainly used for Non Regression Tests between CPU & GPU (when testing Convolution forward/backward) 
            //around 2x slower then FASTEST on WRN-16-10 & WRN-28-10
            FASTEST_DETERMINIST_NO_TRANSFORM,

            //Use the algorithm returned by methods:
            //      cudnnGetConvolutionForwardAlgorithm
            //      cudnnGetConvolutionBackwardFilterAlgorithm
            //      cudnnGetConvolutionBackwardDataAlgorithm
            //it is mainly used for backward compatibility
            //around 1.5x slower then FASTEST on WRN-16-10 & WRN-28-10
            USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS
        }

        /// <summary>
        /// return the Convolution Forward Algorithm to be used
        /// </summary>
        /// <param name="xDesc">input tensor descriptor</param>
        /// <param name="filterDesc">filter descriptor</param>
        /// <param name="convDesc">convolution descriptor</param>
        /// <param name="yDesc">output tensor descriptor</param>
        /// <param name="forwardAlgoPreference"></param>
        /// <returns></returns>
        public cudnnConvolutionFwdAlgo_t ConvolutionForwardAlgorithm(IntPtr xDesc, IntPtr filterDesc, IntPtr convDesc, IntPtr yDesc, ConvolutionAlgoPreference forwardAlgoPreference)
        {
            var key = Tuple.Create(xDesc, yDesc, convDesc, filterDesc, forwardAlgoPreference);
            cudnnStatus_t cudnnStatus;
            if (cacheConvolutionForwardAlgorithm.TryGetValue(key, out var forwardAlgo))
            {
                return forwardAlgo;
            }

            if (forwardAlgoPreference == ConvolutionAlgoPreference.USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS)
            {
                cudnnStatus = CudnnWrapper.cudnnGetConvolutionForwardAlgorithm(CudnnHandle, xDesc, filterDesc, convDesc, yDesc, cudnnConvolutionFwdPreference_t.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, out forwardAlgo);
                CheckStatus(cudnnStatus);
                cacheConvolutionForwardAlgorithm[key] = forwardAlgo;
                return forwardAlgo;
            }

            //we benchmark all available forward algorithms
            int requestedAlgoCount = Enum.GetNames(typeof(cudnnConvolutionFwdAlgo_t)).Length;
            var perfResultsStackalloc = stackalloc cudnnConvolutionFwdAlgoPerf_t[requestedAlgoCount];
            cudnnStatus = CudnnWrapper.cudnnFindConvolutionForwardAlgorithm(CudnnHandle, xDesc, filterDesc, convDesc, yDesc, requestedAlgoCount, out int returnedAlgoCount, perfResultsStackalloc);
            CheckStatus(cudnnStatus);
            //'perfResults' contains KPI for all available algos, starting from the fastest
            var perfResults = ExtractFromStackalloc(perfResultsStackalloc, returnedAlgoCount).Where(p => p.status == cudnnStatus_t.CUDNN_STATUS_SUCCESS).ToList();

            //we apply our algorithms constraints (deterministic, no transform, etc.)
            if (IsDeterminist(forwardAlgoPreference))
            {
                perfResults = perfResults.Where(p => p.determinism == cudnnDeterminism_t.CUDNN_DETERMINISTIC).ToList();
            }
            if (forwardAlgoPreference == ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM)
            {
                perfResults = perfResults.Where(p => p.algo !=cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_FFT && p.algo !=cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING && p.algo != cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD && p.algo != cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_​WINOGRAD_NONFUSED).ToList();
            }

            //we choose the fastest algorithms matching the constraints
            forwardAlgo = perfResults[0].algo;
            cacheConvolutionForwardAlgorithm[key] = forwardAlgo;
            return forwardAlgo;
        }
        /// <summary>
        /// return the Convolution Backward Filter Algorithm to be used
        /// </summary>
        /// <param name="xDesc">input tensor descriptor</param>
        /// <param name="dyDesc">output gradient tensor descriptor</param>
        /// <param name="convDesc">convolution descriptor</param>
        /// <param name="filterDesc">filter descriptor</param>
        /// <param name="backwardAlgoPreference"></param>
        /// <returns></returns>
        public cudnnConvolutionBwdFilterAlgo_t ConvolutionBackwardFilterAlgorithm(IntPtr xDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr filterDesc, ConvolutionAlgoPreference backwardAlgoPreference)
        {
            var key = Tuple.Create(xDesc, dyDesc, convDesc, filterDesc, backwardAlgoPreference);
            cudnnStatus_t cudnnStatus;
            if (cacheConvolutionBackwardFilterAlgorithm.TryGetValue(key, out var backwardFilterAlgo))
            {
                return backwardFilterAlgo;
            }
            if (backwardAlgoPreference == ConvolutionAlgoPreference.USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS)
            {
                cudnnStatus = CudnnWrapper.cudnnGetConvolutionBackwardFilterAlgorithm(CudnnHandle, xDesc, dyDesc, convDesc, filterDesc, cudnnConvolutionBwdFilterPreference_t.CUDNN_CONVOLUTION_BWD_FILTER_​PREFER_FASTEST, 0, out backwardFilterAlgo);
                CheckStatus(cudnnStatus);
                cacheConvolutionBackwardFilterAlgorithm[key] = backwardFilterAlgo;
                return backwardFilterAlgo;
            }

            //We benchmark all available backward filter algorithms
            int requestedAlgoCount = Enum.GetNames(typeof(cudnnConvolutionBwdFilterAlgo_t)).Length;
            var perfResultsStackalloc = stackalloc cudnnConvolutionBwdFilterAlgoPerf_t[requestedAlgoCount];
            cudnnStatus = CudnnWrapper.cudnnFindConvolutionBackwardFilterAlgorithm(CudnnHandle, xDesc, dyDesc, convDesc, filterDesc, requestedAlgoCount, out int returnedAlgoCount, perfResultsStackalloc);
            CheckStatus(cudnnStatus);
            //'perfResults' contains KPI for all available algos, starting from the fastest
            var perfResults = ExtractFromStackalloc(perfResultsStackalloc, returnedAlgoCount).Where(p => p.status == cudnnStatus_t.CUDNN_STATUS_SUCCESS).ToList();

            //we apply our algorithms constraints (deterministic, no transform, etc.)
            if (IsDeterminist(backwardAlgoPreference))
            {
                perfResults = perfResults.Where(p => p.determinism == cudnnDeterminism_t.CUDNN_DETERMINISTIC).ToList();
            }
            if (backwardAlgoPreference == ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM)
            {
                perfResults = perfResults.Where(p => p.algo != cudnnConvolutionBwdFilterAlgo_t.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT && p.algo != cudnnConvolutionBwdFilterAlgo_t.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_​FFT_TILING && p.algo != cudnnConvolutionBwdFilterAlgo_t.CUDNN_CONVOLUTION_BWD_FILTER_​WINOGRAD_NONFUSED).ToList();
            }

            //we choose the fastest algorithms matching the constraints
            backwardFilterAlgo = perfResults[0].algo;
            cacheConvolutionBackwardFilterAlgorithm[key] = backwardFilterAlgo;
            return backwardFilterAlgo;
        }

        /// <summary>
        /// return the Convolution Backward Data Algorithm to be used
        /// </summary>
        /// <param name="filterDesc">filter descriptor</param>
        /// <param name="dyDesc">output gradient tensor descriptor</param>
        /// <param name="convDesc">convolution descriptor</param>
        /// <param name="xDesc">input tensor descriptor</param>
        /// <param name="backwardAlgoPreference"></param>
        /// <returns></returns>
        public cudnnConvolutionBwdDataAlgo_t ConvolutionBackwardDataAlgorithm(IntPtr filterDesc, IntPtr dyDesc, IntPtr convDesc, IntPtr xDesc, ConvolutionAlgoPreference backwardAlgoPreference)
        {
            var key = Tuple.Create(filterDesc, dyDesc, convDesc, xDesc, backwardAlgoPreference);
            cudnnStatus_t cudnnStatus;
            if (cacheFindConvolutionBackwardDataAlgorithm.TryGetValue(key, out var backwardDataAlgo))
            {
                return backwardDataAlgo;
            }
            if (backwardAlgoPreference == ConvolutionAlgoPreference.USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS)
            {
                cudnnStatus = CudnnWrapper.cudnnGetConvolutionBackwardDataAlgorithm(CudnnHandle, filterDesc, dyDesc, convDesc, xDesc, cudnnConvolutionBwdDataPreference_t.CUDNN_CONVOLUTION_BWD_DATA_​PREFER_FASTEST, 0, out backwardDataAlgo);
                CheckStatus(cudnnStatus);
                return backwardDataAlgo;
            }

            //We benchmark all available backward data algorithms
            int requestedAlgoCount = Enum.GetNames(typeof(cudnnConvolutionBwdDataAlgo_t)).Length;
            var perfResultsStackalloc = stackalloc cudnnConvolutionBwdDataAlgoPerf_t[requestedAlgoCount];
            cudnnStatus = CudnnWrapper.cudnnFindConvolutionBackwardDataAlgorithm(CudnnHandle, filterDesc, dyDesc, convDesc, xDesc, requestedAlgoCount, out int returnedAlgoCount, perfResultsStackalloc);
            CheckStatus(cudnnStatus);
            var perfResults = ExtractFromStackalloc(perfResultsStackalloc, returnedAlgoCount).Where(p=>p.status == cudnnStatus_t.CUDNN_STATUS_SUCCESS).ToList();

            //we apply our algorithms constraints (deterministic, no transform, etc.)
            if (IsDeterminist(backwardAlgoPreference))
            {
                perfResults = perfResults.Where(p => p.determinism == cudnnDeterminism_t.CUDNN_DETERMINISTIC).ToList();
            }
            if (backwardAlgoPreference == ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM)
            {
                perfResults = perfResults.Where(p => p.algo != cudnnConvolutionBwdDataAlgo_t.CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT && p.algo != cudnnConvolutionBwdDataAlgo_t.CUDNN_CONVOLUTION_BWD_DATA_ALGO_​FFT_TILING && p.algo != cudnnConvolutionBwdDataAlgo_t.CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD && p.algo != cudnnConvolutionBwdDataAlgo_t.CUDNN_CONVOLUTION_BWD_DATA_ALGO_​WINOGRAD_NONFUSED).ToList();
            }

            //we choose the fastest algorithms matching the constraints
            backwardDataAlgo = perfResults[0].algo;
            cacheFindConvolutionBackwardDataAlgorithm[key] = backwardDataAlgo;

            return backwardDataAlgo;
        }

        private static bool IsDeterminist(ConvolutionAlgoPreference algoPreference)
        {
            return algoPreference == ConvolutionAlgoPreference.FASTEST_DETERMINIST || algoPreference == ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM;
        }

        public IntPtr FilterDesc(cudnnDataType_t cudaType, int[] shape, bool isDepthwiseConvolution)
        {
            CheckThreadId();

            int inputChannels; //Number of input channels
            int outputChannels; //number of output channels
            if (isDepthwiseConvolution)
            {
                //the depthwise Convolution shape: (depthMultiplier=1, channels, F, F)
                inputChannels = 1;
                outputChannels = shape[1];
            }
            else
            {
                //the Convolution shape: (outputChannels, inputChannels, f1,f2)
                inputChannels = shape[1];
                outputChannels = shape[0];
            }
            var h = shape[2]; //Height of each filter
            var w = shape[3]; //Width of each filter

            var key = Tuple.Create(cudaType, outputChannels, inputChannels, h, w);
            if (!cacheFilterDesc.TryGetValue(key, out var desc))
            {
                var res = CudnnWrapper.cudnnCreateFilterDescriptor(out desc);
                CheckStatus(res);
                res = CudnnWrapper.cudnnSetFilter4dDescriptor(desc, cudaType, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, outputChannels, inputChannels, h, w);
                CheckStatus(res);
                cacheFilterDesc[key] = desc;
            }
            return desc;
        }
        public IntPtr ConvDesc(cudnnDataType_t cudaType, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int stride, int groupCount)
        {
            CheckThreadId();

            if ((paddingTop != paddingBottom) || (paddingLeft != paddingRight))
            {
                throw new NotImplementedException("only symmetric padding is supported (padding=[" + paddingTop + "," + paddingBottom + "," + paddingLeft + "," + paddingRight + "])");
            }

            var key = Tuple.Create(cudaType, paddingTop, paddingBottom, paddingLeft, paddingRight, stride, groupCount);
            if (!cacheConvolutionDesc.TryGetValue(key, out var desc))
            {
                var res = CudnnWrapper.cudnnCreateConvolutionDescriptor(out desc);
                CheckStatus(res);
                if (groupCount != 1)
                {
                    res = CudnnWrapper.cudnnSetConvolutionGroupCount(desc, groupCount);
                    CheckStatus(res);
                }
                res = CudnnWrapper.cudnnSetConvolution2dDescriptor(desc, paddingTop, paddingLeft, stride, stride, 1, 1, cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION, cudaType);
                CheckStatus(res);
                cacheConvolutionDesc[key] = desc;
            }
            return desc;
        }


        #endregion

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

        public void Reset()
        {
            CheckThreadId();
            _copyToDeviceCalls = 0;
            _bytesCopiedToDevice = 0;
            _copyToHostCalls = 0;
            _bytesCopiedToHost = 0;
            SwCopyToDevice.Reset();
            SwCopyToHost.Reset();
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
            cacheConvolutionForwardAlgorithm.Clear();
            cacheConvolutionBackwardFilterAlgorithm.Clear();
            cacheFindConvolutionBackwardDataAlgorithm.Clear();
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
            var result = _deviceName + " - driver " + _driverVersion;
            var cudaVersion = CudaVersionFromCudaPath();
            result += string.IsNullOrEmpty(cudaVersion) ? " - no CUDA found" : (" - CUDA " + cudaVersion);
            result += " - cublas " + _cublasVersion + " - deviceId:" + _deviceId + " - threadId:" + _threadId;
            return result;
        }
        public override string ToString()
        {
            return DeviceName() + " - " + MemoryInfo();
        }
        public string MemoryInfo()
        {
            CheckThreadId();
            var result = "Free GPU Memory: " + Utils.MemoryBytesToString(FreeMemoryInBytes()) + "/" + Utils.MemoryBytesToString(TotalMemoryInBytes());
            if (_copyToDeviceCalls!= 0)
            { 
                result += " - " + Utils.MemoryBytesToString(_bytesCopiedToDevice) + " CopiedToDevice (" + _copyToDeviceCalls + "calls, " + SwCopyToDevice.ElapsedMilliseconds + "ms)";
            }
            if (_copyToHostCalls != 0)
            { 
                result += " - " + Utils.MemoryBytesToString(_bytesCopiedToHost) + " CopiedToHost (" + _copyToHostCalls + "calls, " + SwCopyToHost.ElapsedMilliseconds + "ms)";
            }
            return result;
        }

        public size_t AvailableGpuMemoryInBytes()
        {
            CheckThreadId();
            return FreeMemoryInBytes();
        }
        public IntPtr CudaBlasHandle => _cudaBlasHandle;
        public StreamWrapper DefaultStream { get; }
        public static void CheckStatus(cublasStatus_t _status, Func<string> getComment)
        {
            if (_status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            {
                throw new Exception(_status + " " + getComment());
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
            if (cudaPath.ToLowerInvariant().Contains("v10.2"))
            {
                return CUDA_Versions.CUDA_10_2;
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
        [SuppressMessage("ReSharper", "UnusedVariable")]
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
            var cublasRes = CublasWrapper.cublasDestroy_v2(_cudaBlasHandle);
            //CheckStatus(cublasRes);
            _cudaBlasHandle = IntPtr.Zero;
            var cudnnRes = CudnnWrapper.cudnnDestroy(_cudnnHandle);
            //CheckStatus(cudnnRes);
            _cudnnHandle = IntPtr.Zero;
            var cuRes = NVCudaWrapper.cuCtxDestroy_v2(_contextHandle);
            //CheckStatus(cuRes);
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
            _deviceId = deviceId;
            _threadId = System.Threading.Thread.CurrentThread.ManagedThreadId;
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
        }
        private static Version NewVersion(int driverVersion)
        {
            var major = driverVersion / 1000;
            var minor = driverVersion % 100;
            var build = (driverVersion % 1000)/100;
            return (build==0)?new Version(major, minor): new Version(major, minor, build);
        }
        private static void CheckStatus(cublasStatus_t _status)
        {
            if (_status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            {
                throw new Exception(_status.ToString());
            }

        }
        private static void CheckStatus(cudnnStatus_t _status)
        {
            if (_status != cudnnStatus_t.CUDNN_STATUS_SUCCESS)
            {
                throw new Exception(_status.ToString());
            }
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
        private static ulong FreeMemoryInBytes()
        {
            CuMemGetInfoV2(out size_t freeMemoryInBytes, out size_t _);
            return freeMemoryInBytes;
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
