using SharpNet.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using SharpNet.Layers;
using SharpNet.HyperParameters;
using log4net;

namespace SharpNet.GPU
{
    public enum CUDA_Versions { CUDA_10_1, CUDA_10_2, CUDA_11_0, CUDA_11_4 };

    [DebuggerDisplay("{"+nameof(DeviceName)+"()}")]
    public unsafe class GPUWrapper : IDisposable
    {
        #region Private fields
        // ReSharper disable once PrivateFieldCanBeConvertedToLocalVariable
        public static readonly ILog Log = LogManager.GetLogger(typeof(GPUWrapper));
        private readonly IntPtr _deviceHandle;
        private readonly string _deviceName;
        private readonly Version _cublasVersion;
        private readonly Version _cuDNNVersion;
        private readonly KernelManager _kernelManager;
        private readonly IDictionary<Tuple<cudnnDataType_t, int, int, int, int>, cudnnTensorDescriptor_t> cacheTensorDesc = new Dictionary<Tuple<cudnnDataType_t, int, int, int, int>, cudnnTensorDescriptor_t>();
        private readonly IDictionary<Tuple<cudnnDataType_t, cudnnRNNDataLayout_t, int, int, int>, cudnnRNNDataDescriptor_t> cacheRNNDataDesc = new Dictionary<Tuple<cudnnDataType_t, cudnnRNNDataLayout_t, int, int, int>, cudnnRNNDataDescriptor_t>();
        private readonly IDictionary<Tuple<cudnnDataType_t, int, int, int, int>, cudnnFilterDescriptor_t> cacheFilterDesc = new Dictionary<Tuple<cudnnDataType_t, int, int, int, int>, cudnnFilterDescriptor_t>();
        private readonly IDictionary<Tuple<cudnnPoolingMode_t, int, int, int, int>, cudnnPoolingDescriptor_t> cachePoolingDesc = new Dictionary<Tuple<cudnnPoolingMode_t, int, int, int, int>, cudnnPoolingDescriptor_t>();
        private readonly IDictionary<RNNDescriptor, cudnnRNNDescriptor_t> cacheRNNDesc = new Dictionary<RNNDescriptor, cudnnRNNDescriptor_t>();
        private readonly IDictionary<double, cudnnDropoutDescriptor_t> cacheDropoutDesc = new Dictionary<double, cudnnDropoutDescriptor_t>();
        private readonly IDictionary<Tuple<cudnnDataType_t, int, int, int, int, int, int>, cudnnConvolutionDescriptor_t> cacheConvolutionDesc = new Dictionary<Tuple<cudnnDataType_t, int, int, int, int, int, int>, cudnnConvolutionDescriptor_t>();
        private readonly IDictionary<cudnnActivationMode_t, cudnnActivationDescriptor_t> cacheActivationDesc = new Dictionary<cudnnActivationMode_t, cudnnActivationDescriptor_t>();
        private readonly IDictionary<Tuple<cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, ConvolutionAlgoPreference>, cudnnConvolutionBwdFilterAlgo_t> cacheConvolutionBackwardFilterAlgorithm = new Dictionary<Tuple<cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, ConvolutionAlgoPreference>, cudnnConvolutionBwdFilterAlgo_t>();
        private readonly IDictionary<Tuple<cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, ConvolutionAlgoPreference>, cudnnConvolutionBwdDataAlgo_t> cacheFindConvolutionBackwardDataAlgorithm = new Dictionary<Tuple<cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, ConvolutionAlgoPreference>, cudnnConvolutionBwdDataAlgo_t>();
        private readonly IDictionary<Tuple<cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, ConvolutionAlgoPreference>, cudnnConvolutionFwdAlgo_t> cacheConvolutionForwardAlgorithm = new Dictionary<Tuple<cudnnTensorDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnFilterDescriptor_t, ConvolutionAlgoPreference>, cudnnConvolutionFwdAlgo_t>();
        private readonly IDictionary<CUdevice_attribute, int> properties = new Dictionary<CUdevice_attribute, int>();
        private readonly IDictionary<Tuple<int, int>, Tensor> _cacheDevSeqLengths = new Dictionary<Tuple<int, int>, Tensor>();
        private readonly List<Tensor> _randomNumberGeneratorStatesBuffers = new List<Tensor>();
        private IntPtr _cudaBlasHandle;
        private IntPtr _contextHandle;
        private cudnnHandle_t _cudnnHandle;
        private cusolverDnHandle_t _cusolverDnHandle;
        private int _copyHostToDeviceCalls;
        private ulong _bytesCopiedHostToDevice;
        private int _copyDeviceToSameDeviceCalls;
        private int _copyDeviceToOtherDeviceCalls;
        private ulong _bytesCopiedDeviceToSameDevice;
        private ulong _bytesCopiedDeviceToOtherDevice;
        private int _copyDeviceToHostCalls;
        private ulong _bytesCopiedDeviceToHost;
        private int _threadId;
        private static readonly IDictionary<int, GPUWrapper> Cache = new Dictionary<int, GPUWrapper>();
        #endregion

        #region readonly properties
        public int DeviceId { get; }
        public Version CudaVersion { get; }
        public StreamWrapper DefaultStream { get; }
        public int MaxThreadsPerBlock { get; }
        public int MultiProcessorCount { get; }
        public int WarpSize { get; }
        public Stopwatch SwCopyDeviceToSameDevice { get; } = new Stopwatch();
        public Stopwatch SwCopyDeviceToOtherDevice { get; } = new Stopwatch();
        public Stopwatch SwCopyHostToDevice { get; } = new Stopwatch();
        public Stopwatch SwCopyDeviceToHost { get; } = new Stopwatch();
        #endregion

        public CublasWrapper CublasWrapper { get; }
        public CudartWrapper CudartWrapper { get; }

        #region constructor
        private GPUWrapper(int deviceId)
        {
            // We use Deterministic mode for RNN
            // See: https://docs.nvidia.com/deeplearning/cudnn/release-notes/rel_8.html#rel-800-Preview__section_qhc_jc1_5kb
            Environment.SetEnvironmentVariable("CUBLAS_WORKSPACE_CONFIG", ":16:8");

            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("CUDA_PATH")))
            {
                throw new Exception("CUDA_PATH environment variable is missing");
            }

            //We retrieve the cuda version
            var versionFromPath = System.IO.Path.GetFileName(Environment.GetEnvironmentVariable("CUDA_PATH")).Trim('v').Split('.').Select(int.Parse).ToArray();
            int cudaDriverVersion = versionFromPath[0] * 1000 + versionFromPath[1] * 10;
            //var cuRes = NVCudaWrapper.cuDriverGetVersion(out int cudaDriverVersion);
            //CheckStatus(cuRes);

            CudaVersion = Utils.NewVersionXXYY0(cudaDriverVersion);

            CudartWrapper = new CudartWrapper(CudaVersion);
            CudartWrapper.cudaDeviceReset();
            DeviceId = deviceId;
            AssociateCurrentThreadWithDevice();
            CublasWrapper = new CublasWrapper(CudaVersion);
            var cublasRes = CublasWrapper.cublasCreate_v2(ref _cudaBlasHandle);
            CheckStatus(cublasRes);


            //We retrieve the cublas version
            cublasRes = CublasWrapper.cublasGetVersion_v2(CudaBlasHandle, out var cublasVersion);
            CheckStatus(cublasRes);
            _cublasVersion = Utils.NewVersion(cublasVersion);

            //We retrieve the cudnn version
            _cuDNNVersion = Utils.NewVersion((int)(ulong)CudnnWrapper.cudnnGetVersion());

            _deviceHandle = GetDeviceHandle(deviceId);

            //cuRes = NVCudaWrapper.cuCtxCreate_v2(out _contextHandle, 0, _deviceHandle);
            var cuRes = NVCudaWrapper.cuDevicePrimaryCtxRetain(out _contextHandle, _deviceHandle);
            CheckStatus(cuRes);

            var devName = new byte[256];
            cuRes = NVCudaWrapper.cuDeviceGetName(devName, devName.Length, _deviceHandle);
            CheckStatus(cuRes);
            System.Text.ASCIIEncoding enc = new System.Text.ASCIIEncoding();
            _deviceName = enc.GetString(devName).Replace("\0", "");

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

            var cusolverRes = CusolverWrapper.cusolverDnCreate(out _cusolverDnHandle);
            CheckStatus(cusolverRes);

            _kernelManager = new KernelManager(this);
        }
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
        public cudnnHandle_t CudnnHandle => _cudnnHandle;
        public cusolverDnHandle_t CusolverDnHandle => _cusolverDnHandle;


        public void RunKernel(string kernelName, int count, object[] parameterLists)
        {
            CheckThreadId();
            _kernelManager.RunKernel(kernelName, count, parameterLists);
        }
        public cudnnActivationDescriptor_t ActivationDesc(cudnnActivationMode_t activationFunctionType)
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
        public cudnnPoolingDescriptor_t PoolingDesc(cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride)
        {
            var key = Tuple.Create(poolingMode, poolingHeight, poolingWidth, verticalStride, horizontalStride);
            if (!cachePoolingDesc.TryGetValue(key, out var desc))
            {
                var res = CudnnWrapper.cudnnCreatePoolingDescriptor(out desc);
                CheckStatus(res);
                res = CudnnWrapper.cudnnSetPooling2dDescriptor(desc, poolingMode, cudnnNanPropagation_t.CUDNN_NOT_PROPAGATE_NAN, poolingHeight, poolingWidth, 0, 0, verticalStride, horizontalStride);
                CheckStatus(res);
                cachePoolingDesc[key] = desc;
            }
            return desc;
        }

        public cudnnRNNDescriptor_t RNNDesc(RNNDescriptor key)
        {
            if (!cacheRNNDesc.TryGetValue(key, out var rnnDesc))
            {
                var dropoutDesc = DropoutDesc(key.dropoutRate);
                var res = CudnnWrapper.cudnnCreateRNNDescriptor(out rnnDesc);
                CheckStatus(res);
                res = CudnnWrapper.cudnnSetRNNDescriptor_v8(rnnDesc, key.algo, key.cellMode, key.biasMode, key.dirMode, key.inputMode, key.dataType, key.mathPrec, key.mathType, key.inputSize, key.hiddenSize, key.projSize, key.numLayers, dropoutDesc, key.auxFlags);
                CheckStatus(res);
                cacheRNNDesc[key] = rnnDesc;
            }
            return rnnDesc;
        }

        private static List<T> ExtractFromStackalloc<T>(T* stackAllocated, int length) where T : unmanaged
        {
            var result = new List<T>();
            for (int i = 0; i < length; ++i)
            {
                result.Add(stackAllocated[i]);
            }
            return result;
        }

        private static void FillWithSameValue<T>(T* stackAllocated, int length, T newValue) where T : unmanaged
        {
            for (int i = 0; i < length; ++i)
            {
                stackAllocated[i] = newValue;
            }
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
            FASTEST_DETERMINIST_NO_TRANSFORM
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
        public cudnnConvolutionFwdAlgo_t ConvolutionForwardAlgorithm(cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t filterDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc, ConvolutionAlgoPreference forwardAlgoPreference)
        {
            var key = Tuple.Create(xDesc, yDesc, convDesc, filterDesc, forwardAlgoPreference);
            cudnnStatus_t cudnnStatus;
            if (cacheConvolutionForwardAlgorithm.TryGetValue(key, out var forwardAlgo))
            {
                return forwardAlgo;
            }

            //we benchmark all available forward algorithms
            int requestedAlgoCount = Enum.GetNames(typeof(cudnnConvolutionFwdAlgo_t)).Length;
            var perfResultsStackalloc = stackalloc cudnnConvolutionFwdAlgoPerf_t[requestedAlgoCount];
            cudnnStatus = CudnnWrapper.cudnnFindConvolutionForwardAlgorithm(CudnnHandle, xDesc, filterDesc, convDesc, yDesc, requestedAlgoCount, out int returnedAlgoCount, perfResultsStackalloc);
            CheckStatus(cudnnStatus);
            //'perfResults' contains KPI for all available algos, starting from the fastest
            var perfResults = ExtractFromStackalloc(perfResultsStackalloc, returnedAlgoCount).Where(p => p.status == cudnnStatus_t.CUDNN_STATUS_SUCCESS).ToList();

            Debug.Assert(perfResults.Count != 0);

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
        public cudnnConvolutionBwdFilterAlgo_t ConvolutionBackwardFilterAlgorithm(cudnnTensorDescriptor_t xDesc, cudnnTensorDescriptor_t dyDesc, cudnnConvolutionDescriptor_t convDesc, cudnnFilterDescriptor_t filterDesc, ConvolutionAlgoPreference backwardAlgoPreference)
        {
            var key = Tuple.Create(xDesc, dyDesc, convDesc, filterDesc, backwardAlgoPreference);
            cudnnStatus_t cudnnStatus;
            if (cacheConvolutionBackwardFilterAlgorithm.TryGetValue(key, out var backwardFilterAlgo))
            {
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
        public cudnnConvolutionBwdDataAlgo_t ConvolutionBackwardDataAlgorithm(cudnnFilterDescriptor_t filterDesc, cudnnTensorDescriptor_t dyDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t xDesc, ConvolutionAlgoPreference backwardAlgoPreference)
        {
            var key = Tuple.Create(filterDesc, dyDesc, convDesc, xDesc, backwardAlgoPreference);
            cudnnStatus_t cudnnStatus;
            if (cacheFindConvolutionBackwardDataAlgorithm.TryGetValue(key, out var backwardDataAlgo))
            {
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

        public cudnnFilterDescriptor_t FilterDesc(cudnnDataType_t cudaType, int[] shape, bool isDepthwiseConvolution)
        {
            CheckThreadId();

            int inputChannels; //Number of input channels
            int outputChannels; //number of output channels
            if (isDepthwiseConvolution)
            {
                //the depthwise Convolution shape: (depthMultiplier=1, channels, kernelHeight, kernelWidth)
                inputChannels = 1;
                outputChannels = shape[1];
            }
            else
            {
                //the Convolution shape: (outputChannels, inputChannels, kernelHeight, kernelWidth)
                inputChannels = shape[1];
                outputChannels = shape[0];
            }
            var kernelHeight = shape[2]; //Height of each filter
            var kernelWidth = shape[3]; //Width of each filter

            var key = Tuple.Create(cudaType, outputChannels, inputChannels, kernelHeight, kernelWidth);
            if (!cacheFilterDesc.TryGetValue(key, out var desc))
            {
                var res = CudnnWrapper.cudnnCreateFilterDescriptor(out desc);
                CheckStatus(res);
                res = CudnnWrapper.cudnnSetFilter4dDescriptor(desc, cudaType, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, outputChannels, inputChannels, kernelHeight, kernelWidth);
                CheckStatus(res);
                cacheFilterDesc[key] = desc;
            }
            return desc;
        }
        public cudnnConvolutionDescriptor_t ConvDesc(cudnnDataType_t cudaType, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int stride, int groupCount)
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
        public cudnnDropoutDescriptor_t DropoutDesc(double dropoutRate)
        {
            CheckThreadId();
            if (!cacheDropoutDesc.TryGetValue(dropoutRate, out var desc))
            {
                var res = CudnnWrapper.cudnnCreateDropoutDescriptor(out desc);
                CheckStatus(res);

                res = CudnnWrapper.cudnnDropoutGetStatesSize(CudnnHandle, out var dropoutStateSize);
                CheckStatus(res);
                var randomNumberGeneratorStatesBuffer = new GPUTensor<float>(new []{ (int)(ulong)dropoutStateSize/4+1}, null, this);
                _randomNumberGeneratorStatesBuffers.Add(randomNumberGeneratorStatesBuffer);
                res = CudnnWrapper.cudnnSetDropoutDescriptor(desc, _cudnnHandle, (float) dropoutRate, randomNumberGeneratorStatesBuffer, dropoutStateSize, 0);
                CheckStatus(res);
                cacheDropoutDesc[dropoutRate] = desc;
            }
            return desc;
        }

        public static int[] GetTensorShape(cudnnTensorDescriptor_t tensorDesc)
        {
            var res = CudnnWrapper.cudnnGetTensor4dDescriptor(
                tensorDesc,
                out cudnnDataType_t _,
                out int n,
                out int c,
                out int h,
                out int w,
                out int _,
                out int _,
                out int _,
                out int _);
            CheckStatus(res);
            return new[] { n, c, h, w };
        }

        public cudnnTensorDescriptor_t TensorDesc(cudnnDataType_t dataType, int[] shape)
        {
            CheckThreadId();

            shape = shape.Select(i => Math.Max(i, 1)).ToArray();

            var n = shape[0];
            var c = shape.Length >= 2 ? shape[1] : 1;
            var h = shape.Length >= 3 ? shape[2] : 1;
            var w = shape.Length >= 4 ? shape[3] : 1;

            var key = Tuple.Create(dataType, n, c, h, w);
            if (!cacheTensorDesc.TryGetValue(key, out var desc))
            {
                var res = CudnnWrapper.cudnnCreateTensorDescriptor(out desc);
                CheckStatus(res);
                res = CudnnWrapper.cudnnSetTensor4dDescriptor(desc, cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, dataType, n, c, h, w);
                CheckStatus(res);
                cacheTensorDesc[key] = desc;
            }
            return desc;
        }

        /// <summary>
        /// return an int tensor of length 'batchSize' containing only the same int value: timeSteps
        /// </summary>
        /// <param name="batchSize">length of the vector</param>
        /// <param name="timeSteps">element to put on each element of the vector</param>
        /// <returns></returns>
        public Tensor GetDevSeqLengths(int batchSize, int timeSteps)
        {
            var key = Tuple.Create(batchSize, timeSteps);
            if (!_cacheDevSeqLengths.ContainsKey(key))
            {
                _cacheDevSeqLengths[key] = new GPUTensor<int>(new[] { batchSize }, Enumerable.Repeat(timeSteps, batchSize).ToArray(), this);
            }
            return _cacheDevSeqLengths[key];
        }

        public cudnnRNNDataDescriptor_t RNNDataDesc(cudnnDataType_t dataType, int maxSeqLength, int batchSize, int vectorSize)
        {
            CheckThreadId();
            const cudnnRNNDataLayout_t layout = cudnnRNNDataLayout_t.CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
            var key = Tuple.Create(dataType, layout, maxSeqLength, batchSize, vectorSize);
            if (!cacheRNNDataDesc.TryGetValue(key, out var desc))
            {
                var res = CudnnWrapper.cudnnCreateRNNDataDescriptor(out desc);
                CheckStatus(res);
                int* seqLengthArray = stackalloc int[batchSize];
                FillWithSameValue(seqLengthArray, batchSize, maxSeqLength);

                float paddingFill = 0.0f;
                res = CudnnWrapper.cudnnSetRNNDataDescriptor(desc, dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, &paddingFill);
                CheckStatus(res);
                cacheRNNDataDesc[key] = desc;
            }
            return desc;
        }

        public void Reset()
        {
            CheckThreadId();
            _copyHostToDeviceCalls = 0;
            _bytesCopiedHostToDevice = 0;
            _copyDeviceToHostCalls = 0;
            _bytesCopiedDeviceToHost = 0;
            _copyDeviceToSameDeviceCalls = 0;
            _copyDeviceToOtherDeviceCalls = 0;
            _bytesCopiedDeviceToSameDevice = 0;
            _bytesCopiedDeviceToOtherDevice = 0;
            SwCopyHostToDevice.Reset();
            SwCopyDeviceToHost.Reset();
            SwCopyDeviceToSameDevice.Reset();
            SwCopyDeviceToOtherDevice.Reset();
            //_nbChunksInDeviceMemory = 0;
            cacheTensorDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyTensorDescriptor(x)));
            cacheTensorDesc.Clear();
            cacheFilterDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyFilterDescriptor(x)));
            cacheFilterDesc.Clear();
            cachePoolingDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyPoolingDescriptor(x)));
            cachePoolingDesc.Clear();
            cacheConvolutionDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyConvolutionDescriptor(x)));
            cacheConvolutionDesc.Clear();
            cacheActivationDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyActivationDescriptor(x)));
            cacheActivationDesc.Clear();
            _cacheDevSeqLengths.Values.ToList().ForEach(x => x.Dispose());
            _cacheDevSeqLengths.Clear();
            cacheConvolutionForwardAlgorithm.Clear();
            cacheConvolutionBackwardFilterAlgorithm.Clear();
            cacheFindConvolutionBackwardDataAlgorithm.Clear();
            cacheDropoutDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyDropoutDescriptor(x)));
            cacheDropoutDesc.Clear();
            cacheRNNDesc.Values.ToList().ForEach(x => CheckStatus(CudnnWrapper.cudnnDestroyRNNDescriptor(x)));
            cacheRNNDesc.Clear();
            _randomNumberGeneratorStatesBuffers.ForEach(x => x.Dispose());
            _randomNumberGeneratorStatesBuffers.Clear();
        }
        public void LogCopyDeviceToSameDeviceCall(ulong byteCopied)
        {
            ++_copyDeviceToSameDeviceCalls;
            _bytesCopiedDeviceToSameDevice += byteCopied;
        }
        public void LogCopyDeviceToOtherDeviceCall(ulong byteCopied)
        {
            ++_copyDeviceToOtherDeviceCalls;
            _bytesCopiedDeviceToOtherDevice += byteCopied;
        }
        public void LogCopyHostToDeviceCall(ulong byteCopied)
        {
            ++_copyHostToDeviceCalls;
            _bytesCopiedHostToDevice += byteCopied;
        }
        public void LogCopyDeviceToHostCall(ulong byteCopied)
        {
            ++_copyDeviceToHostCalls;
            _bytesCopiedDeviceToHost += byteCopied;
        }
        public string DeviceName()
        {
            var result = _deviceName;
            result += " - cuda " + CudaVersion;
            result += " - cublas " + _cublasVersion + " - cudnn " + _cuDNNVersion + " - deviceId:" + DeviceId;
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
            if (_copyHostToDeviceCalls!= 0)
            { 
                result += " - " + Utils.MemoryBytesToString(_bytesCopiedHostToDevice) + " CopiedHostToDevice (" + _copyHostToDeviceCalls + "calls, " + SwCopyHostToDevice.ElapsedMilliseconds + "ms)";
            }
            if (_copyDeviceToHostCalls != 0)
            { 
                result += " - " + Utils.MemoryBytesToString(_bytesCopiedDeviceToHost) + " CopiedDeviceToHost (" + _copyDeviceToHostCalls + "calls, " + SwCopyDeviceToHost.ElapsedMilliseconds + "ms)";
            }
            if (_copyDeviceToSameDeviceCalls != 0)
            {
                result += " - " + Utils.MemoryBytesToString(_bytesCopiedDeviceToSameDevice) + " CopiedDeviceToSameDevice (" + _copyDeviceToSameDeviceCalls + "calls, " + SwCopyDeviceToSameDevice.ElapsedMilliseconds + "ms)";
            }
            if (_copyDeviceToOtherDeviceCalls != 0)
            {
                result += " - " + Utils.MemoryBytesToString(_bytesCopiedDeviceToOtherDevice) + " CopiedDeviceToOtherDevice (" + _copyDeviceToOtherDeviceCalls + "calls, " + SwCopyDeviceToOtherDevice.ElapsedMilliseconds + "ms)";
            }
            return result;
        }

        // ReSharper disable once UnusedMember.Global
        public size_t AvailableGpuMemoryInBytes()
        {
            CheckThreadId();
            return FreeMemoryInBytes();
        }
        public IntPtr CudaBlasHandle => _cudaBlasHandle;

        public static void CheckStatus(cudnnStatus_t status)
        {
            if (status != cudnnStatus_t.CUDNN_STATUS_SUCCESS)
            {
                throw new Exception(status.ToString());
            }
        }
        public static void CheckStatus(cublasStatus_t status)
        {
            if (status != cublasStatus_t.CUBLAS_STATUS_SUCCESS)
            {
                throw new Exception(status.ToString());
            }
        }
        public static void CheckStatus(cusolverStatus_t status)
        {
            if (status != cusolverStatus_t.CUSOLVER_STATUS_SUCCESS)
            {
                throw new Exception(status.ToString());
            }
        }
        public static void CheckStatus(CUresult status)
        {
            if (status != CUresult.CUDA_SUCCESS)
            {
                throw new Exception(status.ToString());
            }
        }
        public static void CheckStatus(nvrtcResult status)
        {
            if (status != nvrtcResult.NVRTC_SUCCESS)
            {
                throw new Exception(status.ToString());
            }
        }
        public static void CheckStatus(cudaError_t status)
        {
            if (status != cudaError_t.cudaSuccess)
            {
                throw new Exception(status.ToString());
            }
        }

        public static CUDA_Versions ToCUDA_Versions_enum(Version cudaVersion)
        {
            if (cudaVersion.Major == 10)
            {
                if (cudaVersion.Minor == 1) {return CUDA_Versions.CUDA_10_1;}
                if (cudaVersion.Minor == 2) {return CUDA_Versions.CUDA_10_2;}
            }
            else if (cudaVersion.Major == 11)
            {
                if (cudaVersion.Minor == 0) { return CUDA_Versions.CUDA_11_0; }
                if (cudaVersion.Minor == 4) { return CUDA_Versions.CUDA_11_4; }
            }
            throw new Exception("cuda " + cudaVersion + " is not supported");
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
            _cudnnHandle = new cudnnHandle_t();
            var cusolverRes = CusolverWrapper.cusolverDnDestroy(_cusolverDnHandle);
            //CheckStatus(cusolverRes);
            _cusolverDnHandle = new cusolverDnHandle_t();
            //var cuRes = NVCudaWrapper.cuCtxDestroy_v2(_contextHandle);
            var cuRes = NVCudaWrapper.cuDevicePrimaryCtxRelease(_contextHandle);
            //CheckStatus(cuRes);
            _contextHandle = IntPtr.Zero;
        }
        ~GPUWrapper()
        {
            Dispose(false);
        }
        #endregion

        /// <summary>
        /// associate the current running thread with the 'this' Device
        /// </summary>
        public void AssociateCurrentThreadWithDevice()
        {
            var res = CudartWrapper.cudaSetDevice(DeviceId);
            CheckStatus(res);
            _threadId = System.Threading.Thread.CurrentThread.ManagedThreadId;
            //Log.Debug($"{nameof(GPUWrapper)}#{DeviceId} is associated with ManagedThreadId {_threadId}");
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
                var errorMsg = $"invalid Thread Id: expecting {_threadId} but got {System.Threading.Thread.CurrentThread.ManagedThreadId}";
                ISample.Log.Error(errorMsg);
                throw new Exception(errorMsg);
            }
        }
    }
}
