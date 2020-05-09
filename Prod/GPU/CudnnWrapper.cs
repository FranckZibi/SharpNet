using System;
using System.Runtime.InteropServices;
using SharpNet.Data;
// ReSharper disable UnusedMember.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.GPU
{
    public enum cudnnStatus_t
    {
        //The operation completed successfully.
        CUDNN_STATUS_SUCCESS = 0,

        //The cuDNN library was not initialized properly. This error is usually returned when a call to cudnnCreate() fails or
        //when cudnnCreate() has not been called prior to calling another cuDNN routine.
        //In the former case, it is usually due to an error in the CUDA Runtime API called by cudnnCreate() or by an error in the hardware setup.
        CUDNN_STATUS_NOT_INITIALIZED = 1,

        //Resource allocation failed inside the cuDNN library. This is usually caused by an internal cudaMalloc() failure.
        CUDNN_STATUS_ALLOC_FAILED = 2,

        //An incorrect value or parameter was passed to the function.
        CUDNN_STATUS_BAD_PARAM = 3,

        //An internal cuDNN operation failed.
        CUDNN_STATUS_INTERNAL_ERROR = 4,

        CUDNN_STATUS_INVALID_VALUE = 5,

        //The function requires a feature absent from the current GPU device. Note that cuDNN only supports devices with compute capabilities
        //greater than or equal to 3.0.
        CUDNN_STATUS_ARCH_MISMATCH = 6,

        //An access to GPU memory space failed, which is usually caused by a failure to bind a texture.
        //To correct: prior to the function call, unbind any previously bound textures.
        //Otherwise, this may indicate an internal error/b u g in the library.
        CUDNN_STATUS_MAPPING_ERROR = 7,

        //The GPU program failed to execute. This is usually caused by a failure to launch some cuDNN kernel on the GPU, which can occur for multiple reasons.
        //To correct: check that the hardware, an appropriate version of the driver, and the cuDNN library are correctly installed.
        //Otherwise, this may indicate a internal error/b u g in the library.
        CUDNN_STATUS_EXECUTION_FAILED = 8,

        //The functionality requested is not presently supported by cuDNN.
        CUDNN_STATUS_NOT_SUPPORTED = 9,

        //The functionality requested requires some license and an error was detected when trying to check the current licensing.
        //This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly.
        CUDNN_STATUS_LICENSE_ERROR = 10,

        //Runtime library required by RNN calls (libcuda.so or nvcuda.dll) cannot be found in predefined search paths.
        CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,

        //Some tasks in the user stream are not completed.
        CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,

        //Numerical overflow occurred during the GPU kernel execution.
        CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13
    }
    public enum cudnnDataType_t
    {
        CUDNN_DATA_FLOAT,
        CUDNN_DATA_DOUBLE,
        CUDNN_DATA_HALF,
        CUDNN_DATA_INT8,
        CUDNN_DATA_UINT8,
        CUDNN_DATA_INT32,
        CUDNN_DATA_INT8x4,
        CUDNN_DATA_INT8x32
    }
    public enum cudnnNanPropagation_t
    {
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_PROPAGATE_NAN
    }
    public enum cudnnTensorFormat_t
    {
        CUDNN_TENSOR_NCHW,
        CUDNN_TENSOR_NHWC,
        CUDNN_TENSOR_NCHW_VECT_C
    }

    #region convolution
    public enum cudnnConvolutionMode_t
    {
        CUDNN_CONVOLUTION,
        CUDNN_CROSS_CORRELATION
    }
    public enum cudnnMathType_t
    {
        //Tensor Core Operations are not used
        CUDNN_DEFAULT_MATH,
        //The use of Tensor Core Operations is permitted
        CUDNN_TENSOR_OP_MATH
    }

    public enum cudnnDeterminism_t
    {
        //Results are not guaranteed to be reproducible
        CUDNN_NON_DETERMINISTIC,
        //Results are guaranteed to be reproducible
        CUDNN_DETERMINISTIC
    }
    #region convoluton forward
    public enum cudnnConvolutionFwdPreference_t
    {
        CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        CUDNN_CONVOLUTION_FWD_SPECIFY_​WORKSPACE_LIMIT
    }
    public enum cudnnConvolutionFwdAlgo_t
    {
        //This algorithm expresses the convolution as a matrix product without actually explicitly
        //form the matrix that holds the input tensor data.
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,

        //This algorithm expresses the convolution as a matrix product without actually explicitly
        //form the matrix that holds the input tensor data, but still needs some memory workspace
        //to precompute some indices in order to facilitate the implicit construction of the matrix that holds the input tensor data
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_​PRECOMP_GEMM,

        //This algorithm expresses the convolution as an explicit matrix product.
        //A significant memory workspace is needed to store the matrix that holds the input tensor data.
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,

        //This algorithm expresses the convolution as a direct convolution
        //(e.g without implicitly or explicitly doing a matrix multiplication).
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,

        //This algorithm uses the Fast-Fourier Transform approach to compute the convolution.
        //A significant memory workspace is needed to store intermediate results.
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,

        //This algorithm uses the Fast-Fourier Transform approach but splits the inputs into tiles.
        //A significant memory workspace is needed to store intermediate results but less than CUDNN_CONVOLUTION_FWD_ALGO_FFT for large size images.
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,

        //This algorithm uses the Winograd Transform approach to compute the convolution.
        //A reasonably sized workspace is needed to store intermediate results.
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,

        //This algorithm uses the Winograd Transform approach to compute the convolution.
        //Significant workspace may be needed to store intermediate results.
        CUDNN_CONVOLUTION_FWD_ALGO_​WINOGRAD_NONFUSED
    }
    /// <summary>
    /// cudnnConvolutionFwdAlgoPerf_t is a structure containing performance results returned by cudnnFindConvolutionForwardAlgorithm()
    /// or heuristic results returned by cudnnGetConvolutionForwardAlgorithm_v7()
    /// see: https://docs.nvidia.com/deeplearning/sdk/cudnn-archived/cudnn_701/cudnn-user-guide/index.html#cudnnConvolutionFwdAlgoPerf_t
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnConvolutionFwdAlgoPerf_t
    {
        //The algorithm run to obtain the associated performance metrics
        public readonly cudnnConvolutionFwdAlgo_t algo;
        public readonly cudnnStatus_t status;
        //The execution time of cudnnConvolutionForward() (in milliseconds)
        public readonly float time;
        //The workspace size (in bytes)
        public readonly size_t memory;
        //The determinism of the algorithm
        public readonly cudnnDeterminism_t determinism;
        //The math type provided to the algorithm
        public readonly cudnnMathType_t mathType;
        //Reserved space for future properties
        public readonly int reserved1;
        //Reserved space for future properties
        public readonly int reserved2;
        //Reserved space for future properties
        public readonly int reserved3;
    }
    #endregion

    #region convoluton backward
    public enum cudnnConvolutionBwdFilterPreference_t
    {
        //In this configuration, the routine cudnnGetConvolutionBackwardFilterAlgorithm() is guaranteed to return an algorithm
        //that does not require any extra workspace to be provided by the user.
        CUDNN_CONVOLUTION_BWD_FILTER_​NO_WORKSPACE,

        //In this configuration, the routine cudnnGetConvolutionBackwardFilterAlgorithm() will return the fastest algorithm
        //regardless how much workspace is needed to execute it.
        CUDNN_CONVOLUTION_BWD_FILTER_​PREFER_FASTEST,

        //In this configuration, the routine cudnnGetConvolutionBackwardFilterAlgorithm() will return the fastest algorithm
        //that fits within the memory limit that the user provided.
        CUDNN_CONVOLUTION_BWD_FILTER_​SPECIFY_WORKSPACE_LIMIT
    }

    public enum cudnnConvolutionBwdFilterAlgo_t
    {
        //This algorithm expresses the convolution as a sum of matrix product without actually explicitly
        //form the matrix that holds the input tensor data. The sum is done using atomic adds operation,
        //thus the results are non-deterministic.
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,

        //This algorithm expresses the convolution as a matrix product without actually explicitly
        //form the matrix that holds the input tensor data. The results are deterministic.
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,

        //This algorithm uses the Fast-Fourier Transform approach to compute the convolution.
        //Significant workspace is needed to store intermediate results. The results are deterministic.
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,

        //This algorithm is similar to CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 but uses some small workspace
        //to precompute some indices. The results are also non-deterministic.
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,

        //This algorithm uses the Winograd Transform approach to compute the convolution.
        //Significant workspace may be needed to store intermediate results.
        //The results are deterministic.
        CUDNN_CONVOLUTION_BWD_FILTER_​WINOGRAD_NONFUSED,

        //This algorithm uses the Fast-Fourier Transform approach to compute the convolution but splits
        //the input tensor into tiles. Significant workspace may be needed to store intermediate results.
        //The results are deterministic.
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_​FFT_TILING,
    }

    /// <summary>
    /// cudnnConvolutionBwdFilterAlgoPerf_t is a structure containing performance results returned by cudnnFindConvolutionBackwardFilterAlgorithm()
    /// or heuristic results returned by cudnnGetConvolutionBackwardFilterAlgorithm_v7()
    /// see: https://docs.nvidia.com/deeplearning/sdk/cudnn-archived/cudnn_701/cudnn-user-guide/index.html#cudnnConvolutionBwdFilterAlgoPerf_t
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnConvolutionBwdFilterAlgoPerf_t
    {
        //The algorithm run to obtain the associated performance metrics
        public readonly cudnnConvolutionBwdFilterAlgo_t algo;
        public readonly cudnnStatus_t status;
        //The execution time of cudnnConvolutionBackwardFilter() (in milliseconds)
        public readonly float time;
        //The workspace size (in bytes)
        public readonly size_t memory;
        //The determinism of the algorithm
        public readonly cudnnDeterminism_t determinism;
        //The math type provided to the algorithm 
        public readonly cudnnMathType_t mathType;
        //Reserved space for future properties
        public readonly int reserved1;
        //Reserved space for future properties
        public readonly int reserved2;
        //Reserved space for future properties
        public readonly int reserved3;
    }

    public enum cudnnConvolutionBwdDataPreference_t
    {
        //In this configuration, the routine cudnnGetConvolutionBackwardDataAlgorithm() is guaranteed to return an algorithm
        //that does not require any extra workspace to be provided by the user.
        CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,

        //In this configuration, the routine cudnnGetConvolutionBackwardDataAlgorithm() will return the fastest algorithm
        //regardless how much workspace is needed to execute it.
        CUDNN_CONVOLUTION_BWD_DATA_​PREFER_FASTEST,

        //In this configuration, the routine cudnnGetConvolutionBackwardDataAlgorithm() will return the fastest algorithm
        //that fits within the memory limit that the user provided.
        CUDNN_CONVOLUTION_BWD_DATA_​SPECIFY_WORKSPACE_LIMIT
    }
    public enum cudnnConvolutionBwdDataAlgo_t
    {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_​FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_​WINOGRAD_NONFUSED
    }
    /// <summary>
    /// cudnnConvolutionBwdDataAlgoPerf_t is a structure containing performance results returned by cudnnFindConvolutionBackwardDataAlgorithm()
    /// or heuristic results returned by cudnnGetConvolutionBackwardDataAlgorithm_v7()
    /// see: https://docs.nvidia.com/deeplearning/sdk/cudnn-archived/cudnn_701/cudnn-user-guide/index.html#cudnnConvolutionBwdDataAlgoPerf_t
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnConvolutionBwdDataAlgoPerf_t
    {
        //The algorithm run to obtain the associated performance metric
        public readonly cudnnConvolutionBwdDataAlgo_t algo;
        public readonly cudnnStatus_t status;
        //The execution time of cudnnConvolutionBackwardData() (in milliseconds)
        public readonly float time;
        //The workspace size (in bytes)
        public readonly size_t memory;
        //The determinism of the algorithm
        public readonly cudnnDeterminism_t determinism;
        //The math type provided to the algorithm
        public readonly cudnnMathType_t mathType;
        //Reserved space for future properties
        public readonly int reserved1;
        //Reserved space for future properties
        public readonly int reserved2;
        //Reserved space for future properties
        public readonly int reserved3;
    }
    #endregion

    #endregion


    public enum cudnnActivationMode_t
    {
        CUDNN_ACTIVATION_SIGMOID,       //Selects the sigmoid function.
        CUDNN_ACTIVATION_RELU,          //Selects the rectified linear function.
        CUDNN_ACTIVATION_TANH,          //Selects the hyperbolic tangent function.
        CUDNN_ACTIVATION_CLIPPED_RELU,  //Selects the clipped rectified linear function
        CUDNN_ACTIVATION_ELU,           //Selects the exponential linear function
        CUDNN_ACTIVATION_IDENTITY,
        CUDNN_ACTIVATION_SOFTMAX=1000,
        CUDNN_ACTIVATION_SWISH= 1001,     //Selects the swish function ( f(x) = x*sigmoid(x) , see https://arxiv.org/abs/1710.05941)
        CUDNN_ACTIVATION_LEAKY_RELU = 1002     //Leaky Relu (requires an additional alpha coefficient)
    }
    public enum cudnnBatchNormMode_t
    {
        CUDNN_BATCHNORM_PER_ACTIVATION,
        CUDNN_BATCHNORM_SPATIAL,
        CUDNN_BATCHNORM_SPATIAL_PERSISTENT
    }
    public enum cudnnPoolingMode_t
    {
        CUDNN_POOLING_MAX,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_POOLING_MAX_DETERMINISTIC
    }

    public enum cudnnSoftmaxAlgorithm_t
    {
        CUDNN_SOFTMAX_FAST,
        CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_LOG
    }
    public enum cudnnSoftmaxMode_t
    {
        CUDNN_SOFTMAX_MODE_INSTANCE,
        CUDNN_SOFTMAX_MODE_CHANNEL
    }

    public static unsafe class CudnnWrapper
    {
        private const string CUDNN64_7 = "cudnn64_7.dll";

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
            IntPtr cudnnHandle,
            IntPtr xDesc,
            IntPtr wDesc,
            IntPtr convDesc,
            IntPtr yDesc,
            cudnnConvolutionFwdAlgo_t algo,
            out size_t sizeInBytes);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
            IntPtr cudnnHandle,
            IntPtr xDesc,
            IntPtr wDesc,
            IntPtr convDesc,
            IntPtr yDesc,
            cudnnConvolutionFwdPreference_t preference,
            uint memoryLimitInBytes,
            out cudnnConvolutionFwdAlgo_t algo);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnConvolutionForward(
            IntPtr cudnnHandle,
            void* alpha,
            IntPtr xDesc,
            IntPtr x,
            IntPtr wDesc,
            IntPtr w,
            IntPtr convDesc,
            cudnnConvolutionFwdAlgo_t algo,
            IntPtr workSpace,
            size_t workSpaceSizeInBytes,
            void* beta,
            IntPtr yDesc,
            IntPtr y);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnConvolutionBackwardBias(
            IntPtr cudnnHandle,
            void* alpha,
            IntPtr dyDesc,
            IntPtr dy,
            void* beta,
            IntPtr dbDesc,
            IntPtr db);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
            IntPtr cudnnHandle,
            IntPtr xDesc,
            IntPtr dyDesc,
            IntPtr convDesc,
            IntPtr dwDesc,
            cudnnConvolutionBwdFilterPreference_t preference,
            size_t memoryLimitInBytes,
            out cudnnConvolutionBwdFilterAlgo_t algo);


        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
            //[in] Handle to a previously created cuDNN context
            IntPtr cudnnHandle,
            //[in] input tensor descriptor
            IntPtr xDesc,
            //[in] filter descriptor
            IntPtr wDesc,
            //[in] convolution descriptor
            IntPtr convDesc,
            //[in] output tensor descriptor
            IntPtr yDesc,
            //[in] The maximum number of elements to be stored in perfResults
            int requestedAlgoCount,
            //[out] The number of output elements stored in perfResults
            out int returnedAlgoCount,
            //[out] A user-allocated array to store performance metrics sorted ascending by compute time
            cudnnConvolutionFwdAlgoPerf_t* perfResults
            );

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
            //[in] Handle to a previously created cuDNN context
            IntPtr cudnnHandle,
            //[in] input tensor descriptor
            IntPtr xDesc,
            //[in] output gradient tensor descriptor
            IntPtr dyDesc,
            //[in] convolution descriptor
            IntPtr convDesc,
            //[in] filter descriptor
            IntPtr wDesc,
            //[in] The maximum number of elements to be stored in perfResults
            int requestedAlgoCount,
            //[out] The number of output elements stored in perfResults
            out int returnedAlgoCount,
            //[out] A user-allocated array to store performance metrics sorted ascending by compute time
            cudnnConvolutionBwdFilterAlgoPerf_t* perfResults
        );

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
            //[in] Handle to a previously created cuDNN context
            IntPtr cudnnHandle,
            //[in] filter descriptor
            IntPtr wDesc,
            //[in] output tensor descriptor
            IntPtr dyDesc,
            //[in] convolution descriptor
            IntPtr convDesc,
            //[in] input tensor descriptor
            IntPtr xDesc,
            //[in] The maximum number of elements to be stored in perfResults
            int requestedAlgoCount,
            //[out] The number of output elements stored in perfResults
            out int returnedAlgoCount,
            //[out] A user-allocated array to store performance metrics sorted ascending by compute time
            cudnnConvolutionBwdDataAlgoPerf_t* perfResults
        );

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnConvolutionBackwardFilter(
            IntPtr cudnnHandle,
            void* alpha,
            IntPtr xDesc,
            IntPtr x,
            IntPtr dyDesc,
            IntPtr dy,
            IntPtr convDesc,
            cudnnConvolutionBwdFilterAlgo_t algo,
            IntPtr workSpace,
            size_t workSpaceSizeInBytes,
            void* beta,
            IntPtr dwDesc,
            IntPtr dw);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
            IntPtr cudnnHandle,
            IntPtr xDesc,
            IntPtr dyDesc,
            IntPtr convDesc,
            IntPtr dwDesc,
            cudnnConvolutionBwdFilterAlgo_t algo,
            out size_t sizeInBytes);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(
            IntPtr cudnnHandle,
            IntPtr wDesc,
            IntPtr dyDesc,
            IntPtr convDesc,
            IntPtr dxDesc,
            cudnnConvolutionBwdDataPreference_t preference,
            size_t memoryLimitInBytes,
            out cudnnConvolutionBwdDataAlgo_t algo);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
            IntPtr cudnnHandle,
            IntPtr wDesc,
            IntPtr dyDesc,
            IntPtr convDesc,
            IntPtr dxDesc,
            cudnnConvolutionBwdDataAlgo_t algo,
            out size_t sizeInBytes);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnConvolutionBackwardData(
            IntPtr cudnnHandle,
            void* alpha,
            IntPtr wDesc,
            IntPtr w,
            IntPtr dyDesc,
            IntPtr dy,
            IntPtr convDesc,
            cudnnConvolutionBwdDataAlgo_t algo,
            IntPtr workSpace,
            size_t workSpaceSizeInBytes,
            void* beta,
            IntPtr dxDesc,
            IntPtr dx);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnDropoutForward(
            IntPtr cudnnHandle,
            IntPtr dropoutDesc,
            IntPtr xDesc,
            IntPtr x,
            IntPtr yDesc,
            IntPtr y,
            IntPtr reserveSpace,
            size_t reserveSpaceSizeInBytes);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnDropoutGetReserveSpaceSize(
            IntPtr xDesc,
            out size_t sizeInBytes);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnDropoutBackward(
            IntPtr cudnnHandle,
            IntPtr dropoutDesc,
            IntPtr dyDesc,
            IntPtr dy,
            IntPtr dxDesc,
            IntPtr dx,
            IntPtr reserveSpace,
            size_t reserveSpaceSizeInBytes);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnDropoutGetStatesSize(
            IntPtr cudnnHandle,
            out size_t sizeInBytes);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnPoolingBackward(
            IntPtr cudnnHandle,
            IntPtr poolingDesc,
            void* alpha,
            IntPtr yDesc,
            IntPtr y,
            IntPtr dyDesc,
            IntPtr dy,
            IntPtr xDesc,
            IntPtr x,
            void* beta,
            IntPtr dxDesc,
            IntPtr dx);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnPoolingForward(
            IntPtr cudnnHandle,
            IntPtr poolingDesc,
            void* alpha,
            IntPtr xDesc,
            IntPtr x,
            void* beta,
            IntPtr yDesc,
            IntPtr y);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnSetActivationDescriptor(
            IntPtr activationDesc,
            cudnnActivationMode_t mode,
            cudnnNanPropagation_t reluNanOpt,
            double coef);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnActivationForward(
            IntPtr cudnnHandle,
            IntPtr activationDesc,
            void* alpha,
            IntPtr xDesc,
            IntPtr x,
            void* beta,
            IntPtr yDesc,
            IntPtr y);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnActivationBackward(
            IntPtr cudnnHandle,
            IntPtr activationDesc,
            void* alpha,
            IntPtr yDesc,
            IntPtr y,
            IntPtr dyDesc,
            IntPtr dy,
            IntPtr xDesc,
            IntPtr x,
            void* beta,
            IntPtr dxDesc,
            IntPtr dx);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnSoftmaxForward(
            IntPtr cudnnHandle,
            cudnnSoftmaxAlgorithm_t algorithm,
            cudnnSoftmaxMode_t mode,
            void* alpha,
            IntPtr xDesc,
            IntPtr x,
            void* beta,
            IntPtr yDesc,
            IntPtr y);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnSoftmaxBackward(
            IntPtr cudnnHandle,
            cudnnSoftmaxAlgorithm_t algorithm,
            cudnnSoftmaxMode_t mode,
            void* alpha,
            IntPtr yDesc,
            IntPtr yData,
            IntPtr dyDesc,
            IntPtr dy,
            void* beta,
            IntPtr dxDesc,
            IntPtr dx);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnBatchNormalizationBackward(
            IntPtr cudnnHandle,
            cudnnBatchNormMode_t mode,
            void* alphaDataDiff,
            void* betaDataDiff,
            void* alphaParamDiff,
            void* betaParamDiff,
            IntPtr xDesc,
            IntPtr x,
            IntPtr dyDesc,
            IntPtr dy,
            IntPtr dxDesc,
            IntPtr dx,
            IntPtr bnScaleBiasDiffDesc,
            IntPtr bnScale,
            IntPtr resultBnScaleDiff,
            IntPtr resultBnBiasDiff,
            double epsilon,
            IntPtr savedMean,
            IntPtr savedInvVariance);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnBatchNormalizationForwardTraining(
            IntPtr cudnnHandle,
            cudnnBatchNormMode_t mode,
            void* alpha,
            void* beta,
            IntPtr xDesc,
            IntPtr x,
            IntPtr yDesc,
            IntPtr y,
            IntPtr bnScaleBiasMeanVarDesc,
            IntPtr bnScale,
            IntPtr bnBias,
            double exponentialAverageFactor,
            IntPtr resultRunningMean,
            IntPtr resultRunningVariance,
            double epsilon,
            IntPtr resultSaveMean,
            IntPtr resultSaveInvVariance);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnBatchNormalizationForwardInference(
            IntPtr cudnnHandle,
            cudnnBatchNormMode_t mode,
            void* alpha,
            void* beta,
            IntPtr xDesc,
            IntPtr x,
            IntPtr yDesc,
            IntPtr y,
            IntPtr bnScaleBiasMeanVarDesc,
            IntPtr bnScale,
            IntPtr bnBias,
            IntPtr estimatedMean,
            IntPtr estimatedVariance,
            double epsilon);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnScaleTensor(
            IntPtr cudnnHandle,
            IntPtr yDesc,
            IntPtr y,
            void* alpha);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnAddTensor(
            IntPtr cudnnHandle,
            void* alpha,
            IntPtr aDesc,
            IntPtr A,
            void* beta,
            IntPtr cDesc,
            IntPtr C);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnCreateTensorDescriptor(
            out IntPtr tensorDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnSetTensor4dDescriptor(
            IntPtr tensorDesc,
            cudnnTensorFormat_t format,
            cudnnDataType_t dataType,
            int n,
            int c,
            int h,
            int w);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnDestroyTensorDescriptor(
            IntPtr tensorDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnCreateActivationDescriptor(
            out IntPtr activationDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnDestroyActivationDescriptor(
            IntPtr activationDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnCreatePoolingDescriptor(
            out IntPtr poolingDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnSetPooling2dDescriptor(
            IntPtr poolingDesc,
            cudnnPoolingMode_t mode,
            cudnnNanPropagation_t maxPoolingNanOpt,
            int windowHeight,
            int windowWidth,
            int verticalPadding,
            int horizontalPadding,
            int verticalStride,
            int horizontalStride);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnDestroyPoolingDescriptor(
            IntPtr poolingDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnCreateConvolutionDescriptor(out IntPtr convDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnSetConvolutionGroupCount(IntPtr convDesc, int groupCount);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnSetConvolution2dDescriptor(
            IntPtr convDesc,
            int pad_h,
            int pad_w,
            int u,
            int v,
            int dilation_h,
            int dilation_w,
            cudnnConvolutionMode_t mode,
            cudnnDataType_t computeType);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnDestroyConvolutionDescriptor(
            IntPtr convDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnCreateFilterDescriptor(
            out IntPtr filterDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnSetFilter4dDescriptor(
            IntPtr filterDesc,
            cudnnDataType_t dataType,
            cudnnTensorFormat_t format,
            int k,
            int c,
            int h,
            int w);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnDestroyFilterDescriptor(
            IntPtr filterDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnCreateDropoutDescriptor(
            out IntPtr dropoutDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnSetDropoutDescriptor(
            IntPtr dropoutDesc,
            IntPtr cudnnHandle,
            float dropout,
            IntPtr states,
            size_t stateSizeInBytes,
            ulong seed);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnDestroyDropoutDescriptor(
            IntPtr dropoutDesc);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnCreate(
            out IntPtr cudnnHandle);

        [DllImport(CUDNN64_7)]
        public static extern cudnnStatus_t cudnnDestroy(
            IntPtr cudnnHandle);
    }
}
