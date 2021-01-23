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
        CUDNN_ACTIVATION_SOFTMAX = 1000,
        CUDNN_ACTIVATION_SWISH = 1001,     //Selects the swish function ( f(x) = x*sigmoid(x) , see https://arxiv.org/abs/1710.05941)
        CUDNN_ACTIVATION_LEAKY_RELU = 1002,     //Leaky Relu (requires an additional alpha coefficient)
        CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY = 1003     //Softmax with hierarchical categories
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


    /// <summary>
    /// Type of network used in the cudnnRNNForwardInference(), cudnnRNNForwardTraining(), cudnnRNNBackwardData() and cudnnRNNBackwardWeights() routines
    /// see https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_701/cudnn-user-guide/index.html#cudnnRNNMode_t
    /// </summary>
    public enum cudnnRNNMode_t
    {
        //A single-gate recurrent neural network with a ReLU activation function.
        CUDNN_RNN_RELU,

        //A single-gate recurrent neural network with a tanh activation function.
        CUDNN_RNN_TANH,

        //A four-gate Long Short-Term Memory network with no peephole connections.
        CUDNN_LSTM,

        //A three-gate network consisting of Gated Recurrent Units.
        CUDNN_GRU
    }

    /// <summary>
    /// specify the recurrence pattern in the cudnnRNNForwardInference(), cudnnRNNForwardTraining(), cudnnRNNBackwardData() and cudnnRNNBackwardWeights() routines
    /// see: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_701/cudnn-user-guide/index.html#cudnnDirectionMode_t
    /// </summary>
    public enum cudnnDirectionMode_t
    {
        //The network iterates recurrently from the first input to the last.    
        CUDNN_UNIDIRECTIONAL,

        //Each layer of the the network iterates recurrently from the first input to the last and separately from the last input to the first.
        //The outputs of the two are concatenated at each iteration giving the output of the layer.
        CUDNN_BIDIRECTIONAL
    }

    /// <summary>
    /// specify the behavior of the first layer in the cudnnRNNForwardInference(), cudnnRNNForwardTraining(), cudnnRNNBackwardData() and cudnnRNNBackwardWeights() routines
    /// see: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_701/cudnn-user-guide/index.html#cudnnRNNInputMode_t
    /// </summary>
    public enum cudnnRNNInputMode_t
    {
        //A biased matrix multiplication is performed at the input of the first recurrent layer.
        CUDNN_LINEAR_INPUT, 

        //No operation is performed at the input of the first recurrent layer.
        //If CUDNN_SKIP_INPUT is used the leading dimension of the input tensor must be equal to the hidden state size of the network.
        CUDNN_SKIP_INPUT
    }

    /// <summary>
    /// specify the algorithm used in the cudnnRNNForwardInference(), cudnnRNNForwardTraining(), cudnnRNNBackwardData() and cudnnRNNBackwardWeights() routines.
    /// see: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_701/cudnn-user-guide/index.html#cudnnRNNAlgo_t
    /// </summary>
    public enum cudnnRNNAlgo_t
    {
        //Each RNN layer is executed as a sequence of operations.
        //This algorithm is expected to have robust performance across a wide range of network parameters.
        CUDNN_RNN_ALGO_STANDARD,

        //The recurrent parts of the network are executed using a persistent kernel approach.
        //This method is expected to be fast when the first dimension of the input tensor is small(ie.a small minibatch).
        CUDNN_RNN_ALGO_PERSIST_STATIC,

        //The recurrent parts of the network are executed using a persistent kernel approach.
        //This method is expected to be fast when the first dimension of the input tensor is small (ie. a small minibatch)
        CUDNN_RNN_ALGO_PERSIST_DYNAMIC
    }


    /// <summary>
    /// specify inference or training mode in RNN API.
    /// This parameter allows the cuDNN library to tune more precisely the size of the workspace buffer
    /// that could be different in inference and training regimens
    /// see: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnForwardMode_t
    /// </summary>
    public enum cudnnForwardMode_t
    {
        //Selects the inference mode.
        CUDNN_FWD_MODE_INFERENCE,

        //Selects the training mode.
        CUDNN_FWD_MODE_TRAINING
    }

    /// <summary>
    /// number of bias vectors for RNN functions.See the description of the cudnnRNNMode_t enumerated type for
    /// the equations for each cell type based on the bias mode.
    /// see: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNBiasMode_t
    /// </summary>
    public enum cudnnRNNBiasMode_t
    {
        //Applies RNN cell formulas that do not use biases.    
        CUDNN_RNN_NO_BIAS,

        //Applies RNN cell formulas that use one input bias vector in the input GEMM.
        CUDNN_RNN_SINGLE_INP_BIAS,

        //Applies RNN cell formulas that use two bias vectors.
        CUDNN_RNN_DOUBLE_BIAS,

        //Applies RNN cell formulas that use one recurrent bias vector in the recurrent GEMM.
        CUDNN_RNN_SINGLE_REC_BIAS
    }

    /// <summary>
    /// selects how buffers holding gradients of the loss function, computed with respect to trainable parameters,
    /// are updated.Currently, this type is used by the cudnnGetMultiHeadAttnWeights() function only.
    /// see: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnWgradMode_t
    /// </summary>
    public enum cudnnWgradMode_t
    {
        //A weight gradient component corresponding to a new batch of inputs is added to previously evaluated weight gradients.
        //Before using this mode, the buffer holding weight gradients should be initialized to zero.
        //Alternatively, the first API call outputting to an uninitialized buffer should use the CUDNN_WGRAD_MODE_SET option.
        CUDNN_WGRAD_MODE_ADD,

        //A weight gradient component, corresponding to a new batch of inputs,
        //overwrites previously stored weight gradients in the output buffer.
        CUDNN_WGRAD_MODE_SET
    }

    /// <summary>
    /// select the RNN data layout. It is used in the API calls cudnnGetRNNDataDescriptor() and cudnnSetRNNDataDescriptor().
    /// see: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNDataLayout_t
    /// </summary>
    public enum cudnnRNNDataLayout_t
    {
        //Data layout is padded, with outer stride from one time-step to the next.
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,

        //The sequence length is sorted and packed as in the basic RNN API.
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,

        //Data layout is padded, with outer stride from one batch to the next.
        CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnHandle_t
    {
        private readonly IntPtr _pointer;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnRNNDescriptor_t
    {
        private readonly IntPtr _pointer;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnDropoutDescriptor_t
    {
        private readonly IntPtr _pointer;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnTensorDescriptor_t
    {
        private readonly IntPtr _pointer;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnRNNDataDescriptor_t
    {
        private readonly IntPtr _pointer;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnFilterDescriptor_t
    {
        private readonly IntPtr _pointer;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnConvolutionDescriptor_t
    {
        private readonly IntPtr _pointer;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnActivationDescriptor_t
    {
        private readonly IntPtr _pointer;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct cudnnPoolingDescriptor_t
    {
        private readonly IntPtr _pointer;
    }


    public static unsafe class CudnnWrapper
    {
        private const string DLL_NAME = "cudnn64_8.dll";

        public static void EnableAPILogging(string fileName)
        {
            Environment.SetEnvironmentVariable("CUDNN_LOGINFO_DBG", "1");
            Environment.SetEnvironmentVariable("CUDNN_LOGDEST_DBG", fileName);
        }
        public static void DisableAPILogging()
        {
            Environment.SetEnvironmentVariable("CUDNN_LOGINFO_DBG", "0");
        }

        public const uint CUDNN_RNN_PADDED_IO_DISABLED = 0u;
        public const uint CUDNN_RNN_PADDED_IO_ENABLED = 1u;


        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
            cudnnHandle_t cudnnHandle,
            cudnnTensorDescriptor_t xDesc,
            cudnnFilterDescriptor_t wDesc,
            cudnnConvolutionDescriptor_t convDesc,
            cudnnTensorDescriptor_t yDesc,
            cudnnConvolutionFwdAlgo_t algo,
            out size_t sizeInBytes);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnConvolutionForward(
            cudnnHandle_t cudnnHandle,
            void* alpha,
            cudnnTensorDescriptor_t xDesc,
            IntPtr x,
            cudnnFilterDescriptor_t wDesc,
            IntPtr w,
            cudnnConvolutionDescriptor_t convDesc,
            cudnnConvolutionFwdAlgo_t algo,
            IntPtr workSpace,
            size_t workSpaceSizeInBytes,
            void* beta,
            cudnnTensorDescriptor_t yDesc,
            IntPtr y);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnConvolutionBackwardBias(
            cudnnHandle_t cudnnHandle,
            void* alpha,
            cudnnTensorDescriptor_t dyDesc,
            IntPtr dy,
            void* beta,
            cudnnTensorDescriptor_t dbDesc,
            IntPtr db);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
            cudnnHandle_t cudnnHandle,
            cudnnTensorDescriptor_t xDesc,
            cudnnFilterDescriptor_t wDesc,
            cudnnConvolutionDescriptor_t convDesc,
            cudnnTensorDescriptor_t yDesc,
            int requestedAlgoCount,
            out int returnedAlgoCount,
            cudnnConvolutionFwdAlgoPerf_t* perfResults
            );

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
            cudnnHandle_t cudnnHandle,
            cudnnTensorDescriptor_t xDesc,
            cudnnTensorDescriptor_t dyDesc,
            cudnnConvolutionDescriptor_t convDesc,
            cudnnFilterDescriptor_t wDesc,
            int requestedAlgoCount,
            out int returnedAlgoCount,
            cudnnConvolutionBwdFilterAlgoPerf_t* perfResults
        );

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
            //[in] Handle to a previously created cuDNN context
            cudnnHandle_t cudnnHandle,
            //[in] filter descriptor
            cudnnFilterDescriptor_t wDesc,
            //[in] output tensor descriptor
            cudnnTensorDescriptor_t dyDesc,
            //[in] convolution descriptor
            cudnnConvolutionDescriptor_t convDesc,
            //[in] input tensor descriptor
            cudnnTensorDescriptor_t xDesc,
            //[in] The maximum number of elements to be stored in perfResults
            int requestedAlgoCount,
            //[out] The number of output elements stored in perfResults
            out int returnedAlgoCount,
            //[out] A user-allocated array to store performance metrics sorted ascending by compute time
            cudnnConvolutionBwdDataAlgoPerf_t* perfResults
        );

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnConvolutionBackwardFilter(
            cudnnHandle_t cudnnHandle,
            void* alpha,
            cudnnTensorDescriptor_t xDesc,
            IntPtr x,
            cudnnTensorDescriptor_t dyDesc,
            IntPtr dy,
            cudnnConvolutionDescriptor_t convDesc,
            cudnnConvolutionBwdFilterAlgo_t algo,
            IntPtr workSpace,
            size_t workSpaceSizeInBytes,
            void* beta,
            cudnnFilterDescriptor_t dwDesc,
            IntPtr dw);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
            cudnnHandle_t cudnnHandle,
            cudnnTensorDescriptor_t xDesc,
            cudnnTensorDescriptor_t dyDesc,
            cudnnConvolutionDescriptor_t convDesc,
            cudnnFilterDescriptor_t dwDesc,
            cudnnConvolutionBwdFilterAlgo_t algo,
            out size_t sizeInBytes);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
            cudnnHandle_t cudnnHandle,
            cudnnFilterDescriptor_t wDesc,
            cudnnTensorDescriptor_t dyDesc,
            cudnnConvolutionDescriptor_t convDesc,
            cudnnTensorDescriptor_t dxDesc,
            cudnnConvolutionBwdDataAlgo_t algo,
            out size_t sizeInBytes);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnConvolutionBackwardData(
            cudnnHandle_t cudnnHandle,
            void* alpha,
            cudnnFilterDescriptor_t wDesc,
            IntPtr w,
            cudnnTensorDescriptor_t dyDesc,
            IntPtr dy,
            cudnnConvolutionDescriptor_t convDesc,
            cudnnConvolutionBwdDataAlgo_t algo,
            IntPtr workSpace,
            size_t workSpaceSizeInBytes,
            void* beta,
            cudnnTensorDescriptor_t dxDesc,
            IntPtr dx);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDropoutForward(
            cudnnHandle_t cudnnHandle,
            cudnnDropoutDescriptor_t dropoutDesc,
            cudnnTensorDescriptor_t xDesc,
            IntPtr x,
            cudnnTensorDescriptor_t yDesc,
            IntPtr y,
            IntPtr reserveSpace,
            size_t reserveSpaceSizeInBytes);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDropoutGetReserveSpaceSize(
            cudnnTensorDescriptor_t xDesc,
            out size_t sizeInBytes);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDropoutBackward(
            cudnnHandle_t cudnnHandle,
            cudnnDropoutDescriptor_t dropoutDesc,
            cudnnTensorDescriptor_t dyDesc,
            IntPtr dy,
            cudnnTensorDescriptor_t dxDesc,
            IntPtr dx,
            IntPtr reserveSpace,
            size_t reserveSpaceSizeInBytes);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDropoutGetStatesSize(
            cudnnHandle_t cudnnHandle,
            out size_t sizeInBytes);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnPoolingBackward(
            cudnnHandle_t cudnnHandle,
            cudnnPoolingDescriptor_t poolingDesc,
            void* alpha,
            cudnnTensorDescriptor_t yDesc,
            IntPtr y,
            cudnnTensorDescriptor_t dyDesc,
            IntPtr dy,
            cudnnTensorDescriptor_t xDesc,
            IntPtr x,
            void* beta,
            cudnnTensorDescriptor_t dxDesc,
            IntPtr dx);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnPoolingForward(
            cudnnHandle_t cudnnHandle,
            cudnnPoolingDescriptor_t poolingDesc,
            void* alpha,
            cudnnTensorDescriptor_t xDesc,
            IntPtr x,
            void* beta,
            cudnnTensorDescriptor_t yDesc,
            IntPtr y);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnSetActivationDescriptor(
            cudnnActivationDescriptor_t activationDesc,
            cudnnActivationMode_t mode,
            cudnnNanPropagation_t reluNanOpt,
            double coef);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnActivationForward(
            cudnnHandle_t cudnnHandle,
            cudnnActivationDescriptor_t activationDesc,
            void* alpha,
            cudnnTensorDescriptor_t xDesc,
            IntPtr x,
            void* beta,
            cudnnTensorDescriptor_t yDesc,
            IntPtr y);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnActivationBackward(
            cudnnHandle_t cudnnHandle,
            cudnnActivationDescriptor_t activationDesc,
            void* alpha,
            cudnnTensorDescriptor_t yDesc,
            IntPtr y,
            cudnnTensorDescriptor_t dyDesc,
            IntPtr dy,
            cudnnTensorDescriptor_t xDesc,
            IntPtr x,
            void* beta,
            cudnnTensorDescriptor_t dxDesc,
            IntPtr dx);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnSoftmaxForward(
            cudnnHandle_t cudnnHandle,
            cudnnSoftmaxAlgorithm_t algorithm,
            cudnnSoftmaxMode_t mode,
            void* alpha,
            cudnnTensorDescriptor_t xDesc,
            IntPtr x,
            void* beta,
            cudnnTensorDescriptor_t yDesc,
            IntPtr y);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnSoftmaxBackward(
            cudnnHandle_t cudnnHandle,
            cudnnSoftmaxAlgorithm_t algorithm,
            cudnnSoftmaxMode_t mode,
            void* alpha,
            cudnnTensorDescriptor_t yDesc,
            IntPtr yData,
            cudnnTensorDescriptor_t dyDesc,
            IntPtr dy,
            void* beta,
            cudnnTensorDescriptor_t dxDesc,
            IntPtr dx);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnBatchNormalizationBackward(
            cudnnHandle_t cudnnHandle,
            cudnnBatchNormMode_t mode,
            void* alphaDataDiff,
            void* betaDataDiff,
            void* alphaParamDiff,
            void* betaParamDiff,
            cudnnTensorDescriptor_t xDesc,
            IntPtr x,
            cudnnTensorDescriptor_t dyDesc,
            IntPtr dy,
            cudnnTensorDescriptor_t dxDesc,
            IntPtr dx,
            cudnnTensorDescriptor_t bnScaleBiasDiffDesc,
            IntPtr bnScale,
            IntPtr resultBnScaleDiff,
            IntPtr resultBnBiasDiff,
            double epsilon,
            IntPtr savedMean,
            IntPtr savedInvVariance);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnBatchNormalizationForwardTraining(
            cudnnHandle_t cudnnHandle,
            cudnnBatchNormMode_t mode,
            void* alpha,
            void* beta,
            cudnnTensorDescriptor_t xDesc,
            IntPtr x,
            cudnnTensorDescriptor_t yDesc,
            IntPtr y,
            cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
            IntPtr bnScale,
            IntPtr bnBias,
            double exponentialAverageFactor,
            IntPtr resultRunningMean,
            IntPtr resultRunningVariance,
            double epsilon,
            IntPtr resultSaveMean,
            IntPtr resultSaveInvVariance);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnBatchNormalizationForwardInference(
            cudnnHandle_t cudnnHandle,
            cudnnBatchNormMode_t mode,
            void* alpha,
            void* beta,
            cudnnTensorDescriptor_t xDesc,
            IntPtr x,
            cudnnTensorDescriptor_t yDesc,
            IntPtr y,
            cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
            IntPtr bnScale,
            IntPtr bnBias,
            IntPtr estimatedMean,
            IntPtr estimatedVariance,
            double epsilon);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnScaleTensor(
            cudnnHandle_t cudnnHandle,
            cudnnTensorDescriptor_t yDesc,
            IntPtr y,
            void* alpha);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnAddTensor(
            cudnnHandle_t cudnnHandle,
            void* alpha,
            cudnnTensorDescriptor_t aDesc,
            IntPtr A,
            void* beta,
            cudnnTensorDescriptor_t cDesc,
            IntPtr C);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnCreateTensorDescriptor(
            out cudnnTensorDescriptor_t tensorDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnSetTensor4dDescriptor(
            cudnnTensorDescriptor_t tensorDesc,
            cudnnTensorFormat_t format,
            cudnnDataType_t dataType,
            int n,
            int c,
            int h,
            int w);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDestroyTensorDescriptor(
            cudnnTensorDescriptor_t tensorDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnCreateActivationDescriptor(
            out cudnnActivationDescriptor_t activationDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDestroyActivationDescriptor(
            cudnnActivationDescriptor_t activationDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnCreatePoolingDescriptor(
            out cudnnPoolingDescriptor_t poolingDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnSetPooling2dDescriptor(
            cudnnPoolingDescriptor_t poolingDesc,
            cudnnPoolingMode_t mode,
            cudnnNanPropagation_t maxPoolingNanOpt,
            int windowHeight,
            int windowWidth,
            int verticalPadding,
            int horizontalPadding,
            int verticalStride,
            int horizontalStride);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDestroyPoolingDescriptor(
            cudnnPoolingDescriptor_t poolingDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnCreateConvolutionDescriptor(out cudnnConvolutionDescriptor_t convDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnSetConvolution2dDescriptor(
            cudnnConvolutionDescriptor_t convDesc,
            int pad_h,
            int pad_w,
            int u,
            int v,
            int dilation_h,
            int dilation_w,
            cudnnConvolutionMode_t mode,
            cudnnDataType_t computeType);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDestroyConvolutionDescriptor(
            cudnnConvolutionDescriptor_t convDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnCreateFilterDescriptor(
            out cudnnFilterDescriptor_t filterDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnSetFilter4dDescriptor(
            cudnnFilterDescriptor_t filterDesc,
            cudnnDataType_t dataType,
            cudnnTensorFormat_t format,
            int k,
            int c,
            int h,
            int w);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDestroyFilterDescriptor(
            cudnnFilterDescriptor_t filterDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnCreateDropoutDescriptor(
            out cudnnDropoutDescriptor_t dropoutDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnSetDropoutDescriptor(
            cudnnDropoutDescriptor_t dropoutDesc,
            cudnnHandle_t cudnnHandle,
            float dropout,
            IntPtr states,
            size_t stateSizeInBytes,
            ulong seed);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDestroyDropoutDescriptor(
            cudnnDropoutDescriptor_t dropoutDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnCreate(
            out cudnnHandle_t cudnnHandle);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDestroy(
            cudnnHandle_t cudnnHandle);

        [DllImport(DLL_NAME)]
        public static extern size_t cudnnGetVersion();

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnCreateRNNDescriptor(
            out cudnnRNNDescriptor_t rnnDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnDestroyRNNDescriptor(
            cudnnRNNDescriptor_t rnnDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnSetRNNDescriptor_v8(
            cudnnRNNDescriptor_t rnnDesc,
            cudnnRNNAlgo_t algo,
            cudnnRNNMode_t cellMode,
            cudnnRNNBiasMode_t biasMode,
            cudnnDirectionMode_t dirMode,
            cudnnRNNInputMode_t inputMode,
            cudnnDataType_t dataType,
            cudnnDataType_t mathPrec,
            cudnnMathType_t mathType,
            int inputSize,
            int hiddenSize,
            int projSize,
            int numLayers,
            cudnnDropoutDescriptor_t dropoutDesc,
            uint auxFlags);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnCreateRNNDataDescriptor(
            out cudnnRNNDataDescriptor_t RNNDataDesc);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnSetRNNDataDescriptor(
            cudnnRNNDataDescriptor_t RNNDataDesc,
            cudnnDataType_t dataType,
            cudnnRNNDataLayout_t layout,
            int maxSeqLength,
            int batchSize,
            int vectorSize,
            int* seqLengthArray,
            void* paddingFill);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="rnnDesc"></param>
        /// <param name="fMode"></param>
        /// <param name="xDesc"></param>
        /// <param name="workSpaceSize">Output.
        /// Minimum amount of GPU memory in bytes needed as a workspace buffer.
        /// The workspace buffer is not used to pass intermediate results between APIs but as a temporary read/write buffer.</param>
        /// <param name="reserveSpaceSize">Output.
        /// Minimum amount of GPU memory in bytes needed as the reserve-space buffer.
        /// The reserve space buffer is used to pass intermediate results from cudnnRNNForward()
        /// to RNN BackwardData and BackwardWeights routines that compute first order derivatives with respect to RNN inputs
        /// or trainable weight and biases.</param>
        /// <returns></returns>
        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnGetRNNTempSpaceSizes(
            cudnnHandle_t handle,
            cudnnRNNDescriptor_t rnnDesc,
            cudnnForwardMode_t fMode,
            cudnnRNNDataDescriptor_t xDesc,
            out size_t workSpaceSize,       //needed both for training and inference
            out size_t reserveSpaceSize);   //needed only for training

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnGetRNNWeightSpaceSize(
            cudnnHandle_t handle,
            cudnnRNNDescriptor_t rnnDesc,
            out size_t weightSpaceSize);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnRNNForward(
            cudnnHandle_t cudnnHandle,
            cudnnRNNDescriptor_t rnnDesc,
            cudnnForwardMode_t fwdMode,
            IntPtr devSeqLengths,
            cudnnRNNDataDescriptor_t xDesc,
            IntPtr x,
            cudnnRNNDataDescriptor_t yDesc,
            IntPtr y,
            cudnnTensorDescriptor_t hDesc,
            IntPtr hx,
            IntPtr hy,
            cudnnTensorDescriptor_t cDesc,
            IntPtr cx,
            IntPtr cy,
            size_t weightSpaceSize,
            IntPtr weightSpace,
            size_t workSpaceSize,           //needed both for training and inference
            IntPtr workSpace,               //needed both for training and inference
            size_t reserveSpaceSize,        //needed only for training
            IntPtr reserveSpace);           //needed only for training

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnRNNBackwardData_v8(
            cudnnHandle_t cudnnHandle,
            cudnnRNNDescriptor_t rnnDesc,
            IntPtr devSeqLengths,
            cudnnRNNDataDescriptor_t yDesc,
            IntPtr y,
            IntPtr dy,
            cudnnRNNDataDescriptor_t xDesc,
            IntPtr dx,
            cudnnTensorDescriptor_t hDesc,
            IntPtr hx,
            IntPtr dhy,
            IntPtr dhx,
            cudnnTensorDescriptor_t cDesc,
            IntPtr cx,
            IntPtr dcy,
            IntPtr dcx,
            size_t weightSpaceSize,
            IntPtr weightSpace,
            size_t workSpaceSize,
            IntPtr workSpace,
            size_t reserveSpaceSize,
            IntPtr reserveSpace);

        [DllImport(DLL_NAME)]
        public static extern cudnnStatus_t cudnnRNNBackwardWeights_v8(
            cudnnHandle_t cudnnHandle,
            cudnnRNNDescriptor_t rnnDesc,
            cudnnWgradMode_t addGrad,
            IntPtr devSeqLengths,
            cudnnRNNDataDescriptor_t xDesc,
            IntPtr x,
            cudnnTensorDescriptor_t hDesc,
            IntPtr hx,
            cudnnRNNDataDescriptor_t yDesc,
            IntPtr y,
            size_t weightSpaceSize,
            IntPtr dweightSpace,
            size_t workSpaceSize,
            IntPtr workSpace,
            size_t reserveSpaceSize,
            IntPtr reserveSpace);
    }
    
}
