﻿using System.Runtime.InteropServices;
// ReSharper disable UnusedMember.Global
// ReSharper disable InconsistentNaming

namespace SharpNet.GPU
{
    public enum cudaError_t
    {
        cudaSuccess = 0,
        cudaErrorMissingConfiguration,
        cudaErrorMemoryAllocation,
        cudaErrorInitializationError,
        cudaErrorLaunchFailure,
        cudaErrorPriorLaunchFailure,
        cudaErrorLaunchTimeout,
        cudaErrorLaunchOutOfResources,
        cudaErrorInvalidDeviceFunction,
        cudaErrorInvalidConfiguration,
        cudaErrorInvalidDevice,
        cudaErrorInvalidValue,
        cudaErrorInvalidPitchValue,
        cudaErrorInvalidSymbol,
        cudaErrorMapBufferObjectFailed,
        cudaErrorUnmapBufferObjectFailed,
        cudaErrorInvalidHostPointer,
        cudaErrorInvalidDevicePointer,
        cudaErrorInvalidTexture,
        cudaErrorInvalidTextureBinding,
        cudaErrorInvalidChannelDescriptor,
        cudaErrorInvalidMemcpyDirection,
        cudaErrorAddressOfConstant,
        cudaErrorTextureFetchFailed,
        cudaErrorTextureNotBound,
        cudaErrorSynchronizationError,
        cudaErrorInvalidFilterSetting,
        cudaErrorInvalidNormSetting,
        cudaErrorMixedDeviceExecution,
        cudaErrorCudartUnloading,
        cudaErrorUnknown,
        cudaErrorNotYetImplemented,
        cudaErrorMemoryValueTooLarge,
        cudaErrorInvalidResourceHandle,
        cudaErrorNotReady,
        cudaErrorStartupFailure = 0x7f,
        cudaErrorApiFailureBase = 10000
    }



    public static class CudartWrapper
    {
        private static readonly CUDA_Versions InstalledCudaVersion;

        static CudartWrapper()
        {
            InstalledCudaVersion = GPUWrapper.GetInstalledCudaVersion();
        }

        public static cudaError_t cudaGetDevice(out int device)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return CudartWrapper_cudart64_101.cudaGetDevice(out device);
                case CUDA_Versions.CUDA_10_2:
                    return CudartWrapper_cudart64_102.cudaGetDevice(out device);
                default:
                    return CudartWrapper_cudart64_100.cudaGetDevice(out device);
            }
        }
        public static cudaError_t cudaSetDevice(int device)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return CudartWrapper_cudart64_101.cudaSetDevice(device);
                case CUDA_Versions.CUDA_10_2:
                    return CudartWrapper_cudart64_102.cudaSetDevice(device);
                default:
                    return CudartWrapper_cudart64_100.cudaSetDevice(device);
            }
        }
        public static cudaError_t cudaDeviceReset()
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return CudartWrapper_cudart64_101.cudaDeviceReset();
                case CUDA_Versions.CUDA_10_2:
                    return CudartWrapper_cudart64_102.cudaDeviceReset();
                default:
                    return CudartWrapper_cudart64_100.cudaDeviceReset();
            }
        }
    }

    public static class CudartWrapper_cudart64_100
    {
        private const string CUDART64_100 = "cudart64_100";

        [DllImport(CUDART64_100)]
        public static extern cudaError_t cudaGetDevice(out int device);
        [DllImport(CUDART64_100)]
        public static extern cudaError_t cudaSetDevice(int device);
        [DllImport(CUDART64_100)]
        public static extern cudaError_t cudaDeviceReset();
    }
    public static class CudartWrapper_cudart64_101
    {
        private const string CUDART64_101 = "cudart64_101";
        [DllImport(CUDART64_101)]
        public static extern cudaError_t cudaGetDevice(out int device);
        [DllImport(CUDART64_101)]
        public static extern cudaError_t cudaSetDevice(int device);
        [DllImport(CUDART64_101)]
        public static extern cudaError_t cudaDeviceReset();
    }
    public static class CudartWrapper_cudart64_102
    {
        private const string CUDART64_102 = "cudart64_102";
        [DllImport(CUDART64_102)]
        public static extern cudaError_t cudaGetDevice(out int device);
        [DllImport(CUDART64_102)]
        public static extern cudaError_t cudaSetDevice(int device);
        [DllImport(CUDART64_102)]
        public static extern cudaError_t cudaDeviceReset();
    }


}