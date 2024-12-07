using System;
using System.Runtime.InteropServices;
using SharpNet.Data;

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



    public class CudartWrapper
    {
        private readonly CUDA_Versions _cudaVersion;

        public CudartWrapper(Version cudaVersion)
        {
            _cudaVersion = GPUWrapper.ToCUDA_Versions_enum(cudaVersion);
        }

        public cudaError_t cudaGetDevice(out int device)
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return CudartWrapper_cudart64_101.cudaGetDevice(out device);
                case CUDA_Versions.CUDA_10_2:
                    return CudartWrapper_cudart64_102.cudaGetDevice(out device);
                case CUDA_Versions.CUDA_11_0:
                case CUDA_Versions.CUDA_11_4:
                    return CudartWrapper_cudart64_110.cudaGetDevice(out device);
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                case CUDA_Versions.CUDA_12_6:
                    return CudartWrapper_cudart64_12.cudaGetDevice(out device);
                default:
                    throw new ArgumentException("invalid cuda version " + _cudaVersion);
            }
        }
        public cudaError_t cudaSetDevice(int device)
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return CudartWrapper_cudart64_101.cudaSetDevice(device);
                case CUDA_Versions.CUDA_10_2:
                    return CudartWrapper_cudart64_102.cudaSetDevice(device);
                case CUDA_Versions.CUDA_11_0:
                case CUDA_Versions.CUDA_11_4:
                    return CudartWrapper_cudart64_110.cudaSetDevice(device);
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                case CUDA_Versions.CUDA_12_6:
                    return CudartWrapper_cudart64_12.cudaSetDevice(device);
                default:
                    throw new ArgumentException("invalid cuda version " + _cudaVersion);
            }
        }
        public cudaError_t cudaDeviceReset()
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return CudartWrapper_cudart64_101.cudaDeviceReset();
                case CUDA_Versions.CUDA_10_2:
                    return CudartWrapper_cudart64_102.cudaDeviceReset();
                case CUDA_Versions.CUDA_11_0:
                case CUDA_Versions.CUDA_11_4:
                    return CudartWrapper_cudart64_110.cudaDeviceReset();
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                case CUDA_Versions.CUDA_12_6:
                    return CudartWrapper_cudart64_12.cudaDeviceReset();
                default:
                    throw new ArgumentException("invalid cuda version " + _cudaVersion);
            }
        }
        public cudaError_t cudaMemcpyPeerAsync(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count, IntPtr stream)
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return CudartWrapper_cudart64_101.cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
                case CUDA_Versions.CUDA_10_2:
                    return CudartWrapper_cudart64_102.cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
                case CUDA_Versions.CUDA_11_0:
                case CUDA_Versions.CUDA_11_4:
                    return CudartWrapper_cudart64_110.cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                case CUDA_Versions.CUDA_12_6:
                    return CudartWrapper_cudart64_12.cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream);
                default:
                    throw new ArgumentException("invalid cuda version " + _cudaVersion);
            }
        }
        public cudaError_t cudaMemcpyPeer(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count)
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return CudartWrapper_cudart64_101.cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
                case CUDA_Versions.CUDA_10_2:
                    return CudartWrapper_cudart64_102.cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
                case CUDA_Versions.CUDA_11_0:
                case CUDA_Versions.CUDA_11_4:
                    return CudartWrapper_cudart64_110.cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                case CUDA_Versions.CUDA_12_6:
                    return CudartWrapper_cudart64_12.cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
                default:
                    throw new ArgumentException("invalid cuda version " + _cudaVersion);
            }
        }
    }

    public static class CudartWrapper_cudart64_100
    {
        private const string DLL_NAME = "cudart64_100";
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaGetDevice(out int device);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaSetDevice(int device);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaDeviceReset();
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaMemcpyPeerAsync(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count, IntPtr stream);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaMemcpyPeer(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count);
    }
    public static class CudartWrapper_cudart64_101
    {
        private const string DLL_NAME = "cudart64_101";
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaGetDevice(out int device);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaSetDevice(int device);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaDeviceReset();
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaMemcpyPeerAsync(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count, IntPtr stream);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaMemcpyPeer(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count);
    }
    public static class CudartWrapper_cudart64_102
    {
        private const string DLL_NAME = "cudart64_102";
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaGetDevice(out int device);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaSetDevice(int device);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaDeviceReset();
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaMemcpyPeerAsync(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count, IntPtr stream);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaMemcpyPeer(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count);
    }

    public static class CudartWrapper_cudart64_110
    {
        private const string DLL_NAME = "cudart64_110";
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaGetDevice(out int device);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaSetDevice(int device);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaDeviceReset();
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaMemcpyPeerAsync(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count, IntPtr stream);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaMemcpyPeer(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count);
    }

    public static class CudartWrapper_cudart64_12
    {
        private const string DLL_NAME = "cudart64_12";
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaGetDevice(out int device);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaSetDevice(int device);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaDeviceReset();
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaMemcpyPeerAsync(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count, IntPtr stream);
        [DllImport(DLL_NAME)]
        public static extern cudaError_t cudaMemcpyPeer(IntPtr dst, int dstDevice, IntPtr src, int srcDevice, size_t count);
    }

}
