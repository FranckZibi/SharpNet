using System;
using System.Runtime.InteropServices;
// ReSharper disable UnusedMember.Global

namespace SharpNet.GPU
{
    public enum cublasStatus_t
    {
        CUBLAS_STATUS_SUCCESS = 0, //The operation completed successfully.
        CUBLAS_STATUS_NOT_INITIALIZED = 1, //The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the cuBLAS routine, or an error in the hardware setup. To correct: call cublasCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.
        CUBLAS_STATUS_ALLOC_FAILED = 3, //Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure. To correct: prior to the function call, deallocate previously allocated memory as much as possible.
        CUBLAS_STATUS_INVALID_VALUE = 7, //An unsupported value or parameter was passed to the function (a negative vector size, for example).To correct: ensure that all the parameters being passed have valid values.
        CUBLAS_STATUS_ARCH_MISMATCH = 8, //The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.To correct: compile and run the application on a device with appropriate compute capability, which is 1.3 for double precision.
        CUBLAS_STATUS_MAPPING_ERROR = 11, //An access to GPU memory space failed, which is usually caused by a failure to bind a texture.To correct: prior to the function call, unbind any previously bound textures.
        CUBLAS_STATUS_EXECUTION_FAILED = 13, //The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons. To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.
        CUBLAS_STATUS_INTERNAL_ERROR = 14, //An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure. To correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.
        CUBLAS_STATUS_NOT_SUPPORTED = 15, //The functionality requested is not supported
        CUBLAS_STATUS_LICENSE_ERROR =16
    }

    public enum cublasOperation_t
    {
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        CUBLAS_OP_C
    }

    public enum cublasSideMode_t
    {
        CUBLAS_SIDE_LEFT, //the matrix is on the left side in the equation
        CUBLAS_SIDE_RIGHT //he matrix is on the left side in the equation
    }

    public static class CublasWrapper
    {
        private static readonly CUDA_Versions InstalledCudaVersion;

        static CublasWrapper()
        {
            InstalledCudaVersion = GPUWrapper.GetInstalledCudaVersion();
        }

        public static cublasStatus_t cublasCreate_v2(ref IntPtr cublasHandle)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                case CUDA_Versions.CUDA_10_2:
                    return CublasWrapper_cublas64_10.cublasCreate_v2(ref cublasHandle);
                default:
                    return CublasWrapper_cublas64_100.cublasCreate_v2(ref cublasHandle);
            }
        }
        public static cublasStatus_t cublasDestroy_v2(IntPtr cublasHandle)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                case CUDA_Versions.CUDA_10_2:
                    return CublasWrapper_cublas64_10.cublasDestroy_v2(cublasHandle);
                default:
                    return CublasWrapper_cublas64_100.cublasDestroy_v2(cublasHandle);
            }
        }
        public static cublasStatus_t cublasScopy_v2(IntPtr cublasHandle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                case CUDA_Versions.CUDA_10_2:
                    return CublasWrapper_cublas64_10.cublasScopy_v2(cublasHandle, n, x, incx, y, incy);
                default:
                    return CublasWrapper_cublas64_100.cublasScopy_v2(cublasHandle, n, x, incx, y, incy);
            }
        }
        public static cublasStatus_t cublasSgemm_v2(IntPtr cublasHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                case CUDA_Versions.CUDA_10_2:
                    return CublasWrapper_cublas64_10.cublasSgemm_v2(cublasHandle, transa, transb, m, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
                default:
                    return CublasWrapper_cublas64_100.cublasSgemm_v2(cublasHandle, transa, transb, m, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
            }
        }
        public static cublasStatus_t cublasSdgmm(IntPtr cublasHandle, cublasSideMode_t mode, int m, int n, IntPtr A, int lda, IntPtr x, int incx, IntPtr C, int ldc)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                case CUDA_Versions.CUDA_10_2:
                    return CublasWrapper_cublas64_10.cublasSdgmm(cublasHandle, mode, m, n, A, lda, x, incx, C, ldc);
                default:
                    return CublasWrapper_cublas64_100.cublasSdgmm(cublasHandle, mode, m, n, A, lda, x, incx, C, ldc);
            }
        }
        public static cublasStatus_t cublasGetVersion_v2(IntPtr cublasHandle, out int cublasVersion)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                case CUDA_Versions.CUDA_10_2:
                    return CublasWrapper_cublas64_10.cublasGetVersion_v2(cublasHandle, out cublasVersion);
                default:
                    return CublasWrapper_cublas64_100.cublasGetVersion_v2(cublasHandle, out cublasVersion);
            }
        }
    }


    public static class CublasWrapper_cublas64_100
    {
        private const string CUBLAS64_100 = "cublas64_100";
        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasCreate_v2(ref IntPtr cublasHandle);
        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasDestroy_v2(IntPtr cublasHandle);
        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasScopy_v2(IntPtr cublasHandle,int n,IntPtr x,int incx,IntPtr y,int incy);
        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasSgemm_v2(IntPtr cublasHandle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,ref float alpha,IntPtr A,int lda,IntPtr B,int ldb,ref float beta,IntPtr C,int ldc);
        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasSdgmm(IntPtr cublasHandle, cublasSideMode_t mode, int m, int n, IntPtr A, int lda, IntPtr x, int incx, IntPtr C, int ldc);
        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasGetVersion_v2(IntPtr cublasHandle, out int cublasVersion);
    }
    public static class CublasWrapper_cublas64_10
    {
        private const string CUBLAS64_10 = "cublas64_10";
        [DllImport(CUBLAS64_10)]
        public static extern cublasStatus_t cublasCreate_v2(ref IntPtr cublasHandle);
        [DllImport(CUBLAS64_10)]
        public static extern cublasStatus_t cublasDestroy_v2(IntPtr cublasHandle);
        [DllImport(CUBLAS64_10)]
        public static extern cublasStatus_t cublasScopy_v2(IntPtr cublasHandle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS64_10)]
        public static extern cublasStatus_t cublasSgemm_v2(IntPtr cublasHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc);
        [DllImport(CUBLAS64_10)]
        public static extern cublasStatus_t cublasSdgmm(IntPtr cublasHandle, cublasSideMode_t mode, int m, int n, IntPtr A, int lda, IntPtr x, int incx, IntPtr C, int ldc);
        [DllImport(CUBLAS64_10)]
        public static extern cublasStatus_t cublasGetVersion_v2(IntPtr cublasHandle, out int cublasVersion);
    }

}