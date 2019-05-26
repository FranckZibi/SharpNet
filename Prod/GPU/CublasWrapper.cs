using System;
using System.Runtime.InteropServices;
// ReSharper disable UnusedMember.Global

namespace SharpNet.GPU
{
    public enum cublasStatus_t
    {
        CUBLAS_STATUS_SUCCESS,
        CUBLAS_STATUS_NOT_INITIALIZED,
        CUBLAS_STATUS_ALLOC_FAILED,
        CUBLAS_STATUS_INVALID_VALUE,
        CUBLAS_STATUS_ARCH_MISMATCH,
        CUBLAS_STATUS_MAPPING_ERROR,
        CUBLAS_STATUS_EXECUTION_FAILED,
        CUBLAS_STATUS_INTERNAL_ERROR,
        CUBLAS_STATUS_NOT_SUPPORTED,
        CUBLAS_STATUS_LICENSE_ERROR
    }

    public enum cublasOperation_t
    {
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        CUBLAS_OP_C
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
                    return CublasWrapper_cublas64_10.cublasScopy_v2(cublasHandle, n, x, incx, y, incy);
                default:
                    return CublasWrapper_cublas64_100.cublasScopy_v2(cublasHandle, n, x, incx, y, incy);
            }
        }
        public static cublasStatus_t cublasDcopy_v2(IntPtr cublasHandle, int n, IntPtr x, int incx, IntPtr y, int incy)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return CublasWrapper_cublas64_10.cublasDcopy_v2(cublasHandle, n, x, incx, y, incy);
                default:
                    return CublasWrapper_cublas64_100.cublasDcopy_v2(cublasHandle, n, x, incx, y, incy);
            }
        }
        public static cublasStatus_t cublasDgemm_v2(IntPtr cublasHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return CublasWrapper_cublas64_10.cublasDgemm_v2(cublasHandle, transa, transb, m, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
                default:
                    return CublasWrapper_cublas64_100.cublasDgemm_v2(cublasHandle, transa, transb, m, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
            }
        }
        public static cublasStatus_t cublasSgemm_v2(IntPtr cublasHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return CublasWrapper_cublas64_10.cublasSgemm_v2(cublasHandle, transa, transb, m, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
                default:
                    return CublasWrapper_cublas64_100.cublasSgemm_v2(cublasHandle, transa, transb, m, n, k, ref alpha, A, lda, B, ldb, ref beta, C, ldc);
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
        public static extern cublasStatus_t cublasDcopy_v2(IntPtr cublasHandle,int n,IntPtr x,int incx,IntPtr y,int incy);
        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasDgemm_v2(IntPtr cublasHandle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,ref double alpha,IntPtr A,int lda,IntPtr B,int ldb,ref double beta,IntPtr C,int ldc);
        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasSgemm_v2(IntPtr cublasHandle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,ref float alpha,IntPtr A,int lda,IntPtr B,int ldb,ref float beta,IntPtr C,int ldc);
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
        public static extern cublasStatus_t cublasDcopy_v2(IntPtr cublasHandle, int n, IntPtr x, int incx, IntPtr y, int incy);
        [DllImport(CUBLAS64_10)]
        public static extern cublasStatus_t cublasDgemm_v2(IntPtr cublasHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, ref double alpha, IntPtr A, int lda, IntPtr B, int ldb, ref double beta, IntPtr C, int ldc);
        [DllImport(CUBLAS64_10)]
        public static extern cublasStatus_t cublasSgemm_v2(IntPtr cublasHandle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, ref float alpha, IntPtr A, int lda, IntPtr B, int ldb, ref float beta, IntPtr C, int ldc);
    }

}