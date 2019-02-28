using System;
using System.Runtime.InteropServices;

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
        private const string CUBLAS64_100 = "cublas64_100";

        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasCreate_v2(
            ref IntPtr cublasHandle);

        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasDestroy_v2(
            IntPtr cublasHandle);

        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasScopy_v2(
            IntPtr cublasHandle, 
            int n,
            IntPtr x, 
            int incx,
            IntPtr y, 
            int incy);

        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasDcopy_v2(
            IntPtr cublasHandle,
            int n,
            IntPtr x,
            int incx,
            IntPtr y,
            int incy);

        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasDgemm_v2(
            IntPtr cublasHandle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            int k,
            ref double alpha,
            IntPtr A,
            int lda,
            IntPtr B,
            int ldb,
            ref double beta,
            IntPtr C,
            int ldc);

        [DllImport(CUBLAS64_100)]
        public static extern cublasStatus_t cublasSgemm_v2(
            IntPtr cublasHandle,
            cublasOperation_t transa,
            cublasOperation_t transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr A,
            int lda,
            IntPtr B,
            int ldb,
            ref float beta,
            IntPtr C,
            int ldc);
    }
}