using System.Runtime.InteropServices;
using System.Security;

namespace SharpNet.Data
{
    public static class BlasServices
    {
        #region Matrix multiplication
        public static void DotMkl(double[] A, int aH, int aW, bool transposeA, double[] B, int bH, int bW, bool transposeB, double[] C, double alpha, double beta)
        {
            var transA = transposeA ? CBLAS_TRANSPOSE.Trans : CBLAS_TRANSPOSE.NoTrans;
            var transB = transposeB ? CBLAS_TRANSPOSE.Trans : CBLAS_TRANSPOSE.NoTrans;
            int M = transposeA ? aW : aH; //number of rows of the matrix op(A) (= number of rows of matrix C)
            int N = transposeB ? bH : bW; //number of columns of the matrix op(B) (= number of columns of the matrix C)
            int K = transposeA ? aH : aW; //number of columns of the matrix op(A) (= number of rows of the matrix op(B))
            int lda = aW; //number of columns of the matrix A (because order = RowMajor)
            int ldb = bW; //number of colums of the matrix B (because order = RowMajor)
            int ldc = N; //number of columns of the matrix C (because order = RowMajor)
            MKL_BLAS.cblas_dgemm(CBLAS_ORDER.RowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        public static void DotMkl(float[] A, int aH, int aW, bool transposeA, float[] B, int bH, int bW, bool transposeB, float[] C, float alpha, float beta)
        {
            var transA = transposeA ? CBLAS_TRANSPOSE.Trans : CBLAS_TRANSPOSE.NoTrans;
            var transB = transposeB ? CBLAS_TRANSPOSE.Trans : CBLAS_TRANSPOSE.NoTrans;
            int M = transposeA ? aW : aH; //number of rows of the matrix op(A) (= number of rows of matrix C)
            int N = transposeB ? bH : bW; //number of columns of the matrix op(B) (= number of columns of the matrix C)
            int K = transposeA ? aH : aW; //number of columns of the matrix op(A) (= number of rows of the matrix op(B))
            int lda = aW; //number of columns of the matrix A (because order = RowMajor)
            int ldb = bW; //number of colums of the matrix B (because order = RowMajor)
            int ldc = N; //number of columns of the matrix C (because order = RowMajor)
            MKL_BLAS.cblas_sgemm(CBLAS_ORDER.RowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        #endregion
    }

    /** Constants for CBLAS_ORDER enum, file "mkl_cblas.h" */
    static class CBLAS_ORDER
    {
        public const int RowMajor = 101; /* row-major arrays */
        public const int ColMajor = 102; /* column-major arrays */
    }

    /** Constants for CBLAS_TRANSPOSE enum, file "mkl_cblas.h" */
    static class CBLAS_TRANSPOSE
    {
        public const int NoTrans = 111; /* trans='N' */
        public const int Trans = 112; /* trans='T' */
        public const int ConjTrans = 113; /* trans='C' */
    }

    /** MKL BLAS wrappers */
    [SuppressUnmanagedCodeSecurity]
    public static class MKL_BLAS
    {
        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        public static extern void cblas_dgemm(int Order, int TransA, int TransB, int M, int N, int K, double alpha, [In] double[] A, int lda, [In] double[] B, int ldb, double beta, [In, Out] double[] C, int ldc);
        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        public static extern void cblas_sgemm(int Order, int TransA, int TransB, int M, int N, int K, float alpha, [In] float[] A, int lda, [In] float[] B, int ldb, float beta, [In, Out] float[] C, int ldc);
        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        public static extern void cblas_daxpy(int n, double alpha, [In] double[] x, int incx, [In, Out] double[] y, int incy);
        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        public static extern void cblas_saxpy(int n, float alpha, [In] float[] x, int incx, [In, Out] float[] y, int incy);
        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        public static extern void cblas_dscal(int n, double alpha, [In] double[] x, int incx);
        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        public static extern void cblas_sscal(int n, float alpha, [In] float[] x, int incx);
        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        public static extern void cblas_dcopy(int n, [In] double[] x, int incx, [Out] double[] y, int incy);
        [DllImport("mkl_rt.dll", CallingConvention = CallingConvention.Cdecl, ExactSpelling = true, SetLastError = false)]
        public static extern void cblas_scopy(int n, [In] float[] x, int incx, [Out] float[] y, int incy);
    }
}
