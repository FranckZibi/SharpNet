using System;
using System.Runtime.InteropServices;
// ReSharper disable UnusedMember.Global

namespace SharpNet.GPU
{
    public enum cusolverStatus_t
    {
        //The operation completed successfully.
        CUSOLVER_STATUS_SUCCESS = 0,

        //The cuSolver library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSolver routine, or an error in the hardware setup.
        //To correct: call cusolverCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuSolver library are correctly installed.
        CUSOLVER_STATUS_NOT_INITIALIZE = 1,

        //Resource allocation failed inside the cuSolver library. This is usually caused by a cudaMalloc() failure.
        //To correct: prior to the function call, deallocate previously allocated memory as much as possible.
        CUSOLVER_STATUS_ALLOC_FAILED = 2,

        //An unsupported value or parameter was passed to the function (a negative vector size, for example).
        //To correct: ensure that all the parameters being passed have valid values.
        CUSOLVER_STATUS_INVALID_VALUE = 3,

        //The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision.
        //To correct: compile and run the application on a device with compute capability 2.0 or above.
        CUSOLVER_STATUS_ARCH_MISMATCH = 4,

        CUSOLVER_STATUS_MAPPING_ERROR = 5,

        //The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.
        //To correct: check that the hardware, an appropriate version of the driver, and the cuSolver library are correctly installed.
        CUSOLVER_STATUS_EXECUTION_FAILED = 6,

        //An internal cuSolver operation failed.This error is usually caused by a cudaMemcpyAsync() failure.
        //To correct: check that the hardware, an appropriate version of the driver, and the cuSolver library are correctly installed.Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.
        CUSOLVER_STATUS_INTERNAL_ERROR = 7,

        //The matrix type is not supported by this function.This is usually caused by passing an invalid matrix descriptor to the function.
        //To correct: check that the fields in descrA were set correctly.
        CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,

        CUSOLVER_STATUS_NOT_SUPPORTED = 9,
        CUSOLVER_STATUS_ZERO_PIVOT = 10,
        CUSOLVER_STATUS_INVALID_LICENSE = 11,
        CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED = 12,
        CUSOLVER_STATUS_IRS_PARAMS_INVALID = 13,
        CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC = 14,
        CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE = 15,
        CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER = 16,
        CUSOLVER_STATUS_IRS_INTERNAL_ERROR = 20,
        CUSOLVER_STATUS_IRS_NOT_SUPPORTED = 21,
        CUSOLVER_STATUS_IRS_OUT_OF_RANGE = 22,
        CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES = 23,
        CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED = 25,
        CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED = 26,
        CUSOLVER_STATUS_IRS_MATRIX_SINGULAR = 30,
        CUSOLVER_STATUS_INVALID_WORKSPACE = 31
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct cusolverDnHandle_t
    {
        private readonly IntPtr _pointer;
    }


    public static unsafe class CusolverWrapper
    {
        private const string DLL_NAME = "cusolver64_11";


        [DllImport(DLL_NAME)]
        public static extern cusolverStatus_t cusolverDnCreate(out cusolverDnHandle_t cusolverhandle);

        [DllImport(DLL_NAME)]
        public static extern cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t cusolverhandle);


        [DllImport(DLL_NAME)]
        public static extern cusolverStatus_t cusolverDnSgetrf_bufferSize(
            cusolverDnHandle_t cusolverhandle,
            int m,
            int n,
            IntPtr A,
            int lda,
            out int lwork);

        [DllImport(DLL_NAME)]
        public static extern cusolverStatus_t cusolverDnSgeqrf(
            cusolverDnHandle_t cusolverDnHandle, 
            int m, 
            int n,
            IntPtr A,  
            int lda,
            IntPtr TAU,
            IntPtr workSpace, 
            int Lwork,
            IntPtr devInfo);

        [DllImport(DLL_NAME)]
        public static extern cusolverStatus_t cusolverDnSorgqr_bufferSize(
            cusolverDnHandle_t handle,
            int m,
            int n,
            int k,
            IntPtr A,
            int lda,
            IntPtr tau,
            out int lwork);

        [DllImport(DLL_NAME)]
        public static extern cusolverStatus_t cusolverDnSorgqr(
                cusolverDnHandle_t handle,
                int m,
                int n,
                int k,
                IntPtr A,
                int lda,
                IntPtr tau,
                IntPtr workSpace,
                int lwork,
                IntPtr devInfo);

        [DllImport(DLL_NAME)]
        public static extern cusolverStatus_t cusolverDnSormqr_bufferSize(
            cusolverDnHandle_t handle,
            cublasSideMode_t side,
            cublasOperation_t trans,
            int m,
            int n,
            int k,
            IntPtr A,
            int lda,
            IntPtr tau,
            IntPtr C,
            int ldc,
            out int lwork);

        [DllImport(DLL_NAME)]
        public static extern cusolverStatus_t cusolverDnSormqr(
            cusolverDnHandle_t handle,
            cublasSideMode_t side,
            cublasOperation_t trans,
            int m,
            int n,
            int k,
            IntPtr A,           //computed from cusolverDnSgeqrf
            int lda,
            IntPtr tau,         //computed from cusolverDnSgeqrf
            IntPtr C,
            int ldc,
            IntPtr workSpace,        // buffer of length (at least) 'lwork'
            int lwork,          //computed from cusolverDnSormqr_bufferSize
            IntPtr devInfo);    //buffer in device memory for error management

    }
}