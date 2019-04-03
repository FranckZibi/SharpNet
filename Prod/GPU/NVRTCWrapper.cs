using System;
using System.Runtime.InteropServices;
using SharpNet.Data;
// ReSharper disable UnusedMember.Global

namespace SharpNet.GPU
{
    public enum nvrtcResult
    {
        NVRTC_SUCCESS = 0,
        NVRTC_ERROR_OUT_OF_MEMORY = 1,
        NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
        NVRTC_ERROR_INVALID_INPUT = 3,
        NVRTC_ERROR_INVALID_PROGRAM = 4,
        NVRTC_ERROR_INVALID_OPTION = 5,
        NVRTC_ERROR_COMPILATION = 6,
        NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
        NVRTC_ERROR_INTERNAL_ERROR = 11
    }

    public static class NVRTCWrapper
    {
        private static readonly CUDA_Versions InstalledCudaVersion;

        static NVRTCWrapper()
        {
            InstalledCudaVersion = GPUWrapper.GetInstalledCudaVersion();
        }

        public static nvrtcResult nvrtcCreateProgram(out IntPtr programHandle, [MarshalAs(UnmanagedType.LPStr)] string src, [MarshalAs(UnmanagedType.LPStr)] string name, int numHeaders, IntPtr[] headers, IntPtr[] includeNames)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcCreateProgram(out programHandle, src, name, numHeaders, headers, includeNames);
                default:
                    return NVRTCWrapper_nvrtc64_100_0.nvrtcCreateProgram(out programHandle, src, name, numHeaders, headers, includeNames);
            }
        }
        public static nvrtcResult nvrtcCompileProgram(IntPtr programHandle, int numOptions, IntPtr[] options)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcCompileProgram(programHandle, numOptions, options);
                default:
                    return NVRTCWrapper_nvrtc64_100_0.nvrtcCompileProgram(programHandle, numOptions, options);
            }
        }
        public static nvrtcResult nvrtcGetPTXSize(IntPtr programHandle, out size_t ptxSizeRet)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcGetPTXSize(programHandle, out ptxSizeRet);
                default:
                    return NVRTCWrapper_nvrtc64_100_0.nvrtcGetPTXSize(programHandle, out ptxSizeRet);
            }
        }
        public static nvrtcResult nvrtcDestroyProgram(ref IntPtr programHandle)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcDestroyProgram(ref programHandle);
                default:
                    return NVRTCWrapper_nvrtc64_100_0.nvrtcDestroyProgram(ref programHandle);
            }
        }
        public static nvrtcResult nvrtcGetPTX(IntPtr programHandle, byte[] ptx)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcGetPTX(programHandle, ptx);
                default:
                    return NVRTCWrapper_nvrtc64_100_0.nvrtcGetPTX(programHandle, ptx);
            }
        }
        public static nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out size_t logSizeRet)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcGetProgramLogSize(prog, out logSizeRet);
                default:
                    return NVRTCWrapper_nvrtc64_100_0.nvrtcGetProgramLogSize(prog, out logSizeRet);
            }
        }
        public static nvrtcResult nvrtcGetProgramLog(IntPtr programHandle, byte[] log)
        {
            switch (InstalledCudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcGetProgramLog(programHandle, log);
                default:
                    return NVRTCWrapper_nvrtc64_100_0.nvrtcGetProgramLog(programHandle, log);
            }
        }
    }

    public static class NVRTCWrapper_nvrtc64_100_0
    {
        private const string NVRTC_API_DLL_NAME = "nvrtc64_100_0";
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcCreateProgram(out IntPtr programHandle, [MarshalAs(UnmanagedType.LPStr)] string src, [MarshalAs(UnmanagedType.LPStr)] string name,  int numHeaders, IntPtr[] headers, IntPtr[] includeNames);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcCompileProgram(IntPtr programHandle, int numOptions, IntPtr[] options);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTXSize(IntPtr programHandle, out size_t ptxSizeRet);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr programHandle);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTX(IntPtr programHandle, byte[] ptx);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out size_t logSizeRet);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLog(IntPtr programHandle, byte[] log);
    }
    public static class NVRTCWrapper_nvrtc64_101_0
    {
        private const string NVRTC_API_DLL_NAME = "nvrtc64_101_0";
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcCreateProgram(out IntPtr programHandle, [MarshalAs(UnmanagedType.LPStr)] string src, [MarshalAs(UnmanagedType.LPStr)] string name, int numHeaders, IntPtr[] headers, IntPtr[] includeNames);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcCompileProgram(IntPtr programHandle, int numOptions, IntPtr[] options);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTXSize(IntPtr programHandle, out size_t ptxSizeRet);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr programHandle);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTX(IntPtr programHandle, byte[] ptx);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out size_t logSizeRet);
        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLog(IntPtr programHandle, byte[] log);
    }
}
