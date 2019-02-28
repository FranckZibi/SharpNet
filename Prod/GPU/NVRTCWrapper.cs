using System;
using System.Runtime.InteropServices;
using SharpNet.Data;

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
        internal const string NVRTC_API_DLL_NAME = "nvrtc64_100_0";

        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcCreateProgram(
            out IntPtr programHandle,
            [MarshalAs(UnmanagedType.LPStr)] string src, //!D
            [MarshalAs(UnmanagedType.LPStr)] string name, 
            int numHeaders,
            IntPtr[] headers,
            IntPtr[] includeNames);


        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcCompileProgram(
            IntPtr programHandle,
            int numOptions,
            IntPtr[] options);


        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTXSize(
            IntPtr programHandle,
            out size_t ptxSizeRet);

        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcDestroyProgram(
            ref IntPtr programHandle);

        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTX(
            IntPtr programHandle,
            byte[] ptx);

        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLogSize(
            IntPtr prog,
            out size_t logSizeRet);

        [DllImport(NVRTC_API_DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLog(
            IntPtr programHandle,
            byte[] log);
    }
}
