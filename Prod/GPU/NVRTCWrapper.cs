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

    public class NVRTCWrapper
    {
        private readonly CUDA_Versions _cudaVersion;

        public NVRTCWrapper(Version cudaVersion)
        {
            _cudaVersion = GPUWrapper.ToCUDA_Versions_enum(cudaVersion);
        }


        public nvrtcResult nvrtcCreateProgram(out IntPtr programHandle, [MarshalAs(UnmanagedType.LPStr)] string src, [MarshalAs(UnmanagedType.LPStr)] string name, int numHeaders, IntPtr[] headers, IntPtr[] includeNames)
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcCreateProgram(out programHandle, src, name, numHeaders, headers, includeNames);
                case CUDA_Versions.CUDA_10_2:
                    return NVRTCWrapper_nvrtc64_102_0.nvrtcCreateProgram(out programHandle, src, name, numHeaders, headers, includeNames);
                case CUDA_Versions.CUDA_11_0:
                    return NVRTCWrapper_nvrtc64_110_0.nvrtcCreateProgram(out programHandle, src, name, numHeaders, headers, includeNames);
                case CUDA_Versions.CUDA_11_4:
                    return NVRTCWrapper_nvrtc64_112_0.nvrtcCreateProgram(out programHandle, src, name, numHeaders, headers, includeNames);
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                    return NVRTCWrapper_nvrtc64_120_0.nvrtcCreateProgram(out programHandle, src, name, numHeaders, headers, includeNames);
                default:
                    throw new ArgumentException("invalid cuda version "+_cudaVersion);
            }
        }
        public nvrtcResult nvrtcCompileProgram(IntPtr programHandle, int numOptions, IntPtr[] options)
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcCompileProgram(programHandle, numOptions, options);
                case CUDA_Versions.CUDA_10_2:
                    return NVRTCWrapper_nvrtc64_102_0.nvrtcCompileProgram(programHandle, numOptions, options);
                case CUDA_Versions.CUDA_11_0:
                    return NVRTCWrapper_nvrtc64_110_0.nvrtcCompileProgram(programHandle, numOptions, options);
                case CUDA_Versions.CUDA_11_4:
                    return NVRTCWrapper_nvrtc64_112_0.nvrtcCompileProgram(programHandle, numOptions, options);
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                    return NVRTCWrapper_nvrtc64_120_0.nvrtcCompileProgram(programHandle, numOptions, options);
                default:
                    throw new ArgumentException("invalid cuda version " + _cudaVersion);
            }
        }
        public nvrtcResult nvrtcGetPTXSize(IntPtr programHandle, out size_t ptxSizeRet)
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcGetPTXSize(programHandle, out ptxSizeRet);
                case CUDA_Versions.CUDA_10_2:
                    return NVRTCWrapper_nvrtc64_102_0.nvrtcGetPTXSize(programHandle, out ptxSizeRet);
                case CUDA_Versions.CUDA_11_0:
                    return NVRTCWrapper_nvrtc64_110_0.nvrtcGetPTXSize(programHandle, out ptxSizeRet);
                case CUDA_Versions.CUDA_11_4:
                    return NVRTCWrapper_nvrtc64_112_0.nvrtcGetPTXSize(programHandle, out ptxSizeRet);
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                    return NVRTCWrapper_nvrtc64_120_0.nvrtcGetPTXSize(programHandle, out ptxSizeRet);
                default:
                    throw new ArgumentException("invalid cuda version " + _cudaVersion);
            }
        }
        public nvrtcResult nvrtcDestroyProgram(ref IntPtr programHandle)
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcDestroyProgram(ref programHandle);
                case CUDA_Versions.CUDA_10_2:
                    return NVRTCWrapper_nvrtc64_102_0.nvrtcDestroyProgram(ref programHandle);
                case CUDA_Versions.CUDA_11_0:
                    return NVRTCWrapper_nvrtc64_110_0.nvrtcDestroyProgram(ref programHandle);
                case CUDA_Versions.CUDA_11_4:
                    return NVRTCWrapper_nvrtc64_112_0.nvrtcDestroyProgram(ref programHandle);
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                    return NVRTCWrapper_nvrtc64_120_0.nvrtcDestroyProgram(ref programHandle);
                default:
                    throw new ArgumentException("invalid cuda version " + _cudaVersion);
            }
        }
        public nvrtcResult nvrtcGetPTX(IntPtr programHandle, byte[] ptx)
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcGetPTX(programHandle, ptx);
                case CUDA_Versions.CUDA_10_2:
                    return NVRTCWrapper_nvrtc64_102_0.nvrtcGetPTX(programHandle, ptx);
                case CUDA_Versions.CUDA_11_0:
                    return NVRTCWrapper_nvrtc64_110_0.nvrtcGetPTX(programHandle, ptx);
                case CUDA_Versions.CUDA_11_4:
                    return NVRTCWrapper_nvrtc64_112_0.nvrtcGetPTX(programHandle, ptx);
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                    return NVRTCWrapper_nvrtc64_120_0.nvrtcGetPTX(programHandle, ptx);
                default:
                    throw new ArgumentException("invalid cuda version " + _cudaVersion);
            }
        }
        public nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out size_t logSizeRet)
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcGetProgramLogSize(prog, out logSizeRet);
                case CUDA_Versions.CUDA_10_2:
                    return NVRTCWrapper_nvrtc64_102_0.nvrtcGetProgramLogSize(prog, out logSizeRet);
                case CUDA_Versions.CUDA_11_0:
                    return NVRTCWrapper_nvrtc64_110_0.nvrtcGetProgramLogSize(prog, out logSizeRet);
                case CUDA_Versions.CUDA_11_4:
                    return NVRTCWrapper_nvrtc64_112_0.nvrtcGetProgramLogSize(prog, out logSizeRet);
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                    return NVRTCWrapper_nvrtc64_120_0.nvrtcGetProgramLogSize(prog, out logSizeRet);
                default:
                    throw new ArgumentException("invalid cuda version " + _cudaVersion);
            }
        }
        public nvrtcResult nvrtcGetProgramLog(IntPtr programHandle, byte[] log)
        {
            switch (_cudaVersion)
            {
                case CUDA_Versions.CUDA_10_1:
                    return NVRTCWrapper_nvrtc64_101_0.nvrtcGetProgramLog(programHandle, log);
                case CUDA_Versions.CUDA_10_2:
                    return NVRTCWrapper_nvrtc64_102_0.nvrtcGetProgramLog(programHandle, log);
                case CUDA_Versions.CUDA_11_0:
                    return NVRTCWrapper_nvrtc64_110_0.nvrtcGetProgramLog(programHandle, log);
                case CUDA_Versions.CUDA_11_4:
                    return NVRTCWrapper_nvrtc64_112_0.nvrtcGetProgramLog(programHandle, log);
                case CUDA_Versions.CUDA_12_0:
                case CUDA_Versions.CUDA_12_1:
                    return NVRTCWrapper_nvrtc64_120_0.nvrtcGetProgramLog(programHandle, log);
                default:
                    throw new ArgumentException("invalid cuda version " + _cudaVersion);
            }
        }
    }

    public static class NVRTCWrapper_nvrtc64_100_0
    {
        private const string DLL_NAME = "nvrtc64_100_0";
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCreateProgram(out IntPtr programHandle, [MarshalAs(UnmanagedType.LPStr)] string src, [MarshalAs(UnmanagedType.LPStr)] string name,  int numHeaders, IntPtr[] headers, IntPtr[] includeNames);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCompileProgram(IntPtr programHandle, int numOptions, IntPtr[] options);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTXSize(IntPtr programHandle, out size_t ptxSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr programHandle);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTX(IntPtr programHandle, byte[] ptx);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out size_t logSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLog(IntPtr programHandle, byte[] log);
    }
    public static class NVRTCWrapper_nvrtc64_101_0
    {
        private const string DLL_NAME = "nvrtc64_101_0";
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCreateProgram(out IntPtr programHandle, [MarshalAs(UnmanagedType.LPStr)] string src, [MarshalAs(UnmanagedType.LPStr)] string name, int numHeaders, IntPtr[] headers, IntPtr[] includeNames);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCompileProgram(IntPtr programHandle, int numOptions, IntPtr[] options);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTXSize(IntPtr programHandle, out size_t ptxSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr programHandle);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTX(IntPtr programHandle, byte[] ptx);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out size_t logSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLog(IntPtr programHandle, byte[] log);
    }
    public static class NVRTCWrapper_nvrtc64_102_0
    {
        private const string DLL_NAME = "nvrtc64_102_0";
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCreateProgram(out IntPtr programHandle, [MarshalAs(UnmanagedType.LPStr)] string src, [MarshalAs(UnmanagedType.LPStr)] string name, int numHeaders, IntPtr[] headers, IntPtr[] includeNames);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCompileProgram(IntPtr programHandle, int numOptions, IntPtr[] options);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTXSize(IntPtr programHandle, out size_t ptxSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr programHandle);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTX(IntPtr programHandle, byte[] ptx);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out size_t logSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLog(IntPtr programHandle, byte[] log);
    }

    public static class NVRTCWrapper_nvrtc64_110_0
    {
        private const string DLL_NAME = "nvrtc64_110_0";
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCreateProgram(out IntPtr programHandle, [MarshalAs(UnmanagedType.LPStr)] string src, [MarshalAs(UnmanagedType.LPStr)] string name, int numHeaders, IntPtr[] headers, IntPtr[] includeNames);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCompileProgram(IntPtr programHandle, int numOptions, IntPtr[] options);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTXSize(IntPtr programHandle, out size_t ptxSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr programHandle);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTX(IntPtr programHandle, byte[] ptx);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out size_t logSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLog(IntPtr programHandle, byte[] log);
    }
    public static class NVRTCWrapper_nvrtc64_112_0
    {
        private const string DLL_NAME = "nvrtc64_112_0";
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCreateProgram(out IntPtr programHandle, [MarshalAs(UnmanagedType.LPStr)] string src, [MarshalAs(UnmanagedType.LPStr)] string name, int numHeaders, IntPtr[] headers, IntPtr[] includeNames);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCompileProgram(IntPtr programHandle, int numOptions, IntPtr[] options);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTXSize(IntPtr programHandle, out size_t ptxSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr programHandle);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTX(IntPtr programHandle, byte[] ptx);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out size_t logSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLog(IntPtr programHandle, byte[] log);
    }
    public static class NVRTCWrapper_nvrtc64_120_0
    {
        private const string DLL_NAME = "nvrtc64_120_0";
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCreateProgram(out IntPtr programHandle, [MarshalAs(UnmanagedType.LPStr)] string src, [MarshalAs(UnmanagedType.LPStr)] string name, int numHeaders, IntPtr[] headers, IntPtr[] includeNames);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcCompileProgram(IntPtr programHandle, int numOptions, IntPtr[] options);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTXSize(IntPtr programHandle, out size_t ptxSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcDestroyProgram(ref IntPtr programHandle);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetPTX(IntPtr programHandle, byte[] ptx);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLogSize(IntPtr prog, out size_t logSizeRet);
        [DllImport(DLL_NAME)]
        public static extern nvrtcResult nvrtcGetProgramLog(IntPtr programHandle, byte[] log);
    }

}
