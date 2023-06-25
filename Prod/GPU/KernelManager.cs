using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using SharpNet.Data;

namespace SharpNet.GPU
{
    //This below code was inspired by ManagedCuda (http://kunzmi.github.io/managedCuda)
    public class KernelManager : IDisposable
    {
        private class CudaKernel
        {
            #region private fields
            private readonly IntPtr _functionHandle;
            #endregion
            public IntPtr FunctionHandle => _functionHandle;
            public uint ThreadsPerBlock { get; set; }
            public uint BlocksPerGrid { get; set; }
            // ReSharper disable once MemberCanBeMadeStatic.Local
            public uint DynamicSharedMemory { get; set; } = 0;
            public CudaKernel(byte[] fatBinaryObject, string kernelName)
            {
                var res = NVCudaWrapper.cuModuleLoadFatBinary(out IntPtr moduleHandle, fatBinaryObject);
                GPUWrapper.CheckStatus(res);
                res = NVCudaWrapper.cuModuleGetFunction(out _functionHandle, moduleHandle, kernelName);
                GPUWrapper.CheckStatus(res);
            }
        }

        #region private fields
        private readonly GPUWrapper _gpu;
        // kernel used for 32 bits floats
        //private readonly Dictionary<string, CudaKernel> _kernelDoubles;
        private readonly Dictionary<string, CudaKernel> _kernelFloats;
        private readonly Dictionary<string, CudaKernel> _kernelHalfs;
        #endregion

        private Dictionary<string, CudaKernel> GetKernelsForTensorTypeSize(int tensorTypeSize)
        {
            switch (tensorTypeSize)
            {
                //case 8: return _kernelDoubles;
                case 4: return _kernelFloats;
                case 2:  return _kernelHalfs;
                default:
                    throw new Exception($"Unsupported tensor type size {tensorTypeSize}");
            }
        }


        public KernelManager(GPUWrapper gpu)
        {
            _gpu = gpu;
            _kernelFloats = LoadKernels("SharpNet.GPU.Kernels.SinglePrecision.cu", "float");
            _kernelHalfs = LoadKernels("SharpNet.GPU.Kernels.SinglePrecision.cu", "__half");
        }

        public void RunKernel(string kernelName, int count, object[] parameterLists, int dynamicSharedMemory)
        {
            var (blocksPerGrid, threadsPerBlock) = Compute_BlocksPerGrid_ThreadsPerBlock(count, _gpu.MaxThreadsPerBlock, _gpu.MultiProcessorCount, _gpu.WarpSize, _gpu.ThreadsByMultiprocessor);
            RunKernel(kernelName, count, parameterLists, blocksPerGrid, threadsPerBlock, dynamicSharedMemory);
        }

        public void RunKernel(string kernelName, int count, object[] parameterLists, int blocksPerGrid, int threadsPerBlock, int dynamicSharedMemory)
        {
            if (threadsPerBlock <= 0 || threadsPerBlock > _gpu.MaxThreadsPerBlock)
            {
                throw new ArgumentException($"threadsPerBlock must be between 1 and {_gpu.MaxThreadsPerBlock}, not {threadsPerBlock}");
            }

            //all tensor types found in input
            var allTensorTypeSize = new HashSet<int>();
            foreach (var e in parameterLists)
            {
                if (e is Tensor b)
                {
                    allTensorTypeSize.Add(b.TypeSize);
                }
            }
            if (allTensorTypeSize.Count > 1)
            {
                throw new Exception($"All tensors must have the same type size, {string.Join(" ", allTensorTypeSize)}");
            }
            var tensorTypeSize = allTensorTypeSize.Count==0?4:allTensorTypeSize.First();

            for (var i = 0; i < parameterLists.Length; i++)
            {
                var e = parameterLists[i];
                if (e is Tensor)
                {
                    parameterLists[i] = (IntPtr)((Tensor)e);
                    continue;
                }
                if (e is double)
                {
                    if (tensorTypeSize == 4)
                    {
                        parameterLists[i] = (float)(double)e;
                    }
                    else if (tensorTypeSize == 2)
                    {
                        parameterLists[i] = (Half)(double)e;
                    }
                    continue;
                }
                if (e is bool)
                {
                    parameterLists[i] = ((bool)e)?1:0;
                    // ReSharper disable once RedundantJumpStatement
                    continue;
                }
            }
            var parameters = new List<object>();
            parameters.Add(count);
            parameters.AddRange(parameterLists);
            int paramCount = parameters.Count;
            var paramsList = new IntPtr[paramCount];
            var gcHandleList = new GCHandle[paramCount];
            for (int i = 0; i < parameters.Count; i++)
            {
                gcHandleList[i] = GCHandle.Alloc(parameters[i], GCHandleType.Pinned);
                paramsList[i] = gcHandleList[i].AddrOfPinnedObject();
            }
            var stream = _gpu.DefaultStream.StreamHandle;

            var kernel = GetKernelsForTensorTypeSize(tensorTypeSize)[kernelName];
            kernel.BlocksPerGrid = (uint)blocksPerGrid;
            kernel.ThreadsPerBlock = (uint)threadsPerBlock;
            kernel.DynamicSharedMemory = (uint)dynamicSharedMemory;


            var res = NVCudaWrapper.cuLaunchKernel(kernel.FunctionHandle, kernel.BlocksPerGrid, 1u, 1u, kernel.ThreadsPerBlock, 1u, 1u, kernel.DynamicSharedMemory, stream, paramsList, null);

            foreach (var gcHandle in gcHandleList)
            {
                gcHandle.Free();
            }
            GPUWrapper.CheckStatus(res);
        }

        public static (int BlocksPerGrid, int ThreadsPerBlock) Compute_BlocksPerGrid_ThreadsPerBlock_From_rows_cols(int rows, int cols, int threadsByMultiprocessor)
        {
            int targetThreadsPerBlock = 2* threadsByMultiprocessor;
            if (cols < targetThreadsPerBlock)
            {
                return (rows, Utils.PrevPowerOf2(cols));
            }
            return (rows, targetThreadsPerBlock);
        }



        public static (int BlocksPerGrid, int ThreadsPerBlock) Compute_BlocksPerGrid_ThreadsPerBlock(int count, int maxThreadsPerBlock, int multiProcessorCount, int warpSize, int threadsByMultiprocessor)
        {
            count = Math.Max(1, count);

            if (count <= warpSize)
            {
                return (1, count);
            }

            //!D TO OPTIMIZE
            if (count < threadsByMultiprocessor * multiProcessorCount)
            {
                int threadsPerBlockBeforeRoundingUp = (count + multiProcessorCount - 1) / multiProcessorCount;
                //we want 'ThreadsPerBlock' be a multiple of 'warpSize'
                int threadsPerBlockAfterRoundingUp = Utils.FirstMultipleOfAtomicValueAboveOrEqualToMinimum(threadsPerBlockBeforeRoundingUp, warpSize);
                return (multiProcessorCount, threadsPerBlockAfterRoundingUp);
            }

            var threadsPerBlock = maxThreadsPerBlock;

            //!D TO OPTIMIZE
            //!D
            //threadsPerBlock = 2 * threadsByMultiprocessor;

            var blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
            return (blocksPerGrid, threadsPerBlock);
        }

        #region Dispose pattern
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        ~KernelManager()
        {
            Dispose(false);
        }
        // TODO dispose the instantiated kernels (field _kernels.Values)
        // ReSharper disable once MemberCanBeMadeStatic.Local
        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                //managed Memory
            }
            //unmanaged memory
        }
        #endregion

        private Dictionary<string, CudaKernel> LoadKernels(string embeddedResourceWithCudaSourceCode, string targetTypeName)
        {
            Dictionary<string, CudaKernel> res = new();

            var cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (string.IsNullOrEmpty(cudaPath))
            {
                throw new ArgumentException("CUDA_PATH environment variable is not set");
            }

            //We load the embedded resource containing the CUDA source code
            var cudaSourceCode = Utils.LoadResourceContent(typeof(KernelManager).Assembly, embeddedResourceWithCudaSourceCode);
            cudaSourceCode = ReplaceTemplateWithTargetTypeName(cudaSourceCode, targetTypeName);

            using (var rtc = new CudaRuntimeCompiler(cudaSourceCode, "CudaRuntimeCompiler", _gpu.CudaVersion))
            {
                try
                {
                    rtc.Compile(new []
                    {
                        "--include-path="+Path.Combine(cudaPath, "include"),
                        "-arch=sm_"+_gpu.ComputeCapabilityMajor.ToString()+_gpu.ComputeCapabilityMinor,
                    });
                }
                catch (Exception ex)
                {
                    throw new Exception(rtc.GetLogAsString() + Environment.NewLine + ex);
                }


                foreach (var kernelName in ExtractCudaKernelNames(cudaSourceCode))
                {
                    var kernel = new CudaKernel(rtc.FatBinaryObject, kernelName);
                    res[kernelName] = kernel;
                }
            }
            return res;
        }


        /// <summary>
        /// extract all kernels (methods prefixed with __global__) names from a CUDA source code
        /// </summary>
        /// <param name="cudaSrcCode">the source code to process</param>
        /// <returns></returns>
        public static List<string> ExtractCudaKernelNames(string cudaSrcCode)
        {
            char[] delimiterChars = { ' ', ',', '.', ':', '\t','\r', '\n', '(', ')', '{', '}' };
            List<string> res = new();
            string[] splitted = cudaSrcCode.Split(delimiterChars, StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < splitted.Length - 2; ++i)
            {
                if (splitted[i] == "__global__")
                {
                    res.Add(splitted[i+2]);
                    i += 2;
                }
            }
            return res;
        }

        public static string ReplaceTemplateWithTargetTypeName(string cudaSrcCode, string targetTypeName)
        {
            string pattern = @"\bT\b";
            string replace = targetTypeName;
            return Regex.Replace(cudaSrcCode, pattern, replace);
        }
    }
}
