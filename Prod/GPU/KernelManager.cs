using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
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
            public uint DynamicSharedMemory => 0;
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
        private readonly Dictionary<string, CudaKernel> _kernels = new Dictionary<string, CudaKernel>();
        #endregion

        public KernelManager(GPUWrapper gpu)
        {
            _gpu = gpu;
            bool success = TryLoadKernel(new[]
                {
                    "ApplyZeroPaddingForRowId",
                    "ComputeAccuracy",
                    "ComputeMae",
                    "ComputeMse",
                    "BinaryCrossentropyLoss",
                    "CategoricalCrossentropyLoss",
                    "CategoricalCrossentropyWithHierarchyLoss",
                    "HuberLoss",
                    "HuberGradient",
                    "MseLoss",
                    "MseGradient",
                    "MseOfLogLoss",
                    "MseOfLogGradient",
                    "ComputeSingleAccuracyForCategoricalCrossentropyWithHierarchy",
                    "CategoricalCrossentropyWithHierarchyGradient",
                    "Concatenate",
                    "Concatenate3",
                    "MultiplyEachRowIntoSingleValue",
                    "Split",
                    "Split3",
                    "SwishGradient",
                    "Sum",
                    "UpdateAdamOptimizer",
                    "UpSampling2D",
                    "YOLOV3Forward",
                    "ComputeSoftmaxWithHierarchy",
                    "ComputeSoftmaxGradientWitHierarchy",
                    "WordEmbeddingForwardPropagation",
                    "WordEmbeddingBackwardPropagation",
                    "Switch_First_2_axis",
                    "LinearFunction",
                    "ComputeLn",
                    "LnGradient",
                    "Clip"
        },
                "SharpNet.GPU.Kernels.SinglePrecision.cu",
                out var errorMsg);
            if (!success)
            {
                throw new Exception(errorMsg);
            }
        }
        public void RunKernel(string kernelName, int count, object[] parameterLists)
        {
            var kernel = _kernels[kernelName];

            var blocksPerGrid_ThreadsPerBlock = Compute_BlocksPerGrid_ThreadsPerBlock(count, _gpu.MaxThreadsPerBlock, _gpu.MultiProcessorCount, _gpu.WarpSize);
            kernel.BlocksPerGrid = blocksPerGrid_ThreadsPerBlock.Item1;
            kernel.ThreadsPerBlock = blocksPerGrid_ThreadsPerBlock.Item2;

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
                    parameterLists[i] = (float)(double)e;
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
            var res = NVCudaWrapper.cuLaunchKernel(kernel.FunctionHandle, kernel.BlocksPerGrid, 1u, 1u, kernel.ThreadsPerBlock, 1u, 1u, kernel.DynamicSharedMemory, stream, paramsList, null);
            foreach (var gcHandle in gcHandleList)
            {
                gcHandle.Free();
            }
            GPUWrapper.CheckStatus(res);
        }

        //return a Tuple<BlocksPerGrid, ThreadsPerBlock>
        public static Tuple<uint, uint> Compute_BlocksPerGrid_ThreadsPerBlock(int count, int maxThreadsPerBlock, int multiProcessorCount, int warpSize)
        {
            count = Math.Max(1, count);
            if (count <= warpSize)
            {
                return Tuple.Create(1u, (uint)count);
            }
            if (count < maxThreadsPerBlock * multiProcessorCount)
            {
                int threadsPerBlockBeforeRoundingUp = (count + multiProcessorCount - 1) / multiProcessorCount;
                //we want 'ThreadsPerBlock' be a multiple of 'warpSize'
                int threadsPerBlockAfterRoundingUp = Utils.FirstMultipleOfAtomicValueAboveOrEqualToMinimum(threadsPerBlockBeforeRoundingUp, warpSize);
                return Tuple.Create((uint)multiProcessorCount, (uint)threadsPerBlockAfterRoundingUp);
            }
            var threadsPerBlock = maxThreadsPerBlock;
            var blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
            return Tuple.Create((uint)blocksPerGrid, (uint)threadsPerBlock);
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

        private bool TryLoadKernel(string[] kernelNames, string embeddedResourceWithCudaSourceCode, out string errorMsg)
        {
            //We load the embedded resource containing the CUDA source code
            var cudaSourceCode = Utils.LoadResourceContent(typeof(KernelManager).Assembly, embeddedResourceWithCudaSourceCode);

            using (var rtc = new CudaRuntimeCompiler(cudaSourceCode, nameof(_kernels), _gpu.CudaVersion))
            {
                try
                {
                    rtc.Compile();
                }
                catch (Exception ex)
                {
                    errorMsg = rtc.GetLogAsString() + Environment.NewLine + ex;
                    return false;
                }
                foreach (var kernelName in kernelNames)
                {
                    var kernel = new CudaKernel(rtc.FatBinaryObject, kernelName);
                    _kernels[kernelName] = kernel;
                }
            }
            errorMsg = "";
            return true;
        }
    }
}
