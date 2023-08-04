using System;
using System.Diagnostics;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNetTests.CPU;
using SharpNetTests.NonReg;

namespace SharpNetTests.GPU
{
    [TestFixture]
    public class KernelManagerTests
    {
        private static GPUWrapper GpuWrapper => TestGPUTensor.GpuWrapper;
        private readonly Random _rand = new (0);

        [Test]
        public void KernelManagerTest()
        {
            var km = new KernelManager(GpuWrapper);
            const int size = 1 << 20;
            var shape = new[] { 1, size, 1, 1 };
            var aCpu = RandomTensor(shape);
            var bCpu = RandomTensor(shape);
            var resultCpu = new CpuTensor<float>(shape);
            for (int i = 0; i < aCpu.Count; ++i)
            {
                resultCpu[i] = aCpu[i] + bCpu[i];
            }

            //GPU single precision test
            Tensor a = aCpu.ToGPU<float>(GpuWrapper);
            Tensor b = bCpu.ToGPU<float>(GpuWrapper);
            Tensor resultGpu = new GPUTensor<float>(shape, null, GpuWrapper);
            km.RunKernel("Sum", resultGpu.Count, new object[] { a, b, resultGpu }, 0);
            Assert.IsTrue(TensorExtensions.SameFloatContent(resultCpu, resultGpu, 1e-2));
        }

        [Test]
        public void TestCudaParsingKernelManagerTest()
        {
            var cudaSrcCode =
                " //  y = (x-mean)/volatility" + Environment.NewLine +
                " __global__ void StandardizeInPlaceByRow(int N, int cols, int pointsByThread, int threadsByRow, float* __restrict x, float* __restrict row_mean, float* __restrict row_variance, float epsilon)"+ Environment.NewLine +
                "{int idx = blockIdx.x * blockDim.x + threadIdx.x;}" + Environment.NewLine +
                " __device__ inline float sigmoidf(float x) { return 1.0f / (1 + expf(-x)); "+ Environment.NewLine +
                " }" + Environment.NewLine + "   __global__ void " + Environment.NewLine +
                "numpy_sum_RowByRow(int rows, int cols, const float* __restrict x, float* __restrict sum_buffer) " + Environment.NewLine +
                "{ " + Environment.NewLine +
                " //test" + Environment.NewLine +
                " } " + Environment.NewLine;
            var observedResult = KernelManager.ExtractCudaKernelNames(cudaSrcCode);
            var expectedResult = new System.Collections.Generic.List<string>{ "StandardizeInPlaceByRow", "numpy_sum_RowByRow" };
            Assert.AreEqual(expectedResult, observedResult);
        }

        [Test]
        public void TestReplaceTemplateWithTargetTypeName()
        {
            var srcCode = " void A(int N, T cols, T* pointsByThread, aT threadsByRow, float* T1)";
            var observedResult = KernelManager.ReplaceTemplateWithTargetTypeName(srcCode, "double");
            const string expectedResult = " void A(int N, double cols, double* pointsByThread, aT threadsByRow, float* T1)";
            Assert.AreEqual(expectedResult, observedResult);
        }
        

        [TestCase(1,1,0,2048,20,32)]
        [TestCase(1,1,1,2048, 20, 32)]
        [TestCase(1,30,30,2048, 20, 32)]
        [TestCase(20,128,2047,2048, 20, 32)]
        [TestCase(20,224,2*2048-1,2048, 20, 32)]
        [TestCase(20, 2048, 20 * 2047, 2048, 20, 32)]
        [TestCase(20,2048,20*2048,2048, 20, 32)]
        [TestCase(21,2048,20*2049,2048, 20, 32)]
        [TestCase(524288, 2048, 1024*1024*1024, 2048, 20, 32)]
        [TestCase(524288+1, 2048, 1024*1024*1024+1, 2048, 20, 32)]
        public void Compute_BlocksPerGrid_ThreadsPerBlockTest(int expectedBlocksPerGrid, int expectedThreadsPerBlock, int count, int maxThreadsPerBlock, int multiProcessorCount, int warpSize)
        {
            var (BlocksPerGrid, ThreadsPerBlock) = KernelManager.Compute_BlocksPerGrid_ThreadsPerBlock(count, maxThreadsPerBlock, multiProcessorCount, warpSize,256);
            Assert.AreEqual(expectedBlocksPerGrid, BlocksPerGrid);
            Assert.AreEqual(expectedThreadsPerBlock, ThreadsPerBlock);
        }

        [Test, Explicit]
        public void BenchmarkTest()
        {
            var km = new KernelManager(GpuWrapper);

            const int size = 1<<20;
            var shape = new [] {1, size, 1, 1};
            var aCpu = RandomTensor(shape);
            var bCpu = RandomTensor(shape);
            var resultCpu = new CpuTensor<float>(shape);
            const int nbBatchCpu = 10;
            const int nbBatchGPU = 1000;

            var sw = Stopwatch.StartNew();
            for (int batchId = 0; batchId < nbBatchCpu; ++batchId)
            {
                for (int i = 0; i < aCpu.Count; ++i)
                {
                    resultCpu[i] = aCpu[i] + bCpu[i];
                }
            }
            Console.WriteLine("1 CPU Time: " + (sw.Elapsed.TotalMilliseconds / (nbBatchCpu)) + "ms");
            Console.WriteLine("8 CPU Time: " + (sw.Elapsed.TotalMilliseconds / (nbBatchCpu * 8)) + "ms");
            Tensor a = aCpu.ToGPU<float>(GpuWrapper);
            Tensor b = bCpu.ToGPU<float>(GpuWrapper);
            Tensor resultGpu = new GPUTensor<float>(shape, null, GpuWrapper);
            sw = Stopwatch.StartNew();
            for (int batchId = 0; batchId < nbBatchGPU; ++batchId)
            {
                km.RunKernel("Sum", resultGpu.Count, new object[] {a, b, resultGpu}, 0);
            }
            Console.WriteLine("1 GPU Time: " + (sw.Elapsed.TotalMilliseconds / nbBatchGPU) + "ms");
        }

        [Test, Explicit]
        public void BenchmarkTestV2()
        {
            var km = new KernelManager(GpuWrapper);
            const int rows = 1_000;
            const int cols = 1024;
            var shape = new[] { rows, cols };
            var aCpu = TestNetworkPropagation.numpy_array_for_tests(shape); 

            Tensor a = aCpu.ToGPU<float>(GpuWrapper);
            Tensor mean = new GPUTensor<float>(new[]{ rows},null, GpuWrapper);
            Tensor variance = new GPUTensor<float>(new[]{ rows},null, GpuWrapper);
            var sw = Stopwatch.StartNew();
            const int nb_computes = 100_000;
            //const int nb_computes = 1;

            var (blocksPerGrid, threadsPerBlock) = KernelManager.Compute_BlocksPerGrid_ThreadsPerBlock_From_rows_cols(rows, cols, GpuWrapper.ThreadsByMultiprocessor);
            int dynamicSharedMemory = sizeof(float) * (cols+1+cols);
            int nextColsPowerOf2 = Utils.NextPowerOf2(cols);

            for (int batchId = 0; batchId < nb_computes; ++batchId)
            {
                km.RunKernel("Compute_Row_Mean_Variance_V2", blocksPerGrid * threadsPerBlock, new object[] { a, mean, variance, cols, nextColsPowerOf2, false }, blocksPerGrid, threadsPerBlock, dynamicSharedMemory);
                //a.Compute_Row_Mean_Variance(mean, variance, false);
            }
            Console.WriteLine($"Total Time: {sw.Elapsed.TotalMilliseconds}ms, {sw.Elapsed.TotalMilliseconds / nb_computes}ms/compute");

            var expectedSum = 0f;
            var expectedSumSquare = 0f;
            foreach (var v in aCpu.SpanContent)
            {
                expectedSum += v;
                expectedSumSquare += v*v;
            }


            Console.WriteLine($"Expected Sum of mean: {expectedSum/cols}" );
            Console.WriteLine($"Observed Sum of mean: {mean.ContentAsFloatArray().Sum()}");
            Console.WriteLine($"Observed Sum of variance: {variance.ContentAsFloatArray().Sum()}");
        }

        private CpuTensor<float> RandomTensor(int[] shape)
        {
            return TestCpuTensor.RandomFloatTensor(shape, _rand, -1.5, +1.5);
        }
    }
}
