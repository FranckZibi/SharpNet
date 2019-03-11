using System;
using System.Diagnostics;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNetTests.CPU;
using SharpNetTests.Data;

namespace SharpNetTests.GPU
{
    [TestFixture]
    public class KernelManagerTests
    {
        private readonly GPUWrapper _gpuWrapper = GPUWrapper.Default;
        private readonly Random _rand = new Random(0);

        [Test]
        public void KernelManagerTest()
        {
            var km = new KernelManager(_gpuWrapper);
            var size = 1 << 20;
            var shape = new[] { 1, size, 1, 1 };
            var aCpu = RandomTensor(shape, "aCpu");
            var bCpu = RandomTensor(shape, "bCpu");
            var resultCpu = new CpuTensor<double>(shape, "resultCpu");
            for (int i = 0; i < aCpu.Count; ++i)
            {
                resultCpu[i] = aCpu[i] + bCpu[i];
            }

            //GPU double precision test
            Tensor a = aCpu.ToGPU<double>(_gpuWrapper);
            Tensor b = bCpu.ToGPU<double>(_gpuWrapper);
            Tensor resultGpu = new GPUTensor<double>(shape, null, "resultGpu", _gpuWrapper);
            km.RunKernel("Sum", resultGpu.Count, new object[] { a, b, resultGpu });
            Assert.IsTrue(TestTensor.SameContent(resultCpu, resultGpu, 1e-9));

            //GPU single precision test
            a = aCpu.ToSinglePrecision().ToGPU<float>(_gpuWrapper);
            b = bCpu.ToSinglePrecision().ToGPU<float>(_gpuWrapper);
            resultGpu = new GPUTensor<float>(shape, null, "resultGpu", _gpuWrapper);
            km.RunKernel("Sum", resultGpu.Count, new object[] { a, b, resultGpu });
            Assert.IsTrue(TestTensor.SameContent(resultCpu, resultGpu, 1e-2));
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
            var observedBlocksPerGridThreadsPerBlock = KernelManager.Compute_BlocksPerGrid_ThreadsPerBlock(count, maxThreadsPerBlock, multiProcessorCount, warpSize);
            Assert.AreEqual(expectedBlocksPerGrid, observedBlocksPerGridThreadsPerBlock.Item1);
            Assert.AreEqual(expectedThreadsPerBlock, observedBlocksPerGridThreadsPerBlock.Item2);
        }

        [Test, Explicit]
        public void BenchmarkTest()
        {
            var km = new KernelManager(_gpuWrapper);

            int size = 1<<20;
            var shape = new [] {1, size, 1, 1};
            var aCpu = RandomTensor(shape, "aCpu");
            var bCpu = RandomTensor(shape, "bCpu");
            var resultCpu = new CpuTensor<double>(shape, "resultCpu");
            int nbBatchCpu = 10;
            int nbBatchGPU = 1000;

            var sw = Stopwatch.StartNew();
            for (int batchid = 0; batchid < nbBatchCpu; ++batchid)
            {
                for (int i = 0; i < aCpu.Count; ++i)
                {
                    resultCpu[i] = aCpu[i] + bCpu[i];
                }
            }
            Console.WriteLine("1 CPU Double Time: " + (sw.Elapsed.TotalMilliseconds / (nbBatchCpu)) + "ms");
            Console.WriteLine("8 CPU Double Time: " + (sw.Elapsed.TotalMilliseconds / (nbBatchCpu * 8)) + "ms");
            Tensor a = aCpu.ToGPU<double>(_gpuWrapper);
            Tensor b = bCpu.ToGPU<double>(_gpuWrapper);
            Tensor resultGpu = new GPUTensor<double>(shape, null, "resultGpu", _gpuWrapper);
            sw = Stopwatch.StartNew();
            for (int batchid = 0; batchid < nbBatchGPU; ++batchid)
            {
                km.RunKernel("Sum", resultGpu.Count, new object[] {a, b, resultGpu});
            }
            Console.WriteLine("1 GPU Double Time: " + (sw.Elapsed.TotalMilliseconds / nbBatchGPU) + "ms");
            a = aCpu.ToSinglePrecision().ToGPU<float>(_gpuWrapper);
            b = bCpu.ToSinglePrecision().ToGPU<float>(_gpuWrapper);
            resultGpu = new GPUTensor<float>(shape, null, "resultGpu", _gpuWrapper);
            sw = Stopwatch.StartNew();
            for (int batchid = 0; batchid < nbBatchGPU; ++batchid)
            {
                km.RunKernel("Sum", resultGpu.Count, new object[] { a, b, resultGpu });
            }
            Console.WriteLine("1 GPU Float Time: " + (sw.Elapsed.TotalMilliseconds / nbBatchGPU) + "ms");
        }

        private CpuTensor<double> RandomTensor(int[] shape, string description)
        {
            return TestCpuTensor.RandomDoubleTensor(shape, _rand, -1.5, +1.5, description);
        }
    }
}
