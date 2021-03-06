﻿using System;
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
        private static GPUWrapper GpuWrapper => TestGPUTensor.GpuWrapper;
        private readonly Random _rand = new Random(0);

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
                km.RunKernel("Sum", resultGpu.Count, new object[] {a, b, resultGpu});
            }
            Console.WriteLine("1 GPU Time: " + (sw.Elapsed.TotalMilliseconds / nbBatchGPU) + "ms");
        }

        private CpuTensor<float> RandomTensor(int[] shape)
        {
            return TestCpuTensor.RandomFloatTensor(shape, _rand, -1.5, +1.5);
        }
    }
}
