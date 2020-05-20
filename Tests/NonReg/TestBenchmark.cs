using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestBenchmark
    {
        [Test, Explicit]
        public void TestGPUBenchmark_Memory()
        {
            //check RAM => GPU Copy perf
            var tmp_2GB = new float[500 * 1000000];

            var gpuContext = GPUWrapper.FromDeviceId(0);
            Console.WriteLine(gpuContext.ToString());
            double maxSpeed = 0;
            for (int i = 1; i <= 3; ++i)
            {
                Console.WriteLine(Environment.NewLine + "Loop#" + i);
                var sw = Stopwatch.StartNew();
                var tensors = new GPUTensor<float>[1];
                for(int t=0;t<tensors.Length;++t)
                {
                    tensors[t] = new GPUTensor<float>(new[] { tmp_2GB.Length}, tmp_2GB, gpuContext);
                }
                Console.WriteLine(gpuContext.ToString());
                foreach (var t in tensors)
                {
                    t.Dispose();
                }
                var speed = (tensors.Length*((double)tensors[0].CapacityInBytes) / sw.Elapsed.TotalSeconds)/1e9;
                maxSpeed = Math.Max(speed, maxSpeed);
                Console.WriteLine("speed: " + speed + " GB/s");
            }

            System.IO.File.AppendAllText(Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory, "GPUBenchmark_Memory.csv"),
                DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";"
                + "2GB Copy CPU=>GPU;"
                + gpuContext.DeviceName()+";"
#if DEBUG
                +"DEBUG;"
#else
                + "RELEASE;"
#endif
                + maxSpeed + ";"
                + Environment.NewLine
                );
        }


     

        //gpu=>gpu (same device)
        [TestCase("gpu0", "gpu0"), Explicit]
        //gpu=>gpu (different device)
        [TestCase("gpu0", "gpu1")]
        //gpu=>cpu
        [TestCase("gpu0", "cpu")]
        //cpu=>gpu
        [TestCase("cpu", "gpu0")]
        public void Test_MemoryCopy_Benchmark(string srcDescription, string destDescription)
        {
            var chunkSize = new[] { 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000, 1_000_000_000 };
            var maxChunkSize = chunkSize.Max();
            var src = GetTensor(srcDescription, maxChunkSize);
            var dest = GetTensor(destDescription, maxChunkSize);
            foreach (var byteCount in chunkSize)
            { 
                ulong loopId = 0;
                src.Reshape(new[] { byteCount });
                dest.Reshape(new[] { byteCount });
                var sw = Stopwatch.StartNew();
                while (sw.ElapsedMilliseconds < 5000)
                {
                    src.CopyTo(dest);
                    ++loopId;
                }
                sw.Stop();
                var speed = (loopId* src.ReallyNeededMemoryInBytes / sw.Elapsed.TotalSeconds) / 1e9;
                Console.WriteLine("ByteCount: "+Utils.MemoryBytesToString((ulong)byteCount) + ", Avg speed: " + speed + " GB/s");
                System.IO.File.AppendAllText(Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory, "MemoryCopy_Benchmark.csv"), DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";"+ srcDescription + ";"+ destDescription + ";"+ byteCount + ";"+ speed + ";"+ Environment.NewLine);
            }
        }
        private static Tensor GetTensor(string tensorDescription, int chunkSize)
        {
            if (tensorDescription == "gpu1" && GPUWrapper.GetDeviceCount() < 2)
            {
                tensorDescription = "gpu0";
            }
            switch (tensorDescription)
            {
                default:
                    //case "gpu":
                    //case "gpu0":
                    return new GPUTensor<byte>(new[] { chunkSize }, null, GPUWrapper.FromDeviceId(0));
                case "gpu1":
                    return new GPUTensor<byte>(new[] { chunkSize }, null, GPUWrapper.FromDeviceId(1));
                case "cpu":
                    return new CpuTensor<byte>(new[] { chunkSize }, null);
            }
        }
        [Test, Explicit]
        public void TestGPUBenchmark_Speed()
        {
            const int batchSize = 64;
            const int numEpochs = 5;
            var mnist = new MNISTDataSet();

            var network = new Network(new NetworkConfig() { LogFile = "GPUBenchmark", DisableReduceLROnPlateau =true}.WithAdam(), new List<int> {0});
            network
                .Input(mnist.Training.Channels, mnist.Training.Height, mnist.Training.Width)

                .Convolution(16, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true)
                .BatchNorm(0.99, 1e-5)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.2)
                .MaxPooling(2, 2, 2)

                .Convolution(32, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)

                .Dense_Activation(1000, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.2)

                .Output(mnist.Training.CategoryCount, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var sw = Stopwatch.StartNew();
            var learningRate = 0.01;
            var learningRateComputer = new LearningRateComputer(LearningRateScheduler.Constant(learningRate), network.Config.ReduceLROnPlateau(), network.Config.MinimumLearningRate);
            network.Fit(mnist.Training, learningRateComputer, numEpochs, batchSize, mnist.Test);
            var elapsedMs = sw.Elapsed.TotalSeconds;
            var lossAndAccuracy = network.ComputeLossAndAccuracyForTestDataSet(batchSize, mnist.Test);

            System.IO.File.AppendAllText(Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory, "GPUBenchmark_Speed.csv" ), 
                DateTime.Now.ToString("F", CultureInfo.InvariantCulture) +";"
                +"MNIST;"
                + network.DeviceName() + ";"
                + network.TotalParams + ";"
                + numEpochs + ";"
                + batchSize + ";"
                + learningRate + ";"
#if DEBUG
                +"DEBUG;"
#else
                + "RELEASE;"
#endif
                +elapsedMs+";"
                +lossAndAccuracy.Item1+";"
                +lossAndAccuracy.Item2
                +Environment.NewLine
                );
        }
    }
}
