using System;
using System.Diagnostics;
using System.Globalization;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Networks;
using SharpNet.Optimizers;
using SharpNet.Pictures;

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestBenchmark
    {
        private static string LogFileName => Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory, "GPUBenchmark" + "_" + Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");

        [Test, Explicit]
        public void TestGPUBenchmark_Memory()
        {
            var logger = new Logger(LogFileName, true);

            //check RAM => GPU Copy perf
            var tmp_2GB = new double[250 * 1000000];
            var hostPinnedMemory = new HostPinnedMemory<double>(tmp_2GB);

            var gpuContext = GPUWrapper.FromDeviceId(0);
            logger.Info(gpuContext.ToString());
            double maxSpeed = 0;
            for (int i = 1; i <= 3; ++i)
            {
                logger.Info(Environment.NewLine + "Loop#" + i);
                var sw = Stopwatch.StartNew();
                var tensors = new GPUTensor<double>[1];
                for(int t=0;t<tensors.Length;++t)
                {
                    tensors[t] = new GPUTensor<double>(new[] { tmp_2GB.Length}, hostPinnedMemory.Pointer, "test", gpuContext);
                }
                logger.Info(gpuContext.ToString());
                foreach (var t in tensors)
                {
                    t.Dispose();
                }
                var speed = (tensors.Length*((double)tensors[0].CapacityInBytes) / sw.Elapsed.TotalSeconds)/1e9;
                maxSpeed = Math.Max(speed, maxSpeed);
                logger.Info("speed: " + speed + " GB/s");
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
            hostPinnedMemory.Dispose();
        }


        [Test, Explicit]
        public void TestGPUBenchmark_Speed()
        {
            var logger = new Logger(LogFileName, true);
            const int batchSize = 64;
            const int numEpochs = 5;
            var imageDataGenerator = ImageDataGenerator.NoDataAugmentation;
            var loader = new MNISTDataLoader<double>();
            var network = new Network(new NetworkConfig() { Logger = logger, UseDoublePrecision = false }.WithAdam(), imageDataGenerator, 0);
            network
                .Input(loader.Training.Channels, loader.Training.CurrentHeight, loader.Training.CurrentWidth)

                .Convolution(16, 3, 1, 1, 0.0, true)
                .BatchNorm()
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.2)
                .MaxPooling(2, 2)

                .Convolution(32, 3, 1, 1, 0.0, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)

                .Dense_Activation(1000, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.2)

                .Output(loader.Training.Categories, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var sw = Stopwatch.StartNew();
            var learningRate = 0.01;
            network.Fit(loader.Training, LearningRateScheduler.Constant(learningRate), null, numEpochs, batchSize, loader.Test);
            var elapsedMs = sw.Elapsed.TotalSeconds;
            var lossAndAccuracy = network.ComputeLossAndAccuracy(batchSize, loader.Test);

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
