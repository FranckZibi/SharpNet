using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using NUnit.Framework;
using SharpNet;
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
        private static string LogFileName => Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory, "GPUBenchmark" + "_" + Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");

        [Test, Explicit]
        public void TestGPUBenchmark_Memory()
        {
            var logger = new Logger(LogFileName, true);

            //check RAM => GPU Copy perf
            var tmp_2GB = new float[500 * 1000000];

            var gpuContext = GPUWrapper.FromDeviceId(0);
            logger.Info(gpuContext.ToString());
            double maxSpeed = 0;
            for (int i = 1; i <= 3; ++i)
            {
                logger.Info(Environment.NewLine + "Loop#" + i);
                var sw = Stopwatch.StartNew();
                var tensors = new GPUTensor<float>[1];
                for(int t=0;t<tensors.Length;++t)
                {
                    tensors[t] = new GPUTensor<float>(new[] { tmp_2GB.Length}, tmp_2GB, gpuContext, "test");
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
        }


        [Test, Explicit]
        public void TestGPUBenchmark_Speed()
        {
            var logger = new Logger(LogFileName, true);
            const int batchSize = 64;
            const int numEpochs = 5;
            var mnist = new MNISTDataSet();
            var network = new Network(new NetworkConfig() { Logger = logger, DisableReduceLROnPlateau =true}.WithAdam(), new List<int> {0});
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
