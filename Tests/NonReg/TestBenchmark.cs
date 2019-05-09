﻿using System;
using System.Diagnostics;
using System.Globalization;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.GPU;

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

            var gpuContext = GPUWrapper.Default;
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
                for (int t = 0; t < tensors.Length; ++t)
                {
                    tensors[t].Dispose();
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
            MNIST.Load(out var X_train, out var Y_train, out var X_test, out var Y_test);
            const int batchSize = 64;
            const int numEpochs = 5;
            var network = new Network(new NetworkConfig(true) { Logger = logger, UseDoublePrecision = false }
                .WithAdam()
            );
            network
                .Input(X_train.Shape[1], X_train.Shape[2], X_train.Shape[3])

                .Convolution(16, 3, 1, 1, 0.0, true)
                .BatchNorm()
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.2)
                .MaxPooling(2, 2)

                .Convolution(32, 3, 1, 1, 0.0, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)

                .Dense_Activation(1000, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.2)

                .Output(Y_train.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var sw = Stopwatch.StartNew();
            var learningRate = 0.01;
            network.Fit(X_train, Y_train, learningRate, numEpochs, batchSize, X_test, Y_test);
            var elapsedMs = sw.Elapsed.TotalSeconds;
            var lossAndAccuracy = network.ComputeLossAndAccuracy(batchSize, X_test, Y_test);

            System.IO.File.AppendAllText(Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory, "GPUBenchmark_Speed.csv" ), 
                DateTime.Now.ToString("F", CultureInfo.InvariantCulture) +";"
                +"MNIST;"
                + network.Config.GpuWrapper.DeviceName() + ";"
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
