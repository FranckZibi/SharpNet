using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNet.Pictures;

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestMNIST
    {
        [Test, Explicit]
        [SuppressMessage("ReSharper", "ConditionIsAlwaysTrueOrFalse")]
        public void Test()
        {
            /*
            var dllDirectory = @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp";
            Environment.SetEnvironmentVariable("PATH", dllDirectory + ";" + Environment.GetEnvironmentVariable("PATH"));
            Environment.SetEnvironmentVariable("CUDA_PATH", @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0");
            Environment.SetEnvironmentVariable("NVCUDASAMPLES_ROOT  ", @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0");
            */
            //Environment.SetEnvironmentVariable("TF_FP16_CONV_USE_FP32_COMPUTE", "1");
            //Environment.SetEnvironmentVariable("TF_FP16_MATMUL_USE_FP32_COMPUTE", "1");

            var logger = new Logger(LogFileName, true);

            Load(out var X_train, out var Y_train, out var X_test, out var Y_test);

            var useGpu = true;
            int batchSize = 32;
            const int numEpochs = 1000;

            /*
            //this code is used to use CPU instead of GPU. It uses a smaller sample
            int nbTests = 100;
            X_test = X_test.ExtractSubTensor(0, nbTests);
            Y_test = Y_test.ExtractSubTensor(0, nbTests);
            X_train = X_train.ExtractSubTensor(0, nbTests);
            Y_train = Y_train.ExtractSubTensor(0, nbTests);
            useGpu = false;
            */

            //var network = Network.ValueOf("c:/temp/ml/save_2632_2.txt");network.Config.Logger = logger;


            var imageDataGenerator = new ImageDataGenerator(0.1, 0.1, false, false, ImageDataGenerator.FillModeEnum.Nearest, 0.0);
            var network = new Network(
                new NetworkConfig(useGpu) { Logger = logger, UseDoublePrecision = false }
                //.WithAdam()
                .WithSGD(0.99,0,true)
                ,
                imageDataGenerator
            );

            double lambdaL2Regularization = 0.0;

            network
                .AddInput(X_train.Shape[1], X_train.Shape[2], X_train.Shape[3])

                .AddConvolution_BatchNorm_Activation(16, 3, 1, 1, lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .AddMaxPooling(2,2)
                //.AddBatchNorm()
                //.AddPooling(2, 2)

                //.AddDropout(0.2)

                .AddConvolution_BatchNorm_Activation(32, 3, 1, 1, lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .AddMaxPooling(2, 2)

                //.AddBatchNorm()
                //.AddPooling(2, 2)

                //.AddDropout(0.4)

                //.AddConvolution_Activation_Pooling(64, 3, 1, 1, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 2, 2)
                //.AddPooling(2, 2)

                //.AddConvolution_BatchNorm_Activation_Pooling(16, 5, 1, 2, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 2, 2)
                //.AddConvolution_BatchNorm_Activation_Pooling(32, 5, 1, 2, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 2, 2)

                .AddDense_Activation(1000, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .AddDropout(0.5)
                //.AddBatchNorm()

                .AddOutput(Y_train.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);
                var learningRate = LearningRateScheduler.DivideByConstantEveryXEpoch(0.01, 2, 5, true);
                //var learningRate = 0.1;


            network.Fit(X_train, Y_train, learningRate, numEpochs, batchSize, X_test, Y_test);
            
        }

        public static void Load(out CpuTensor<double> X_train, out CpuTensor<double> Y_train, out CpuTensor<double> X_test, out CpuTensor<double> Y_test)
        {
            var trainTuple = ToWorkingSet(TrainingSet);
            X_train = trainTuple.Item1;
            Y_train = trainTuple.Item2;
            var testTuple = ToWorkingSet(TestSet);
            X_test = testTuple.Item1;
            Y_test = testTuple.Item2;
        }

        private static string LogFileName => Utils.ConcatenatePathWithFileName(@"c:\temp\ML\", "MNIST" + "_" + Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
        private static Tuple<CpuTensor<double>, CpuTensor<double>> ToWorkingSet(List<KeyValuePair<CpuTensor<byte>, int>> t)
        {
            int setSize = t.Count;

            //setSize = Math.Min(5000,setSize);

            var X = new CpuTensor<double>(new[] { setSize, 1, t[0].Key.Height, t[0].Key.Width}, "X");
            var Y = new CpuTensor<double>(new[] { setSize, 10 }, "Y");
            for (int m = 0; m < setSize; ++m)
            {
                var matrix = t[m].Key;
                for (int row = 0; row < matrix.Height; ++row)
                {
                    for (int col = 0; col < matrix.Width; ++col)
                    {
                        X.Set(m, 0, row, col, matrix.Get(row,col) / 255.0);
                    }
                }
                Y.Set(m, t[m].Value, 1);
            }
            return Tuple.Create(X,Y);
        }
        private static List<KeyValuePair<CpuTensor<byte>, int>> TrainingSet => PictureTools.ReadInputPictures(@"C:\Projects\SharpNet\Tests\Data\MNIST\train-images.idx3-ubyte", @"C:\Projects\SharpNet\Tests\Data\MNIST\train-labels.idx1-ubyte");
        private static List<KeyValuePair<CpuTensor<byte>, int>> TestSet => PictureTools.ReadInputPictures(@"C:\Projects\SharpNet\Tests\Data\MNIST\t10k-images.idx3-ubyte", @"C:\Projects\SharpNet\Tests\Data\MNIST\t10k-labels.idx1-ubyte");
    }
}
