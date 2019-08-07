using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using NUnit.Framework;
using SharpNet;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Networks;
using SharpNet.Optimizers;
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
            var useGpu = true;
            int batchSize = 32;
            const int numEpochs = 1000;

            var imageDataGenerator = new ImageDataGenerator(0.1, 0.1, false, false, ImageDataGenerator.FillModeEnum.Nearest, 0.0, 0.0, 0.0, 0.0);
            var logFileName = Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory, "MNIST" + "_" + Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            var network = new Network(
                new NetworkConfig{ Logger = new Logger(logFileName, true), UseDoublePrecision = false }
                //.WithAdam()
                .WithSGD(0.99,true)
                ,
                imageDataGenerator,
                useGpu?0:-1
            );

            var loader = new MNISTDataLoader<double>(imageDataGenerator);

            double lambdaL2Regularization = 0.0;

            network
                .Input(loader.Training.Channels, loader.Training.CurrentHeight, loader.Training.CurrentWidth)

                .Convolution_BatchNorm_Activation(16, 3, 1, 1, lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .MaxPooling(2,2)
                //.AddBatchNorm()
                //.AddPooling(2, 2)

                //.AddDropout(0.2)

                .Convolution_BatchNorm_Activation(32, 3, 1, 1, lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .MaxPooling(2, 2)

                //.AddBatchNorm()
                //.AddPooling(2, 2)

                //.AddDropout(0.4)

                //.AddConvolution_Activation_Pooling(64, 3, 1, 1, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 2, 2)
                //.AddPooling(2, 2)

                //.AddConvolution_BatchNorm_Activation_Pooling(16, 5, 1, 2, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 2, 2)
                //.AddConvolution_BatchNorm_Activation_Pooling(32, 5, 1, 2, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 2, 2)

                .Dense_Activation(1000, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.5)
                //.AddBatchNorm()

                .Output(loader.Training.Categories, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            var learningRate = LearningRateScheduler.DivideByConstantEveryXEpoch(0.01, 2, 5, true);
            
            //learningRate = LearningRateScheduler.Constant(network.FindBestLearningRate(xTrain, yTrain, 128));
            //learningRate = LearningRateScheduler.Constant(0.00774263682681115);


            network.Fit(loader.Training, learningRate, null, numEpochs, batchSize, loader.Test);
        }

    }
}
