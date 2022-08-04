using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestMNIST
    {
        [Test, Explicit]
        [SuppressMessage("ReSharper", "ConditionIsAlwaysTrueOrFalse")]
        public void Test()
        {
            const bool useGpu = true;

            var network = Network.NewForTests(
                new NetworkConfig
                    {
                        ModelName = "MNIST",
                        BatchSize = 32,
                        NumEpochs = 1000,
                        DisableReduceLROnPlateau = true,
                        ResourceIds = new List<int> { useGpu ? 0 : -1 },
                        InitialLearningRate = 0.01
                }
                    //.WithAdam()
                    .WithSGD(0.99, true)
                    .WithCyclicCosineAnnealingLearningRateScheduler(10,2),
                new DataAugmentationSample()
            );

            //Data Augmentation
            network.DA.WidthShiftRangeInPercentage = 0.1;
            network.DA.HeightShiftRangeInPercentage = 0.1;

            var mnist = new MnistDataset();

            const double lambdaL2Regularization = 0.0;

            network
                .Input(MnistDataset.Shape_CHW[0], MnistDataset.Shape_CHW[1], MnistDataset.Shape_CHW[2])

                .Convolution_BatchNorm_Activation(16, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .MaxPooling(2, 2, 2, 2)
                //.AddBatchNorm()
                //.AddPooling(2, 2)

                //.AddDropout(0.2)

                .Convolution_BatchNorm_Activation(32, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .MaxPooling(2, 2, 2, 2)

                //.AddBatchNorm()
                //.AddPooling(2, 2)

                //.AddDropout(0.4)

                //.AddConvolution_Activation_Pooling(64, 3, 1, 1, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 2, 2)
                //.AddPooling(2, 2)

                //.AddConvolution_BatchNorm_Activation_Pooling(16, 5, 1, 2, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 2, 2)
                //.AddConvolution_BatchNorm_Activation_Pooling(32, 5, 1, 2, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 2, 2)

                .Dense_Activation(1000, 0.0, false, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.5)
                //.AddBatchNorm()

                .Output(MnistDataset.CategoryCount, 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            network.Fit(mnist.Training, mnist.Test);
        }
    }
}
