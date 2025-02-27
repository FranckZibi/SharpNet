﻿using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using NUnit.Framework;
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

            var network = TestNetwork.NewForTests(
                new NetworkSample
                    {
                        BatchSize = 32,
                        num_epochs = 1000,
                        DisableReduceLROnPlateau = true,
                        ResourceIds = new List<int> { useGpu ? 0 : -1 },
                }
                    //.WithAdam()
                    .WithSGD(0.01, 0.99, 0,true)
                    .WithCyclicCosineAnnealingLearningRateScheduler(10,2),
                NetworkSample.DefaultWorkingDirectory,
                "MNIST"
                );

            //Data Augmentation
            network.Sample.WidthShiftRangeInPercentage = 0.1;
            network.Sample.HeightShiftRangeInPercentage = 0.1;

            var mnist = new MnistDataset();

            network
                .Input(MnistDataset.Shape_CHW[0], MnistDataset.Shape_CHW[1], MnistDataset.Shape_CHW[2])

                .Convolution_BatchNorm_Activation(16, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .MaxPooling(2, 2, 2, 2)
                //.AddBatchNorm()
                //.AddPooling(2, 2)

                //.AddDropout(0.2)

                .Convolution_BatchNorm_Activation(32, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .MaxPooling(2, 2, 2, 2).Linear(1000, true, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.5).Linear(MnistDataset.NumClass, true, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

            network.Fit(mnist.Training, mnist.Test);
        }
    }
}
