using System;
using System.Diagnostics;
using System.Linq;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;

// ReSharper disable UnusedMember.Global

/*
SharpNet on 8-may-2019
LearningRate = Orig Paper
BatchSize = 64
EpochCount = 300
SGD with momentum = 0.9 & Nesterov & L2 = 1-e4 
Cutout 16 / FillMode = Reflect
Orig Paper = https://arxiv.org/pdf/1608.06993.pdf
# ------------------------------------------------------------
# Model     | k |Depth|Epochs| SharpNet  | Orig Paper| sec/epoch
#           |   |     |      | %Accuracy | %Accuracy | GTX1080
# ---------------------------|--------------------------------
# DenseNet  | 12|   40|  300 | 91.48     | 94.76     |  94.0        (9-may-2019, 1 078 018 params)
# DenseNet  | 12|  100|      | -----     | 95.90     | -----
# DenseNet  | 24|  100|      | -----     | 96.26     | -----
# DenseNetBC| 12|   40|  300 | 89.18     | NA        |  36.5        (9-may-2019, 181 210 params)
# DenseNetBC| 12|  100|  300 | 93.76     | 95.49     |   156        (9-may-2019, 793 150 params)
# DenseNetBC| 24|  250|      | -----     | 96.38     | -----
# DenseNetBC| 40|  190|      | -----     | 97.54     | -----
*/

namespace SharpNet.Networks
{
    //DenseNet implementation as described in https://arxiv.org/pdf/1608.06993.pdf
    public class DenseNetBuilder : NetworkBuilder
    {
        public static DenseNetBuilder DenseNet_CIFAR10()
        {
            var builder = new DenseNetBuilder
            {
                Config = new NetworkConfig
                    {
                        LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                        lambdaL2Regularization = 1e-4
                    }
                    .WithSGD(0.9, true)
                    .WithCifar10DenseNetLearningRateScheduler(false, true, false),

                //Data augmentation
                WidthShiftRange = 0.1,
                HeightShiftRange = 0.1,
                HorizontalFlip = true,
                VerticalFlip = false,
                FillMode = ImageDataGenerator.FillModeEnum.Reflect,
                CutoutPatchPercentage = 0.5, //by default we use a cutout of 1/2 of the image width

                NumEpochs = 300,
                BatchSize = 64,
                InitialLearningRate = 0.1
            };
            return builder;
        }

        public Network DenseNet_12_40_CIFAR10()
        {
            //return Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\Network_15576_14.txt");
            return Build(
                nameof(DenseNet_12_40_CIFAR10),
                new[] { 1, CIFAR10DataSet.Channels, CIFAR10DataSet.Height, CIFAR10DataSet.Width },
                CIFAR10DataSet.Categories,
                false,
                new[] { 12, 12, 12 },
                false,
                12,
                1.0,
                null);
        }
        public Network DenseNetBC_12_40_CIFAR10()
        {
            return Build(
                nameof(DenseNetBC_12_40_CIFAR10),
                new[] { 1, CIFAR10DataSet.Channels, CIFAR10DataSet.Height, CIFAR10DataSet.Width },
                CIFAR10DataSet.Categories,
                false,
                new[] { 12 / 2, 12 / 2, 12 / 2 },
                true,
                12,
                0.5,
                null);
        }
        public Network DenseNetBC_12_100_CIFAR10()
        {
            return Build(
                nameof(DenseNetBC_12_100_CIFAR10),
                new[] { 1, CIFAR10DataSet.Channels, CIFAR10DataSet.Height, CIFAR10DataSet.Width },
                CIFAR10DataSet.Categories,
                false,
                new[] { 32 / 2, 32 / 2, 32 / 2 },
                true,
                12,
                0.5,
                null);
        }
        //public Network DenseNet_Fast_CIFAR10()
        //{
        //    return AddDenseNet(
        //        nameof(DenseNet_Fast_CIFAR10),
        //        new[] { 1, CIFAR10DataLoader.Channels, CIFAR10DataLoader.Height, CIFAR10DataLoader.Width },
        //        CIFAR10DataLoader.Categories,
        //        false,
        //        new[] { 2, 2},
        //        false,
        //        12,
        //        0.5,
        //        null);
        //}

        /// <summary>
        /// build a DenseNet
        /// </summary>
        /// <param name="networkName"></param>
        /// <param name="xShape"></param>
        /// <param name="nbCategories"></param>
        /// <param name="subSampleInitialBlock"></param>
        /// <param name="nbConvBlocksInEachDenseBlock"></param>
        /// <param name="useBottleneckInEachConvBlock"></param>
        /// <param name="growthRate"></param>
        /// <param name="compression"> 1.0 = no compression</param>
        /// <param name="dropProbability"></param>
        /// <returns></returns>
        public Network Build(
            string networkName,
            int[] xShape, 
            int nbCategories,
            bool subSampleInitialBlock,
            int[] nbConvBlocksInEachDenseBlock,
            bool useBottleneckInEachConvBlock,
            int growthRate,
            double compression,
            double? dropProbability)
        {
            var net = BuildEmptyNetwork(networkName);

            Debug.Assert(net.Layers.Count == 0);
            net.Input(xShape[1], xShape[2], xShape[3]);
            var filtersCount = 2 * growthRate;
            if (subSampleInitialBlock)
            {
                net.Convolution(filtersCount, 7, 2, 3, Config.lambdaL2Regularization, false)
                    .BatchNorm()
                    .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                    .MaxPooling(3, 2);
            }
            else
            {
                net.Convolution(filtersCount, 3, 1, 1, Config.lambdaL2Regularization, false);
                //net.Convolution(filtersCount, 3, 1, 1, 0.0, false);
            }

            for (int denseBlockId = 0; denseBlockId < nbConvBlocksInEachDenseBlock.Length; ++denseBlockId)
            {
                AddDenseBlock(net, nbConvBlocksInEachDenseBlock[denseBlockId], growthRate, useBottleneckInEachConvBlock, dropProbability, Config.lambdaL2Regularization);
                if (denseBlockId != nbConvBlocksInEachDenseBlock.Length - 1)
                {
                    //the last dense block does not have a transition block
                    AddTransitionBlock(net, compression, Config.lambdaL2Regularization);
                }
                filtersCount = (int)Math.Round(filtersCount * compression);
            }

            //we add the classification layer part
            net
                .BatchNorm()
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .GlobalAvgPooling()
                //.Dense(nbCategories, 0.0) //!D check if lambdaL2Regularization should be 0
                .Dense(nbCategories, Config.lambdaL2Regularization)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return net;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="network"></param>
        /// <param name="nbConvBlocksInDenseBlock"></param>
        /// <param name="growthRate"></param>
        /// <param name="bottleneck"></param>
        /// <param name="dropProbability"></param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        private void AddDenseBlock(
            Network network,
            int nbConvBlocksInDenseBlock,
            int growthRate,
            bool bottleneck,
            double? dropProbability,
            double lambdaL2Regularization)
        {
            for (int convBlockId = 0; convBlockId < nbConvBlocksInDenseBlock; ++convBlockId)
            {
                var previousLayerIndex1 = network.Layers.Last().LayerIndex;
                AddConvolutionBlock(network, growthRate, bottleneck, dropProbability, lambdaL2Regularization);
                var previousLayerIndex2 = network.Layers.Last().LayerIndex;
                network.ConcatenateLayer(previousLayerIndex1, previousLayerIndex2);
            }
        }
        /// <summary>
        /// Add a Convolution block in a Dense Network
        /// </summary>
        /// <param name="network"></param>
        /// <param name="growthRate"></param>
        /// <param name="bottleneck"></param>
        /// <param name="dropProbability">optional value, if presents will add a Dropout layer at the end of the block</param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        private void AddConvolutionBlock(Network network, int growthRate, bool bottleneck, double? dropProbability, double lambdaL2Regularization)
        {
            if (bottleneck)
            {
                network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 4 * growthRate, 1, 1, 0, lambdaL2Regularization, true);
            }
            //network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, growthRate, 3, 1, 1, lambdaL2Regularization, true);
            network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, growthRate, 3, 1, 1, 0.0, true);
            if (dropProbability.HasValue)
            {
                network.Dropout(dropProbability.Value);
            }
        }
        /// <summary>
        /// Add a transition block in a Dense Net network
        /// </summary>
        /// <param name="network"></param>
        /// <param name="compression"></param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        private void AddTransitionBlock(Network network, double compression, double lambdaL2Regularization)
        {
            var filtersCount = network.Layers.Last().OutputShape(1)[1];
            network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, (int)Math.Round(filtersCount * compression), 1, 1, 0, lambdaL2Regularization, true)
                .AvgPooling(2, 2);
        }
    }
}
