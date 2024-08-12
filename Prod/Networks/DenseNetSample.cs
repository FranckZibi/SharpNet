using System;
using System.Diagnostics;
using System.Linq;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;

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

    public class DenseNetNetworkSample : NetworkSample
    {
        // ReSharper disable once MemberCanBePrivate.Global
        public DenseNetNetworkSample()
        {
        }

        public static readonly string Cifar10WorkingDirectory = System.IO.Path.Combine(DefaultWorkingDirectory, CIFAR10DataSet.NAME);
        public static readonly string Cifar100WorkingDirectory = System.IO.Path.Combine(DefaultWorkingDirectory, CIFAR100DataSet.NAME);
        public static readonly string CancelWorkingDirectory = System.IO.Path.Combine(DefaultWorkingDirectory, "Cancel");

        public static DenseNetNetworkSample CIFAR10()
        {
            var config = (DenseNetNetworkSample)new DenseNetNetworkSample
            {
                    LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                    lambdaL2Regularization = 1e-4,
                    num_epochs = 300,
                    BatchSize = 64,
                    InitialLearningRate = 0.1,

                    //Data AUgmentation
                    DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
                    WidthShiftRangeInPercentage = 0.1,
                    HeightShiftRangeInPercentage = 0.1,
                    HorizontalFlip = true,
                    VerticalFlip = false,
                    FillMode = ImageDataGenerator.FillModeEnum.Reflect,
                    //by default we use a cutout of 1/2 of the image width
                    CutoutPatchPercentage = 0.5

            }
                .WithSGD(0.9, true)
                .WithCifar10DenseNetLearningRateScheduler(false, true, false);

            return config;

        }

        public static Network DenseNet_12_40(NetworkSample sample)
        {
            return Build(
                sample,
                nameof(DenseNet_12_40) + "_" + CIFAR10DataSet.NAME,
                new[] { 1, CIFAR10DataSet.Shape_CHW[0], CIFAR10DataSet.Shape_CHW[1], CIFAR10DataSet.Shape_CHW[2]},
                CIFAR10DataSet.NumClass,
                false,
                new[] { 12, 12, 12 },
                false,
                12,
                1.0,
                null);
        }
        public static Network DenseNetBC_12_100(NetworkSample sample)
        {
            return Build(
                sample,
                nameof(DenseNetBC_12_100) + "_" + CIFAR10DataSet.NAME,
                new[] { 1, CIFAR10DataSet.Shape_CHW[0], CIFAR10DataSet.Shape_CHW[1], CIFAR10DataSet.Shape_CHW[2] },
                CIFAR10DataSet.NumClass,
                false,
                new[] { 32 / 2, 32 / 2, 32 / 2 },
                true,
                12,
                0.5,
                null);
        }

        /// <summary>
        /// build a DenseNet
        /// </summary>
        /// <param name="sample"></param>
        /// <param name="networkName"></param>
        /// <param name="xShape"></param>
        /// <param name="numClass"></param>
        /// <param name="subSampleInitialBlock"></param>
        /// <param name="nbConvBlocksInEachDenseBlock"></param>
        /// <param name="useBottleneckInEachConvBlock"></param>
        /// <param name="growthRate"></param>
        /// <param name="compression"> 1.0 = no compression</param>
        /// <param name="dropoutRate"></param>
        /// <returns></returns>
        private static Network Build(
            NetworkSample sample,
            string networkName,
            int[] xShape, 
            int numClass,
            bool subSampleInitialBlock,
            int[] nbConvBlocksInEachDenseBlock,
            bool useBottleneckInEachConvBlock,
            int growthRate,
            double compression,
            double? dropoutRate)
        {
            var net = sample.BuildNetworkWithoutLayers(Cifar10WorkingDirectory, networkName);

            Debug.Assert(net.Layers.Count == 0);
            net.Input(xShape[1], xShape[2], xShape[3]);
            var filtersCount = 2 * growthRate;
            var lambdaL2Regularization = sample.lambdaL2Regularization;

            if (subSampleInitialBlock)
            {
                net.Convolution(filtersCount, 7, 2, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, false)
                    .BatchNorm(0.99, 1e-5)
                    .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                    .MaxPooling(3, 3, 2, 2);
            }
            else
            {
                net.Convolution(filtersCount, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, false);
                //net.Convolution(filtersCount, 3, 1, 1, 0.0, false);
            }

            for (int denseBlockId = 0; denseBlockId < nbConvBlocksInEachDenseBlock.Length; ++denseBlockId)
            {
                AddDenseBlock(net, nbConvBlocksInEachDenseBlock[denseBlockId], growthRate, useBottleneckInEachConvBlock, dropoutRate, lambdaL2Regularization);
                if (denseBlockId != nbConvBlocksInEachDenseBlock.Length - 1)
                {
                    //the last dense block does not have a transition block
                    AddTransitionBlock(net, compression, lambdaL2Regularization);
                }
                filtersCount = (int)Math.Round(filtersCount * compression);
            }

            //we add the classification layer part
            net
                .BatchNorm(0.99, 1e-5)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .GlobalAvgPooling()
                //.Dense(numClass, 0.0) //!D check if lambdaL2Regularization should be 0
                .Dense(numClass, lambdaL2Regularization, false)
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
        /// <param name="dropoutRate"></param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        private static void AddDenseBlock(
            Network network,
            int nbConvBlocksInDenseBlock,
            int growthRate,
            bool bottleneck,
            double? dropoutRate,
            double lambdaL2Regularization)
        {
            for (int convBlockId = 0; convBlockId < nbConvBlocksInDenseBlock; ++convBlockId)
            {
                var previousLayerIndex1 = network.LastLayerIndex;
                AddConvolutionBlock(network, growthRate, bottleneck, dropoutRate, lambdaL2Regularization);
                var previousLayerIndex2 = network.LastLayerIndex;
                network.ConcatenateLayer(previousLayerIndex1, previousLayerIndex2);
            }
        }
        /// <summary>
        /// Add a Convolution block in a Dense Network
        /// </summary>
        /// <param name="network"></param>
        /// <param name="growthRate"></param>
        /// <param name="bottleneck"></param>
        /// <param name="dropoutRate">optional value, if presents will add a Dropout layer at the end of the block</param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        private static void AddConvolutionBlock(Network network, int growthRate, bool bottleneck, double? dropoutRate, double lambdaL2Regularization)
        {
            if (bottleneck)
            {
                network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 4 * growthRate, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true);
            }
            //network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, growthRate, 3, 1, 1, lambdaL2Regularization, true);
            network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, growthRate, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true);
            if (dropoutRate.HasValue)
            {
                network.Dropout(dropoutRate.Value);
            }
        }
        /// <summary>
        /// Add a transition block in a Dense Net network
        /// </summary>
        /// <param name="network"></param>
        /// <param name="compression"></param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        private static void AddTransitionBlock(Network network, double compression, double lambdaL2Regularization)
        {
            var filtersCount = network.Layers.Last().OutputShape(1)[1];
            network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, (int)Math.Round(filtersCount * compression), 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .AvgPooling(2, 2, 2, 2);
        }
    }
}
