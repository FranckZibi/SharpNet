﻿using System.Linq;
using SharpNet.DataAugmentation;
using SharpNet.GPU;

// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnusedMember.Global

namespace SharpNet.Networks
{
    /// <summary>
    /// Wide Residual Network support, as described in https://arxiv.org/pdf/1605.07146.pdf
    /// </summary>
    public class WideResNetBuilder : NetworkBuilder
    {

        /// <summary>
        /// The default WRN Meta Parameters for Recursion Cellular Image Classification
        /// </summary>
        /// <returns></returns>
        public static WideResNetBuilder RecursionCellularImageClassification()
        {
            var builder = new WideResNetBuilder
            {
                Config = new NetworkConfig
                    {
                        LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                        lambdaL2Regularization = 0.0005
                    }
                    .WithSGD(0.9, false)
                    .WithCyclicCosineAnnealingLearningRateScheduler(10, 2)
                ,

                //Data augmentation
                //WidthShiftRange = 0.1,
                //HeightShiftRange = 0.1,
                HorizontalFlip = true,
                VerticalFlip = true,
                //RotationRangeInDegrees = 180,
                FillMode = ImageDataGenerator.FillModeEnum.Reflect,
                AlphaCutMix = 0.0, //no CutMix
                AlphaMixup = 1.0, //with mixup
                CutoutPatchPercentage = 0.0, //no cutout

                WRN_AvgPoolingSize = 8,
                NumEpochs = 70,
                BatchSize = 128,
                WRN_DropOut = 0.0, //by default we disable dropout
                InitialLearningRate = 0.1,
            };
            return builder;
        }

        public static WideResNetBuilder WRN_Aptos2019Blindness()
        {
            return WRN_CIFAR10();
        }


        /// <summary>
        /// The default WRN Meta Parameters for CIFAR10
        /// </summary>
        /// <returns></returns>
        public static WideResNetBuilder WRN_CIFAR10()
        {
            return new WideResNetBuilder
            {
                Config = new NetworkConfig
                    {
                        LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                        lambdaL2Regularization = 0.0005
                    }
                    .WithSGD(0.9, false)
                    //.WithCifar10WideResNetLearningRateScheduler(true, true, false) : discarded on 14-aug-2019 : Cyclic annealing is better
                    .WithCyclicCosineAnnealingLearningRateScheduler(10, 2) //new default value on 14-aug-2019
                ,

                //Data augmentation
                DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
                WidthShiftRange = 0.1,
                HeightShiftRange = 0.1,
                HorizontalFlip = true,
                VerticalFlip = false,
                FillMode = ImageDataGenerator.FillModeEnum.Reflect,
                //Mixup is discarded
                AlphaMixup = 0.0,
                //We use CutMix, lambda will follow a uniform distribution in [0,1]
                AlphaCutMix = 1.0, //validated on 14-aug-2019 : +15 bps
                //Cutout discarded on 14-aug-2019: do not improve the use of CutMix
                CutoutPatchPercentage = 0.0,
                //CutoutPatchPercentage = 0.5; //validated on 04-aug-2019 for CIFAR-10: +75 bps vs no cutout (= 0.0)
                //CutoutPatchPercentage = 0.25; //discarded on 04-aug-2019 for CIFAR-10: -60 bps vs 0.5




                NumEpochs = 150, //changed on 8-aug-2019 : new default batch size : 150 (was 200)
                BatchSize = 128,
                WRN_DropOut = 0.0, //by default we disable dropout
                InitialLearningRate = 0.1,
                WRN_AvgPoolingSize = 2, //validated on 04-june-2019: +37 bps

                //DropOutAfterDenseLayer = 0.3; //discarded on 05-june-2019: -136 bps
                WRN_DropOutAfterDenseLayer = 0,
            };
        }

        /// <summary>
        /// The default WRN Meta Parameters for CIFAR100
        /// </summary>
        /// <returns></returns>
        public static WideResNetBuilder WRN_CIFAR100()
        {
            return new WideResNetBuilder
            {
                Config = new NetworkConfig
                    {
                        LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                        lambdaL2Regularization = 0.0005
                    }
                    .WithSGD(0.9, false)
                    .WithCyclicCosineAnnealingLearningRateScheduler(10, 2),

                //Data augmentation
                DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
                WidthShiftRange = 0.1,
                HeightShiftRange = 0.1,
                HorizontalFlip = true,
                VerticalFlip = false,
                FillMode = ImageDataGenerator.FillModeEnum.Reflect,
                AlphaMixup = 0.0,
                AlphaCutMix = 1.0,
                CutoutPatchPercentage = 0.0,

                NumEpochs = 150,
                BatchSize = 128,
                WRN_DropOut = 0.0,
                InitialLearningRate = 0.1,
                WRN_AvgPoolingSize = 2,
                WRN_DropOutAfterDenseLayer = 0
            };
        }

        /// <summary>
        /// 0 to disable dropout
        /// any value > 0 will enable dropout
        /// </summary>
        public double WRN_DropOut { get; set; }
        public double WRN_DropOutAfterDenseLayer { get; set; }

        public int WRN_AvgPoolingSize { get; set; }

        /// <summary>
        /// returns a Wide Residual network, as described in https://arxiv.org/pdf/1605.07146.pdf
        /// There are always 3 stages in a Wide ResNet.
        /// Number of convolutions in each stage = (depth-1)/3      (1 of them is used to change dimension)
        /// Number of convolutions in each residual block = 2       (3 for the 1st residual block of each stage)
        /// Number of residual blocks in each stage = (depth-4)/6
        /// Each residual block is in the form:
        ///     for the 1st one at each stage:
        ///         BatchNorm+Activ+Conv + BatchNorm+Activ+Conv + Conv(to change dimension) + Add
        ///     for the other residual blocks ones:
        ///         BatchNorm+Activ+Conv + BatchNorm+Activ+Conv +        Add
        /// For each stage,
        ///     if the input is of dimension
        ///         (N,C,H,W)
        ///     the output will:
        ///         (N,k*C,H,W)         => for the 1st stage
        ///         (N,2C,H/2,W/2)      => for other stages
        /// </summary>
        /// <param name="depth">total number of convolutions in the network</param>
        /// <param name="k">widening parameter</param>
        /// <param name="inputShape_CHW">input shape of a single element in format (channels, height, width)</param>
        /// <param name="categories">number of distinct categories</param>
        /// <returns></returns>

        public Network WRN(int depth, int k, int[] inputShape_CHW, int categories)
        {
            return WRN(depth, k, inputShape_CHW, categories, false);
        }
        public Network WRN_ImageNet(int depth, int k, int[] inputShape_CHW, int categories)
        {
            return WRN(depth, k, inputShape_CHW, categories, true);
        }

        public Network WRN(int depth, int k, int[] inputShape_CHW, int categories, bool reduceInputSize)
        {
            int convolutionsCountByStage = (depth - 1) / 3;
            int residualBlocksCountByStage = (convolutionsCountByStage-1) / 2;

            var networkName = "WRN-"+depth+"-"+k;
            var net = BuildEmptyNetwork(networkName);
            var config = net.Config;
            var layers = net.Layers;
            var channelCount = inputShape_CHW[0];
            var height = inputShape_CHW[1];
            var width = inputShape_CHW[2];
            net.Input(channelCount, height, width);

            if (reduceInputSize)
            {
                net.Convolution_BatchNorm_Activation(64, 7, 2, 3, config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                net.MaxPooling(2, 2);
            }

            net.Convolution(16, 3, 1, 1, config.lambdaL2Regularization, false);

            int stageC = 16* k; //number of channels for current stage
            for (int stageId = 0; stageId < 3; ++stageId)
            {
                //residualBlockId : id of the residual block in the current stage
                for (int residualBlockId = 0; residualBlockId < residualBlocksCountByStage; ++residualBlockId)
                {
                    int stride = ((residualBlockId == 0)&&(stageId != 0)) ? 2 : 1;
                    var startOfBlockLayerIndex = layers.Last().LayerIndex;
                    net.BatchNorm_Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                    if (residualBlockId == 0) // first residual block in stage
                    {
                        startOfBlockLayerIndex = layers.Last().LayerIndex;
                    }
                    net.Convolution(stageC, 3, stride, 1, config.lambdaL2Regularization, false);
                    if ((WRN_DropOut > 0.0)&& (residualBlockId != 0))
                    {
                        net.Dropout(WRN_DropOut);
                    }

                    net.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageC, 3, 1, 1, config.lambdaL2Regularization, false);
                    net.Shortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, config.lambdaL2Regularization);
                }
                stageC *= 2;
            }
            net.BatchNorm_Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            if (WRN_AvgPoolingSize>=1)
            { 
                net.AvgPooling(WRN_AvgPoolingSize, WRN_AvgPoolingSize);
            }

            if (WRN_DropOutAfterDenseLayer > 0)
            {
                net.Dense_DropOut_Activation(categories, config.lambdaL2Regularization, WRN_DropOutAfterDenseLayer, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            }
            else
            {
                net.Output(categories, config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            }
            return net;
        }
    }
}
