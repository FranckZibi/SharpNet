using System.Linq;
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
                WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.AveragePooling_8,
                NumEpochs = 70,
                BatchSize = 128,
                WRN_DropOut = 0.0, //by default we disable dropout
                InitialLearningRate = 0.1,
            };

            //Data augmentation
            var da = builder.Config.DataAugmentation;
            //da.WidthShiftRangeInPercentage = 0.1;
            //da.HeightShiftRangeInPercentage = 0.1;
            da.HorizontalFlip = true;
            da.VerticalFlip = true;
            //da.RotationRangeInDegrees = 180;
            da.FillMode = ImageDataGenerator.FillModeEnum.Reflect;
            da.AlphaCutMix = 0.0; //no CutMix
            da.AlphaMixup = 1.0; //with mixup
            da.CutoutPatchPercentage = 0.0; //no cutout


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
            var builder = new WideResNetBuilder
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

                NumEpochs = 150, //changed on 8-aug-2019 : new default batch size : 150 (was 200)
                BatchSize = 128,
                WRN_DropOut = 0.0, //by default we disable dropout
                InitialLearningRate = 0.1,

                //Validated in 24-feb-2020 : +5bps, and independent of input picture size 
                WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling_And_GlobalMaxPooling,

                //discarded on 24-feb-2020: using Global Average Pooling + Global Max Pooling Instead
                //WRN_AvgPoolingSize = 2, 

                //DropOutAfterDenseLayer = 0.3; //discarded on 05-june-2019: -136 bps
                WRN_DropOutAfterDenseLayer = 0,
            };

            //Data augmentation
            var da = builder.Config.DataAugmentation;
            da.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT;
            da.WidthShiftRangeInPercentage = 0.1;
            da.HeightShiftRangeInPercentage = 0.1;
            da.HorizontalFlip = true;
            da.VerticalFlip = false;
            da.FillMode = ImageDataGenerator.FillModeEnum.Reflect;
            //Mixup is discarded
            da.AlphaMixup = 0.0;
            //We use CutMix, lambda will follow a uniform distribution in [0,1]
            da.AlphaCutMix = 1.0; //validated on 14-aug-2019 : +15 bps
            //Cutout discarded on 14-aug-2019: do not improve the use of CutMix
            da.CutoutPatchPercentage = 0.0;
            //da.CutoutPatchPercentage = 0.5; //validated on 04-aug-2019 for CIFAR-10: +75 bps vs no cutout (= 0.0)
            //da.CutoutPatchPercentage = 0.25; //discarded on 04-aug-2019 for CIFAR-10: -60 bps vs 0.5


            return builder;
        }

        /// <summary>
        /// The default WRN Meta Parameters for CIFAR100
        /// </summary>
        /// <returns></returns>
        public static WideResNetBuilder WRN_CIFAR100()
        {
            var builder = new WideResNetBuilder
            {
                Config = new NetworkConfig
                    {
                        LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                        lambdaL2Regularization = 0.0005
                    }
                    .WithSGD(0.9, false)
                    .WithCyclicCosineAnnealingLearningRateScheduler(10, 2),

                NumEpochs = 150,
                BatchSize = 128,
                WRN_DropOut = 0.0,
                InitialLearningRate = 0.1,
                WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.AveragePooling_2,
                WRN_DropOutAfterDenseLayer = 0
            };

            //Data augmentation
            var da = builder.Config.DataAugmentation;
            da.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT;
            da.WidthShiftRangeInPercentage = 0.1;
            da.HeightShiftRangeInPercentage = 0.1;
            da.HorizontalFlip = true;
            da.VerticalFlip = false;
            da.FillMode = ImageDataGenerator.FillModeEnum.Reflect;
            da.AlphaMixup = 0.0;
            da.AlphaCutMix = 1.0;
            da.CutoutPatchPercentage = 0.0;

            return builder;
        }


        /// <summary>
        /// The default WRN Meta Parameters for SVHN
        /// </summary>
        /// <returns></returns>
        public static WideResNetBuilder WRN_SVHN()
        {
            var builder = new WideResNetBuilder
                          {
                              Config = new NetworkConfig
                                       {
                                           LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                                           lambdaL2Regularization = 0.0005
                                       }
                                  .WithSGD(0.9, false)
                                  .WithCyclicCosineAnnealingLearningRateScheduler(10, 2),

                              NumEpochs = 70,
                              BatchSize = 128,
                              WRN_DropOut = 0.0,
                              InitialLearningRate = 0.1,

                              //discarded on 7-march-2020 : -5bps vs GlobalAveragePooling_And_GlobalMaxPooling
                              //WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.AveragePooling_2,
                              //discarded on 7-march-2020 : no change vs GlobalAveragePooling_And_GlobalMaxPooling
                              //WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling,
                              //discarded on 7-march-2020 : -5bps vs GlobalAveragePooling_And_GlobalMaxPooling
                              //WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling,
                              //validated on 7-march-2020 : +5bps vs AveragePooling_2
                              WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling_And_GlobalMaxPooling,
                
                              WRN_DropOutAfterDenseLayer = 0
                          };

            //Data augmentation
            var da = builder.Config.DataAugmentation;
            da.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT;
            da.WidthShiftRangeInPercentage = 0.1;
            da.HeightShiftRangeInPercentage = 0.1;
            da.HorizontalFlip = false;
            da.VerticalFlip = false;
            da.FillMode = ImageDataGenerator.FillModeEnum.Reflect;
            da.AlphaMixup = 0.0;
            da.AlphaCutMix = 0.0;
            da.CutoutPatchPercentage = 0.0;

            return builder;
        }


        /// <summary>
        /// 0 to disable dropout
        /// any value > 0 will enable dropout
        /// </summary>
        public double WRN_DropOut { get; set; }
        public double WRN_DropOutAfterDenseLayer { get; set; }
        
        public POOLING_BEFORE_DENSE_LAYER WRN_PoolingBeforeDenseLayer { get; set; } = POOLING_BEFORE_DENSE_LAYER.AveragePooling_2;

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
                net.MaxPooling(2, 2, 2);
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

            if (WRN_PoolingBeforeDenseLayer == POOLING_BEFORE_DENSE_LAYER.AveragePooling_2)
            {
                net.AvgPooling(2, 2, 2);
            }
            else if (WRN_PoolingBeforeDenseLayer == POOLING_BEFORE_DENSE_LAYER.AveragePooling_8)
            {
                net.AvgPooling(8, 8, 8);
            }
            else if (WRN_PoolingBeforeDenseLayer == POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling)
            {
                net.GlobalAvgPooling();
            }
            else if (WRN_PoolingBeforeDenseLayer == POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling_And_GlobalMaxPooling)
            {
                net.GlobalAvgPooling_And_GlobalMaxPooling();
            }
            else if (WRN_PoolingBeforeDenseLayer == POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling)
            {
                net.GlobalMaxPooling(net.Layers.Count-1);
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
