using System.IO;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.HyperParameters;
using SharpNet.Layers;

// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnusedMember.Global

namespace SharpNet.Networks
{
    /// <summary>
    /// Wide Residual Network support, as described in https://arxiv.org/pdf/1605.07146.pdf
    /// </summary>
    public class WideResNetSample : NetworkSample
    {
        private WideResNetSample(ISample[] samples) : base(samples)
        {
        }

        public WideResNetHyperParameters WideResNetHyperParameters => (WideResNetHyperParameters)Samples[2];
        public static WideResNetSample ValueOfWideResNetSample(string workingDirectory, string modelName)
        {
            return new WideResNetSample(new ISample[]
            {
                NetworkConfig.ValueOf(workingDirectory, ISample.SampleName(modelName, 0)),
                DataAugmentationSample.ValueOf(workingDirectory, ISample.SampleName(modelName, 1)),
                WideResNetHyperParameters.ValueOf(workingDirectory, ISample.SampleName(modelName, 2))
            });
        }

        /// <summary>
        /// default WRN Hyper-Parameters for CIFAR-10
        /// </summary>
        /// <returns></returns>
        public static WideResNetSample CIFAR10()
        {
            var config = new NetworkConfig
                {
                    LossFunction = LossFunctionEnum.CategoricalCrossentropy,
                    ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST,
                    lambdaL2Regularization = 0.0005,
                    WorkingDirectory = Path.Combine(NetworkConfig.DefaultWorkingDirectory, CIFAR10DataSet.NAME),
                    NumEpochs = 150, //changed on 8-aug-2019 : new default batch size : 150 (was 200)
                    BatchSize = 128,
                    InitialLearningRate = 0.1
                }
                .WithSGD(0.9, false)
                //.WithCifar10WideResNetLearningRateScheduler(true, true, false) : discarded on 14 - aug - 2019 : Cyclic annealing is better
                .WithCyclicCosineAnnealingLearningRateScheduler(10, 2); //new default value on 14-aug-2019

            //Data augmentation
            DataAugmentationSample da = new()
            {
                DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
                WidthShiftRangeInPercentage = 0.1,
                HeightShiftRangeInPercentage = 0.1,
                HorizontalFlip = true,
                VerticalFlip = false,
                FillMode = ImageDataGenerator.FillModeEnum.Reflect,
                //Mixup is discarded
                AlphaMixup = 0.0,
                //We use CutMix, lambda will follow a uniform distribution in [0,1]
                AlphaCutMix = 1.0, //validated on 14-aug-2019 : +15 bps
                //Cutout discarded on 14-aug-2019: do not improve the use of CutMix
                CutoutPatchPercentage = 0.0
                //CutoutPatchPercentage = 0.5; //validated on 04-aug-2019 for CIFAR-10: +75 bps vs no cutout (= 0.0)
                //CutoutPatchPercentage = 0.25; //discarded on 04-aug-2019 for CIFAR-10: -60 bps vs 0.5
            };


            WideResNetHyperParameters wideResNetHyperParameters = new()
            {
                WRN_DropOut = 0.0, //by default we disable dropout
                //Validated in 24-feb-2020 : +5bps, and independent of input picture size 
                WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling_And_GlobalMaxPooling,
                //DropOutAfterDenseLayer = 0.3; //discarded on 05-june-2019: -136 bps
                //WRN_AvgPoolingSize = 2, 
                //discarded on 24-feb-2020: using Global Average Pooling + Global Max Pooling Instead
                WRN_DropOutAfterDenseLayer = 0
            };

            return new WideResNetSample(new ISample[] { config, da, wideResNetHyperParameters});
        }

        /// <summary>
        /// default WRN Hyper-Parameters for CIFAR-100
        /// </summary>
        /// <returns></returns>
        public static WideResNetSample CIFAR100()
        {
            var config = new NetworkConfig
                {
                    LossFunction = LossFunctionEnum.CategoricalCrossentropy,
                    ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST,
                    lambdaL2Regularization = 0.0005,
                    WorkingDirectory = Path.Combine(NetworkConfig.DefaultWorkingDirectory, CIFAR100DataSet.NAME),
                    NumEpochs = 150,
                    BatchSize = 128,
                    InitialLearningRate = 0.1
            }
                .WithSGD(0.9, false)
                .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);

            //Data augmentation
            var da = new DataAugmentationSample();
            da.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT;
            da.WidthShiftRangeInPercentage = 0.1;
            da.HeightShiftRangeInPercentage = 0.1;
            da.HorizontalFlip = true;
            da.VerticalFlip = false;
            da.FillMode = ImageDataGenerator.FillModeEnum.Reflect;
            da.AlphaMixup = 0.0;
            da.AlphaCutMix = 1.0;
            da.CutoutPatchPercentage = 0.0;

            WideResNetHyperParameters wideResNetHyperParameters = new();
            wideResNetHyperParameters.WRN_DropOut = 0.0;
            wideResNetHyperParameters.WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.AveragePooling_2;
            wideResNetHyperParameters.WRN_DropOutAfterDenseLayer = 0;

            return new WideResNetSample(new ISample[] { config, da, wideResNetHyperParameters });
        }


        /// <summary>
        /// default WRN Hyper-Parameters for SVHN
        /// </summary>
        /// <returns></returns>
        public static WideResNetSample WRN_SVHN()
        {
            var config = new NetworkConfig
                {
                    LossFunction = LossFunctionEnum.CategoricalCrossentropy,
                    ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST,
                    lambdaL2Regularization = 0.0005,
                    WorkingDirectory = Path.Combine(NetworkConfig.DefaultWorkingDirectory, "SVHN"),
                    NumEpochs = 70,
                    BatchSize = 128,
                    InitialLearningRate = 0.1
            }
                .WithSGD(0.9, false)
                .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);

            //Data augmentation
            var da = new DataAugmentationSample();
            da.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT;
            da.WidthShiftRangeInPercentage = 0.1;
            da.HeightShiftRangeInPercentage = 0.1;
            da.HorizontalFlip = false;
            da.VerticalFlip = false;
            da.FillMode = ImageDataGenerator.FillModeEnum.Reflect;
            da.AlphaMixup = 0.0;
            da.AlphaCutMix = 0.0;
            da.CutoutPatchPercentage = 0.0;

            WideResNetHyperParameters wideResNetHyperParameters = new();
            wideResNetHyperParameters.WRN_DropOut = 0.0;
            //discarded on 7-march-2020 : -5bps vs GlobalAveragePooling_And_GlobalMaxPooling
            //wideResNetHyperParameters.WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.AveragePooling_2,
            //discarded on 7-march-2020 : no change vs GlobalAveragePooling_And_GlobalMaxPooling
            //wideResNetHyperParameters.WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling,
            //discarded on 7-march-2020 : -5bps vs GlobalAveragePooling_And_GlobalMaxPooling
            //wideResNetHyperParameters.WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling,
            //validated on 7-march-2020 : +5bps vs AveragePooling_2
            wideResNetHyperParameters.WRN_PoolingBeforeDenseLayer = POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling_And_GlobalMaxPooling;
            wideResNetHyperParameters.WRN_DropOutAfterDenseLayer = 0;

            return new WideResNetSample(new ISample[] { config, da, wideResNetHyperParameters });
        }


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
        /// <param name="categoryCount">number of distinct categoryCount</param>
        /// <returns></returns>

        public Network WRN(int depth, int k, int[] inputShape_CHW, int categoryCount)
        {
            // ReSharper disable once IntroduceOptionalParameters.Global
            return WRN(depth, k, inputShape_CHW, categoryCount, false);
        }
        public Network WRN_ImageNet(int depth, int k, int[] inputShape_CHW, int categoryCount)
        {
            return WRN(depth, k, inputShape_CHW, categoryCount, true);
        }
        public Network WRN(int depth, int k, int[] inputShape_CHW, int categoryCount, bool reduceInputSize)
        {
            int convolutionsCountByStage = (depth - 1) / 3;
            int residualBlocksCountByStage = (convolutionsCountByStage-1) / 2;

            var networkName = "WRN-"+depth+"-"+k;
            var network = BuildEmptyNetwork(networkName);
            var config = network.Config;
            var channelCount = inputShape_CHW[0];
            var height = inputShape_CHW[1];
            var width = inputShape_CHW[2];
            network.Input(channelCount, height, width);

            if (reduceInputSize)
            {
                network.Convolution_BatchNorm_Activation(64, 7, 2, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                network.MaxPooling(2, 2, 2, 2);
            }

            network.Convolution(16, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, false);

            int stageC = 16* k; //number of channels for current stage
            for (int stageId = 0; stageId < 3; ++stageId)
            {
                //residualBlockId : id of the residual block in the current stage
                for (int residualBlockId = 0; residualBlockId < residualBlocksCountByStage; ++residualBlockId)
                {
                    int stride = ((residualBlockId == 0)&&(stageId != 0)) ? 2 : 1;
                    var startOfBlockLayerIndex = network.LastLayerIndex;
                    network.BatchNorm_Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                    if (residualBlockId == 0) // first residual block in stage
                    {
                        startOfBlockLayerIndex = network.LastLayerIndex;
                    }
                    network.Convolution(stageC, 3, stride, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, false);
                    if ((WideResNetHyperParameters.WRN_DropOut > 0.0)&& (residualBlockId != 0))
                    {
                        network.Dropout(WideResNetHyperParameters.WRN_DropOut);
                    }

                    network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageC, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, false);
                    network.Shortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, config.lambdaL2Regularization);
                }
                stageC *= 2;
            }
            network.BatchNorm_Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);

            if (WideResNetHyperParameters.WRN_PoolingBeforeDenseLayer == POOLING_BEFORE_DENSE_LAYER.AveragePooling_2)
            {
                network.AvgPooling(2, 2, 2, 2);
            }
            else if (WideResNetHyperParameters.WRN_PoolingBeforeDenseLayer == POOLING_BEFORE_DENSE_LAYER.AveragePooling_8)
            {
                network.AvgPooling(8, 8, 8, 8);
            }
            else if (WideResNetHyperParameters.WRN_PoolingBeforeDenseLayer == POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling)
            {
                network.GlobalAvgPooling();
            }
            else if (WideResNetHyperParameters.WRN_PoolingBeforeDenseLayer == POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling_And_GlobalMaxPooling)
            {
                network.GlobalAvgPooling_And_GlobalMaxPooling();
            }
            else if (WideResNetHyperParameters.WRN_PoolingBeforeDenseLayer == POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling)
            {
                network.GlobalMaxPooling(network.Layers.Count-1);
            }
           

            if (WideResNetHyperParameters.WRN_DropOutAfterDenseLayer > 0)
            {
                network.Dense_DropOut_Activation(categoryCount, config.lambdaL2Regularization, WideResNetHyperParameters.WRN_DropOutAfterDenseLayer, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            }
            else
            {
                network.Output(categoryCount, config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            }
            return network;
        }
    }
}
