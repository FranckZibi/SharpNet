using SharpNet.DataAugmentation;
using SharpNet.GPU;
using SharpNet.Layers;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks;

public class WideResNetNetworkSample : NetworkSample
{

    // ReSharper disable once EmptyConstructor
    public WideResNetNetworkSample()
    {
    }
    
    #region Hyper-Parameters
    /// <summary>
    /// 0 to disable dropout
    /// any value > 0 will enable dropout
    /// </summary>
    public double WRN_DropOut;
    public double WRN_DropOutAfterLinearLayer;
    public POOLING_BEFORE_DENSE_LAYER WRN_PoolingBeforeLinearLayer = POOLING_BEFORE_DENSE_LAYER.AveragePooling_2;
    #endregion



    /// <summary>
    /// default WRN Hyper-Parameters for CIFAR-10
    /// </summary>
    /// <returns></returns>
    public static WideResNetNetworkSample CIFAR10()
    {
        var networkSample = (WideResNetNetworkSample)new WideResNetNetworkSample()
        {
            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
            ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST,
            //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, CIFAR10DataSet.NAME),
            num_epochs = 150, //changed on 8-aug-2019 : new default batch size : 150 (was 200)
            BatchSize = 128,

            //specific to WideResNetNetworkSample
            WRN_DropOut = 0.0, //by default we disable dropout
                               //Validated in 24-feb-2020 : +5bps, and independent of input picture size 
            WRN_PoolingBeforeLinearLayer = POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling_And_GlobalMaxPooling,
            //DropOutAfterLinearLayer = 0.3; //discarded on 05-june-2019: -136 bps
            //WRN_AvgPoolingSize = 2, 
            //discarded on 24-feb-2020: using Global Average Pooling + Global Max Pooling Instead
            WRN_DropOutAfterLinearLayer = 0,

            //Data augmentation
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
            WidthShiftRangeInPercentage = 0.1,
            HeightShiftRangeInPercentage = 0.1,
            HorizontalFlip = true,
            VerticalFlip = false,
            FillMode = ImageDataGenerator.FillModeEnum.Reflect,
            //MixUp is discarded
            AlphaMixUp = 0.0,
            //We use CutMix, lambda will follow a uniform distribution in [0,1]
            AlphaCutMix = 1.0, //validated on 14-aug-2019 : +15 bps
                               //Cutout discarded on 14-aug-2019: do not improve the use of CutMix
            CutoutPatchPercentage = 0.0,
            //CutoutPatchPercentage = 0.5; //validated on 04-aug-2019 for CIFAR-10: +75 bps vs no cutout (= 0.0)
            //CutoutPatchPercentage = 0.25; //discarded on 04-aug-2019 for CIFAR-10: -60 bps vs 0.5

        }
            .WithSGD(0.1, 0.9, 0.0005, false)
            //.WithCifar10WideResNetLearningRateScheduler(true, true, false) : discarded on 14 - aug - 2019 : Cyclic annealing is better
            .WithCyclicCosineAnnealingLearningRateScheduler(10, 2); //new default value on 14-aug-2019

        return networkSample;
    }

    /// <summary>
    /// default WRN Hyper-Parameters for CIFAR-100
    /// </summary>
    /// <returns></returns>
    public static WideResNetNetworkSample CIFAR100()
    {
        var config = (WideResNetNetworkSample)new WideResNetNetworkSample()
        {
            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
            ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST,
            num_epochs = 150,
            BatchSize = 128,

            //wideResNetNetworkSample.
            WRN_DropOut = 0.0,
            WRN_PoolingBeforeLinearLayer = POOLING_BEFORE_DENSE_LAYER.AveragePooling_2,
            WRN_DropOutAfterLinearLayer = 0,

            //Data augmentation
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
            WidthShiftRangeInPercentage = 0.1,
            HeightShiftRangeInPercentage = 0.1,
            HorizontalFlip = true,
            VerticalFlip = false,
            FillMode = ImageDataGenerator.FillModeEnum.Reflect,
            AlphaMixUp = 0.0,
            AlphaCutMix = 1.0,
            CutoutPatchPercentage = 0.0,
        }
            .WithSGD(0.1, 0.9, 0.0005, false)
            .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
        return config;
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
    /// <param name="workingDirectory"></param>
    /// <param name="depth">total number of convolutions in the network</param>
    /// <param name="k">widening parameter</param>
    /// <param name="inputShape_CHW">input shape of a single element in format (channels, height, width)</param>
    /// <param name="numClass">number of distinct numClass</param>
    /// <returns></returns>
    public Network WRN(string workingDirectory, int depth, int k, int[] inputShape_CHW, int numClass)
    {
        // ReSharper disable once IntroduceOptionalParameters.Global
        return WRN(workingDirectory, depth, k, inputShape_CHW, numClass, false);
    }
    public Network WRN(string workingDirectory, int depth, int k, int[] inputShape_CHW, int numClass, bool reduceInputSize)
    {
        int convolutionsCountByStage = (depth - 1) / 3;
        int residualBlocksCountByStage = (convolutionsCountByStage - 1) / 2;

        var networkName = "WRN-" + depth + "-" + k;
        var network = BuildNetworkWithoutLayers(workingDirectory, networkName);
        var channelCount = inputShape_CHW[0];
        var height = inputShape_CHW[1];
        var width = inputShape_CHW[2];
        network.Input(channelCount, height, width);

        if (reduceInputSize)
        {
            network.Convolution_BatchNorm_Activation(64, 7, 2, ConvolutionLayer.PADDING_TYPE.SAME, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            network.MaxPooling(2, 2, 2, 2);
        }

        network.Convolution(16, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, false);

        int stageC = 16 * k; //number of channels for current stage
        for (int stageId = 0; stageId < 3; ++stageId)
        {
            //residualBlockId : id of the residual block in the current stage
            for (int residualBlockId = 0; residualBlockId < residualBlocksCountByStage; ++residualBlockId)
            {
                int stride = ((residualBlockId == 0) && (stageId != 0)) ? 2 : 1;
                var startOfBlockLayerIndex = network.LastLayerIndex;
                network.BatchNorm_Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                if (residualBlockId == 0) // first residual block in stage
                {
                    startOfBlockLayerIndex = network.LastLayerIndex;
                }
                network.Convolution(stageC, 3, stride, ConvolutionLayer.PADDING_TYPE.SAME, false);
                if ((WRN_DropOut > 0.0) && (residualBlockId != 0))
                {
                    network.Dropout(WRN_DropOut);
                }

                network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageC, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, false);
                network.Shortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride);
            }
            stageC *= 2;
        }
        network.BatchNorm_Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);

        if (WRN_PoolingBeforeLinearLayer == POOLING_BEFORE_DENSE_LAYER.AveragePooling_2)
        {
            network.AvgPooling(2, 2, 2, 2);
        }
        else if (WRN_PoolingBeforeLinearLayer == POOLING_BEFORE_DENSE_LAYER.AveragePooling_8)
        {
            network.AvgPooling(8, 8, 8, 8);
        }
        else if (WRN_PoolingBeforeLinearLayer == POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling)
        {
            network.GlobalAvgPooling();
        }
        else if (WRN_PoolingBeforeLinearLayer == POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling_And_GlobalMaxPooling)
        {
            network.GlobalAvgPooling_And_GlobalMaxPooling();
        }
        else if (WRN_PoolingBeforeLinearLayer == POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling)
        {
            network.GlobalMaxPooling(network.Layers.Count - 1);
        }


        if (WRN_DropOutAfterLinearLayer > 0)
        {
            network.Linear(numClass, true, false)
                .Dropout(WRN_DropOutAfterLinearLayer)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
        }
        else
        {
            network.Linear(numClass, true, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
        }
        return network;
    }
}
