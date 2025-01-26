using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Models;
// ReSharper disable UnusedMember.Global
// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable ConvertToConstant.Global
// ReSharper disable FieldCanBeMadeReadOnly.Global

namespace SharpNet.Networks;

public class EfficientNetNetworkSample : NetworkSample
{
    // ReSharper disable once EmptyConstructor
    public EfficientNetNetworkSample()
    {
    }

    #region Hyperparameters
    public cudnnActivationMode_t DefaultActivation = cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH;
    public cudnnActivationMode_t LastActivationLayer = cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX;
    public double BatchNormMomentum = 0.99;
    public double BatchNormEpsilon = 0.001;
    /// <summary>
    /// dropout rate before final classifier layer
    /// </summary>
    public float TopDropoutRate = 0.2f;
    /// <summary>
    /// dropout rate at skip connections
    /// </summary>
    public float SkipConnectionsDropoutRate = 0.2f;
    /// <summary>
    /// name of the trained network to load the weights from.
    /// used for transfer learning
    /// </summary>
    public string WeightForTransferLearning = "";
    /// <summary>
    /// number of elements in the Default MobileBlocksDescription List
    /// -1 means all 7 blocks
    /// </summary>
    public int DefaultMobileBlocksDescriptionCount = -1;

    public enum enum_efficientnet_name { EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7 };

    public enum_efficientnet_name EfficientNetName = enum_efficientnet_name.EfficientNetB0;

    #endregion



    /// <summary>
    /// The default EfficientNet Hyperparameters for ImageNet
    /// </summary>
    /// <returns></returns>
    public static EfficientNetNetworkSample ImageNet()
    {
        var config = (EfficientNetNetworkSample)new EfficientNetNetworkSample()
        {
            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
            CompatibilityMode = CompatibilityModeEnum.TensorFlow,
            lambdaL2Regularization = 0.0005,
            //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, "ImageNet"),
            num_epochs = 150,
            BatchSize = 128,
            InitialLearningRate = 0.1,

            //specific to EfficientNetNetworkSample
            BatchNormMomentum = 0.99,
            BatchNormEpsilon = 0.001,
        }
            .WithSGD(0.9, false)
            //.WithCifar10WideResNetLearningRateScheduler(true, true, false) : discarded on 14-aug-2019 : Cyclic annealing is better
            .WithCyclicCosineAnnealingLearningRateScheduler(10, 2); //new default value on 14-aug-2019


        return config;
    }

    /// <summary>
    /// The default EfficientNet Hyperparameters for Cancel Dataset
    /// </summary>
    /// <returns></returns>
    public static EfficientNetNetworkSample Cancel()
    {
        var config = (EfficientNetNetworkSample)new EfficientNetNetworkSample
        {
            LossFunction = EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy,
            CompatibilityMode = CompatibilityModeEnum.TensorFlow,
            lambdaL2Regularization = 0.0005,
            //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, "Cancel"),
            AlwaysUseFullTestDataSetForLossAndAccuracy = false,
            num_epochs = 150,
            BatchSize = 128,
            InitialLearningRate = 0.03,

            //EfficientNetNetworkSample
            BatchNormMomentum = 0.99,
            BatchNormEpsilon = 0.001,

            //Data Augmentation
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_IMAGENET,
            //da.HeightShiftRangeInPercentage = 0.1;
            //da.WidthShiftRangeInPercentage = 0.1;
            HorizontalFlip = false,
            VerticalFlip = false,
            Rotate180Degrees = true,
            FillMode = ImageDataGenerator.FillModeEnum.Reflect,
            AlphaMixUp = 0.0, //MixUp is discarded
            AlphaCutMix = 0.0, //CutMix is discarded
            CutoutPatchPercentage = 0.1
        }
            .WithSGD(0.9, false)
            .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
        return config;
    }


    /// <summary>
    /// The default EfficientNet Hyperparameters for CIFAR10
    /// </summary>
    /// <returns></returns>
    public static EfficientNetNetworkSample CIFAR10()
    {
        var config = (EfficientNetNetworkSample)new EfficientNetNetworkSample()
        {
            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
            CompatibilityMode = CompatibilityModeEnum.TensorFlow,
            lambdaL2Regularization = 0.0005,
            //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, CIFAR10DataSet.NAME),
            num_epochs = 150,
            BatchSize = 128,
            InitialLearningRate = 0.1,

            //Data augmentation
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
            WidthShiftRangeInPercentage = 0.1,
            HeightShiftRangeInPercentage = 0.1,
            HorizontalFlip = true,
            VerticalFlip = false,
            FillMode = ImageDataGenerator.FillModeEnum.Reflect,
            //MixUp is discarded
            AlphaMixUp = 0.0,
            AlphaCutMix = 1.0,
            CutoutPatchPercentage = 0.0
        }
            .WithSGD(0.9, false)
            .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
        return config;

    }

    /// <param name="widthCoefficient">scaling coefficient for network width</param>
    /// <param name="depthCoefficient">scaling coefficient for network depth</param>
    /// <param name="topDropoutRate">dropout rate before final classifier layer</param>
    /// <param name="blocks">A list of BlockDescription to construct block modules</param>
    /// <param name="skipConnectionsDropoutRate">dropout rate at skip connections</param>
    /// <param name="depthDivisor"></param>
    /// <param name="network"></param>
    /// <param name="includeTop">whether to include the fully-connected layer at the top of the network</param>
    /// <param name="weights"></param>
    /// <param name="inputShape_CHW">optional shape tuple, only to be specified if `includeTop` is False.</param>
    /// <param name="pooling">pooling: optional pooling mode for feature extraction (used only when 'includeTop' is false)
    /// when `includeTop` is false:
    ///       - `None`
    ///                 means that the output of the model will be the 4D tensor output of the last convolutional layer.
    ///       - `avg`
    ///                 means that global average pooling will be applied to the output of the last convolutional layer,
    ///                 and thus the output of the model will be a 2D tensor.
    ///       - `max`
    ///                 means that global max pooling will be applied
    /// </param>
    /// <param name="numClass">optional number of classes to classify images  into, only to be specified if `includeTop` is True, and if no `weights` argument is specified</param>
    /// <returns></returns>
    private Network  EfficientNet(
            float widthCoefficient,
            float depthCoefficient,
            float topDropoutRate,
            float skipConnectionsDropoutRate,
            int depthDivisor,
            List<MobileBlocksDescription> blocks,
            Network network, /* network without layers */
            bool includeTop,
            string weights,
            int[] inputShape_CHW,
            POOLING_BEFORE_DENSE_LAYER pooling,
            int numClass //= 1000
        )
    {
        if (network.Layers.Count != 0)
        {
            throw new ArgumentException($"The input network has {network.Layers.Count} layers");
        }

        //dropoutRate = dropConnectRate = 0; //to disable all Dropout layers

        blocks = blocks.Select(x => x.ApplyScaling(widthCoefficient, depthDivisor, depthCoefficient)).ToList();
        //var network = BuildNetworkWithoutLayers(workingDirectory, modelName);
        var config = network.Sample;

        network.Input(inputShape_CHW);


        //Build stem
        var stemChannels = MobileBlocksDescription.RoundFilters(32, widthCoefficient, depthDivisor);
        const int stemStride = 2;

        //if (Math.Min(height, width) <= 32)
        //{
        //    stemStride = 1;
        //}
        network.Convolution(stemChannels, 3, stemStride, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, false, "stem_conv")
            .BatchNorm(BatchNormMomentum, BatchNormEpsilon, "stem_bn")
            .Activation(DefaultActivation, "stem_activation");

        //Build blocks
        int numBlocksTotal = blocks.Select(x => x.NumRepeat).Sum();
        int blockNum = 0;
        for (int idx = 0; idx < blocks.Count; ++idx)
        {
            var block_arg = blocks[idx];
            for (int bidx = 0; bidx < block_arg.NumRepeat; ++bidx)
            {
                var layerPrefix = "block" + (idx + 1) + (char)('a' + bidx) + "_";
                var blockSkipConnectionsDropoutRate = (skipConnectionsDropoutRate * blockNum) / numBlocksTotal;
                //The first block needs to take care of stride and filter size increase.
                var mobileBlocksDescription = bidx == 0 ? block_arg : block_arg.WithStride(1, 1);
                AddMBConvBlock(network, mobileBlocksDescription, blockSkipConnectionsDropoutRate, layerPrefix);
                ++blockNum;
            }
        }

        //# Build top
        var outputChannelsTop = MobileBlocksDescription.RoundFilters(1280, widthCoefficient, depthDivisor);
        network.Convolution(outputChannelsTop, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, false, "top_conv");
        network.BatchNorm(BatchNormMomentum, BatchNormEpsilon, "top_bn");
        network.Activation(DefaultActivation, "top_activation");
        if (includeTop)
        {
            network.GlobalAvgPooling("avg_pool");
            if (topDropoutRate > 0)
            {
                network.Dropout(topDropoutRate, "top_dropout");
            }

            network.Flatten();
            network.Linear(numClass, true, config.lambdaL2Regularization, false, "probs");
            network.Activation(LastActivationLayer);
        }
        else
        {
            if (pooling == POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling)
            {
                network.GlobalAvgPooling("avg_pool");
            }
            else if (pooling == POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling)
            {
                network.GlobalMaxPooling("max_pool");
            }
        }

        //We load weights if needed
        if (0 == string.Compare(weights, "imagenet", StringComparison.OrdinalIgnoreCase))
        {
            if (numClass != 1000)
            {
                throw new ArgumentException("numClass must be 1000 for " + weights);
            }
            var modelPath = GetKerasModelPath(network.ModelName + "_weights_tf_dim_ordering_tf_kernels_autoaugment.h5");
            if (!File.Exists(modelPath))
            {
                throw new ArgumentException("missing " + weights + " model file " + modelPath);
            }
            network.LoadParametersFromH5File(modelPath, CompatibilityModeEnum.TensorFlow);
        }

        network.FreezeSelectedLayers();
        return network;
    }

    /// <summary>
    /// Mobile Inverted Residual Bottleneck
    /// </summary>
    /// <param name="network"></param>
    /// <param name="block_args"></param>
    /// <param name="blockSkipConnectionsDropoutRate"></param>
    /// <param name="layerPrefix"></param>
    private void AddMBConvBlock(Network network, MobileBlocksDescription block_args, float blockSkipConnectionsDropoutRate, string layerPrefix)
    {
        int in_channels = network.Layers.Last().OutputShape(1)[1];
        var inputLayerIndex = network.LastLayerIndex;

        //Expansion phase
        var config = network.Sample;
        var hidden_dim = in_channels * block_args.ExpandRatio;
        if (block_args.ExpandRatio != 1)
        {
            network.Convolution(hidden_dim, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, false, layerPrefix + "expand_conv")
                .BatchNorm(BatchNormMomentum, BatchNormEpsilon, layerPrefix + "expand_bn")
                .Activation(DefaultActivation, layerPrefix + "expand_activation");
        }

        //Depthwise Convolution
        network.DepthwiseConvolution(block_args.KernelSize, block_args.ColStride, ConvolutionLayer.PADDING_TYPE.SAME, 1, config.lambdaL2Regularization, false, layerPrefix + "dwconv")
            .BatchNorm(BatchNormMomentum, BatchNormEpsilon, layerPrefix + "bn")
            .Activation(DefaultActivation, layerPrefix + "activation");

        //Squeeze and Excitation phase
        bool hasSe = block_args.SeRatio > 0 && block_args.SeRatio <= 1.0;
        if (hasSe)
        {
            var xLayerIndex = network.LastLayerIndex;
            var num_reduced_filters = Math.Max(1, (int)(in_channels * block_args.SeRatio));
            network.GlobalAvgPooling(layerPrefix + "se_squeeze");
            network.Convolution(num_reduced_filters, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, true, layerPrefix + "se_reduce")
                .Activation(DefaultActivation);
            network.Convolution(hidden_dim, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, true, layerPrefix + "se_expand")
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);
            network.MultiplyLayer(xLayerIndex, network.LastLayerIndex /*diagonal matrix */, layerPrefix + "se_excite");
        }

        //Output phase
        network.Convolution(block_args.OutputFilters, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, false, layerPrefix + "project_conv");
        network.BatchNorm(BatchNormMomentum, BatchNormEpsilon, layerPrefix + "project_bn");
        if (block_args.IdSkip && block_args.RowStride == 1 && block_args.ColStride == 1 && block_args.OutputFilters == in_channels)
        {
            if (blockSkipConnectionsDropoutRate > 0.000001)
            {
                network.Dropout(blockSkipConnectionsDropoutRate, layerPrefix + "drop");
            }
            network.AddLayer(network.LastLayerIndex, inputLayerIndex, layerPrefix + "add");
        }
    }

    /// <summary>
    /// construct an EfficientNet B0 network for CIFAR10 training
    /// if weight is provided (ex: imagenet):
    ///      will load the weight from the provided source,
    ///      and will set the network category count to 10
    ///      (resetting the last Linear layer weights if required to have 10 output numClass)
    /// </summary>
    /// <param name="weight"></param>
    /// <param name="inputShape_CHW"></param>
    /// <returns></returns>
    public Network EfficientNetB0_CIFAR10(string weight, int[] inputShape_CHW)
    {
        var net = EfficientNetB0(DenseNetNetworkSample.Cifar10WorkingDirectory, true, weight, inputShape_CHW);
        Model.Log.Info("setting number of output numClass to 10");
        net.SetNumClass(10);
        return net;
    }

    public override void BuildLayers(Network nn, AbstractDatasetSample datasetSample)
    {
        var mobileBlocksDescriptions = MobileBlocksDescription.Default();
        if (DefaultMobileBlocksDescriptionCount != -1)
        {
            mobileBlocksDescriptions = mobileBlocksDescriptions.Take(DefaultMobileBlocksDescriptionCount).ToList();
        }


        var inputShapeOfSingleElement = datasetSample.X_Shape(1).Skip(1).ToArray();
        switch (EfficientNetName)
        {
            case enum_efficientnet_name.EfficientNetB0:
                EfficientNetB0(nn, mobileBlocksDescriptions, true, WeightForTransferLearning, inputShapeOfSingleElement, datasetSample.NumClass);
                break;
            case enum_efficientnet_name.EfficientNetB1:
                EfficientNetB1(nn, mobileBlocksDescriptions, true, WeightForTransferLearning, inputShapeOfSingleElement, datasetSample.NumClass);
                break;
            case enum_efficientnet_name.EfficientNetB2:
                EfficientNetB2(nn, mobileBlocksDescriptions, true, WeightForTransferLearning, inputShapeOfSingleElement, datasetSample.NumClass);
                break;
            case enum_efficientnet_name.EfficientNetB3:
                EfficientNetB3(nn, mobileBlocksDescriptions, true, WeightForTransferLearning, inputShapeOfSingleElement, datasetSample.NumClass);
                break;
            case enum_efficientnet_name.EfficientNetB4:
                EfficientNetB4(nn, mobileBlocksDescriptions, true, WeightForTransferLearning, inputShapeOfSingleElement, datasetSample.NumClass);
                break;
            case enum_efficientnet_name.EfficientNetB5:
                EfficientNetB5(nn, mobileBlocksDescriptions, true, WeightForTransferLearning, inputShapeOfSingleElement, datasetSample.NumClass);
                break;
            case enum_efficientnet_name.EfficientNetB6:
                EfficientNetB6(nn, mobileBlocksDescriptions, true, WeightForTransferLearning, inputShapeOfSingleElement, datasetSample.NumClass);
                break;
            case enum_efficientnet_name.EfficientNetB7:
                EfficientNetB7(nn, mobileBlocksDescriptions, true, WeightForTransferLearning, inputShapeOfSingleElement, datasetSample.NumClass);
                break;
            default:
                throw new ArgumentException($"unknown network name {EfficientNetName}");
        }

    }

    public Network EfficientNetB0(
        string workingDirectory,
        bool includeTop, //= True,
        string weights, //= 'imagenet',
        int[] inputShape_CHW, //= None,
        int numClass = 1000,
        POOLING_BEFORE_DENSE_LAYER pooling = POOLING_BEFORE_DENSE_LAYER.NONE)
    {
        return EfficientNetB0(
            BuildNetworkWithoutLayers(workingDirectory, "efficientnet-b0"),
            MobileBlocksDescription.Default(),
            includeTop,
            weights,
            inputShape_CHW,
            numClass,
            pooling);
    }

    public Network EfficientNetB0(
            Network network, /* network without layers */
            List<MobileBlocksDescription> blocks,
            bool includeTop, //= True,
            string weights, //= 'imagenet',
            int[] inputShape_CHW, //= None,
            int numClass = 1000,
            POOLING_BEFORE_DENSE_LAYER pooling = POOLING_BEFORE_DENSE_LAYER.NONE)
    {
        return EfficientNet(1.0f, 1.0f, TopDropoutRate, SkipConnectionsDropoutRate, 8, blocks, network, includeTop, weights, inputShape_CHW, pooling, numClass);
    }

    public Network EfficientNetB1(
        Network network, /* network without layers */
        List<MobileBlocksDescription> blocks,
        bool includeTop, //= True,
        string weights, //= 'imagenet',
        int[] inputShape_CHW, //= None,
        int numClass = 1000,
        POOLING_BEFORE_DENSE_LAYER pooling = POOLING_BEFORE_DENSE_LAYER.NONE)
    {
        return EfficientNet(1.0f, 1.0f, /*224, */ TopDropoutRate, SkipConnectionsDropoutRate, 8, blocks, network, includeTop, weights, inputShape_CHW, pooling, numClass);
    }

    public Network EfficientNetB2(
        Network network, /* network without layers */
        List<MobileBlocksDescription> blocks,
        bool includeTop, //= True,
        string weights, //= 'imagenet',
        int[] inputShape_CHW, //= None,
        int numClass = 1000,
        POOLING_BEFORE_DENSE_LAYER pooling = POOLING_BEFORE_DENSE_LAYER.NONE)
    {
        return EfficientNet(1.1f, 1.2f, 1.5f* TopDropoutRate, SkipConnectionsDropoutRate, 8, blocks, network, includeTop, weights, inputShape_CHW, pooling, numClass);
    }


    public Network EfficientNetB3(
        Network network, /* network without layers */
        List<MobileBlocksDescription> blocks,
        bool includeTop, //= True,
        string weights, //= 'imagenet',
        int[] inputShape_CHW, //= None,
        int numClass = 1000,
        POOLING_BEFORE_DENSE_LAYER pooling = POOLING_BEFORE_DENSE_LAYER.NONE)
    {
        return EfficientNet(1.2f, 1.4f, /*300,*/ 1.5f* TopDropoutRate, SkipConnectionsDropoutRate, 8, blocks, network, includeTop, weights, inputShape_CHW, pooling, numClass);
    }

    public Network EfficientNetB4(
        Network network, /* network without layers */
        List<MobileBlocksDescription> blocks,
        bool includeTop, //= True,
        string weights, //= 'imagenet',
        int[] inputShape_CHW, //= None,
        int numClass = 1000,
        POOLING_BEFORE_DENSE_LAYER pooling = POOLING_BEFORE_DENSE_LAYER.NONE)
    {
        return EfficientNet(1.4f, 1.8f, /*380,*/ 2* TopDropoutRate, SkipConnectionsDropoutRate, 8, blocks, network, includeTop, weights, inputShape_CHW, pooling, numClass);
    }

    public Network EfficientNetB5(
        Network network, /* network without layers */
        List<MobileBlocksDescription> blocks,
        bool includeTop, //= True,
        string weights, //= 'imagenet',
        int[] inputShape_CHW, //= None,
        int numClass = 1000,
        POOLING_BEFORE_DENSE_LAYER pooling = POOLING_BEFORE_DENSE_LAYER.NONE)
    {
        return EfficientNet(1.6f, 2.2f, /*456,*/ 2* TopDropoutRate, SkipConnectionsDropoutRate, 8, blocks, network, includeTop, weights, inputShape_CHW, pooling, numClass);
    }

    public Network EfficientNetB6(
        Network network, /* network without layers */
        List<MobileBlocksDescription> blocks,
        bool includeTop, //= True,
        string weights, //= 'imagenet',
        int[] inputShape_CHW, //= None,
        int numClass = 1000,
        POOLING_BEFORE_DENSE_LAYER pooling = POOLING_BEFORE_DENSE_LAYER.NONE)
    {
        return EfficientNet(1.8f, 2.6f, /*528, */ 2.5f* TopDropoutRate, SkipConnectionsDropoutRate, 8, blocks, network, includeTop, weights, inputShape_CHW, pooling, numClass);
    }

    public Network EfficientNetB7(
        Network network, /* network without layers */
        List<MobileBlocksDescription> blocks,
        bool includeTop, //= True,
        string weights, //= 'imagenet',
        int[] inputShape_CHW, //= None,
        int numClass = 1000,
        POOLING_BEFORE_DENSE_LAYER pooling = POOLING_BEFORE_DENSE_LAYER.NONE)
    {
        return EfficientNet(2.0f, 3.1f, /*600,*/ 2.5f* TopDropoutRate, SkipConnectionsDropoutRate, 8, blocks, network, includeTop, weights, inputShape_CHW, pooling, numClass);
    }


    public override string ToPytorchModule(Network model)
    {
        var inputShape = "["+string.Join(", ",  model.Layers[0].OutputShape(1).Skip(1).ToList())+"]";
        double widthCoefficient = 1.0;
        double depthCoefficient = 1.0;
        int depthDivisor = 8;
        if (EfficientNetName != enum_efficientnet_name.EfficientNetB0)
        {
            throw new NotImplementedException("Only EfficientNetB0 is implemented");
        }
        var res = "model = EfficientNet(" + Environment.NewLine;
        res += "        widthCoefficient = "+ widthCoefficient.ToString(CultureInfo.InvariantCulture) + ","+Environment.NewLine;
        res += "        depthCoefficient = "+ depthCoefficient.ToString(CultureInfo.InvariantCulture) + ","+Environment.NewLine;
        res += "        topDropoutRate = "+TopDropoutRate.ToString(CultureInfo.InvariantCulture) + "," + Environment.NewLine;
        res += "        skipConnectionsDropoutRate = "+SkipConnectionsDropoutRate.ToString(CultureInfo.InvariantCulture) + "," + Environment.NewLine;
        res += "        depthDivisor = "+ depthDivisor + "," + Environment.NewLine;
        if (DefaultMobileBlocksDescriptionCount == -1 && DefaultMobileBlocksDescriptionCount == 7)
        {
            res += "        blocks = MobileBlocksDescription.default()" + "," + Environment.NewLine;
        }
        else
        {
            res += "        blocks = MobileBlocksDescription.default()[0:"+ DefaultMobileBlocksDescriptionCount+"]," + Environment.NewLine;
        }

        res += "        inputShape_CHW = "+ inputShape + "," + Environment.NewLine;
        res += "        numClass = " + model.YPredicted_MiniBatch_Shape(1)[1] + "," + Environment.NewLine;
        res += "        DefaultActivation = " + PyTorchUtils.ToPytorch(DefaultActivation, null) + "," + Environment.NewLine;
        res += "        LastActivationLayer = " + PyTorchUtils.ToPytorch(LastActivationLayer, null) + "," + Environment.NewLine;
        res += ").to(device)";
        return res;
    }

    public static string GetKerasModelPath(string modelFileName)
    {
        return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), @".keras\models\", modelFileName);
    }
}
