using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpNet.DataAugmentation;
using SharpNet.GPU;
using SharpNet.Layers;

// ReSharper disable UnusedMember.Global

namespace SharpNet.Networks
{
    /// <summary>
    /// EfficientNet support, as described in https://arxiv.org/abs/1905.11946
    /// </summary>
    public class EfficientNetBuilder : NetworkBuilder
    {
        private cudnnActivationMode_t DefaultActivation { get; } = cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH;
        private double BatchNormMomentum { get; set; } = 0.99;
        private double BatchNormEpsilon { get; set; } = 0.001;

        /// <summary>
        /// name of the trained network to load the weights from.
        /// used for transfer learning
        /// </summary>
        public string WeightForTransferLearning { get; } = "";


        /// <summary>
        /// The default EfficientNet Meta Parameters for ImageNet
        /// </summary>
        /// <returns></returns>
        public static EfficientNetBuilder ImageNet()
        {
            var builder = new EfficientNetBuilder
            {
                Config = new NetworkConfig
                {
                    LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                    CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1,
                    lambdaL2Regularization = 0.0005,
                    LogDirectory = Path.Combine(NetworkConfig.DefaultLogDirectory, "ImageNet")
                }
                    .WithSGD(0.9, false)
                    //.WithCifar10WideResNetLearningRateScheduler(true, true, false) : discarded on 14-aug-2019 : Cyclic annealing is better
                    .WithCyclicCosineAnnealingLearningRateScheduler(10, 2) //new default value on 14-aug-2019
                ,
                NumEpochs = 150,
                BatchSize = 128,
                InitialLearningRate = 0.1,
                BatchNormMomentum =  0.99,
                BatchNormEpsilon = 0.001
            };
            return builder;
        }

        /// <summary>
        /// The default EfficientNet Meta Parameters for Cancel Dataset
        /// </summary>
        /// <returns></returns>
        public static EfficientNetBuilder Cancel()
        {
            var builder = new EfficientNetBuilder
            {
                Config = new NetworkConfig
                    {
                        LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropyWithHierarchy,
                        CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1,
                        lambdaL2Regularization = 0.0005,
                        LogDirectory = Path.Combine(NetworkConfig.DefaultLogDirectory, "Cancel"),
                        AlwaysUseFullTestDataSetForLossAndAccuracy= false

                }
                    .WithSGD(0.9, false)
                    .WithCyclicCosineAnnealingLearningRateScheduler(10, 2)
                ,
                NumEpochs = 150,
                BatchSize = 128,
                InitialLearningRate = 0.03,
                BatchNormMomentum = 0.99,
                BatchNormEpsilon = 0.001
            };

            //Data augmentation
            var da = builder.Config.DataAugmentation;
            da.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT;
            //da.WidthShiftRangeInPercentage = 0.1;
            //da.HeightShiftRangeInPercentage = 0.1;
            da.HorizontalFlip = false;
            da.VerticalFlip = false;
            da.Rotate180Degrees = true;
            da.FillMode = ImageDataGenerator.FillModeEnum.Reflect;
            da.AlphaMixup = 0.0;    //Mixup is discarded
            da.AlphaCutMix = 0.0;   //CutMix is discarded
            da.CutoutPatchPercentage = 0.1;
            return builder;
        }


        /// <summary>
        /// The default EfficientNet Meta Parameters for CIFAR10
        /// </summary>
        /// <returns></returns>
        public static EfficientNetBuilder CIFAR10()
        {
            var builder = new EfficientNetBuilder
            {
                Config = new NetworkConfig
                {
                    LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                    CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1,
                    lambdaL2Regularization = 0.0005,
                    LogDirectory = Path.Combine(NetworkConfig.DefaultLogDirectory, "CIFAR-10")
                }
                    .WithSGD(0.9, false)
                    .WithCyclicCosineAnnealingLearningRateScheduler(10, 2)
                ,
                NumEpochs = 150,
                BatchSize = 128,
                InitialLearningRate = 0.1,
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
            da.AlphaCutMix = 1.0;
            da.CutoutPatchPercentage = 0.0;
            return builder;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="widthCoefficient">scaling coefficient for network width</param>
        /// <param name="depthCoefficient">scaling coefficient for network depth</param>
        /// <param name="defaultResolution">default input image size</param>
        /// <param name="dropoutRate">dropout rate before final classifier layer</param>
        /// <param name="blocks">A list of BlockDescription to construct block modules</param>
        /// <param name="dropConnectRate">dropout rate at skip connections</param>
        /// <param name="depthDivisor"></param>
        /// <param name="modelName">model name</param>
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
        /// <param name="categoryCount">optional number of classes to classify images  into, only to be specified if `includeTop` is True, and if no `weights` argument is specified</param>
        /// <returns></returns>
        private Network EfficientNet(
            float widthCoefficient,
            float depthCoefficient,
            // ReSharper disable once UnusedParameter.Local
            int defaultResolution,
            float dropoutRate,
            float dropConnectRate,
            int depthDivisor,
            List<MobileBlocksDescription> blocks,
            string modelName,
            bool includeTop,
            string weights,
            int[] inputShape_CHW,
            POOLING_BEFORE_DENSE_LAYER pooling,
            int categoryCount //= 1000
            )
        {
            //dropoutRate = dropConnectRate = 0; //to disable all Dropout layers

            blocks = blocks.Select(x => x.ApplyScaling(widthCoefficient, depthDivisor, depthCoefficient)).ToList();
            var net = BuildEmptyNetwork(modelName);
            var config = net.Config;

            //TODO compute actual inputShape_CHW
            //inputShape_CHW = _obtain_input_shape(inputShape_CHW, default_size=defaultResolution, min_size=32, data_format="NCHW", require_flatten=includeTop, weights=weights)

            net.Input(inputShape_CHW);


            //Build stem
            var stemChannels = MobileBlocksDescription.RoundFilters(32, widthCoefficient, depthDivisor);
            var stemStride = 2;

            //if (Math.Min(height, width) <= 32)
            //{
            //    stemStride = 1;
            //}
            net.Convolution(stemChannels, 3, stemStride, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, false, "stem_conv")
                .BatchNorm(BatchNormMomentum, BatchNormEpsilon, "stem_bn")
                .Activation(DefaultActivation, "stem_activation");

            //Build blocks
            int numBlocksTotal = blocks.Select(x => x.NumRepeat).Sum();
            int blockNum = 0;
            for (int idx = 0; idx < blocks.Count ;++idx)
            {
                var block_arg = blocks[idx];
                for (int bidx = 0; bidx < block_arg.NumRepeat; ++bidx)
                {
                    var layerPrefix = "block" + (idx + 1) + (char)('a' + bidx) + "_";
                    var dropRate = (dropConnectRate * blockNum) / numBlocksTotal;
                    //The first block needs to take care of stride and filter size increase.
                    var mobileBlocksDescription = bidx == 0 ? block_arg : block_arg.WithStride(1, 1);
                    AddMBConvBlock(net, mobileBlocksDescription, dropRate, layerPrefix);
                    ++blockNum;
                }
            }

            //# Build top
            var outputChannelsTop = MobileBlocksDescription.RoundFilters(1280, widthCoefficient, depthDivisor);
            net.Convolution(outputChannelsTop, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, false, "top_conv");
            net.BatchNorm(BatchNormMomentum, BatchNormEpsilon, "top_bn");
            net.Activation(DefaultActivation, "top_activation");
            if (includeTop)
            {
                net.GlobalAvgPooling("avg_pool");
                if (dropoutRate > 0)
                {
                    net.Dropout(dropoutRate, "top_dropout");
                }
                net.Dense(categoryCount, config.lambdaL2Regularization, "probs");
                net.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            }
            else
            {
                if (pooling == POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling)
                {
                    net.GlobalAvgPooling("avg_pool");
                }
                else if (pooling == POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling)
                {
                    net.GlobalMaxPooling("max_pool");
                }
            }


            //We load weights if needed
            if (0 == string.Compare(weights, "imagenet", StringComparison.OrdinalIgnoreCase))
            {
                if (categoryCount != 1000)
                {
                    throw new ArgumentException("categoryCount must be 1000 for " + weights);
                }
                var modelPath = GetKerasModelPath(modelName + "_weights_tf_dim_ordering_tf_kernels_autoaugment.h5");
                if (!File.Exists(modelPath))
                {
                    throw new ArgumentException("missing "+weights+" model file "+modelPath);
                }
                net.LoadParametersFromH5File(modelPath, NetworkConfig.CompatibilityModeEnum.TensorFlow1);
            }



            net.FreezeSelectedLayers();
            return net;
        }

        /// <summary>
        /// Mobile Inverted Residual Bottleneck
        /// </summary>
        /// <param name="network"></param>
        /// <param name="block_args"></param>
        /// <param name="dropRate"></param>
        /// <param name="layerPrefix"></param>
        private void AddMBConvBlock(Network network, MobileBlocksDescription block_args, float dropRate, string layerPrefix)
        {
            int inputChannels = network.Layers.Last().OutputShape(1)[1];
            var inputLayerIndex = network.LastLayerIndex;
            
            //Expansion phase
            var config = network.Config;
            var filters = inputChannels * block_args.ExpandRatio;
            if (block_args.ExpandRatio != 1)
            {
                network.Convolution(filters, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, false, layerPrefix+"expand_conv")
                    .BatchNorm(BatchNormMomentum, BatchNormEpsilon, layerPrefix+"expand_bn")
                    .Activation(DefaultActivation, layerPrefix+ "expand_activation");
            }

            //Depthwise Convolution
            network.DepthwiseConvolution(block_args.KernelSize, block_args.ColStride, ConvolutionLayer.PADDING_TYPE.SAME, 1, config.lambdaL2Regularization, false, layerPrefix+"dwconv")
                .BatchNorm(BatchNormMomentum, BatchNormEpsilon, layerPrefix + "bn")
                .Activation(DefaultActivation, layerPrefix + "activation");

            //Squeeze and Excitation phase
            bool hasSe = block_args.SeRatio > 0 && block_args.SeRatio <= 1.0;
            if (hasSe)
            {
                var xLayerIndex = network.LastLayerIndex;
                var num_reduced_filters = Math.Max(1, (int) (inputChannels * block_args.SeRatio));
                network.GlobalAvgPooling(layerPrefix + "se_squeeze");
                network.Convolution(num_reduced_filters, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, true, layerPrefix + "se_reduce")
                    .Activation(DefaultActivation);
                network.Convolution(filters, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, true, layerPrefix + "se_expand")
                    .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);
                network.MultiplyLayer(xLayerIndex, network.LastLayerIndex /*diagonal matrix */, layerPrefix + "se_excite");
            }

            //Output phase
            network.Convolution(block_args.OutputFilters, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, false, layerPrefix + "project_conv");
            network.BatchNorm(BatchNormMomentum, BatchNormEpsilon, layerPrefix + "project_bn");
            if (block_args.IdSkip && block_args.RowStride == 1 && block_args.ColStride == 1 && block_args.OutputFilters == inputChannels)
            {
                if (dropRate > 0.000001)
                {
                    network.Dropout(dropRate, layerPrefix + "drop");
                }
                network.AddLayer(network.LastLayerIndex, inputLayerIndex, layerPrefix + "add");
            }
        }


        /// <summary>
        /// construct an EfficientNet B0 network for CIFAR10 training
        /// if weight is provided (ex: imagenet):
        ///      will load the weight from the provided source,
        ///      and will set the network category count to 10
        ///      (resetting the last Dense layer weights if required to have 10 output categoryCount)
        /// </summary>
        /// <param name="weight"></param>
        /// <param name="inputShape_CHW"></param>
        /// <returns></returns>
        public Network EfficientNetB0_CIFAR10(string weight, int[] inputShape_CHW)
        {
            var net = EfficientNetB0(true, weight, inputShape_CHW);
            Network.Log.Info("setting number of output categoryCount to 10");
            net.SetCategoryCount(10);
            return net;
        }

        public Network EfficientNetB0(
            bool includeTop, //= True,
            string weights, //= 'imagenet',
            int[] inputShape_CHW, //= None,
            int categoryCount = 1000,
            POOLING_BEFORE_DENSE_LAYER pooling = POOLING_BEFORE_DENSE_LAYER.NONE)
        {
            return EfficientNet(1.0f, 1.0f, 224, 0.2f, 
                0.2f, 8, MobileBlocksDescription.Default(), "efficientnet-b0", includeTop, weights, inputShape_CHW, pooling, categoryCount);
        }

        public Network EfficientNetB1(
            bool includeTop, //= True,
            string weights, //= 'imagenet',
            int[] inputShape_CHW, //= None,
            POOLING_BEFORE_DENSE_LAYER pooling, //= None,
            int categoryCount //= 1000
        )
        {
            return EfficientNet(1.0f, 1.0f, 224, 0.2f,
                0.2f, 8, MobileBlocksDescription.Default(), "efficientnet-b1", includeTop, weights, inputShape_CHW, pooling, categoryCount);
        }

        public Network EfficientNetB2(
            bool includeTop, //= True,
            string weights, //= 'imagenet',
            int[] inputShape_CHW, //= None,
            POOLING_BEFORE_DENSE_LAYER pooling, //= None,
            int categoryCount //= 1000
        )
        {
            return EfficientNet(1.1f, 1.2f,  260, 0.3f, 
                0.2f, 8, MobileBlocksDescription.Default(), "efficientnet-b2", includeTop, weights, inputShape_CHW,  pooling, categoryCount);
        }

        public Network EfficientNetB3(
            bool includeTop, //= True,
            string weights, //= 'imagenet',
            int[] inputShape_CHW, //= None,
            POOLING_BEFORE_DENSE_LAYER pooling, //= None,
            int categoryCount //= 1000
        )
        {
            return EfficientNet(1.2f, 1.4f, 300, 0.3f,
                0.2f, 8, MobileBlocksDescription.Default(), "efficientnet-b3", includeTop, weights, inputShape_CHW, pooling, categoryCount);
        }

        public Network EfficientNetB4(
            bool includeTop, //= True,
            string weights, //= 'imagenet',
            int[] inputShape_CHW, //= None,
            POOLING_BEFORE_DENSE_LAYER pooling, //= None,
            int categoryCount //= 1000
        )
        {
            return EfficientNet(1.4f, 1.8f, 380, 0.4f,
                0.2f, 8, MobileBlocksDescription.Default(), "efficientnet-b4", includeTop, weights, inputShape_CHW, pooling, categoryCount);
        }

        public Network EfficientNetB5(
            bool includeTop, //= True,
            string weights, //= 'imagenet',
            int[] inputShape_CHW, //= None,
            POOLING_BEFORE_DENSE_LAYER pooling, //= None,
            int categoryCount //= 1000
        )
        {
            return EfficientNet(1.6f, 2.2f, 456, 0.4f,
                0.2f, 8, MobileBlocksDescription.Default(), "efficientnet-b5", includeTop, weights, inputShape_CHW, pooling, categoryCount);
        }

        public Network EfficientNetB6(
            bool includeTop, //= True,
            string weights, //= 'imagenet',
            int[] inputShape_CHW, //= None,
            POOLING_BEFORE_DENSE_LAYER pooling, //= None,
            int categoryCount //= 1000
        )
        {
            return EfficientNet(1.8f, 2.6f, 528, 0.5f,
                0.2f, 8, MobileBlocksDescription.Default(), "efficientnet-b6", includeTop, weights, inputShape_CHW, pooling, categoryCount);
        }

        public Network EfficientNetB7(
            bool includeTop, //= True,
            string weights, //= 'imagenet',
            int[] inputShape_CHW, //= None,
            POOLING_BEFORE_DENSE_LAYER pooling, //= None,
            int categoryCount //= 1000
        )
        {
            return EfficientNet(2.0f, 3.1f, 600, 0.5f,
                0.2f, 8, MobileBlocksDescription.Default(), "efficientnet-b7", includeTop, weights, inputShape_CHW, pooling, categoryCount);
        }

        public Network EfficientNetL2(
            bool includeTop, //= True,
            string weights, //= 'imagenet',
            int[] inputShape_CHW, //= None,
            POOLING_BEFORE_DENSE_LAYER pooling, //= None,
            int categoryCount //= 1000
        )
        {
            return EfficientNet(4.3f, 5.3f, 800, 0.5f,
                0.2f, 8, MobileBlocksDescription.Default(), "efficientnet-l2", includeTop, weights, inputShape_CHW, pooling, categoryCount);
        }
    }
}