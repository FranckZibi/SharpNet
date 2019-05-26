using System.Linq;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Pictures;

// ReSharper disable UnusedMember.Global


/*
BatchSize = 128
EpochCount = 200
SGD with momentum = 0.9 & L2 = 0.5* 1-e4
Cutout 16 / FillMode = Reflect / Disable DivideBy10OnPlateau
# ----------------------------------------------------------------
#           |             |    200-epoch  |   Orig Paper  | sec/epoch
# Model     |   #Params   |       WRN     |      WRN      | GTX1080
#           |             |   %Accuracy   |   %Accuracy   | 
#           |             |   (dropout)   |   (dropout)   | 
# ---------------------------------------------------------------
# WRN-16-4  |   2,752,506 | ----- (-----) | 94.98 (94.76) |  27.0
# WRN-40-4  |   8,959,994 | ----- (-----) | 95.43 (-----) |  81.0
# WRN-16-8  |  10,968,570 | ----- (-----) | 95.73         |  83.0
# WRN-16-10 |  17,125,626 | ----- (-----) | NA            | 136.0
# WRN-28-8  |  23,369,210 | ----- (-----) | NA            | 173.0
# WRN-28-10 |  36,497,146 | ----- (-----) | 96.00 (96.11) | 300.0
# ---------------------------------------------------------------
*/

namespace SharpNet.Networks
{
    /// <summary>
    /// Wide Residual Network support, as described in https://arxiv.org/pdf/1605.07146.pdf
    /// </summary>
    public class WideResNetBuilder : NetworkBuilder
    {
        public WideResNetBuilder()
        {
            Config = new NetworkConfig
                {
                    UseDoublePrecision = false,
                    LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                    lambdaL2Regularization = 0.0005
                }
                .WithSGD(0.9, false)
                .WithCifar10WideResNetLearningRateScheduler(true, true, false);

            //Data augmentation
            WidthShiftRange = 0.1;
            HeightShiftRange = 0.1;
            HorizontalFlip = true;
            VerticalFlip = false;
            FillMode = ImageDataGenerator.FillModeEnum.Reflect;
            CutoutPatchlength = 16;

            NumEpochs = 200;
            BatchSize = 128;
            DropOut = 0.0; //by default we disable dropout
            InitialLearningRate = 0.1;
        }

        /// <summary>
        /// 0 to disable dropout
        /// any value > 0 will enable dropout
        /// </summary>
        public double DropOut { get; set; }

        public Network WRN_16_4_CIFAR10() { return WRN_CIFAR10(16, 4); }
        public Network WRN_40_4_CIFAR10() { return WRN_CIFAR10(40, 4); }
        public Network WRN_16_8_CIFAR10() { return WRN_CIFAR10(16, 8); }
        public Network WRN_16_10_CIFAR10() { return WRN_CIFAR10(16, 10); }
        public Network WRN_28_8_CIFAR10() { return WRN_CIFAR10(28, 8); }
        public Network WRN_28_10_CIFAR10() { return WRN_CIFAR10(28, 10); }

        /// <summary>
        /// returns a Wide Residual network, as described in https://arxiv.org/pdf/1605.07146.pdf
        /// </summary>
        /// <param name="depth">total number of convolutions in the network
        /// There are always 3 stages in a Wide ResNet.
        /// Number of convolutions in each stage = (depth-1)/3      (one of them is used to change dimension)
        /// Number of convolutions in each residual block = 2
        /// Number of residual blocks in each stage = (depth-4)/6
        /// </param>
        /// <param name="k">widening parameter</param>
        /// <returns></returns>
        private Network WRN_CIFAR10(int depth, int k)
        {
            int convolutionsCountByStage = (depth - 1) / 3;
            int residualBlocksCountByStage = (convolutionsCountByStage-1) / 2;

            var networkName = "WRN-"+depth+"-"+k;
            var net = BuildEmptyNetwork(networkName);
            var config = net.Config;
            var layers = net.Layers;
            net.Input(CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width);

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
                    if ((DropOut > 0.0)&& (residualBlockId != 0))
                    {
                        net.Dropout(DropOut);
                    }

                    net.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageC, 3, 1, 1, config.lambdaL2Regularization, false);
                    net.Shortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, config.lambdaL2Regularization);
                }
                stageC *= 2;
            }
            net.BatchNorm_Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            net.AvgPooling(8, 8);
            net.Output(CIFAR10.Categories, config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return net;
        }
    }
}
