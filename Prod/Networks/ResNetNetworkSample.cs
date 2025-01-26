using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
// ReSharper disable UnusedMember.Global

/*
SharpNet on 12-march-2019
LearningRate = LearningRateScheduler.ConstantByInterval(1, initialLearningRate, 80, initialLearningRate / 10, 120, initialLearningRate / 100);
BatchSize = 128
EpochCount = 160
SGD with momentum = 0.9 & L2 = 1-e4
Cutout 16 / FillMode = Reflect / DivideBy10OnPlateau
# ------------------------------------------------------------------------
#           |      |   160-epoch   |    Orig Paper      | sec/epoch
# Model     |  n   | ResNetV1 (V2) |   ResNetV1 (V2)    | GTX1080
#           |v1(v2)|   %Accuracy   |     %Accuracy      | v1 (v2)  
# ------------------------------------------------------------------------
# ResNet11  | - (1)| NA    (88.82) | NA (-----)         |   NA (8.4) 
# ResNet20  | 3 (2)| 91.96 (92.24) | 91.25 (-----)      |  8.8 (15.1) 
# ResNet32  | 5(NA)| 93.28 (NA   ) | 92.49 (NA   )      | 13.8 ( NA) 
# ResNet44  | 7(NA)| 93.41 (NA   ) | 92.83 (NA   )      | 18.8 ( NA) 
# ResNet56  | 9 (6)| 93.92 (94.53) | 93.03 (-----)      | 23.8 (41.0) 
# ResNet110 |18(12)| 94.07 (95.38) | 93.39+-.16 (93.63) | 47.1 (80.2)
# ResNet164 |27(18)| 93.51 (92.92) | 94.07 (94.54)      | 78.9 (125)
# ResNet1001| (111)| ----- (-----) | 92.39 (95.08+-.14) | ---- (---)
# ------------------------------------------------------------------------
*/

namespace SharpNet.Networks
{
    public class ResNetNetworkSample : NetworkSample
    {
        private ResNetNetworkSample()
        {
        }

        /// <summary>
        /// default Hyper-Parameters for CIFAR10
        /// </summary>
        /// <returns></returns>
        public static ResNetNetworkSample CIFAR10()
        {
            var config = (ResNetNetworkSample)new ResNetNetworkSample
            {
                    LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                    lambdaL2Regularization = 1e-4,
                    num_epochs = 160, //64k iterations
                    BatchSize = 128,
                    InitialLearningRate = 0.1,

                    //Data Augmentation
                    DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
                    WidthShiftRangeInPercentage = 0.1, //validated on 18-apr-2019: +300 bps (for both using WidthShiftRange & HeightShiftRange)
                    HeightShiftRangeInPercentage = 0.1,
                    HorizontalFlip = true, // 'true' : validated on 18-apr-2019: +70 bps
                    VerticalFlip = false,
                    FillMode = ImageDataGenerator.FillModeEnum.Reflect, //validated on 18-apr-2019: +50 bps
                    CutoutPatchPercentage = 0.5, // validated on 17-apr-2019 for CIFAR-10: +70 bps (a cutout of the 1/2 of the image width)

            }
                .WithSGD(0.9, false) // SGD : validated on 19-apr-2019: +70 bps
                //config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2), //Tested on 28-may-2019: +16bps on ResNetV2 / +2bps on ResNetV1
                .WithCifar10ResNetLearningRateScheduler(true, true, false);

            return config;

        }

        //implementation described in: https://arxiv.org/pdf/1512.03385.pdf
        #region ResNet V1
        public Network ResNet18_V1(int[] xShape, int numClass)
        {
            return ResNetV1(nameof(ResNet18_V1), new[] { 2, 2, 2, 2 }, false, xShape, numClass);
        }
        public Network ResNet34_V1(int[] xShape, int numClass)
        {
            return ResNetV1(nameof(ResNet34_V1), new[] { 3, 4, 6, 3 }, false, xShape, numClass);
        }
        public Network ResNet50_V1(int[] xShape, int numClass)
        {
            return ResNetV1(nameof(ResNet50_V1), new[] { 3, 4, 6, 3 }, true, xShape, numClass);
        }
        public Network ResNet101_V1(int[] xShape, int numClass)
        {
            return ResNetV1(nameof(ResNet101_V1), new[] { 3, 4, 23, 3 }, true, xShape, numClass);
        }
        public Network ResNet152_V1(int[] xShape, int numClass)
        {
            return ResNetV1(nameof(ResNet152_V1), new[] { 3, 8, 36, 3 }, true, xShape, numClass);
        }
        private Network ResNetV1(string networkName, int[] nbResBlocks, bool useBottleNeck, int[] xShape, int numClass)
        {
            var network = BuildNetworkWithoutLayers(DenseNetNetworkSample.Cifar10WorkingDirectory, networkName);
            var config = network.Sample;
            const cudnnActivationMode_t activationFunction = cudnnActivationMode_t.CUDNN_ACTIVATION_RELU;
            network.Input(xShape[1], xShape[2], xShape[3]);

            network.Convolution(64, 7, 2, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, true);
            network.MaxPooling(2, 2, 2, 2);

            int stageC = 64; //number of channels for current stage
            for (int stageId = 0; stageId < nbResBlocks.Length; ++stageId)
            {
                int num_res_blocks = nbResBlocks[stageId];
                for (int res_block = 0; res_block < num_res_blocks; res_block += 1)
                {
                    int stride = (res_block == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = network.LastLayerIndex;
                    if (useBottleNeck)
                    {
                        network.Convolution_BatchNorm_Activation(stageC, 1, stride, ConvolutionLayer.PADDING_TYPE.VALID, config.lambdaL2Regularization, activationFunction);
                        network.Convolution_BatchNorm_Activation(stageC, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, activationFunction);
                        network.Convolution_BatchNorm(4 * stageC, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, config.lambdaL2Regularization);
                        network.Shortcut_IdentityConnection(startOfBlockLayerIndex, 4 * stageC, stride, config.lambdaL2Regularization);
                        network.Activation(activationFunction);
                    }
                    else
                    {
                        network.Convolution_BatchNorm_Activation(stageC, 3, stride, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, activationFunction);
                        network.Convolution_BatchNorm(stageC, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization);
                        network.Shortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, config.lambdaL2Regularization);
                        network.Activation(activationFunction);
                    }
                }
                stageC *= 2;
            }
            network.GlobalAvgPooling();
            network.Linear(numClass, true, config.lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return network;
        }
        #endregion

        //implementation described in: https://arxiv.org/pdf/1512.03385.pdf
        #region ResNetV1 for CIFAR-10
        public Network ResNet20V1_CIFAR10() {return ResNetV1_CIFAR10(3);}
        public Network ResNet32V1_CIFAR10() {return ResNetV1_CIFAR10(5);}
        public Network ResNet44V1_CIFAR10() {return ResNetV1_CIFAR10(7);}
        public Network ResNet56V1_CIFAR10() {return ResNetV1_CIFAR10(9);}
        public Network ResNet110V1_CIFAR10() {return ResNetV1_CIFAR10(18);}
        public Network ResNet164V1_CIFAR10() {return ResNetV1_CIFAR10(27);}
        public Network ResNet1202V1_CIFAR10() {return ResNetV1_CIFAR10(200);}
        private Network ResNetV1_CIFAR10(int numResBlocks)
        {
            var networkName = "ResNet" + (6 * numResBlocks + 2) + "V1_"+ CIFAR10DataSet.NAME;
            var network = BuildNetworkWithoutLayers(DenseNetNetworkSample.Cifar10WorkingDirectory, networkName);
            var config = network.Sample;

            network.Input(CIFAR10DataSet.Shape_CHW);

            network.Convolution_BatchNorm_Activation(16, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);

            int stageC = 16; //number of channels for current stage
            for (int stageId = 0; stageId < 3; ++stageId)
            {
                for (int res_block = 0; res_block < numResBlocks; res_block += 1)
                {
                    int stride = (res_block == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = network.LastLayerIndex;
                    network.Convolution_BatchNorm_Activation(stageC, 3, stride, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                    network.Convolution_BatchNorm(stageC, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization);
                    network.Shortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, config.lambdaL2Regularization);
                    network.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                }
                stageC *= 2;
            }
            network.AvgPooling(8, 8, 8, 8);
            network.Linear(CIFAR10DataSet.NumClass, true, config.lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return network;
        }
        #endregion

        //implementation described in: https://arxiv.org/pdf/1603.05027.pdf
        #region ResNetV2 for CIFAR-10
        public Network ResNet11V2_CIFAR10() {return ResNetV2_CIFAR10(1);}
        public Network ResNet20V2_CIFAR10() {return ResNetV2_CIFAR10(2);}
        public Network ResNet29V2_CIFAR10() {return ResNetV2_CIFAR10(3);}
        public Network ResNet56V2_CIFAR10() {return ResNetV2_CIFAR10(6);}
        public Network ResNet110V2_CIFAR10() {return ResNetV2_CIFAR10(12);}
        public Network ResNet164V2_CIFAR10() {return ResNetV2_CIFAR10(18);}
        public Network ResNet1001V2_CIFAR10() {return ResNetV2_CIFAR10(111);}
        private Network ResNetV2_CIFAR10(int numResBlocks)
        {
            var networkName = "ResNet" + (9 * numResBlocks + 2) + "V2_"+ CIFAR10DataSet.NAME;
            var network = BuildNetworkWithoutLayers(DenseNetNetworkSample.Cifar10WorkingDirectory, networkName);
            var config = network.Sample;

            network.Input(CIFAR10DataSet.Shape_CHW);
            network.Convolution_BatchNorm_Activation(16, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);

            int stageCIn = 16; //number of channels for current stage
            int stageCOut = 4 * stageCIn;

            for (int stageId = 0; stageId < 3; ++stageId)
            {
                for (int resBlock = 0; resBlock < numResBlocks; resBlock += 1)
                {
                    int stride = (resBlock == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = network.LastLayerIndex;
                    if (stageId == 0 && resBlock == 0)
                    {
                        network.Convolution(stageCIn, 1, stride, ConvolutionLayer.PADDING_TYPE.VALID, config.lambdaL2Regularization, true);
                    }
                    else
                    {
                        network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCIn, 1, stride, ConvolutionLayer.PADDING_TYPE.VALID, config.lambdaL2Regularization, true);
                    }
                    network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCIn, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, config.lambdaL2Regularization, true);
                    network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCOut, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, config.lambdaL2Regularization, true);
                    network.Shortcut_IdentityConnection(startOfBlockLayerIndex, stageCOut, stride, config.lambdaL2Regularization);
                }
                stageCIn = stageCOut;
                stageCOut = 2 * stageCIn;
            }
            network.BatchNorm(0.99, 1e-5).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            network.AvgPooling(8, 8, 8, 8);
            network.Linear(CIFAR10DataSet.NumClass, true, config.lambdaL2Regularization, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return network;
        }
        #endregion
    }
}