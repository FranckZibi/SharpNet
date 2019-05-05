using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.GPU;

namespace SharpNet
{
    public static class ResNetUtils
    {
        public static Network ResNet18_V1(int[] xShape, int nbCategories, ResNetMetaParameters param, Logger logger = null)
        {
            return ResNetV1(new[] { 2, 2, 2, 2 }, false, xShape, nbCategories, param, logger);
        }
        public static Network ResNet34_V1(int[] xShape, int nbCategories, ResNetMetaParameters param, Logger logger = null)
        {
            return ResNetV1(new[] { 3, 4, 6, 3 }, false, xShape, nbCategories, param, logger);
        }
        public static Network ResNet50_V1(int[] xShape, int nbCategories, ResNetMetaParameters param, Logger logger = null)
        {
            return ResNetV1(new[] { 3, 4, 6, 3 }, true, xShape, nbCategories, param, logger);
        }
        public static Network ResNet101_V1(int[] xShape, int nbCategories, ResNetMetaParameters param, Logger logger = null)
        {
            return ResNetV1(new[] { 3, 4, 23, 3 }, true, xShape, nbCategories, param, logger);
        }
        public static Network ResNet152_V1(int[] xShape, int nbCategories, ResNetMetaParameters param, Logger logger = null)
        {
            return ResNetV1(new[] { 3, 8, 36, 3 }, true, xShape, nbCategories, param, logger);
        }
        public static Network ResNet20V1_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(3, param, logger);
        }
        public static Network ResNet32V1_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(5, param, logger);
        }
        public static Network ResNet44V1_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(7, param, logger);
        }
        public static Network ResNet56V1_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(9, param, logger);
        }
        public static Network ResNet110V1_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(18, param, logger);
        }
        public static Network ResNet164V1_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(27, param, logger);
        }
        public static Network ResNet1202V1_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(200, param, logger);
        }



        public static Network ResNet11V2_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(1, param, logger);
        }
        public static Network ResNet20V2_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(2, param, logger);
        }
        public static Network ResNet29V2_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(3, param, logger);
        }
        public static Network ResNet56V2_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(6, param, logger);
        }
        public static Network ResNet110V2_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(12, param, logger);
        }
        public static Network ResNet164V2_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(18, param, logger);
        }
        public static Network ResNet1001V2_CIFAR10(ResNetMetaParameters param, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(111, param, logger);
        }
        public static void Load(out CpuTensor<byte> xTrainingSet, out CpuTensor<byte> yTrainingSet, out CpuTensor<byte> xTestSet, out CpuTensor<byte> yTestSet)
        {
            var path = @"C:\Projects\SharpNet\Tests\Data\cifar-10-batches-bin\";
            xTrainingSet = new CpuTensor<byte>(new[] { 50000, 3, 32, 32 }, "xTrainingSet");
            yTrainingSet = new CpuTensor<byte>(new[] { 50000, 1, 1, 1 }, "yTrainingSet");
            xTestSet = new CpuTensor<byte>(new[] { 10000, 3, 32, 32 }, "xTestSet");
            yTestSet = new CpuTensor<byte>(new[] { 10000, 1, 1, 1 }, "yTestSet");
            for (int i = 0; i < 5; ++i)
            {
                LoadAt(Path.Combine(path, "data_batch_" + (i + 1) + ".bin"), xTrainingSet, yTrainingSet, 10000 * i);
            }
            LoadAt(Path.Combine(path, "test_batch.bin"), xTestSet, yTestSet, 0);
        }
        public static void ToWorkingSet(CpuTensor<byte> x, CpuTensor<byte> y, out CpuTensor<float> xWorkingSet, out CpuTensor<float> yWorkingSet, List<Tuple<double, double>> meanAndVolatilityOfEachChannel)
        {
            xWorkingSet = x.Select((n, c, val) => (float)((val - meanAndVolatilityOfEachChannel[c].Item1) / Math.Max(meanAndVolatilityOfEachChannel[c].Item2, 1e-9)));
            yWorkingSet = y.ToCategorical(1.0f, out _);
        }
        private static Network ResNetV1(int[] nbResBlocks, bool useBottleNeck, int[] xShape, int nbCategories, ResNetMetaParameters param, Logger logger)
        {
            var activationFunction = cudnnActivationMode_t.CUDNN_ACTIVATION_RELU;

            var networkConfig = new NetworkConfig(param.UseGPU) { UseDoublePrecision = param.UseDoublePrecision, LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy, Logger = logger ?? Logger.ConsoleLogger };
            var network = new Network(networkConfig.WithSGD(0.9, 0.0, true), param.ResNetImageDataGenerator());
            network.Input(xShape[1], xShape[2], xShape[3]);

            network.Convolution(64, 7, 2, 3, param.lambdaL2Regularization, true);
            network.MaxPooling(2, 2);

            int stageC = 64; //number of channels for current stage
            for (int stageId = 0; stageId < nbResBlocks.Length; ++stageId)
            {
                int num_res_blocks = nbResBlocks[stageId];
                for (int res_block = 0; res_block < num_res_blocks; res_block += 1)
                {
                    int stride = (res_block == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = network.Layers.Last().LayerIndex;
                    if (useBottleNeck)
                    {
                        network.Convolution_BatchNorm_Activation(stageC, 1, stride, 0, param.lambdaL2Regularization, activationFunction);
                        network.Convolution_BatchNorm_Activation(stageC, 3, 1, 1, param.lambdaL2Regularization, activationFunction);
                        network.Convolution_BatchNorm(4 * stageC, 1, 1, 0, param.lambdaL2Regularization);
                        network.Shortcut_IdentityConnection(startOfBlockLayerIndex, 4 * stageC, stride, param.lambdaL2Regularization);
                        network.Activation(activationFunction);
                    }
                    else
                    {
                        network.Convolution_BatchNorm_Activation(stageC, 3, stride, 1, param.lambdaL2Regularization, activationFunction);
                        network.Convolution_BatchNorm(stageC, 3, 1, 1, param.lambdaL2Regularization);
                        network.Shortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, param.lambdaL2Regularization);
                        network.Activation(activationFunction);
                    }
                }
                stageC *= 2;
            }
            network.GlobalAvgPooling();
            network.Output(nbCategories, param.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return network;
        }
     
        private const int ChannelsCifar10 = 3;
        private const int HeightCifar10 = 32;
        private const int WidthCifar10 = HeightCifar10;
        private const int CategoriesCifar10 = 10;

        //implementation described in: https://arxiv.org/pdf/1512.03385.pdf
        private static Network GetResNetV1_CIFAR10(int numResBlocks, ResNetMetaParameters param, Logger logger)
        {
            var networkConfig = new NetworkConfig(param.UseGPU) { UseDoublePrecision = param.UseDoublePrecision, LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy, Logger = logger ?? Logger.ConsoleLogger };
        
            networkConfig = param.UseAdam ? networkConfig.WithAdam() :  networkConfig.WithSGD(0.9, 0, param.UseNesterov);

            var network = new Network(networkConfig, param.ResNetImageDataGenerator());
            network.Input(ChannelsCifar10, HeightCifar10, WidthCifar10);

            network.Convolution_BatchNorm_Activation(16, 3, 1, 1, param.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);

            int stageC = 16; //number of channels for current stage
            for (int stageId = 0; stageId < 3; ++stageId)
            {
                for (int res_block = 0; res_block < numResBlocks; res_block += 1)
                {
                    int stride = (res_block == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = network.Layers.Last().LayerIndex;
                    network.Convolution_BatchNorm_Activation(stageC, 3, stride, 1, param.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                    network.Convolution_BatchNorm(stageC, 3, 1, 1, param.lambdaL2Regularization);
                    network.Shortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, param.lambdaL2Regularization);
                    network.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                }
                stageC *= 2;
            }
            network.AvgPooling(8, 8);
            network.Output(CategoriesCifar10, param.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            network.Description = "ResNet" + (6*numResBlocks+2) + "V1_CIFAR10" + param.ExtraDescription;
            return network;
        }

        //implementation described in: https://arxiv.org/pdf/1603.05027.pdf
        private static Network GetResNetV2_CIFAR10(int numResBlocks, ResNetMetaParameters param, Logger logger)
        {
            var description = "ResNet"+(9* numResBlocks +2)+ "V2_CIFAR10" + param.ExtraDescription;
            var networkConfig = new NetworkConfig(param.UseGPU) { UseDoublePrecision = param.UseDoublePrecision, LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy, Logger = logger ?? Logger.ConsoleLogger };

            networkConfig = param.UseAdam ? networkConfig.WithAdam() : networkConfig.WithSGD(0.9, 0, param.UseNesterov);
            var network = new Network(networkConfig, param.ResNetImageDataGenerator());

            network.Input(ChannelsCifar10, HeightCifar10, WidthCifar10);
            network.Convolution_BatchNorm_Activation(16, 3, 1, 1, param.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);

            int stageCIn = 16; //number of channels for current stage
            int stageCOut = 4 * stageCIn;

            for (int stageId = 0; stageId < 3; ++stageId)
            {
                for (int resBlock = 0; resBlock < numResBlocks; resBlock += 1)
                {
                    int stride = (resBlock == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = network.Layers.Last().LayerIndex;
                    if (stageId == 0 && resBlock == 0)
                    {
                        network.Convolution(stageCIn, 1, stride, 0, param.lambdaL2Regularization, true);
                    }
                    else
                    {
                        network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCIn, 1, stride, 0, param.lambdaL2Regularization);
                    }
                    network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCIn, 3, 1, 1, param.lambdaL2Regularization);
                    network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCOut, 1, 1, 0, param.lambdaL2Regularization);
                    network.Shortcut_IdentityConnection(startOfBlockLayerIndex, stageCOut, stride, param.lambdaL2Regularization);
                }
                stageCIn = stageCOut;
                stageCOut = 2 * stageCIn;
            }
            network.BatchNorm().Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            network.AvgPooling(8, 8);
            network.Output(CategoriesCifar10, param.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            network.Description = description;
            return network;
        }

        private static void LoadAt(string path, CpuTensor<byte> x, CpuTensor<byte> y, int indexFirst)
        {
            var b = File.ReadAllBytes(path);
            for (int count = 0; count < 10000; ++count)
            {
                int bIndex = count * (1 + 32 * 32 * 3);
                int xIndex = (count + indexFirst) * x.MultDim0;
                int yIndex = (count + indexFirst) * y.MultDim0;
                y[yIndex] = b[bIndex];
                for (int j = 0; j < 32 * 32 * 3; ++j)
                {
                    x[xIndex + j] = b[bIndex + 1 + j];
                }
            }
        }
    }
}
