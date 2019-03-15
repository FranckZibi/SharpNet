using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNet.Pictures;

namespace SharpNet
{
    public static class ResNetUtils
    {
        public static Network ResNet18_V1(int[] xShape, int nbCategories, bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return ResNetV1(new[] { 2, 2, 2, 2 }, false, xShape, nbCategories, useGPU, useDoublePrecision, logger);
        }
        public static Network ResNet34_V1(int[] xShape, int nbCategories, bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return ResNetV1(new[] { 3, 4, 6, 3 }, false, xShape, nbCategories, useGPU, useDoublePrecision, logger);
        }
        public static Network ResNet50_V1(int[] xShape, int nbCategories, bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return ResNetV1(new[] { 3, 4, 6, 3 }, true, xShape, nbCategories, useGPU, useDoublePrecision, logger);
        }
        public static Network ResNet101_V1(int[] xShape, int nbCategories, bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return ResNetV1(new[] { 3, 4, 23, 3 }, true, xShape, nbCategories, useGPU, useDoublePrecision, logger);
        }
        public static Network ResNet152_V1(int[] xShape, int nbCategories, bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return ResNetV1(new[] { 3, 8, 36, 3 }, true, xShape, nbCategories, useGPU, useDoublePrecision, logger);
        }
        public static Network ResNet20V1_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(3, useGPU, useDoublePrecision, nameof(ResNet20V1_CIFAR10), logger);
        }
        public static Network ResNet32V1_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(5, useGPU, useDoublePrecision, nameof(ResNet32V1_CIFAR10), logger);
        }
        public static Network ResNet44V1_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(7, useGPU, useDoublePrecision, nameof(ResNet44V1_CIFAR10), logger);
        }
        public static Network ResNet56V1_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(9, useGPU, useDoublePrecision, nameof(ResNet56V1_CIFAR10), logger);
        }
        public static Network ResNet110V1_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(18, useGPU, useDoublePrecision, nameof(ResNet110V1_CIFAR10), logger);
        }
        public static Network ResNet164V1_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(27, useGPU, useDoublePrecision, nameof(ResNet164V1_CIFAR10), logger);
        }
        public static Network ResNet1202V1_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV1_CIFAR10(200, useGPU, useDoublePrecision, nameof(ResNet1202V1_CIFAR10), logger);
        }



        public static Network ResNet20V2_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(2, useGPU, useDoublePrecision, nameof(ResNet20V2_CIFAR10), logger);
        }
        public static Network ResNet56V2_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(6, useGPU, useDoublePrecision, nameof(ResNet56V2_CIFAR10), logger);
        }
        public static Network ResNet110V2_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(12, useGPU, useDoublePrecision, nameof(ResNet110V2_CIFAR10), logger);
        }
        public static Network ResNet164V2_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(18, useGPU, useDoublePrecision, nameof(ResNet164V2_CIFAR10), logger);
        }
        public static Network ResNet1001V2_CIFAR10(bool useGPU = true, bool useDoublePrecision = false, Logger logger = null)
        {
            return GetResNetV2_CIFAR10(111, useGPU, useDoublePrecision, nameof(ResNet1001V2_CIFAR10), logger);
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
        public static void ToWorkingSet(CpuTensor<byte> x, CpuTensor<byte> y, out CpuTensor<float> xWorkingSet, out CpuTensor<float> yWorkingSet)
        {
            xWorkingSet = x.Select(b => b / 255.0f);
            yWorkingSet = y.ToCategorical(1.0f, out _);
        }

        public static LearningRateScheduler Cifar10LearningRateScheduler()
        {
            return UpdatedCifar10LearningRateScheduler();
            //var initialLearningRate = 0.1;
            //return LearningRateScheduler.ConstantByInterval(1, initialLearningRate, 80, initialLearningRate / 10, 120, initialLearningRate / 100);
        }

        public static LearningRateScheduler UpdatedCifar10LearningRateScheduler()
        {
            var initialLearningRate = 0.1;
            return LearningRateScheduler.ConstantByInterval(1, initialLearningRate/10.0, 2, initialLearningRate, 80, initialLearningRate / 10, 120, initialLearningRate / 100, 160, initialLearningRate / 1000, 180, initialLearningRate / 2000);
        }
        public static LearningRateScheduler ResNet110LearningRateScheduler()
        {
            return UpdatedCifar10LearningRateScheduler();
            //var initialLearningRate = 0.1;
            //return LearningRateScheduler.ConstantByInterval(1, initialLearningRate/10, 2, initialLearningRate, 80, initialLearningRate / 10, 120, initialLearningRate / 100);
        }

        private static Network ResNetV1(int[] nbResBlocks, bool useBottleNeck, int[] xShape, int nbCategories, bool useGPU, bool useDoublePrecision, Logger logger)
        {
            var activationFunction = cudnnActivationMode_t.CUDNN_ACTIVATION_RELU;

            var networkConfig = new NetworkConfig(useGPU) { UseDoublePrecision = useDoublePrecision, LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy, Logger = logger ?? Logger.ConsoleLogger };
            var network = new Network(networkConfig.WithSGD(0.9, 0.0001, true), ResNetImageDataGenerator());
            network.AddInput(xShape[1], xShape[2], xShape[3]);

            const double lambdaL2Regularization = 1e-4;
            network.AddConvolution(64, 7, 2, 3, lambdaL2Regularization);
            network.AddMaxPooling(2, 2);

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
                        network.AddConvolution_BatchNorm_Activation(stageC, 1, stride, 0, lambdaL2Regularization, activationFunction);
                        network.AddConvolution_BatchNorm_Activation(stageC, 3, 1, 1, lambdaL2Regularization, activationFunction);
                        network.AddConvolution_BatchNorm(4 * stageC, 1, 1, 0, lambdaL2Regularization);
                        network.AddShortcut_IdentityConnection(startOfBlockLayerIndex, 4 * stageC, stride, lambdaL2Regularization);
                        network.AddActivation(activationFunction);
                    }
                    else
                    {
                        network.AddConvolution_BatchNorm_Activation(stageC, 3, stride, 1, lambdaL2Regularization, activationFunction);
                        network.AddConvolution_BatchNorm(stageC, 3, 1, 1, lambdaL2Regularization);
                        network.AddShortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, lambdaL2Regularization);
                        network.AddActivation(activationFunction);
                    }
                }
                stageC *= 2;
            }
            network.AddGlobalAvgPooling();
            network.AddOutput(nbCategories, lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return network;
        }
        private static ImageDataGenerator ResNetImageDataGenerator()
        {
            return new ImageDataGenerator(0.1, 0.1, true, false, ImageDataGenerator.FillModeEnum.Nearest, 0.0);
        }
        private const int ChannelsCifar10 = 3;
        private const int HeightCifar10 = 32;
        private const int WidthCifar10 = HeightCifar10;
        private const int CategoriesCifar10 = 10;

        //implementation described in: https://arxiv.org/pdf/1512.03385.pdf
        private static Network GetResNetV1_CIFAR10(int numResBlocks, bool useGpu, bool useDoublePrecision, string description, Logger logger)
        {
            var networkConfig = new NetworkConfig(useGpu) { UseDoublePrecision = useDoublePrecision, LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy, Logger = logger ?? Logger.ConsoleLogger };

            const double lambdaL2Regularization = 1e-4;
            const bool useNesterov = false;

            var network = new Network(networkConfig.WithSGD(0.9, 0, useNesterov), ResNetImageDataGenerator());
            network.AddInput(ChannelsCifar10, HeightCifar10, WidthCifar10);

            network.AddConvolution_BatchNorm_Activation(16, 3, 1, 1, lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);

            int stageC = 16; //number of channels for current stage
            for (int stageId = 0; stageId < 3; ++stageId)
            {
                for (int res_block = 0; res_block < numResBlocks; res_block += 1)
                {
                    int stride = (res_block == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = network.Layers.Last().LayerIndex;
                    network.AddConvolution_BatchNorm_Activation(stageC, 3, stride, 1, lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                    network.AddConvolution_BatchNorm(stageC, 3, 1, 1, lambdaL2Regularization);
                    network.AddShortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, lambdaL2Regularization);
                    network.AddActivation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                }
                stageC *= 2;
            }
            network.AddAvgPooling(8, 8);
            network.AddOutput(CategoriesCifar10, lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            network.Description = description;
            return network;
        }

        //implementation described in: https://arxiv.org/pdf/1603.05027.pdf
        private static Network GetResNetV2_CIFAR10(int numResBlocks, bool useGpu, bool useDoublePrecision, string description, Logger logger)
        {
            var networkConfig = new NetworkConfig(useGpu) { UseDoublePrecision = useDoublePrecision, LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy, Logger = logger ?? Logger.ConsoleLogger };

            const double lambdaL2Regularization = 1e-4;
            const bool useNesterov = false;

            var network = new Network(networkConfig.WithSGD(0.9, 0, useNesterov), ResNetImageDataGenerator());
            network.AddInput(ChannelsCifar10, HeightCifar10, WidthCifar10);
            network.AddConvolution(16, 3, 1, 1, lambdaL2Regularization);

            int stageCIn = 16; //number of channels for current stage
            int stageCOut = 4 * stageCIn;

            for (int stageId = 0; stageId < 3; ++stageId)
            {
                for (int resBlock = 0; resBlock < numResBlocks; resBlock += 1)
                {
                    int stride = (resBlock == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = network.Layers.Last().LayerIndex;
                    network.AddBatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCIn, 1, stride, 0, lambdaL2Regularization);
                    network.AddBatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCIn, 3, 1, 1, lambdaL2Regularization);
                    network.AddBatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCOut, 1, 1, 0, lambdaL2Regularization);
                    network.AddShortcut_IdentityConnection(startOfBlockLayerIndex, stageCOut, stride, lambdaL2Regularization);
                }
                stageCIn = stageCOut;
                stageCOut = 2 * stageCIn;
            }
            network.AddBatchNorm().AddActivation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            network.AddAvgPooling(8, 8);
            network.AddOutput(CategoriesCifar10, lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
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
