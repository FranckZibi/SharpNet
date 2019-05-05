using System;
using System.Linq;
using SharpNet.GPU;

namespace SharpNet
{
    //DenseNet implementation as described in https://arxiv.org/pdf/1608.06993.pdf
    public static class DenseNetUtils
    {
        //!D TO REMOVE
        public static Network DenseNet10_CIFAR10(DenseNetMetaParameters param, Logger logger = null)
        {
            var network = DenseNet(
                new[] { 1, ChannelsCifar10, HeightCifar10, WidthCifar10 },
                CategoriesCifar10,
                false,
                new[] { 2, 2, 2},
                false,
                12,
                -1,
                1.0,
                null,
                param, logger);
            network.Description = "DenseNet10_CIFAR10" + param.ExtraDescription;
            return network;
        }



        public static Network DenseNet40_CIFAR10(DenseNetMetaParameters param, Logger logger = null)
        {
            //return Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\Network_15576_14.txt");
            var network = DenseNet(
                new [] {1, ChannelsCifar10, HeightCifar10, WidthCifar10},
                CategoriesCifar10,
                false,
                new [] {12, 12, 12},
                false,
                12,
                -1,
                1.0,
                null,
                param, logger);
            network.Description = "DenseNet40_CIFAR10"+param.ExtraDescription;
            return network;
        }
        public static Network DenseNet100_BC_k_12_CIFAR10(DenseNetMetaParameters param, Logger logger = null)
        {
            var network = DenseNet(
                new [] { 1, ChannelsCifar10, HeightCifar10, WidthCifar10 },
                10,
                false,
                new [] { 32, 32, 32},
                true,
                12,
                -1,
                0.5,
                null,
                param, logger);
            network.Description = "DenseNet100_BC_k_12_CIFAR10" + param.ExtraDescription;
            return network;
        }

        private const int CategoriesCifar10 = 10;
        private const int ChannelsCifar10 = 3;
        private const int HeightCifar10 = 32;
        private const int WidthCifar10 = HeightCifar10;

        /// <summary>
        /// buid a DenseNet
        /// </summary>
        /// <param name="xShape"></param>
        /// <param name="nbCategories"></param>
        /// <param name="nbConvBlocksInEachDenseBlock"></param>
        /// <param name="growthRate"></param>
        /// <param name="filtersCount">-1 means 2*growthRate</param>
        /// <param name="useBottleneckInEachConvBlock"></param>
        /// <param name="compression"> 1.0 = no compression</param>
        /// <param name="dropProbability"></param>
        /// <param name="subsampleInitialBlock"></param>
        /// <param name="param"></param>
        /// <param name="logger"></param>
        /// <returns></returns>
        public static Network DenseNet(int[] xShape, int nbCategories,
                bool subsampleInitialBlock,
                int[] nbConvBlocksInEachDenseBlock,
                bool useBottleneckInEachConvBlock,
                int growthRate,
                int filtersCount,
                double compression,
                double? dropProbability,
                DenseNetMetaParameters param, 
                Logger logger)
        {

            var networkConfig = param.Config();
            networkConfig.Logger = logger ?? Logger.ConsoleLogger;
            var network = new Network(networkConfig, param.DataGenerator());
            network.Input(xShape[1], xShape[2], xShape[3]);

            if (filtersCount <= 0)
            {
                filtersCount = 2 * growthRate;
            }
            if (subsampleInitialBlock)
            {
                network.Convolution(filtersCount, 7, 2, 3, param.lambdaL2Regularization, false)
                    .BatchNorm()
                    .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                    .MaxPooling(3, 2);
            }
            else
            {
                network.Convolution(filtersCount, 3, 1, 1, param.lambdaL2Regularization, false);
                //network.Convolution(filtersCount, 3, 1, 1, 0.0, false);
            }

            for (int denseBlockId = 0; denseBlockId < nbConvBlocksInEachDenseBlock.Length; ++denseBlockId)
            {
                AddDenseBlock(network, nbConvBlocksInEachDenseBlock[denseBlockId], 
                    growthRate, useBottleneckInEachConvBlock, dropProbability, param.lambdaL2Regularization);
                if (denseBlockId != nbConvBlocksInEachDenseBlock.Length - 1)
                {
                    //the last dense block does not have a transition block
                    AddTransitionBlock(network, compression, param.lambdaL2Regularization);
                }
                filtersCount = (int)Math.Round(filtersCount * compression);
            }

            //we add the classification layer part
            network
                .BatchNorm()
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .GlobalAvgPooling()
                //!D check if lambdaL2Regularization should be 0
                .Dense(nbCategories, 0.0)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return network;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="network"></param>
        /// <param name="nbConvBlocksInDenseBlock"></param>
        /// <param name="growthRate"></param>
        /// <param name="bottleneck"></param>
        /// <param name="dropProbability"></param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        public static Network AddDenseBlock(Network network, int nbConvBlocksInDenseBlock, 
            int growthRate,
            bool bottleneck,
            double? dropProbability, double lambdaL2Regularization)
        {
            for (int convBlockId = 0; convBlockId < nbConvBlocksInDenseBlock; ++convBlockId)
            {
                var previousLayerIndex1 = network.Layers.Last().LayerIndex;
                AddConvolutionBlock(network, growthRate, bottleneck, dropProbability, lambdaL2Regularization);
                var previousLayerIndex2 = network.Layers.Last().LayerIndex;
                network.ConcatenateLayer(previousLayerIndex1, previousLayerIndex2);
            }

            return network;
        }


        /// <summary>
        /// Add a Convolution block in a Dense Network
        /// </summary>
        /// <param name="network"></param>
        /// <param name="growthRate"></param>
        /// <param name="bottleneck"></param>
        /// <param name="dropProbability">optional value, if presents will add a Dropout layer at the end of the block</param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        public static Network AddConvolutionBlock(Network network, int growthRate, bool bottleneck, double? dropProbability, double lambdaL2Regularization)
        {
            if (bottleneck)
            {
                network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 4 * growthRate, 1, 1, 0, lambdaL2Regularization);
            }
            //network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, growthRate, 3, 1, 1, lambdaL2Regularization);
            network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, growthRate, 3, 1, 1, 0.0);
            if (dropProbability.HasValue)
            {
                network.Dropout(dropProbability.Value);
            }
            return network;
        }


        /// <summary>
        /// Add a transition block in a Dense Net network
        /// </summary>
        /// <param name="network"></param>
        /// <param name="compression"></param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        private static void AddTransitionBlock(Network network, double compression, double lambdaL2Regularization)
        {
            var filtersCount = network.Layers.Last().OutputShape(1)[1];
            network
                .BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, (int)Math.Round(filtersCount*compression), 1, 1, 0, lambdaL2Regularization)
                .AvgPooling(2, 2);
        }
    }
}
