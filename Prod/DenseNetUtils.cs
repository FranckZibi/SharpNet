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
                10,
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
            //return Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\Network_15272_21.txt");

            var network = DenseNet(
                new [] {1, ChannelsCifar10, HeightCifar10, WidthCifar10},
                10,
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

        private const int ChannelsCifar10 = 3;
        private const int HeightCifar10 = 32;
        private const int WidthCifar10 = HeightCifar10;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="xShape"></param>
        /// <param name="nbCategories"></param>
        /// <param name="nbConvBlocksInEachDenseBlock"></param>
        /// <param name="growthRate"></param>
        /// <param name="filtersCount"></param>
        /// <param name="useBottleneckInEachConvBlock"></param>
        /// <param name="compression"></param>
        /// <param name="dropProbability"></param>
        /// <param name="subsampleInitialBlock"></param>
        /// <param name="param"></param>
        /// <param name="logger"></param>
        /// <returns></returns>
        /*
                    ''' Build the DenseNet model
                    Args:
                        nb_classes: number of classes
                        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
                        include_top: flag to include the final Dense layer
                        depth: number or layers
                        nb_dense_block: number of dense blocks to add to end (generally = 3)
                        growth_rate: number of filters to add per dense block
                        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
                        nb_layers_per_block: number of layers in each dense block.
                                Can be a -1, positive integer or a list.
                                If -1, calculates nb_layer_per_block from the depth of the network.
                                If positive integer, a set number of layers per dense block.
                                If list, nb_layer is used as provided. Note that list size must
                                be (nb_dense_block + 1)
                        bottleneck: add bottleneck blocks
                        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
                        dropout_rate: dropout rate
                        weight_decay: weight decay rate
                        subsample_initial_block: Set to True to subsample the initial convolution and
                                add a MaxPool2D before the dense blocks are added.
                        subsample_initial:
                        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                                Note that if sigmoid is used, classes must be 1.
                    Returns: keras tensor with nb_layers of conv_block appended

         */
        public static Network DenseNet(int[] xShape, int nbCategories,
                bool subsampleInitialBlock,
                int[] nbConvBlocksInEachDenseBlock,
                bool useBottleneckInEachConvBlock,
                int growthRate,
                int filtersCount,
                double compression, // 1.0 = no compression
                double? dropProbability,
                DenseNetMetaParameters param, 
                Logger logger)
        {

            var networkConfig = param.Config();
            networkConfig.Logger = logger ?? Logger.ConsoleLogger;
            //networkConfig.ForceTensorflowCompatibilityMode = true;


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
                .Dense(nbCategories, param.lambdaL2Regularization)
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
            /*
            ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            Args:
                x: keras tensor
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                growth_rate: growth rate
                bottleneck: bottleneck block
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                grow_nb_filters: flag to decide to allow number of filters to grow
                return_concat_list: return the list of feature maps along with the actual output
            Returns: keras tensor with nb_layers of conv_block appended
            '''
             */
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
            /*
            ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
            Args:
            ip: Input keras tensor
            nb_filter: number of filters
            bottleneck: add bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
            */
            if (bottleneck)
            {
                //!D check why  4*growthRate
                network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 4 * growthRate, 1, 1, 0, lambdaL2Regularization);
            }
            network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, growthRate, 3, 1, 1, lambdaL2Regularization);
            if (dropProbability.HasValue)
            {
                network.Dropout(dropProbability.Value);
            }
            return network;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="network"></param>
        /// <param name="compression"></param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        /*     ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
        Args:
        ip: keras tensor
        nb_filter: number of filters
            compression: calculated as 1 - reduction.Reduces the number of feature maps
        in the transition block.
            dropout_rate: dropout rate
        weight_decay: weight decay factor
            Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
        */
        private static Network AddTransitionBlock(Network network, double compression, double lambdaL2Regularization)
        {
            var filtersCount = network.Layers.Last().OutputShape(1)[1];
            return network
                .BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, (int)Math.Round(filtersCount*compression), 1, 1, 0, lambdaL2Regularization)
                .AvgPooling(2, 2);
        }
    }
}
