using System;
using System.Diagnostics;
using System.Linq;
using SharpNet.GPU;

namespace SharpNet
{
    //DenseNet implementation as described in https://arxiv.org/pdf/1608.06993.pdf
    public partial class Network
    {
        /// <summary>
        /// buid a DenseNet
        /// </summary>
        /// <param name="xShape"></param>
        /// <param name="nbCategories"></param>
        /// <param name="subsampleInitialBlock"></param>
        /// <param name="nbConvBlocksInEachDenseBlock"></param>
        /// <param name="useBottleneckInEachConvBlock"></param>
        /// <param name="growthRate"></param>
        /// <param name="compression"> 1.0 = no compression</param>
        /// <param name="dropProbability"></param>
        /// <returns></returns>
        public void AddDenseNet(int[] xShape, int nbCategories,
            bool subsampleInitialBlock,
            int[] nbConvBlocksInEachDenseBlock,
            bool useBottleneckInEachConvBlock,
            int growthRate,
            double compression,
            double? dropProbability)
        {
            Debug.Assert(Layers.Count == 0);
            Input(xShape[1], xShape[2], xShape[3]);
            var filtersCount = 2 * growthRate;
            if (subsampleInitialBlock)
            {
                Convolution(filtersCount, 7, 2, 3, Config.lambdaL2Regularization, false)
                    .BatchNorm()
                    .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                    .MaxPooling(3, 2);
            }
            else
            {
                Convolution(filtersCount, 3, 1, 1, Config.lambdaL2Regularization, false);
                //Convolution(filtersCount, 3, 1, 1, 0.0, false);
            }

            for (int denseBlockId = 0; denseBlockId < nbConvBlocksInEachDenseBlock.Length; ++denseBlockId)
            {
                AddDenseBlock(nbConvBlocksInEachDenseBlock[denseBlockId], growthRate, useBottleneckInEachConvBlock, dropProbability, Config.lambdaL2Regularization);
                if (denseBlockId != nbConvBlocksInEachDenseBlock.Length - 1)
                {
                    //the last dense block does not have a transition block
                    AddTransitionBlock(compression, Config.lambdaL2Regularization);
                }
                filtersCount = (int)Math.Round(filtersCount * compression);
            }

            //we add the classification layer part
            this
                .BatchNorm()
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .GlobalAvgPooling()
                //.Dense(nbCategories, 0.0) //!D check if lambdaL2Regularization should be 0
                .Dense(nbCategories, Config.lambdaL2Regularization)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="nbConvBlocksInDenseBlock"></param>
        /// <param name="growthRate"></param>
        /// <param name="bottleneck"></param>
        /// <param name="dropProbability"></param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        private void AddDenseBlock(
            int nbConvBlocksInDenseBlock, 
            int growthRate,
            bool bottleneck,
            double? dropProbability, 
            double lambdaL2Regularization)
        {
            for (int convBlockId = 0; convBlockId < nbConvBlocksInDenseBlock; ++convBlockId)
            {
                var previousLayerIndex1 = Layers.Last().LayerIndex;
                AddConvolutionBlock(growthRate, bottleneck, dropProbability, lambdaL2Regularization);
                var previousLayerIndex2 = Layers.Last().LayerIndex;
                ConcatenateLayer(previousLayerIndex1, previousLayerIndex2);
            }
        }


        /// <summary>
        /// Add a Convolution block in a Dense Network
        /// </summary>
        /// <param name="growthRate"></param>
        /// <param name="bottleneck"></param>
        /// <param name="dropProbability">optional value, if presents will add a Dropout layer at the end of the block</param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        private void AddConvolutionBlock(int growthRate, bool bottleneck, double? dropProbability, double lambdaL2Regularization)
        {
            if (bottleneck)
            {
                BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 4 * growthRate, 1, 1, 0, lambdaL2Regularization);
            }
            //network.BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, growthRate, 3, 1, 1, lambdaL2Regularization);
            BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, growthRate, 3, 1, 1, 0.0);
            if (dropProbability.HasValue)
            {
                Dropout(dropProbability.Value);
            }
        }


        /// <summary>
        /// Add a transition block in a Dense Net network
        /// </summary>
        /// <param name="compression"></param>
        /// <param name="lambdaL2Regularization"></param>
        /// <returns></returns>
        private void AddTransitionBlock(double compression, double lambdaL2Regularization)
        {
            var filtersCount = Layers.Last().OutputShape(1)[1];
            BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, (int)Math.Round(filtersCount*compression), 1, 1, 0, lambdaL2Regularization)
                .AvgPooling(2, 2);
        }
    }
}
