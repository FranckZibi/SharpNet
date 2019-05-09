using System.Diagnostics;
using System.Linq;
using SharpNet.Datasets;
using SharpNet.GPU;

namespace SharpNet
{
    public partial class Network
    {
        public Network ResNetV1(int[] nbResBlocks, bool useBottleNeck, int[] xShape, int nbCategories)
        {
            Debug.Assert(Layers.Count == 0);
            var activationFunction = cudnnActivationMode_t.CUDNN_ACTIVATION_RELU;
            Input(xShape[1], xShape[2], xShape[3]);

            Convolution(64, 7, 2, 3, Config.lambdaL2Regularization, true);
            MaxPooling(2, 2);

            int stageC = 64; //number of channels for current stage
            for (int stageId = 0; stageId < nbResBlocks.Length; ++stageId)
            {
                int num_res_blocks = nbResBlocks[stageId];
                for (int res_block = 0; res_block < num_res_blocks; res_block += 1)
                {
                    int stride = (res_block == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = Layers.Last().LayerIndex;
                    if (useBottleNeck)
                    {
                        Convolution_BatchNorm_Activation(stageC, 1, stride, 0, Config.lambdaL2Regularization, activationFunction);
                        Convolution_BatchNorm_Activation(stageC, 3, 1, 1, Config.lambdaL2Regularization, activationFunction);
                        Convolution_BatchNorm(4 * stageC, 1, 1, 0, Config.lambdaL2Regularization);
                        Shortcut_IdentityConnection(startOfBlockLayerIndex, 4 * stageC, stride, Config.lambdaL2Regularization);
                        Activation(activationFunction);
                    }
                    else
                    {
                        Convolution_BatchNorm_Activation(stageC, 3, stride, 1, Config.lambdaL2Regularization, activationFunction);
                        Convolution_BatchNorm(stageC, 3, 1, 1, Config.lambdaL2Regularization);
                        Shortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, Config.lambdaL2Regularization);
                        Activation(activationFunction);
                    }
                }
                stageC *= 2;
            }
            GlobalAvgPooling();
            Output(nbCategories, Config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return this;
        }
     
        //implementation described in: https://arxiv.org/pdf/1512.03385.pdf
        public Network GetResNetV1_CIFAR10(int numResBlocks)
        {
            Debug.Assert(Layers.Count == 0);
            Input(CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width);

            Convolution_BatchNorm_Activation(16, 3, 1, 1, Config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);

            int stageC = 16; //number of channels for current stage
            for (int stageId = 0; stageId < 3; ++stageId)
            {
                for (int res_block = 0; res_block < numResBlocks; res_block += 1)
                {
                    int stride = (res_block == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = Layers.Last().LayerIndex;
                    Convolution_BatchNorm_Activation(stageC, 3, stride, 1, Config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                    Convolution_BatchNorm(stageC, 3, 1, 1, Config.lambdaL2Regularization);
                    Shortcut_IdentityConnection(startOfBlockLayerIndex, stageC, stride, Config.lambdaL2Regularization);
                    Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
                }
                stageC *= 2;
            }
            AvgPooling(8, 8);
            Output(CIFAR10.Categories, Config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return this;
        }

        //implementation described in: https://arxiv.org/pdf/1603.05027.pdf
        public Network GetResNetV2_CIFAR10(int numResBlocks)
        {
            //var networkName = "ResNet"+(9* numResBlocks +2)+ "V2_CIFAR10";
            Input(CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width);
            Convolution_BatchNorm_Activation(16, 3, 1, 1, Config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);

            int stageCIn = 16; //number of channels for current stage
            int stageCOut = 4 * stageCIn;

            for (int stageId = 0; stageId < 3; ++stageId)
            {
                for (int resBlock = 0; resBlock < numResBlocks; resBlock += 1)
                {
                    int stride = (resBlock == 0 && stageId != 0) ? 2 : 1;
                    var startOfBlockLayerIndex = Layers.Last().LayerIndex;
                    if (stageId == 0 && resBlock == 0)
                    {
                        Convolution(stageCIn, 1, stride, 0, Config.lambdaL2Regularization, true);
                    }
                    else
                    {
                        BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCIn, 1, stride, 0, Config.lambdaL2Regularization);
                    }
                    BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCIn, 3, 1, 1, Config.lambdaL2Regularization);
                    BatchNorm_Activation_Convolution(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, stageCOut, 1, 1, 0, Config.lambdaL2Regularization);
                    Shortcut_IdentityConnection(startOfBlockLayerIndex, stageCOut, stride, Config.lambdaL2Regularization);
                }
                stageCIn = stageCOut;
                stageCOut = 2 * stageCIn;
            }
            BatchNorm().Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            AvgPooling(8, 8);
            Output(CIFAR10.Categories, Config.lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);
            return this;
        }

    }
}
