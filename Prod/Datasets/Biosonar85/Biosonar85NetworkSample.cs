using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;

namespace SharpNet.Datasets.Biosonar85;

public class Biosonar85NetworkSample : NetworkSample
{
    public double batchNorm_momentum = 0.99;
    public bool Use_MaxPooling = false;
    public bool Use_AvgPooling = false;


    public override bool FixErrors()
    {
        if (!base.FixErrors())
        {
            return false;
        }

        if (Use_AvgPooling && Use_MaxPooling)
        {
            if (Utils.RandomCoinFlip())
            {
                Use_MaxPooling = true;
                Use_AvgPooling = false;
            }
            else
            {
                Use_MaxPooling = false;
                Use_AvgPooling = true;
            }
        }

        return true;
    }

    public Biosonar85NetworkSample()
    {
    }
    public override void BuildLayers(Network nn, AbstractDatasetSample datasetSample)
    {
        nn.Input(datasetSample.GetInputShapeOfSingleElement());

        nn.Convolution(8, 5, 2, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true); //padding =2
        nn.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
        nn.BatchNorm(batchNorm_momentum, 1e-5);

        nn.Convolution(16, 3, 2, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true); //padding =1
        nn.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
        nn.BatchNorm(batchNorm_momentum, 1e-5);

        nn.Convolution(32, 3, 2, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true); //padding =1
        nn.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
        nn.BatchNorm(batchNorm_momentum, 1e-5);

        nn.Convolution(64, 3, 2, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true); //padding =1
        nn.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
        nn.BatchNorm(batchNorm_momentum, 1e-5);

        if (Use_AvgPooling)
        {
            nn.GlobalAvgPooling();
        }
        if (Use_MaxPooling)
        {
            nn.GlobalMaxPooling();
        }
        nn.Flatten();
        nn.Dense(datasetSample.NumClass, lambdaL2Regularization, true);
        nn.Activation(datasetSample.GetActivationForLastLayer(nn.ModelSample.GetObjective()));
    }

}