using System.Linq;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;

// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks;

public class NetworkSample_1DCNN : NetworkSample
{
    // ReSharper disable once EmptyConstructor
    public NetworkSample_1DCNN()
    {
    }

    #region Hyper-Parameters
    //Embedding of the categorical features
    public int DefaultEmbeddingDim = 10;
    public int hidden_size = 512;
    //n_categories: [ 3, 3, 3, 5, 136, 5, 60, 3, 3],
    public int channel_1 = 256;
    public int channel_2 = 512;
    public int channel_3 = 512;
    public float dropout_top = 0.1f;
    public float dropout_mid = 0.3f;
    public float dropout_bottom = 0.2f;
    public bool weight_norm = false;
    public bool two_stage = false;
    public int kernel1 = 5;
    public bool leaky_relu = false;
    //public double  batchNorm_momentum = 0.1;
    public double batchNorm_momentum = 0.99;

    public bool NetworkSample_1DCNN_UseGPU = true;

    #endregion

    public override bool UseGPU => NetworkSample_1DCNN_UseGPU;


    public override void BuildLayers(Network nn, AbstractDatasetSample datasetSample)
    {
        //nn.PropagationManager.LogPropagation = true;
        //nn.Config.DisplayTensorContentStats = true;

        //int channel_1_reshape = hidden_size / channel_1;
        int cha_po_1 = hidden_size / channel_1 / 2;
        //int cha_po_2 = hidden_size / channel_1 / 2 / 2 * channel_3;

        nn.Input_and_Embedding_if_required(datasetSample, DefaultEmbeddingDim, lambdaL2Regularization);

        //expand(x)
        nn.BatchNorm(batchNorm_momentum, 1e-5);
        nn.Dropout(dropout_top);
        nn.Dense(hidden_size, lambdaL2Regularization, false);
        nn.BatchNorm(batchNorm_momentum, 1e-5);//nn.utils.weight_norm(, dim = None), //!D was Weight Norm
        nn.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU); //nn.CELU(0.06) if celu else nn.ReLU()

        nn.Reshape(channel_1, hidden_size / channel_1, -1);  //x = x.reshape(x.shape[0], channel_1, channel_1_reshape)

        //conv1 
        nn.BatchNorm(batchNorm_momentum, 1e-5);
        nn.Dropout(dropout_top);
        nn.Conv1D(channel_2, kernel1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, false); //nn.Conv1d(channel_1, channel_2, kernel_size = kernel1, stride = 1, padding = kernel1 / 2, bias = False);
        if (weight_norm) { nn.BatchNorm(batchNorm_momentum, 1e-5); }
        nn.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
        int AdaptiveAvgPool1d_input_size = nn.Layers.Last().OutputShape(1).Last();
        int AdaptiveAvgPool1d_Stride = (AdaptiveAvgPool1d_input_size / cha_po_1);
        int AdaptiveAvgPool1d_Kernel_size = AdaptiveAvgPool1d_input_size - (cha_po_1 - 1) * AdaptiveAvgPool1d_Stride;
        //Padding = 0
        nn.AvgPooling(1, AdaptiveAvgPool1d_Kernel_size, 1, AdaptiveAvgPool1d_Stride); //nn.AdaptiveAvgPool1d(output_size = cha_po_1);
        if (weight_norm) { nn.BatchNorm(batchNorm_momentum, 1e-5); }
        nn.Dropout(dropout_top);

        nn.Conv1D(channel_2, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true); //nn.Conv1d(channel_2, channel_2, kernel_size = 3, stride = 1, padding = 1, bias = True);
        nn.BatchNorm(batchNorm_momentum, 1e-5); //_norm();
        nn.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);

        if (two_stage)
        {
            //conv2 
            nn.BatchNorm(batchNorm_momentum, 1e-5);
            nn.Dropout(dropout_mid);
            nn.Conv1D(channel_2, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true); //nn.Conv1d(channel_2, channel_2, kernel_size = 3, stride = 1, padding = 1, bias = True);
            if (weight_norm) { nn.BatchNorm(batchNorm_momentum, 1e-5); }
            nn.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            nn.BatchNorm(batchNorm_momentum, 1e-5);
            nn.Dropout(dropout_bottom);
            nn.Conv1D(channel_3, 5, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true); //nn.Conv1d(channel_2, channel_3, kernel_size = 5, stride = 1, padding = 2, bias = True);
            if (weight_norm) { nn.BatchNorm(batchNorm_momentum, 1e-5); }
            nn.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU); // * x
        }

        nn.MaxPooling(1, 2, 1, 2); //MaxPool1d(kernel_size = 4, stride = 2, padding = 1);

        nn.Flatten();

        if (leaky_relu)
        {
            nn.BatchNorm(batchNorm_momentum, 1e-5);
            nn.Dropout(dropout_bottom);
            nn.Dense(datasetSample.NumClass, lambdaL2Regularization, false);
            if (weight_norm) { nn.BatchNorm(batchNorm_momentum, 1e-5); }
            nn.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU); // * x
        }
        else
        {
            nn.BatchNorm(batchNorm_momentum, 1e-5);
            nn.Dropout(dropout_bottom);
            nn.Dense(datasetSample.NumClass, lambdaL2Regularization, false);
            if (weight_norm) { nn.BatchNorm(batchNorm_momentum, 1e-5); }
        }
        nn.Activation(datasetSample.ActivationForLastLayer);
    }
}
