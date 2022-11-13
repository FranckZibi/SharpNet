using System;
using System.Collections.Generic;
using SharpNet.Datasets;
using SharpNet.Datasets.CFM60;
using SharpNet.GPU;
using SharpNet.HPO;
using SharpNet.Layers;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks;

public class EncoderDecoder_NetworkSample: NetworkSample
{
    // ReSharper disable once EmptyConstructor
    public EncoderDecoder_NetworkSample()
    {
    }



    public int Total_TimeSteps => Use_Decoder ? Encoder_TimeSteps + Decoder_TimeSteps : Encoder_TimeSteps;


    #region Hyper-Parameters


    // ReSharper disable once UnusedParameter.Global
    public override void ApplyDataset(AbstractDatasetSample datasetSample)
    {
        base.ApplyDataset(datasetSample);
        if (datasetSample is CFM60DatasetSample cfm60DatasetSample)
        {
            cfm60DatasetSample.encoderDecoder_NetworkSample = this;
        }
    }


    #region Encoder
    private List<string> Encoder_FeatureNames;
    /// <summary>
    /// numb er of layer in the encoder.
    /// </summary>
    public void Encoder(int numLayers, int timeSteps, double dropoutRate)
    {
        if (numLayers == 1 && dropoutRate > 1e-6)
        {
            throw new ArgumentException("invalid dropoutRate (" + dropoutRate + ") for a 1 Layer Encoder");
        }
        Encoder_NumLayers = numLayers;
        Encoder_TimeSteps = timeSteps;
        Encoder_DropoutRate = dropoutRate;
    }

    // ReSharper disable once UnusedMember.Global
    public string EncDesc()
    {
        var res = "_encoder_" + Encoder_NumLayers + "_" + Encoder_TimeSteps;
        if (Encoder_DropoutRate > 1e-6)
        {
            res += "_" + Encoder_DropoutRate + "drop";
        }

        if (DropoutRate_Between_Encoder_And_Decoder > 1e-6)
        {
            res += "_then_" + DropoutRate_Between_Encoder_And_Decoder + "drop";
        }
        if (Use_Decoder)
        {
            res += "_then_decoder_" + Decoder_NumLayers + "_" + Decoder_TimeSteps;
            if (Decoder_DropoutRate > 1e-6)
            {
                res += "_" + Decoder_DropoutRate + "drop";
            }
        }
        if (DropoutRate_After_EncoderDecoder > 1e-6)
        {
            res += "_then_" + DropoutRate_After_EncoderDecoder + "drop";
        }

        res = res.Replace(".", "");
        return res;
    }

    public int Encoder_NumLayers = 2;
    public int Encoder_TimeSteps = 20;
    /// <summary>
    /// dropout to use for the encoder. A value of 0 means no dropout
    /// </summary>
    public double Encoder_DropoutRate = 0.2;
    #endregion



    #region Decoder 
    private List<string> Decoder_FeatureNames;
    public void Decoder(int numLayers, int timeSteps, double dropoutRate)
    {
        if (numLayers == 1 && dropoutRate > 1e-6)
        {
            throw new ArgumentException("invalid dropoutRate (" + dropoutRate + ") for a 1 Layer Decoder");
        }
        Decoder_NumLayers = numLayers;
        Decoder_TimeSteps = timeSteps;
        Decoder_DropoutRate = dropoutRate;
    }
    public bool Use_Decoder => Decoder_NumLayers >= 1 && Decoder_TimeSteps >= 1;
    public int Decoder_NumLayers = 0;
    /// <summary>
    /// Number of time steps for the decoder. 0 means that we are not using a decoder
    /// </summary>
    public int Decoder_TimeSteps = 0;
    public double Decoder_DropoutRate = 0.2;
    #endregion




    public bool EncoderDecoder_NetworkSample_UseGPU = true;

    public float ClipValueForGradients = 0;
    public bool DivideGradientsByTimeSteps = false;
    public bool Use_GRU_instead_of_LSTM = false;
    public bool Use_Bidirectional_RNN = true;

    public int Pid_EmbeddingDim = 4;
    public double DropoutRate_Between_Encoder_And_Decoder = 0.2;       //validated on 15-jan-2021
    public double DropoutRate_After_EncoderDecoder = 0.0;
    public bool UseBatchNorm2 = false;

    public int HiddenSize = 128;               //validated on 15-jan-2021

    public int DenseUnits = 200;              //validated on 21-jan-2021: -0.0038
    //public int DenseUnits = 100;

    public int Conv1DKernelWidth = 3;

    public ConvolutionLayer.PADDING_TYPE Conv1DPaddingType = ConvolutionLayer.PADDING_TYPE.SAME;

    public cudnnActivationMode_t ActivationFunctionAfterFirstDense = cudnnActivationMode_t.CUDNN_ACTIVATION_CLIPPED_RELU;
    public bool WithSpecialEndV1 = false;


    //public double PercentageInTraining = 0.68; //discarded on 16-jan-2021: +0.2759

    public bool UseConv1D = false;
    public bool UseBatchNormAfterConv1D = false;
    public bool UseReluAfterConv1D = false;
    //normalization
    public enum InputNormalizationEnum
    {
        NONE,
        Z_SCORE,
        DEDUCE_MEAN,
        BATCH_NORM_LAYER, //we'll use a batch norm layer for normalization
        DEDUCE_MEAN_AND_BATCH_NORM_LAYER
    }
    //entry.rel_vol.Length - 12

    public InputNormalizationEnum InputNormalizationType = InputNormalizationEnum.NONE;



    #endregion

    public override bool UseGPU => EncoderDecoder_NetworkSample_UseGPU;

    public override void BuildLayers(Network network, AbstractDatasetSample datasetSample)
    {
        network.Input_and_Embedding_if_required(datasetSample, Pid_EmbeddingDim, lambdaL2Regularization, ClipValueForGradients);
        //network.Input(Encoder_TimeSteps, Encoder_InputSize, -1);
        //if (Pid_EmbeddingDim >= 1)
        //{
        //    network.Embedding(new[] { CFM60Entry.DISTINCT_PID_COUNT }, new[] { Pid_EmbeddingDim }, new[] { 0 }, network.Sample.lambdaL2Regularization, ClipValueForGradients, DivideGradientsByTimeSteps);
        //}

        if (InputNormalizationType == InputNormalizationEnum.BATCH_NORM_LAYER || InputNormalizationType == InputNormalizationEnum.DEDUCE_MEAN_AND_BATCH_NORM_LAYER)
        {
            network.SwitchSecondAndThirdDimension(true);
            network.BatchNorm(0.99, 1e-5);
            network.SwitchSecondAndThirdDimension(false);
        }

        if (UseConv1D)
        {
            network.Conv1D(Encoder_TimeSteps, Conv1DKernelWidth, 1, Conv1DPaddingType, network.Sample.lambdaL2Regularization, true);

            if (UseBatchNormAfterConv1D)
            {
                network.BatchNorm(0.99, 1e-5);
            }
            if (UseReluAfterConv1D)
            {
                network.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_CLIPPED_RELU, null);
            }
        }

        //We add the Encoder
        if (Use_GRU_instead_of_LSTM)
        {
            network.GRU(HiddenSize, false, Use_Bidirectional_RNN, Encoder_NumLayers, Encoder_DropoutRate, Use_Decoder);
        }
        else
        {
            network.LSTM(HiddenSize, false, Use_Bidirectional_RNN, Encoder_NumLayers, Encoder_DropoutRate, Use_Decoder);
        }

        if (UseBatchNorm2)
        {
            network.SwitchSecondAndThirdDimension(true);

            //TO TEST:  network.BatchNorm(0.0, 1e-5);
            network.BatchNorm(0.99, 1e-5);
            network.SwitchSecondAndThirdDimension(false);
        }

        if (Use_Decoder && DropoutRate_Between_Encoder_And_Decoder >= 1e-6)
        {
            network.Dropout(DropoutRate_Between_Encoder_And_Decoder);
        }

        //We add the Decoder
        if (Use_Decoder)
        {
            int encoderLayerIndex = network.Layers.Count - 1;
            network.Input(Decoder_TimeSteps, ((DatasetSampleForTimeSeries)datasetSample).GetInputSize(false), -1);
            if (Pid_EmbeddingDim >= 1)
            {
                network.Embedding(new[] { CFM60Entry.DISTINCT_PID_COUNT }, new[] { Pid_EmbeddingDim }, new[] { 0 }, network.Sample.lambdaL2Regularization, ClipValueForGradients, DivideGradientsByTimeSteps);
            }
            network.DecoderLayer(encoderLayerIndex, Decoder_NumLayers, Decoder_DropoutRate);
        }


        if (DropoutRate_After_EncoderDecoder >= 1e-6)
        {
            network.Dropout(DropoutRate_After_EncoderDecoder);
        }

        network.Dense(DenseUnits, network.Sample.lambdaL2Regularization, true)
            .Activation(ActivationFunctionAfterFirstDense);
        network.Dense(1, network.Sample.lambdaL2Regularization, true);
        network.Flatten();

        if (WithSpecialEndV1)
        {
            network.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);
            network.Linear(2, 0);
            network.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_LN);
        }
    }


    public override bool FixErrors()
    {
        if (!base.FixErrors())
        {
            return false;
        }
        return true;
    }



    /// <summary>
    /// The default Search Space for this Model
    /// </summary>
    /// <returns></returns>
    // ReSharper disable once UnusedMember.Global
    public static Dictionary<string, object> DefaultSearchSpace()
    {
        var searchSpace = new Dictionary<string, object>
        {
            //uncomment appropriate one
            //{"LossFunction", "Rmse"},                     //for Regression Tasks: Rmse, Mse, Mae, etc.
            //{"LossFunction", "BinaryCrossentropy"},       //for binary classification
            //{"LossFunction", "CategoricalCrossentropy"},  //for multi class classification

            // Optimizer 
            { "OptimizerType", new[] { "AdamW", "SGD", "Adam" /*, "VanillaSGD", "VanillaSGDOrtho"*/ } },
            { "AdamW_L2Regularization", new[] { 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 } },
            { "SGD_usenesterov", new[] { true, false } },
            { "lambdaL2Regularization", new[] { 0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 } },

            // Learning Rate
            { "InitialLearningRate", AbstractHyperParameterSearchSpace.Range(1e-5f, 1f, AbstractHyperParameterSearchSpace.range_type.normal) },
            // Learning Rate Scheduler
            { "LearningRateSchedulerType", new[] { "CyclicCosineAnnealing", "OneCycle", "Linear" } },
            { "EmbeddingDim", new[] { 0, 4, 8, 12 } },
            //{"weight_norm", new[]{true, false}},
            //{"leaky_relu", new[]{true, false}},
            { "dropout_top", new[] { 0, 0.1, 0.2 } },
            { "dropout_mid", new[] { 0, 0.3, 0.5 } },
            { "dropout_bottom", new[] { 0, 0.2, 0.4 } },
            { "BatchSize", new[] { 256, 512, 1024, 2048 } },
            { "NumEpochs", new[] { 15 } },

            //Dataset specific
            { "KFold", 2 },
            //{"PercentageInTraining", new[]{0.8}},
        };
        return searchSpace;
    }

}