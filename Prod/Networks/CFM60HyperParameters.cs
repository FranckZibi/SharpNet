﻿using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using SharpNet.Datasets.CFM60;
using SharpNet.GPU;
using SharpNet.HyperParameters;
using SharpNet.Layers;
using SharpNet.MathTools;

namespace SharpNet.Networks;

/// <summary>
/// Network support for CFM60 challenge
/// </summary>
[SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
public class CFM60HyperParameters : AbstractSample
{
    public CFM60HyperParameters() : base(new HashSet<string>())
    {
    }

    public override bool PostBuild()
    {
        return true; //TODO
    }

    private int GetInputSize(bool isEncoderInputSize)
    {
        if (isEncoderInputSize && Encoder_FeatureNames != null)
        {
            return Encoder_FeatureNames.Count;
        }
        if (!isEncoderInputSize && Decoder_FeatureNames != null)
        {
            return Decoder_FeatureNames.Count;
        }

        var featureNames = new List<string>();

        int result = 0;

        //pid embedding
        if (Pid_EmbeddingDim >= 1) { ++result; featureNames.Add(nameof(Pid_EmbeddingDim)); }

        //y estimate
        if (Use_prev_Y && isEncoderInputSize) { ++result; featureNames.Add("prev_y"); }
        if (Use_y_LinearRegressionEstimate) { ++result; featureNames.Add("y_LinearRegressionEstimate"); }
        if (Use_mean_pid_y) { ++result; featureNames.Add("mean(pid_y)"); }
        if (Use_volatility_pid_y) { ++result; featureNames.Add("vol(pid_y)"); }
        if (Use_variance_pid_y) { ++result; featureNames.Add("var(pid_y)"); }

        //day/year
        if (Use_day) {++result; featureNames.Add("day/"+(int)Use_day_Divider);}
        if (Use_fraction_of_year) { ++result; featureNames.Add("fraction_of_year"); }
        if (Use_year_Cyclical_Encoding) {
            ++result;
            featureNames.Add("sin_year");
            ++result;
            featureNames.Add("cos_year");
        }
        if (Use_EndOfYear_flag) { ++result; featureNames.Add("EndOfYear_flag"); }
        if (Use_Christmas_flag) { ++result; featureNames.Add("Christmas_flag"); }
        if (Use_EndOfTrimester_flag) { ++result; featureNames.Add("EndOfTrimester_flag"); }

        //abs_ret
        if (Use_abs_ret)
        {
            result += CFM60Entry.POINTS_BY_DAY;
            for (int i = 0; i < CFM60Entry.POINTS_BY_DAY; ++i)
            {
                featureNames.Add(FeatureImportancesCalculator.VectorFeatureName("abs_ret", i));
            }
        }
        if (Use_mean_abs_ret) {++result; featureNames.Add("mean(abs_ret)");}
        if (Use_volatility_abs_ret) { ++result; featureNames.Add("vol(abs_ret)"); }

        //rel_vol
        if (Use_rel_vol)
        {
            result += Use_rel_vol_start_and_end_only?(2*12):CFM60Entry.POINTS_BY_DAY;
            for (int i = 0; i < CFM60Entry.POINTS_BY_DAY; ++i)
            {
                if (i < 12 || !Use_rel_vol_start_and_end_only || i >= CFM60Entry.POINTS_BY_DAY - 12)
                {
                    featureNames.Add(FeatureImportancesCalculator.VectorFeatureName("rel_vol", i));
                }
            }
        }
        if (Use_volatility_rel_vol) { ++result; featureNames.Add("vol(rel_vol)"); }

        //LS
        if (Use_LS)
        {
            ++result;
            featureNames.Add("LS");
        }

        //NLV
        if (Use_NLV)
        {
            ++result;
            featureNames.Add("NLV");
        }
        if (result != featureNames.Count)
        {
            throw new Exception("invalid " + nameof(Encoder_InputSize));
        }

        if (isEncoderInputSize)
        {
            Encoder_FeatureNames = featureNames;
        }
        else
        {
            Decoder_FeatureNames = featureNames;
        }

        return result;
    }

    public void Initialize(Network network)
    {
        network.Input(Encoder_TimeSteps, Encoder_InputSize, -1);

        if (Pid_EmbeddingDim >= 1)
        {
            network.Embedding(CFM60Entry.DISTINCT_PID_COUNT, Pid_EmbeddingDim, 0, network.Config.lambdaL2Regularization, ClipValueForGradients, DivideGradientsByTimeSteps);
        }

        if (InputNormalizationType == InputNormalizationEnum.BATCH_NORM_LAYER || InputNormalizationType == InputNormalizationEnum.DEDUCE_MEAN_AND_BATCH_NORM_LAYER)
        {
            network.SwitchSecondAndThirdDimension(true);
            network.BatchNorm(0.99, 1e-5);
            network.SwitchSecondAndThirdDimension(false);
        }

        if (UseConv1D)
        {
            network.Conv1D(Encoder_TimeSteps, Conv1DKernelWidth, 1, Conv1DPaddingType, network.Config.lambdaL2Regularization, true);

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
            network.Input(Decoder_TimeSteps, Decoder_InputSize, -1);
            if (Pid_EmbeddingDim >= 1)
            {
                network.Embedding(CFM60Entry.DISTINCT_PID_COUNT, Pid_EmbeddingDim, 0, network.Config.lambdaL2Regularization, ClipValueForGradients, DivideGradientsByTimeSteps);
            }
            network.DecoderLayer(encoderLayerIndex, Decoder_NumLayers, Decoder_DropoutRate);
        }


        if (DropoutRate_After_EncoderDecoder >= 1e-6)
        {
            network.Dropout(DropoutRate_After_EncoderDecoder);
        }

        network.Dense_Activation(DenseUnits, network.Config.lambdaL2Regularization, true, ActivationFunctionAfterFirstDense);
        network.Dense(1, network.Config.lambdaL2Regularization, true);
        network.Flatten();

        if (WithSpecialEndV1)
        {
            network.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);
            network.Linear(2, 0);
            network.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_LN);
        }
    }


    //pid embedding
    public int Pid_EmbeddingDim = 4;  //validated on 16-jan-2021: -0.0236

    //y estimate
    public bool Use_y_LinearRegressionEstimate = true; //validated on 19-jan-2021: -0.0501 (with other changes)
    /// <summary>
    /// should we use the average observed 'y' outcome of the company (computed in the training dataSet) in the input tensor
    /// </summary>
    public bool Use_mean_pid_y = false; //discarded on 19-jan-2021: +0.0501 (with other changes)
    public bool Use_volatility_pid_y = true; //validated on 17-jan-2021: -0.0053
    public bool Use_variance_pid_y = false;
    public bool Use_prev_Y = true;
        
        
    //rel_vol fields
    public bool Use_rel_vol = true;
    public bool Use_rel_vol_start_and_end_only = false; //use only the first 12 and last 12 elements of rel_vol
    public bool Use_volatility_rel_vol = false; //'true' discarded on 22-jan-2020: +0.0065

    //abs_ret
    public bool Use_abs_ret = true;  //validated on 16-jan-2021: -0.0515
    public bool Use_mean_abs_ret = false;
    public bool Use_volatility_abs_ret = false;

    //LS
    public bool Use_LS = true; //validated on 16-jan-2021: -0.0164
    /// <summary>
    /// normalize LS value between in [0,1] interval
    /// </summary>

    //NLV
    public bool Use_NLV = true; //validated on 5-july-2021: -0.0763

    // year/day field
    /// <summary>
    /// add the fraction of the year of the current day as an input
    /// it is a value between 1/250f (1-jan) to 1f (31-dec)
    /// </summary>
    public bool Use_fraction_of_year = false;

    /// <summary>
    /// add a cyclical encoding of the current year into 2 new features:
    ///     year_sin: sin(2*pi*fraction_of_year)
    ///and 
    ///     year_cos: cos (2*pi*fraction_of_year)
    ///where
    ///     fraction_of_year: a real between 0 (beginning of the year(, and 1 (end of the year)
    /// see: https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
    /// </summary>
    public bool Use_year_Cyclical_Encoding = false; //discarded on 18-sep-2021: +0.01

    /// <summary>
    /// when 'Use_day' is true, we add the following feature value:   day / Use_day_Divider 
    /// </summary>
    //public float Use_day_Divider = 1151f;
    public float Use_day_Divider = 650f; //validated on 18-sept-2021 -0.002384 vs 1151f

    public bool Use_day = false; //discarded on 19-jan-2021: +0.0501 (with other changes)
    public bool Use_EndOfYear_flag = true;  //validated on 19-jan-2021: -0.0501 (with other changes)
    public bool Use_Christmas_flag = true;  //validated on 19-jan-2021: -0.0501 (with other changes)
    public bool Use_EndOfTrimester_flag = true;  //validated on 19-jan-2021: -0.0501 (with other changes)

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




    /// <summary>
    /// the optional serialized network to load for training
    /// </summary>
    public string SerializedNetwork = "";

    public float ClipValueForGradients = 0f;
    public bool DivideGradientsByTimeSteps = false;
    public bool Use_GRU_instead_of_LSTM = false;
    public bool Use_Bidirectional_RNN = true;
    public int Total_TimeSteps => Use_Decoder ? Encoder_TimeSteps + Decoder_TimeSteps : Encoder_TimeSteps;

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
        if (DropoutRate_After_EncoderDecoder >1e-6)
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
    public int Encoder_InputSize => GetInputSize(true);
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
    public int Decoder_InputSize => GetInputSize(false);
    #endregion

    public double DropoutRate_Between_Encoder_And_Decoder = 0.2;       //validated on 15-jan-2021
    public double DropoutRate_After_EncoderDecoder = 0.0;
    public bool UseBatchNorm2 = false;

    public int HiddenSize = 128;               //validated on 15-jan-2021
        
    public int DenseUnits = 200;              //validated on 21-jan-2021: -0.0038
    //public int DenseUnits = 100;


    //public double PercentageInTraining = 0.68; //discarded on 16-jan-2021: +0.2759
    public double PercentageInTraining = 0.9;
    public bool UseConv1D = false;
    public bool UseBatchNormAfterConv1D = false;
    public bool UseReluAfterConv1D = false;
        
    // max value of the loss to consider saving the network
    public double MaxLossToSaveTheNetwork => IsTryingToPredictErrors?0.356:0.36;

    public string[] predictionFilesIfComputeErrors = null;

    //the kind of value embedded in Y (and that we are trying to predict)
    public ValueToPredictEnum ValueToPredict = ValueToPredictEnum.Y_TRUE;

    public enum ValueToPredictEnum { 
        Y_TRUE,  // we try to predict the true value 'Y'
        Y_TRUE_MINUS_LR, //we try to predict the spread between the true value and a linear regression estimate
        Y_TRUE_MINUS_ADJUSTED_LR, //we try to predict the spread between the true value and a linear regression estimate adjusted by the estimate error
        ERROR, //we try to predict the error (Y_TRUE-Y_PRED) made by a previously trained network
    }

    public bool IsTryingToPredictErrors => predictionFilesIfComputeErrors != null;

    public string DatasetName => IsTryingToPredictErrors ? "CFM60Errors" : "CFM60";

    // ReSharper disable once UnusedMember.Global
    public void WithConv1D(int kernelWidth, ConvolutionLayer.PADDING_TYPE paddingType, bool useBatchNormAfterConv1D, bool useReluAfterConv1D)
    {
        UseConv1D = true;
        Conv1DKernelWidth = kernelWidth;
        Conv1DPaddingType = paddingType;
        UseBatchNormAfterConv1D = useBatchNormAfterConv1D;
        UseReluAfterConv1D = useReluAfterConv1D;
    }

    public int Conv1DKernelWidth = 3;

    public ConvolutionLayer.PADDING_TYPE Conv1DPaddingType = ConvolutionLayer.PADDING_TYPE.SAME;

    public cudnnActivationMode_t ActivationFunctionAfterFirstDense = cudnnActivationMode_t.CUDNN_ACTIVATION_CLIPPED_RELU;
    public bool WithSpecialEndV1 = false;


    public Tuple<double, double, double, double, double, double>[] Compute_Encoder_FeaturesStatistics()
    {
        var featureImportance = FeatureImportancesCalculator.LoadFromFile(Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "featureimportances.csv"));
        var res = new Tuple<double, double, double, double, double, double>[Encoder_InputSize];
        for (int featureId = 0; featureId < res.Length; ++featureId)
        {
            var featureName = FeatureIdToEncoderFeatureName(featureId);
            res[featureId] = featureImportance.TryGetValue(featureName, out var result) ? result : null;
        }
        return res;
    }
    public Tuple<double, double, double, double, double, double>[] Compute_Decoder_FeaturesStatistics()
    {
        if (!Use_Decoder)
        {
            return null;
        }
        var featureImportance = FeatureImportancesCalculator.LoadFromFile(Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "featureimportances.csv"));
        var res = new Tuple<double, double, double, double, double, double>[Decoder_InputSize];
        for (int featureId = 0; featureId < res.Length; ++featureId)
        {
            var featureName = FeatureIdToDecoderFeatureName(featureId);
            res[featureId] = featureImportance.TryGetValue(featureName, out var result) ? result : null;
        }
        return res;
    }



    public string[] ComputeFeatureNames()
    {
        return null; //TODO
    }


    public string FeatureIdToEncoderFeatureName(int featureId)
    {
        return Encoder_FeatureNames[featureId];
    }
    public string FeatureIdToDecoderFeatureName(int featureId)
    {
        return Decoder_FeatureNames[featureId];
    }

    public static CFM60HyperParameters ValueOf(string workingDirectory, string modelName)
    {
        return (CFM60HyperParameters)ISample.LoadConfigIntoSample(() => new CFM60HyperParameters(), workingDirectory, modelName);
    }
}
