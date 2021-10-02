using System;
using System.Collections.Generic;
using System.IO;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.MathTools;

// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks
{
    /// <summary>
    /// Network support for CFM60 challenge
    /// </summary>
    public class CFM60NetworkBuilder : NetworkBuilder
    {
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

            //ret_vol
            if (Use_ret_vol)
            {
                result += Use_ret_vol_start_and_end_only?(2*12):CFM60Entry.POINTS_BY_DAY;
                for (int i = 0; i < CFM60Entry.POINTS_BY_DAY; ++i)
                {
                    if (i < 12 || !Use_ret_vol_start_and_end_only || i >= CFM60Entry.POINTS_BY_DAY - 12)
                    {
                        featureNames.Add(FeatureImportancesCalculator.VectorFeatureName("ret_vol", i));
                    }
                }
            }
            if (Use_volatility_ret_vol) { ++result; featureNames.Add("vol(ret_vol)"); }

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

        //pid embedding
        public int Pid_EmbeddingDim { get; set; } = 4;  //validated on 16-jan-2021: -0.0236

        //y estimate
        public bool Use_y_LinearRegressionEstimate { get; set; } = true; //validated on 19-jan-2021: -0.0501 (with other changes)
        /// <summary>
        /// should we use the average observed 'y' outcome of the company (computed in the training dataSet) in the input tensor
        /// </summary>
        public bool Use_mean_pid_y { get; set; } = false; //discarded on 19-jan-2021: +0.0501 (with other changes)
        public bool Use_volatility_pid_y { get; set; } = true; //validated on 17-jan-2021: -0.0053
        public bool Use_variance_pid_y { get; set; } = false;
        public bool Use_prev_Y { get; set; } = true;
        
        
        //ret_vol fields
        public bool Use_ret_vol { get; set; } = true;
        public bool Use_ret_vol_start_and_end_only { get; set; } = false; //use only the first 12 and last 12 elements of ret_vol
        public bool Use_volatility_ret_vol { get; set; } = false; //'true' discarded on 22-jan-2020: +0.0065

        //abs_ret
        public bool Use_abs_ret { get; set; } = true;  //validated on 16-jan-2021: -0.0515
        public bool Use_mean_abs_ret { get; set; } = false;
        public bool Use_volatility_abs_ret { get; set; } = false;

        //LS
        public bool Use_LS { get; set; } = true; //validated on 16-jan-2021: -0.0164
        /// <summary>
        /// normalize LS value between in [0,1] interval
        /// </summary>

        //NLV
        public bool Use_NLV { get; set; } = true; //validated on 5-july-2021: -0.0763

        // year/day field
        /// <summary>
        /// add the fraction of the year of the current day as an input
        /// it is a value between 1/250f (1-jan) to 1f (31-dec)
        /// </summary>
        public bool Use_fraction_of_year { get; set; } = false;

        /// <summary>
        /// add a cyclical encoding of the current year into 2 new features:
        ///     year_sin: sin(2*pi*fraction_of_year)
        ///and 
        ///     year_cos: cos (2*pi*fraction_of_year)
        ///where
        ///     fraction_of_year: a real between 0 (beginning of the year(, and 1 (end of the year)
        /// see: https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
        /// </summary>
        public bool Use_year_Cyclical_Encoding { get; set; } = false; //discarded on 18-sep-2021: +0.01

        /// <summary>
        /// when 'Use_day' is true, we add the following feature value:   day / Use_day_Divider 
        /// </summary>
        //public float Use_day_Divider { get; set; } = 1151f;
        public float Use_day_Divider { get; set; } = 650f; //validated on 18-sept-2021 -0.002384 vs 1151f

        public bool Use_day { get; set; } = false; //discarded on 19-jan-2021: +0.0501 (with other changes)
        public bool Use_EndOfYear_flag { get; set; } = true;  //validated on 19-jan-2021: -0.0501 (with other changes)
        public bool Use_Christmas_flag { get; set; } = true;  //validated on 19-jan-2021: -0.0501 (with other changes)
        public bool Use_EndOfTrimester_flag { get; set; } = true;  //validated on 19-jan-2021: -0.0501 (with other changes)

        //normalization
        public enum InputNormalizationEnum
        {
            NONE,
            Z_SCORE,
            DEDUCE_MEAN,
            BATCH_NORM_LAYER, //we'll use a batch norm layer for normalization
            DEDUCE_MEAN_AND_BATCH_NORM_LAYER
        }
        //entry.ret_vol.Length - 12

        public InputNormalizationEnum InputNormalizationType { get; set; } = InputNormalizationEnum.NONE;




        /// <summary>
        /// the optional serialized network to load for training
        /// </summary>
        public string SerializedNetwork { get; set; } = "";

        public float ClipValueForGradients { get; set; } = 0f;
        public bool DivideGradientsByTimeSteps { get; set; } = false;
        public bool Use_GRU_instead_of_LSTM { get; set; } = false;
        public bool Use_Bidirectional_RNN { get; set; } = true;
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

        public int Encoder_NumLayers { get; set; } = 2;
        public int Encoder_TimeSteps { get; set; } = 20;
        /// <summary>
        /// dropout to use for the encoder. A value of 0 means no dropout
        /// </summary>
        public double Encoder_DropoutRate { get; set; } = 0.2;
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
        public int Decoder_NumLayers { get; set; } = 0;
        /// <summary>
        /// Number of time steps for the decoder. 0 means that we are not using a decoder
        /// </summary>
        public int Decoder_TimeSteps { get; set; } = 0;
        public double Decoder_DropoutRate { get; set; } = 0.2;
        public int Decoder_InputSize => GetInputSize(false);
        #endregion

        public double DropoutRate_Between_Encoder_And_Decoder { get; set; } = 0.2;       //validated on 15-jan-2021
        public double DropoutRate_After_EncoderDecoder { get; set; } = 0.0;
        public bool UseBatchNorm2 { get; set; } = false;

        public int HiddenSize { get; set; } = 128;               //validated on 15-jan-2021
        
        public int DenseUnits { get; set; } = 200;              //validated on 21-jan-2021: -0.0038
        //public int DenseUnits { get; set; } = 100;


        //public double PercentageInTraining { get; set; } = 0.68; //discarded on 16-jan-2021: +0.2759
        public double PercentageInTraining { get; set; } = 0.9;
        public bool UseConv1D { get; set; } = false;
        public bool UseBatchNormAfterConv1D { get; set; } = false;
        public bool UseReluAfterConv1D { get; set; } = false;
        
        // max value of the loss to consider saving the network
        public double MaxLossToSaveTheNetwork { get; set; } = 0.36;
        
        // ReSharper disable once UnusedMember.Global
        public void WithConv1D(int kernelWidth, ConvolutionLayer.PADDING_TYPE paddingType, bool useBatchNormAfterConv1D, bool useReluAfterConv1D)
        {
            UseConv1D = true;
            Conv1DKernelWidth = kernelWidth;
            Conv1DPaddingType = paddingType;
            UseBatchNormAfterConv1D = useBatchNormAfterConv1D;
            UseReluAfterConv1D = useReluAfterConv1D;
        }

        public int Conv1DKernelWidth { get; set; } = 3;

        public ConvolutionLayer.PADDING_TYPE Conv1DPaddingType { get; set; } = ConvolutionLayer.PADDING_TYPE.SAME;

        public bool Shuffle { get; set; } = true;

        public cudnnActivationMode_t ActivationFunctionAfterFirstDense { get; set; } = cudnnActivationMode_t.CUDNN_ACTIVATION_CLIPPED_RELU;
        public bool WithSpecialEndV1 { get; set; } = false;

        public static CFM60NetworkBuilder Default()
        {
            var builder = new CFM60NetworkBuilder
                          {
                              Config = new NetworkConfig
                                       {
                                           LogFile = "TimeSeries",
                                           LossFunction = NetworkConfig.LossFunctionEnum.Mse,
                                           RandomizeOrder = true,
                                           CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                                           Metrics = new List<NetworkConfig.Metric> { NetworkConfig.Metric.Loss },
                                           LogDirectory = Path.Combine(NetworkConfig.DefaultLogDirectory, "CFM60"),
                                           DataAugmentation = new DataAugmentationConfig()
                                       }
                                  //.WithCyclicCosineAnnealingLearningRateScheduler(10, 2)
                          };
            //builder.BatchSize = 1024; //updated on 13-june-2021
            //builder.InitialLearningRate = 0.001;
            builder.Use_day = true;
            builder.Use_y_LinearRegressionEstimate = false;
            builder.Use_volatility_pid_y = false;
            //builder.Encoder_TimeSteps = 20;
            builder.Use_LS = false; //validated on 2-june-2021: -0.011155486
            builder.Pid_EmbeddingDim = 8; //validated on 6-june-2021: -0.0226
            builder.DenseUnits = 50; //validated on 6-june-2021: -0.0121
            builder.BatchSize = 2048; //validated on 13-june-2021: -0.01
            builder.ClipValueForGradients = 1000;  //validated on 4-july-2021: -0.0110
            builder.Config.WithOneCycleLearningRateScheduler(200, 0.1); //validated on 14-july-2021: -0.0078
            builder.HiddenSize = 64; //validated on 14-july-2021: very small degradation (+0.0010) but much less parameters
            //builder.Config.WithAdamW(0.00005); //validated on 19-july-2021: very small degradation (+0.0016) but better expected results for bigger data set
            builder.NumEpochs = 30; //validated on 20-july-2021: speed up tests
            builder.Config.WithAdamW(0.0001); //validated on 20-july-2021: small change but better generalization
            //builder.Config.AlwaysUseFullTestDataSetForLossAndAccuracy = false;
            //validated on 2-aug-2021:  -0.0078
            builder.Encoder(1, 20, 0.0); //validated on 2-aug-2021:  -0.0078
            builder.DropoutRate_Between_Encoder_And_Decoder = 0.2; //validated on 2-aug-2021:  -0.0078
            builder.Decoder(2, 1, 0.2); //validated on 2-aug-2021:  -0.0078
            builder.DropoutRate_After_EncoderDecoder = 0.2; //validated on 2-aug-2021:  -0.0078
            builder.InitialLearningRate = 0.002;  //validated on 4-aug-2021:  -0.0011

            return builder;
        }

        public Network CFM60()
        {
            if (!string.IsNullOrEmpty(SerializedNetwork))
            {
                return Network.ValueOf(SerializedNetwork);
            }
            const string networkName = "CFM60";
            var network = BuildEmptyNetwork(networkName);
            network.Config.RandomizeOrder = Shuffle;

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

            return network;
        }


        public Tuple<double, double, double, double, double, double>[] Compute_Encoder_FeaturesStatistics()
        {
            var featureImportances = FeatureImportancesCalculator.LoadFromFile(Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "featureimportances.csv"));
            var res = new Tuple<double, double, double, double, double, double>[Encoder_InputSize];
            for (int featureId = 0; featureId < res.Length; ++featureId)
            {
                var featureName = FeatureIdToEncoderFeatureName(featureId);
                res[featureId] = featureImportances.TryGetValue(featureName, out var result) ? result : null;
            }
            return res;
        }
        public Tuple<double, double, double, double, double, double>[] Compute_Decoder_FeaturesStatistics()
        {
            if (!Use_Decoder)
            {
                return null;
            }
            var featureImportances = FeatureImportancesCalculator.LoadFromFile(Path.Combine(NetworkConfig.DefaultDataDirectory, "CFM60", "featureimportances.csv"));
            var res = new Tuple<double, double, double, double, double, double>[Decoder_InputSize];
            for (int featureId = 0; featureId < res.Length; ++featureId)
            {
                var featureName = FeatureIdToDecoderFeatureName(featureId);
                res[featureId] = featureImportances.TryGetValue(featureName, out var result) ? result : null;
            }
            return res;
        }

        public string FeatureIdToEncoderFeatureName(int featureId)
        {
            return Encoder_FeatureNames[featureId];
        }
        public string FeatureIdToDecoderFeatureName(int featureId)
        {
            return Decoder_FeatureNames[featureId];
        }


    }
}