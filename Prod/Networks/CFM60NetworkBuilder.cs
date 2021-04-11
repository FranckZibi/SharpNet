using System;
using System.Collections.Generic;
using System.IO;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks
{
    /// <summary>
    /// Network support for CFM60 challenge
    /// </summary>
    public class CFM60NetworkBuilder : NetworkBuilder
    {
        public int TimeSteps { get; set;  } = 20;
        public int InputSize 
        { 
            get
            {
                int result = 0;
                if (Pid_EmbeddingDim>=1) { ++result; }
                if (Use_ret_vol_in_InputTensor) { result += CFM60Entry.POINTS_BY_DAY; }
                if (Use_abs_ret_in_InputTensor) { result += CFM60Entry.POINTS_BY_DAY; }
                if (Use_y_LinearRegressionEstimate_in_InputTensor) { ++result; }
                if (Use_pid_y_avg_in_InputTensor) {++result;}
                if (Use_pid_y_vol_in_InputTensor) {++result;}
                if (Use_pid_y_variance_in_InputTensor) {++result;}
                if (Use_ret_vol_CoefficientOfVariation_in_InputTensor) {++result;}
                if (Use_ret_vol_Volatility_in_InputTensor) {++result;}
                if (Use_LS_in_InputTensor) {++result;}
                if (Use_NLV_in_InputTensor) {++result;}
                if (Use_day_in_InputTensor) {++result;}
                if (Use_EndOfYear_flag_in_InputTensor) {++result;}
                if (Use_Christmas_flag_in_InputTensor) { ++result; }
                if (Use_EndOfTrimester_flag_in_InputTensor) {++result;}
                return result;
            }
        }

        public bool Use_y_LinearRegressionEstimate_in_InputTensor { get; set; } = true; //validated on 19-jan-2021: -0.0501 (with other changes)
        /// <summary>
        /// should we use the average observed 'y' outcome of the company (computed in the training dataSet) in the input tensor
        /// </summary>
        public bool Use_pid_y_avg_in_InputTensor { get; set; } = false; //discarded on 19-jan-2021: +0.0501 (with other changes)
        //public bool Use_pid_y_avg_in_InputTensor { get; set; } = true; //validated on 17-jan-2021: -0.0385

        public bool Use_pid_y_vol_in_InputTensor { get; set; } = true; //validated on 17-jan-2021: -0.0053
        public bool Use_pid_y_variance_in_InputTensor { get; set; } = false;
        public bool Use_ret_vol_in_InputTensor { get; set; } = true;
        public bool Use_abs_ret_in_InputTensor { get; set; } = true;  //validated on 16-jan-2021: -0.0515
        public bool Use_LS_in_InputTensor { get; set; } = true; //validated on 16-jan-2021: -0.0164
        
        public bool Use_ret_vol_CoefficientOfVariation_in_InputTensor { get; set; } = false; //'true' discarded on 22-jan-2020: +0.0043
        public bool Use_ret_vol_Volatility_in_InputTensor { get; set; } = false; //'true' discarded on 22-jan-2020: +0.0065


        //embedding dim associated with the 'pid'
        public int Pid_EmbeddingDim { get; set; } = 4;  //validated on 16-jan-2021: -0.0236

        public bool Use_day_in_InputTensor { get; set; } = false; //discarded on 19-jan-2021: +0.0501 (with other changes)
        //public bool Use_day_in_InputTensor { get; set; } = true; //validated on 16-jan-2021: -0.0274
        public bool Use_EndOfYear_flag_in_InputTensor { get; set; } = true;  //validated on 19-jan-2021: -0.0501 (with other changes)
        public bool Use_Christmas_flag_in_InputTensor { get; set; } = true;  //validated on 19-jan-2021: -0.0501 (with other changes)
        public bool Use_EndOfTrimester_flag_in_InputTensor { get; set; } = true;  //validated on 19-jan-2021: -0.0501 (with other changes)
        
        public bool Use_CustomLinearFunctionLayer  { get; set; } = false;
        public float Beta_for_CustomLinearFunctionLayer  { get; set; } = 1f;


        public bool Use_GRU_instead_of_LSTM { get; set; } = false;
        public bool Use_Bidirectional_RNN { get; set; } = true;


        /// <summary>
        /// the optional serialized network to load for training
        /// </summary>
        public string SerializedNetwork { get; set; } = "";

        // ReSharper disable once UnusedMember.Global
        public void WithCustomLinearFunctionLayer(float alpha, cudnnActivationMode_t activationFunctionAfterSecondDense)
        {
            Use_CustomLinearFunctionLayer = true;
            Beta_for_CustomLinearFunctionLayer = alpha;
            LinearLayer_slope = 1f;
            LinearLayer_intercept = 0f;
            ActivationFunctionAfterSecondDense = activationFunctionAfterSecondDense;
        }

        /// <summary>
        /// normalize LS value between in [0,1] interval
        /// </summary>
        public bool NormalizeLS { get; set; } = false;
        public bool NormalizeLS_V2 { get; set; } = false;
        public bool Use_NLV_in_InputTensor { get; set; } = true; //validated on 16-jan-2021: -0.0364
        /// <summary>
        /// normalize NLV value between in [0,1] interval
        /// </summary>
        public bool NormalizeNLV { get; set; } = false;
        public bool NormalizeNLV_V2 { get; set; } = false;

        public float LinearLayer_slope { get; set; } = 1.0f;
        public float LinearLayer_intercept { get; set; } = 0.0f;

        public int LSTMLayersReturningFullSequence { get; set; } = 1;
        public double DropProbability { get; set; } = 0.2;       //validated on 15-jan-2021
        
        //public double PercentageInTraining { get; set; } = 0.68; //discarded on 16-jan-2021: +0.2759
        public double PercentageInTraining { get; set; } = 0.9;
        public bool SplitTrainingAndValidationBasedOnDays { get; set; } = true;
        public bool UseConv1D { get; set; } = false;
        public bool UseBatchNormAfterConv1D { get; set; } = false;
        public bool UseReluAfterConv1D { get; set; } = false;
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



        public bool UseBatchNorm { get; set; } = false;
        public int HiddenSize { get; set; } = 128;               //validated on 15-jan-2021
        public int DenseUnits { get; set; } = 200;              //validated on 21-jan-2021: -0.0038
        //public int DenseUnits { get; set; } = 100;
        public bool Shuffle { get; set; } = true;


        public cudnnActivationMode_t ActivationFunctionAfterFirstDense { get; set; } = cudnnActivationMode_t.CUDNN_ACTIVATION_CLIPPED_RELU;
        public cudnnActivationMode_t ActivationFunctionAfterSecondDense { get; set; } = cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY;


        /// <summary>
        /// The default Network for CFM60
        /// </summary>
        /// <returns></returns>
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
                        Metrics = new List<NetworkConfig.Metric> { NetworkConfig.Metric.Loss},
                        LogDirectory = Path.Combine(NetworkConfig.DefaultLogDirectory, "CFM60"),
                        DataAugmentation = DataAugmentationConfig.NoDataAugmentation
                    }
                    .WithCyclicCosineAnnealingLearningRateScheduler(10, 2)
                    
                    //.WithSGD(0.9, false)
                    .WithAdam()

                ,
                NumEpochs = 150,
                BatchSize = 1024,
                //InitialLearningRate = 0.0005, //new default 16-jan-2021
                InitialLearningRate = 0.001, //validated on 17-jan-2021 : -0.0040
            };

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

            network.Input(TimeSteps, InputSize, -1);

            if (Pid_EmbeddingDim >= 1)
            {
                network.Embedding(CFM60Entry.DISTINCT_PID_COUNT, Pid_EmbeddingDim, 0, 0);
            }


            if (UseConv1D)
            {
                network.Conv1D(TimeSteps, Conv1DKernelWidth, 1, Conv1DPaddingType, 0, true);

                if (UseBatchNormAfterConv1D)
                {
                    network.BatchNorm(0.99, 1e-5);
                }
                if (UseReluAfterConv1D)
                {
                    network.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_CLIPPED_RELU, null);
                }
            }

            for (int i = 0; i < LSTMLayersReturningFullSequence; ++i)
            {
                if (Use_GRU_instead_of_LSTM)
                {
                    network.GRU(HiddenSize, true, Use_Bidirectional_RNN);
                }
                else
                {
                    network.LSTM(HiddenSize, true, Use_Bidirectional_RNN);
                }
                if (DropProbability >= 1e-6)
                {
                    network.Dropout(DropProbability);
                }
                if (UseBatchNorm)
                {
                    network.BatchNorm(0.99, 1e-5);
                }
            }

            //network.Linear(aNormalization, bNormalization);
            if (Use_GRU_instead_of_LSTM)
            {
                network.GRU(HiddenSize, false, Use_Bidirectional_RNN);
            }
            else
            {
                network.LSTM(HiddenSize, false, Use_Bidirectional_RNN);
            }
            if (DropProbability >= 1e-6)
            {
                network.Dropout(DropProbability);
            }

            if (UseBatchNorm)
            {
                network.BatchNorm(0.99, 1e-5);
            }
            network.Dense_Activation(DenseUnits, 0.0, true, ActivationFunctionAfterFirstDense);
            network.Dense(1, 0.0, true);

            if (ActivationFunctionAfterSecondDense != cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY)
            {
                network.Activation(ActivationFunctionAfterSecondDense);
            }

            if (Use_CustomLinearFunctionLayer)
            {
                network.CustomLinear(Beta_for_CustomLinearFunctionLayer);
            }

            if (Math.Abs(LinearLayer_slope - 1f) > 1e-5 || Math.Abs(LinearLayer_intercept) > 1e-5)
            {
                network.Linear(LinearLayer_slope, LinearLayer_intercept);
            }

            return network;
        }
    }
}