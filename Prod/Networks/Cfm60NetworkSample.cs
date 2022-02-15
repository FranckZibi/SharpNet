using System.Collections.Generic;
using System.IO;
using SharpNet.DataAugmentation;
using SharpNet.HyperParameters;

// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Networks
{
    public class Cfm60NetworkSample : NetworkSample
    {
        private Cfm60NetworkSample(ISample[] samples) : base(samples)
        {
        }

        public CFM60HyperParameters CFM60HyperParameters => (CFM60HyperParameters)Samples[2];

        public static Cfm60NetworkSample ValueOfCfm60NetworkSample(string workingDirectory, string modelName)
        {
            return new Cfm60NetworkSample(new ISample[]
            {
                NetworkConfig.ValueOf(workingDirectory, modelName+"_0"),
                DataAugmentationSample.ValueOf(workingDirectory, modelName+"_1"),
                CFM60HyperParameters.ValueOf(workingDirectory, modelName+"_2")
            });
        }

        public static Cfm60NetworkSample Default()
        {
            var config = new NetworkConfig
            {
                LogFile = "TimeSeries",
                LossFunction = LossFunctionEnum.Mse,
                RandomizeOrder = true,
                CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                Metrics = new List<MetricEnum> { MetricEnum.Loss },
                WorkingDirectory = Path.Combine(NetworkConfig.DefaultWorkingDirectory, "CFM60"),
                BatchSize = 2048, //validated on 13-june-2021: -0.01
                NumEpochs = 30, //validated on 20-july-2021: speed up tests
                InitialLearningRate = 0.002  //validated on 4-aug-2021:  -0.0011
                //AlwaysUseFullTestDataSetForLossAndAccuracy = false;
        }
            //.WithAdamW(0.00005); //validated on 19-july-2021: very small degradation (+0.0016) but better expected results for bigger data set
            .WithAdamW(0.0001) //validated on 20-july-2021: small change but better generalization
            //.WithCyclicCosineAnnealingLearningRateScheduler(10, 2)
            .WithOneCycleLearningRateScheduler(200, 0.1); //validated on 14-july-2021: -0.0078

            var cfm60HyperParameters = new CFM60HyperParameters();
            cfm60HyperParameters.ValueToPredict = CFM60HyperParameters.ValueToPredictEnum.Y_TRUE;
            cfm60HyperParameters.predictionFilesIfComputeErrors = null;
            //builder.BatchSize = 1024; //updated on 13-june-2021
            //builder.InitialLearningRate = 0.001;
            cfm60HyperParameters.Use_day = true;
            cfm60HyperParameters.Use_y_LinearRegressionEstimate = false;
            cfm60HyperParameters.Use_volatility_pid_y = false;
            //builder.Encoder_TimeSteps = 20;
            cfm60HyperParameters.Use_LS = false; //validated on 2-june-2021: -0.011155486
            cfm60HyperParameters.Pid_EmbeddingDim = 8; //validated on 6-june-2021: -0.0226
            cfm60HyperParameters.DenseUnits = 50; //validated on 6-june-2021: -0.0121
            cfm60HyperParameters.ClipValueForGradients = 1000;  //validated on 4-july-2021: -0.0110
            cfm60HyperParameters.HiddenSize = 64; //validated on 14-july-2021: very small degradation (+0.0010) but much less parameters
            //validated on 2-aug-2021:  -0.0078
            cfm60HyperParameters.Encoder(1, 20, 0.0); //validated on 2-aug-2021:  -0.0078
            cfm60HyperParameters.DropoutRate_Between_Encoder_And_Decoder = 0.2; //validated on 2-aug-2021:  -0.0078
            cfm60HyperParameters.Decoder(2, 1, 0.2); //validated on 2-aug-2021:  -0.0078
            cfm60HyperParameters.DropoutRate_After_EncoderDecoder = 0.2; //validated on 2-aug-2021:  -0.0078

            return new Cfm60NetworkSample(new ISample[] { config, new DataAugmentationSample(), cfm60HyperParameters });
        }

        public static Cfm60NetworkSample DefaultToPredictError()
        {
            var builder = Default();
            builder.CFM60HyperParameters.ValueToPredict = CFM60HyperParameters.ValueToPredictEnum.ERROR;
            builder.CFM60HyperParameters.predictionFilesIfComputeErrors = new[]
                                                        {
                                                            @"C:\Users\Franck\AppData\Local\SharpNet\CFM60\train_predictions\CFM60_30_0_3099_0_3595_20211024_1207_4.csv",
                                                            @"C:\Users\Franck\AppData\Local\SharpNet\CFM60\validation_predictions\CFM60_30_0_3099_0_3595_20211024_1207_4.csv"
                                                        };
            //builder.Config.LossFunction = LossFunctionEnum.Mae;
            //builder.NumEpochs = 10;
            //builder.Use_fraction_of_year = true;
            //builder.Use_year_Cyclical_Encoding = true;
            //builder.Use_rel_vol = false;
            //builder.Use_abs_ret = false;
            //builder.BatchSize= 1024;
            builder.CFM60HyperParameters.Pid_EmbeddingDim = 20; //validated on 26/10/2021 : 
            return builder;
        }

        public Network CFM60()
        {
            if (!string.IsNullOrEmpty(CFM60HyperParameters.SerializedNetwork))
            {
                var workingDirectory = Path.GetDirectoryName(CFM60HyperParameters.SerializedNetwork);
                var modelName = Path.GetFileNameWithoutExtension(CFM60HyperParameters.SerializedNetwork);
                return Network.ValueOf(workingDirectory, modelName);
            }
            var network = BuildEmptyNetwork(CFM60HyperParameters.DatasetName);
            CFM60HyperParameters.Initialize(network);
            return network;
        }

    }
}