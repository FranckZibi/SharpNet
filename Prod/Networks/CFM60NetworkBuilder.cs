using System.Collections.Generic;
using System.IO;
using SharpNet.DataAugmentation;
using SharpNet.GPU;

namespace SharpNet.Networks
{
    /// <summary>
    /// Network support for CFM60 challenge
    /// </summary>
    public class CFM60NetworkBuilder : NetworkBuilder
    {
        /// <summary>
        /// The default Network for CFM60
        /// </summary>
        /// <returns></returns>
        public static CFM60NetworkBuilder Default()
        {
            const bool shuffle = true;
            const double momentum = 0.9;

            var builder = new CFM60NetworkBuilder
            {
                Config = new NetworkConfig
                    {
                        LogFile = "TimeSeries",
                        LossFunction = NetworkConfig.LossFunctionEnum.Mse,
                        RandomizeOrder = shuffle,
                        CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                        Metrics = new List<NetworkConfig.Metric> { NetworkConfig.Metric.Loss},
                        LogDirectory = Path.Combine(NetworkConfig.DefaultLogDirectory, "CFM60"),
                        DataAugmentation = DataAugmentationConfig.NoDataAugmentation
                    }
                    .WithCyclicCosineAnnealingLearningRateScheduler(10, 2)
                    
                    //.WithSGD(momentum, false)
                    .WithAdam()

                ,
                NumEpochs = 70,
                BatchSize = 2,
                InitialLearningRate = 0.1,
            };

            return builder;
        }

        public Network CFM60(int depth, int k, int timeSteps, int inputSize)
        {
            var networkName = "CFM60-" + depth + "-" + k;
            var network = BuildEmptyNetwork(networkName);
            var config = network.Config;
            const int hiddenSize = 64;


            const bool isBidirectional = true;
            network
                .Input(timeSteps, inputSize, -1)
                //.Linear(aNormalization, bNormalization)
                .LSTM(hiddenSize, true, isBidirectional)
                .LSTM(hiddenSize, false, isBidirectional)
                //.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID, null)
                .Dense_Activation(100, 0.0, true, cudnnActivationMode_t.CUDNN_ACTIVATION_CLIPPED_RELU)
                .Dense(1, 0.0, true)
                ;

            return network;
        }
    }
}