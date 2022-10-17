﻿using System;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using SharpNet.DataAugmentation;
using SharpNet.HyperParameters;
using SharpNet.Models;

// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnusedAutoPropertyAccessor.Global
// ReSharper disable MemberCanBeProtected.Global

namespace SharpNet.Networks
{
    public class NetworkSample : MultiSamples, IModelSample
    {

        public NetworkSample(ISample[] samples) : base(samples)
        {
        }


        public NetworkSample CopyWithNewConfig(NetworkConfig newConfig)
        {
            var newSamples = (ISample[])Samples.Clone();
            newSamples[0] = newConfig;
            return new NetworkSample(newSamples);
        }

        public NetworkConfig Config => (NetworkConfig)Samples[0];
        public DataAugmentationSample DA => (DataAugmentationSample)Samples[1];


        public virtual void SaveExtraModelInfos(Model model, string workingDirectory, string modelName)
        {
        }

        public enum POOLING_BEFORE_DENSE_LAYER
        {
            /* we'll use an Average Pooling layer of size [2 x 2] before the Dense layer*/
            AveragePooling_2,
            /* we'll use an Average Pooling layer of size [8 x 8] before the Dense layer*/
            AveragePooling_8,

            /* We'll use a Global Average Pooling (= GAP) layer just before the last Dense (= fully connected) Layer
            This GAP layer will transform the input feature map of shape (n,c,h,w)
            to an output feature map of shape (n,c,1,1),
            so that this output feature map is independent of the the size of the input */
            GlobalAveragePooling,

            /* we'll use a Global Average Pooling layer concatenated with a Global Max Pooling layer before the Dense Layer
            This will transform the input feature map of shape (n,c,h,w)
            to an output feature map of shape (n, 2*c, 1, 1),
            so that this output feature map is independent of the the size of the input */
            GlobalAveragePooling_And_GlobalMaxPooling,

            /* We'll use a Global Max Pooling layer just before the last Dense (= fully connected) Layer
            This will transform the input feature map of shape (n,c,h,w)
            to an output feature map of shape (n,c,1,1),
            so that this output feature map is independent of the the size of the input */
            GlobalMaxPooling,

            /* no pooling */
             NONE
        };

        public Network BuildEmptyNetwork(string networkName)
        {
            Config.ModelName = networkName + Config.ExtraDescription;
            var network = new Network(this, Config.WorkingDirectory, Config.ModelName);
            network.Description = Config.ModelName;
            return network;
        }


        public virtual void BuildNetwork(Network network)
        {
            network.Description = Config.ModelName;
        }





        private static NetworkSample ValueOfNetworkSample(string workingDirectory, string modelName)
        {
            return new NetworkSample(new ISample[]
            {
                ISample.LoadSample<NetworkConfig>(workingDirectory, ISample.SampleName(modelName, 0)),
                ISample.LoadSample<DataAugmentationSample>(workingDirectory, ISample.SampleName(modelName, 1)),
            });
        }

        [SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
        public static NetworkSample ValueOf(string workingDirectory, string modelName)
        {
            try { return EfficientNetSample.ValueOfEfficientNetSample(workingDirectory, modelName); } catch { }
            try { return Cfm60NetworkSample.ValueOfCfm60NetworkSample(workingDirectory, modelName); } catch { }
            try { return WideResNetSample.ValueOfWideResNetSample(workingDirectory, modelName); } catch { }
            try { return ValueOfNetworkSample(workingDirectory, modelName); } catch { }
            throw new Exception($"can't load sample from model {modelName} in directory {workingDirectory}");
        }

        public EvaluationMetricEnum GetMetric()
        {
            if (Config.Metrics.Any())
            {
                return Config.Metrics[0];
            }
            switch (Config.LossFunction)
            {
                case EvaluationMetricEnum.BinaryCrossentropy: return EvaluationMetricEnum.BinaryCrossentropy;
                case EvaluationMetricEnum.CategoricalCrossentropy: return EvaluationMetricEnum.CategoricalCrossentropy;
                case EvaluationMetricEnum.Mse: return EvaluationMetricEnum.Mse;
                case EvaluationMetricEnum.Rmse: return EvaluationMetricEnum.Rmse;
                case EvaluationMetricEnum.Mae: return EvaluationMetricEnum.Mae;
                default: throw new NotImplementedException($"can't retrieve metric from {Config.LossFunction}");
            }
        }

        public EvaluationMetricEnum GetLoss()
        {
            return Config.LossFunction;
        }
    }
}
