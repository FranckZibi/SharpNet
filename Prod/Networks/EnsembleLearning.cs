﻿using System;
using System.IO;
using SharpNet.CPU;
using SharpNet.Datasets;

namespace SharpNet.Networks
{
    public class EnsembleLearning
    {
        private readonly string[] _modelFilesPath;

        public EnsembleLearning(string[] modelFilesPath)
        {
            _modelFilesPath = modelFilesPath;
        }

        public Tuple<CpuTensor<float>, double> Predict(DataSet testDataSet, int miniBatchSize)
        {
            var yCpuPredictedAllNetworks = new CpuTensor<float>(testDataSet.Y.Shape);
            var buffer = new CpuTensor<float>(new []{ testDataSet.Count});
            var lossFunction = EvaluationMetricEnum.CategoricalCrossentropy;
            foreach (var modelFilePath in _modelFilesPath)
            {
                Console.WriteLine("Loading " + modelFilePath + " ...");
                var workingDirectory = Path.GetDirectoryName(modelFilePath);
                var modelName = Path.GetFileNameWithoutExtension(modelFilePath);
                using var network = Network.LoadTrainedNetworkModel(workingDirectory, modelName);
                Console.WriteLine("File loaded");
                Console.WriteLine("Computing accuracy for single network...");
                    
                var yPredictedSingleNetwork = network.MiniBatchGradientDescentForSingleEpoch(testDataSet, miniBatchSize);
                var yCpuPredictedSingleNetwork = yPredictedSingleNetwork.ToCpuFloat();
                lossFunction = network.Config.LossFunction;
                var accuracy = (lossFunction == EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy)
                    ?testDataSet.Y.ComputeAccuracyCategoricalCrossentropyWithHierarchy(yCpuPredictedSingleNetwork, buffer)
                    :testDataSet.Y.ComputeAccuracy(yCpuPredictedSingleNetwork, buffer);
                Console.WriteLine("Single Network Accuracy=" + accuracy);
                yCpuPredictedAllNetworks.Update_Adding_Alpha_X(1f/ _modelFilesPath.Length, yCpuPredictedSingleNetwork);
            }
            var accuracyEnsembleNetwork = (lossFunction == EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy)
                ?testDataSet.Y.ComputeAccuracyCategoricalCrossentropyWithHierarchy(yCpuPredictedAllNetworks, buffer)
                :testDataSet.Y.ComputeAccuracy(yCpuPredictedAllNetworks, buffer);

            Console.WriteLine("Ensemble Network Accuracy=" + accuracyEnsembleNetwork);
            return Tuple.Create(yCpuPredictedAllNetworks, accuracyEnsembleNetwork);
        }
    }
}