using System;
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
            var testDataSet_YIfAny = testDataSet.LoadFullY();
            var yCpuPredictedAllNetworks = new CpuTensor<float>(testDataSet_YIfAny.Shape);
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

                (var yPredictedSingleNetwork, _) = network.MiniBatchGradientDescentForSingleEpoch(testDataSet, miniBatchSize, returnPredictionsForFullDataset: true, computeMetricsForFullDataset: false);
                var yCpuPredictedSingleNetwork = yPredictedSingleNetwork.ToCpuFloat();
                lossFunction = network.Sample.LossFunction;
                var accuracy = (lossFunction == EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy)
                    ? buffer.ComputeAccuracyCategoricalCrossentropyWithHierarchy(testDataSet_YIfAny, yCpuPredictedSingleNetwork)
                    : buffer.ComputeAccuracy(testDataSet_YIfAny, yCpuPredictedSingleNetwork);
                Console.WriteLine("Single Network Accuracy=" + accuracy);
                yCpuPredictedAllNetworks.Update_Adding_Alpha_X(1f/ _modelFilesPath.Length, yCpuPredictedSingleNetwork);
            }
            var accuracyEnsembleNetwork = (lossFunction == EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy)
                ? buffer.ComputeAccuracyCategoricalCrossentropyWithHierarchy(testDataSet_YIfAny, yCpuPredictedAllNetworks)
                : buffer.ComputeAccuracy(testDataSet_YIfAny, yCpuPredictedAllNetworks);
            
            Console.WriteLine("Ensemble Network Accuracy=" + accuracyEnsembleNetwork);
            return Tuple.Create(yCpuPredictedAllNetworks, accuracyEnsembleNetwork);
        }
    }
}