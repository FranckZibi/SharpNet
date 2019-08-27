using System;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;

namespace SharpNet.Networks
{
    public class EnsembleLearning
    {
        private readonly string[] _files;

        public EnsembleLearning(string[] files)
        {
            _files = files;
        }

        public Tuple<Tensor, double> Predict(IDataSetLoader testDataSet)
        {
            var yCpuPredictedAllNetworks = new CpuTensor<float>(testDataSet.Y_Shape, "yCpuPredictedAllNetworks");
            foreach (var file in _files)
            {
                Console.WriteLine("Loading " + file + " ...");
                var network = Network.ValueOf(file, 0);
                Console.WriteLine("File loaded");
                Console.WriteLine("Computing accuracy for single network...");
                var yPredictedSingleNetwork = network.MiniBatchGradientDescent(128, testDataSet);
                var yCpuPredictedSingleNetwork = yPredictedSingleNetwork.ToCpuFloat();
                var accuracy = testDataSet.Y.ComputeAccuracy(yCpuPredictedSingleNetwork, null);
                Console.WriteLine("Single Network Accuracy=" + accuracy);
                yCpuPredictedAllNetworks.Update_Adding_Alpha_X(1f/ _files.Length, yCpuPredictedSingleNetwork);
                network.Dispose();
            }
            var accuracyEnsembleNetwork = testDataSet.Y.ComputeAccuracy(yCpuPredictedAllNetworks, null);
            Console.WriteLine("Ensemble Network Accuracy=" + accuracyEnsembleNetwork);
            return Tuple.Create((Tensor)yCpuPredictedAllNetworks, accuracyEnsembleNetwork);
        }
    }
}