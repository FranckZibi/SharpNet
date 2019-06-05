using System;
using SharpNet.CPU;
using SharpNet.Data;

namespace SharpNet.Networks
{
    public class EnsembleLearning
    {
        private readonly string[] _files;

        public EnsembleLearning(string[] files)
        {
            _files = files;
        }

        public Tuple<Tensor, double> Predict(Tensor xTestCpu, Tensor yExpectedCpu)
        {
            CpuTensor<float> yPredictedEnsembleNetwork = null;
            Tensor xTest = null;
            Tensor yExpected = null;
            foreach (var file in _files)
            {
                Console.WriteLine("Loading " + file + " ...");
                var net = Network.ValueOf(file, 0);
                Console.WriteLine("File loaded");

                Console.WriteLine("Computing accuracy for single network...");
                xTest = xTest ?? net.ReformatToCorrectType(xTestCpu);
                yExpected = yExpected ?? net.ReformatToCorrectType(yExpectedCpu);
                var yPredicted = net.MiniBatchGradientDescent(128, xTest, yExpected);
                var accuracy = net.ComputeLossAndAccuracy_From_Expected_vs_Predicted(yExpected, yPredicted).Item2;
                Console.WriteLine("Single Network Accuracy=" + accuracy);

                var yPredictedAsCpu = new CpuTensor<float>(yPredicted.Shape, yPredicted.ContentAsFloatArray(), yPredicted.Description);
                if (yPredictedEnsembleNetwork == null)
                {
                    yPredictedEnsembleNetwork = yPredictedAsCpu;
                }
                else
                {
                    yPredictedEnsembleNetwork.Update_Adding_Alpha_X(1.0, yPredictedAsCpu);
                }
                net.ClearMemory();
            }
            yPredictedEnsembleNetwork?.Update_Multiplying_By_Alpha(1.0 / _files.Length);
            var accuracyEnsembleNetwork = yExpectedCpu.ComputeAccuracy(yPredictedEnsembleNetwork, null);
            Console.WriteLine("Ensemble Network Accuracy=" + accuracyEnsembleNetwork);
            return Tuple.Create((Tensor)yPredictedEnsembleNetwork, accuracyEnsembleNetwork);
        }
    }
}