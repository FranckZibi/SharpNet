using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.Networks;
using SharpNet.Optimizers;

namespace SharpNetTests
{
    [TestFixture]
    public class TestNetwork
    {
        public static void Fit(Network network, CpuTensor<float> X, CpuTensor<float> Y, double learningRate, int numEpochs, int batchSize, IDataSet testDataSet = null)
        {
            network.Config.DisableReduceLROnPlateau = true;
            var trainingDataSet = new InMemoryDataSet(X, Y, Y_to_Categories(Y), "", null);
            var learningRateComputer = new LearningRateComputer(LearningRateScheduler.Constant(learningRate), network.Config.ReduceLROnPlateau(), network.Config.MinimumLearningRate);
            network.Fit(trainingDataSet, learningRateComputer, numEpochs, batchSize, testDataSet);
        }

        private static int[] Y_to_Categories<T>(CpuTensor<T> Y) where T: struct
        {
            var result = new int[Y.Shape[0]];
            for (int m = 0;m < Y.Shape[0]; ++m)
            {
                for (int category = 0; category < Y.Shape[1]; ++category)
                {
                    if (!Equals(Y.Get(m, category), default(T)))
                    {
                        result[m] = category;
                        break;
                    }
                }
            }
            return result;
        }

      
    }
}
