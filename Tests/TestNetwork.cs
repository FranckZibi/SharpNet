using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.Networks;
using SharpNet.Optimizers;
using SharpNetTests.Data;

namespace SharpNetTests
{
    [TestFixture]
    public class TestNetwork
    {

        [Test]
        public void TestSaveParametersToH5File()
        {
            //we build an efficientNet-B0 network loading the weights from in Keras
            var networkBuilder = EfficientNetBuilder.CIFAR10();
            networkBuilder.Config.LogDirectory = "";
            var network = networkBuilder.EfficientNetB0(true, "imagenet", new[] { 3, 224, 224 });

            //we save the network parameters
            var saveParametersFile = Path.Combine(NetworkConfig.DefaultLogDirectory, "test_EfficientNetB0.h5");
            network.SaveParameters(saveParametersFile);
            network.Dispose();

            //we ensure that the saved parameters are the same as the original one in Keras
            var kerasParametersFile = NetworkBuilder.GetKerasModelPath("efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment.h5");
            using (var kerasParameters = new H5File(kerasParametersFile))
            {
                using var sharpNetParameters = new H5File(saveParametersFile);
                var datasetKeras = kerasParameters.Datasets();
                var datasetSharpNet = sharpNetParameters.Datasets();
                Debug.Assert(datasetKeras.Count == datasetSharpNet.Count);
                foreach (var a in datasetKeras)
                {
                    Assert.IsTrue(TestTensor.SameContent(datasetKeras[a.Key], datasetSharpNet[a.Key], 1e-5));
                }
            }
            File.Delete(saveParametersFile);
        }

        public static void Fit(Network network, CpuTensor<float> X, CpuTensor<float> Y, double learningRate, int numEpochs, int batchSize, IDataSet testDataSet = null)
        {
            network.Config.DisableReduceLROnPlateau = true;
            using var trainingDataSet = new InMemoryDataSet(X, Y, Y_to_Categories(Y), "", null);
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
