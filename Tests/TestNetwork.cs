using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.Networks;
using SharpNetTests.Data;

namespace SharpNetTests
{
    [TestFixture]
    public class TestNetwork
    {

        [Test]
        public void TestAdaptResourceIdsToCurrentComputer()
        {
            // no gpu available
            Assert.AreEqual(new[] {-2, -1 }, Network.AdaptResourceIdsToCurrentComputer(new[] { -2, -1 }, 0));
            Assert.AreEqual(new []{-1}, Network.AdaptResourceIdsToCurrentComputer(new []{0,1}, 0));
            Assert.AreEqual(new []{-1}, Network.AdaptResourceIdsToCurrentComputer(new []{0}, 0));

            // 1 gpu available
            Assert.AreEqual(new[] { -2, -1 }, Network.AdaptResourceIdsToCurrentComputer(new[] { -2, -1 }, 1));
            Assert.AreEqual(new[] { 0 }, Network.AdaptResourceIdsToCurrentComputer(new[] { 0, 1 }, 1));
            Assert.AreEqual(new[] { 0 }, Network.AdaptResourceIdsToCurrentComputer(new[] { 0 }, 1));

            // 2 gpus available
            Assert.AreEqual(new[] { -2, -1 }, Network.AdaptResourceIdsToCurrentComputer(new[] { -2, -1 }, 2));
            Assert.AreEqual(new[] { 0, 1 }, Network.AdaptResourceIdsToCurrentComputer(new[] { 0, 1 }, 2));
            Assert.AreEqual(new[] { 0 }, Network.AdaptResourceIdsToCurrentComputer(new[] { 0 }, 2));
        }

        [Test]
        public void TestSaveParametersToH5File()
        {
            //we build an efficientNet-B0 network loading the weights from in Keras
            var networkBuilder = EfficientNetSample.CIFAR10();
            networkBuilder.Config.WorkingDirectory = NetworkConfig.DefaultWorkingDirectory;
            var workingDirectory = networkBuilder.Config.WorkingDirectory;
            var network = networkBuilder.EfficientNetB0(true, "imagenet", new[] { 3, 224, 224 });

            //we save the network parameters
            network.Save(workingDirectory, network.ModelName);
            var networkFiles = network.AllFiles();
            network.Dispose();

            //we ensure that the saved parameters are the same as the original one in Keras
            var kerasParametersFile = EfficientNetSample.GetKerasModelPath("efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment.h5");
            using (var kerasParameters = new H5File(kerasParametersFile))
            {
                var parametersFile = Network.ToParameterFilePath(workingDirectory, network.ModelName);
                using var sharpNetParameters = new H5File(parametersFile);
                var datasetKeras = kerasParameters.Datasets();
                var datasetSharpNet = sharpNetParameters.Datasets();
                Debug.Assert(datasetKeras.Count == datasetSharpNet.Count);
                foreach (var a in datasetKeras)
                {
                    Assert.IsTrue(TestTensor.SameContent(datasetKeras[a.Key], datasetSharpNet[a.Key], 1e-5));
                }
            }
            networkFiles.ForEach(File.Delete);
        }

        public static void Fit(Network network, CpuTensor<float> X, CpuTensor<float> Y, double learningRate, int numEpochs, int batchSize, IDataSet testDataSet = null)
        {
            network.Config.DisableReduceLROnPlateau = true;
            using var trainingDataSet = new InMemoryDataSet(X, Y);

            network.Config.InitialLearningRate = learningRate;
            network.Config.NumEpochs = numEpochs;
            network.Config.BatchSize = batchSize;

            Fit(network, trainingDataSet, learningRate, numEpochs, batchSize, testDataSet);
        }

        public static void Fit(Network network, IDataSet trainingDataSet, double learningRate, int numEpochs, int batchSize, IDataSet testDataSet = null)
        {
            network.Config.DisableReduceLROnPlateau = true;
            network.Config.InitialLearningRate = learningRate;
            network.Config.NumEpochs = numEpochs;
            network.Config.BatchSize = batchSize;
            network.Fit(trainingDataSet, testDataSet);
        }
    }
}
