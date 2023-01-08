using System.Diagnostics;
using System.IO;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.Networks;

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




        /// <summary>
        /// used for tests only
        /// </summary>
        /// <param name="sample"></param>
        /// <param name="workingDirectory"></param>
        /// <param name="modelName"></param>
        /// <returns></returns>
        public static Network NewForTests(NetworkSample sample, string workingDirectory, string modelName)
        {
            return new Network(sample, null, workingDirectory, modelName, true);
        }


        [Test]
        public void TestSaveParametersToH5File()
        {
            //we build an efficientNet-B0 network loading the weights from in Keras
            var networkBuilder = EfficientNetNetworkSample.CIFAR10();
            var workingDirectory = NetworkSample.DefaultWorkingDirectory;
            var network = networkBuilder.EfficientNetB0(workingDirectory, true, "imagenet", new[] { 3, 224, 224 });

            //we save the network parameters
            network.Save(workingDirectory, network.ModelName);
            var networkFiles = network.AllFiles();
            network.Dispose();

            //we ensure that the saved parameters are the same as the original one in Keras
            var kerasParametersFile = EfficientNetNetworkSample.GetKerasModelPath("efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment.h5");
            using (var kerasParameters = new H5File(kerasParametersFile))
            {
                var parametersFile = Network.ToParameterFilePath(workingDirectory, network.ModelName);
                using var sharpNetParameters = new H5File(parametersFile);
                var datasetKeras = kerasParameters.Datasets();
                var datasetSharpNet = sharpNetParameters.Datasets();
                Debug.Assert(datasetKeras.Count == datasetSharpNet.Count);
                foreach (var a in datasetKeras)
                {
                    Assert.IsTrue(TensorExtensions.SameFloatContent(datasetKeras[a.Key], datasetSharpNet[a.Key], 1e-5));
                }
            }
            networkFiles.ForEach(File.Delete);
        }

        public static void Fit(Network network, CpuTensor<float> X, CpuTensor<float> Y, double learningRate, int numEpochs, int batchSize, DataSet testDataSet = null)
        {
            network.Sample.DisableReduceLROnPlateau = true;
            using var trainingDataSet = new InMemoryDataSet(X, Y);

            network.Sample.InitialLearningRate = learningRate;
            network.Sample.NumEpochs = numEpochs;
            network.Sample.BatchSize = batchSize;

            Fit(network, trainingDataSet, learningRate, numEpochs, batchSize, testDataSet);
        }

        public static void Fit(Network network, DataSet trainingDataSet, double learningRate, int numEpochs, int batchSize, DataSet testDataSet = null)
        {
            network.Sample.DisableReduceLROnPlateau = true;
            network.Sample.InitialLearningRate = learningRate;
            network.Sample.NumEpochs = numEpochs;
            network.Sample.BatchSize = batchSize;
            network.Fit(trainingDataSet, testDataSet);
        }
    }
}
