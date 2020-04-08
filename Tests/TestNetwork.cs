using System;
using System.IO;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Networks;
using SharpNet.Optimizers;
using SharpNetTests.CPU;

namespace SharpNetTests
{
    [TestFixture]
    public class TestNetwork
    {
        [TestCase(false)]
        [TestCase(true)]
        public void TestSave(bool useGPU)
        {
            PerformTest(useGPU, CheckSave);
        }

        [TestCase(false)]
        [TestCase(true)]
        public void TestClone(bool useGPU)
        {
            PerformTest(useGPU, CheckClone);
        }

        private static void PerformTest(bool useGPU, Action<Network> testToPerform)
        {
            var rand = new Random(0);
            int m = 4;
            int categoryCount = 2;
            var xTrain = TestCpuTensor.RandomFloatTensor(new[] { m, 2, 4, 4 }, rand, -1.0, +1.0, "xTrain");
            var yTrain = TestCpuTensor.RandomOneHotTensor(new[] { m, categoryCount }, rand, "yTrain");
            var param = DenseNetBuilder.DenseNet_CIFAR10();
            param.GpuDeviceId = useGPU?0:-1;
            param.DisableLogging = true;
            var network = param.Build(nameof(TestSave), xTrain.Shape, categoryCount, false, new[] { 2, 2 }, true, 8, 1.0, null);
            network.Config.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM;
            testToPerform(network);
            Fit(network, xTrain,yTrain, 0.1, 10, 2);
            network.Description = "after training";
            testToPerform(network);
        }

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

        private static void CheckSave(Network network)
        {
            var fileName = Path.Combine(Path.GetTempPath(), "network.txt");
            try
            {
                network.Save(fileName);
                var networkFromFile = Network.ValueOf(fileName);
                var areEquals = network.Equals(networkFromFile, out var errors);
                Assert.IsTrue(areEquals, errors);
            }
            finally
            {
                if (File.Exists(fileName))
                {
                    File.Delete(fileName);
                }
            }
        }

        private static void CheckClone(Network network)
        {
            var clone = network.Clone(network.GpuWrapper);
            var areEquals = network.Equals(clone, out var errors);
            Assert.IsTrue(areEquals, errors);
        }
    }
}
