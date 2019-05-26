using System;
using System.IO;
using NUnit.Framework;
using SharpNet.Networks;
using SharpNetTests.CPU;

namespace SharpNetTests
{
    [TestFixture]
    public class TestNetwork
    {
        [TestCase(false, false)]
        [TestCase(false, true)]
        [TestCase(true, false)]
        [TestCase(true, true)]
        public void TestSave(bool useDoublePrecision, bool useGPU)
        {
            PerformTest(useDoublePrecision, useGPU, CheckSave);
        }

        [TestCase(false, false)]
        [TestCase(false, true)]
        [TestCase(true, false)]
        [TestCase(true, true)]
        public void TestClone(bool useDoublePrecision, bool useGPU)
        {
            PerformTest(useDoublePrecision, useGPU, CheckClone);
        }

        private static void PerformTest(bool useDoublePrecision, bool useGPU, Action<Network> testToPerform)
        {
            var rand = new Random(0);
            int m = 4;
            int nbCategories = 2;
            var xTrain = TestCpuTensor.RandomDoubleTensor(new[] { m, 2, 4, 4 }, rand, -1.0, +1.0, "xTrain");
            var yTrain = TestCpuTensor.RandomOneHotTensor(new[] { m, nbCategories }, rand, "yTrain");
            var param = new DenseNetBuilder();
            param.GpuDeviceId = useGPU?0:-1;
            param.Config.UseDoublePrecision = useDoublePrecision;
            param.DisableLogging = true;
            var network = param.Build(nameof(TestSave), xTrain.Shape, nbCategories, false, new[] { 2, 2 }, true, 8, 1.0, null);
            testToPerform(network);
            network.Fit(xTrain, yTrain, 0.1, 10, 2);
            network.Description = "after training";
            testToPerform(network);
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
