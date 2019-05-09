using System;
using System.IO;
using NUnit.Framework;
using SharpNet;
using SharpNetTests.CPU;

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestNetworkSave
    {
        [TestCase(false, false)]
        [TestCase(false, true)]
        [TestCase(true, false)]
        [TestCase(true, true)]
        public void TestSave(bool useDoublePrecision, bool useGPU)
        {
            var rand = new Random(0);
            int m = 4;
            int nbCategories = 2;
            var xTrain = TestCpuTensor.RandomDoubleTensor(new[]{m, 2, 4, 4}, rand, -1.0, +1.0, "xTrain");
            var yTrain = TestCpuTensor.RandomOneHotTensor(new[]{m, nbCategories }, rand, "yTrain");
            var param = new DenseNetConfig();
            param.Config.UseGPU = useGPU;
            param.Config.UseDoublePrecision = useDoublePrecision;
            param.DisableLogging = true;
            var network = param.GetNetwork(nameof(TestSave)).DenseNet(xTrain.Shape, nbCategories, false, new[] {2, 2}, true, 8, 1.0, null);
            CheckSave(network);
            network.Fit(xTrain, yTrain, 0.1, 10,2);
            network.Description = "after training";
            CheckSave(network);
        }

        private static void CheckSave(Network net)
        {
            var fileName = Path.Combine(Path.GetTempPath(), "network.txt");
            try
            {
                net.Save(fileName);
                var net2 = Network.ValueOf(fileName);
                var areEquals = net.Equals(net2, out var errors);
                if (File.Exists(fileName))
                {
                    File.Delete(fileName);
                }
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

    }
}
