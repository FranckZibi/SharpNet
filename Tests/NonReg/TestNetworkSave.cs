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
            var param = new DenseNetMetaParameters {UseGPU = useGPU, UseDoublePrecision = useDoublePrecision };
            var network = DenseNetUtils.DenseNet(xTrain.Shape, nbCategories, false, new[] {2, 2}, true, 8, -1, 1.0, null, param, Logger.NullLogger);
            CheckSave(network);
            network.Fit(xTrain, yTrain, 0.1, 10,2);
            network.Description = "after training";
            CheckSave(network);
        }

        private static void CheckSave(Network net)
        {
            var path = "";
            try
            {
                path = net.Save(Path.GetTempPath());
                var net2 = Network.ValueOf(path);
                var areEquals = net.Equals(net2, out var errors);
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
                Assert.IsTrue(areEquals, errors);
            }
            finally
            {
                if (File.Exists(path))
                {
                    File.Delete(path);
                }
            }
        }

    }
}
