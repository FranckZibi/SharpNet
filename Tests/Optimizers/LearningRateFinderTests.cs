using System.Diagnostics;
using NUnit.Framework;
using SharpNet;
using SharpNetTests.NonReg;

namespace SharpNetTests.Optimizers
{
    [TestFixture]
    public class LearningRateFinderTests
    {

        [Test, Explicit]
        public void TestLearningRateFinderTests_ResNet20V1_CIFAR10()
        {
            TestResNetCIFAR10.LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet20V1_CIFAR10(true, false, Logger(nameof(TestLearningRateFinderTests_ResNet20V1_CIFAR10)));
            network.FindBestLearningRate(xTrain, yTrain, 128);
            network.ClearMemory();
        }

        private Logger Logger(string networkName)
        {
            var logFileName = Utils.ConcatenatePathWithFileName(@"c:\temp\ML\",
                networkName + "_" + Process.GetCurrentProcess().Id + "_" +
                System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            return new Logger(logFileName, true);
        }

    }
}