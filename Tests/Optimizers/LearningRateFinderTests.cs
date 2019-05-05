using System.Diagnostics;
using SharpNet;
using SharpNetTests.NonReg;

namespace SharpNetTests.Optimizers
{
    public static class LearningRateFinderTests
    {

        public static void TestLearningRateFinderTests_ResNet20V1_CIFAR10(ResNetMetaParameters param)
        {
            TrainResNet.LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet20V1_CIFAR10(param, Logger(nameof(TestLearningRateFinderTests_ResNet20V1_CIFAR10)));
            network.FindBestLearningRate(xTrain, yTrain, 128);
            network.ClearMemory();
        }

        private static Logger Logger(string networkName)
        {
            var logFileName = Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory,
                networkName + "_" + Process.GetCurrentProcess().Id + "_" +
                System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            return new Logger(logFileName, true);
        }

    }
}