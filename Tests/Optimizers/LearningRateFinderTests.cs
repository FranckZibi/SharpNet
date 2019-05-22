using SharpNet;
using SharpNet.Datasets;

namespace SharpNetTests.Optimizers
{
    public static class LearningRateFinderTests
    {
        public static void TestLearningRateFinderTests_ResNet20V1_CIFAR10(ResNetBuilder param)
        {
            CIFAR10.LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = param.ResNet20V1_CIFAR10();
            network.FindBestLearningRate(xTrain, yTrain, 128);
            network.ClearMemory();
        }
    }
}