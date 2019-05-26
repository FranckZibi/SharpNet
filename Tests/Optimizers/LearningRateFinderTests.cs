using SharpNet.Datasets;
using SharpNet.Networks;

namespace SharpNetTests.Optimizers
{
    // ReSharper disable once UnusedMember.Global
    public static class LearningRateFinderTests
    {
        // ReSharper disable once UnusedMember.Global
        public static void TestLearningRateFinderTests_ResNet20V1_CIFAR10(ResNetBuilder param)
        {
            CIFAR10.LoadCifar10(out var xTrain, out var yTrain, out var _, out var _);
            var network = param.ResNet20V1_CIFAR10();
            network.FindBestLearningRate(xTrain, yTrain, 128);
            network.ClearMemory();
        }
    }
}