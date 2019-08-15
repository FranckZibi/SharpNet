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
            var network = param.ResNet20V1_CIFAR10();
            var loader = new CIFAR10DataLoader();
            network.FindBestLearningRate(loader.Training, 128);
            network.ClearMemory();
            loader.Dispose();
        }
    }
}