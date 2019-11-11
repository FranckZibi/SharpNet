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
            var cifar10 = new CIFAR10DataSet();
            network.FindBestLearningRate(cifar10.Training, 128);
            network.Dispose();
            cifar10.Dispose();
        }
    }
}