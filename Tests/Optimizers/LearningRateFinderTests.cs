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
            using (var cifar10 = new CIFAR10DataSet())
            using (var network = param.ResNet20V1_CIFAR10(cifar10))
            {
                network.FindBestLearningRate(cifar10.Training, 128);
            }
        }
    }
}