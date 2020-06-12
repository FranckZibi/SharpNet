using NUnit.Framework;
using SharpNet.Datasets;
using SharpNet.Networks;

namespace SharpNetTests.Optimizers
{
    [TestFixture]
    public class LearningRateFinderTests
    {
        [Test, Explicit]
        public void TestLearningRateFinderTests_ResNet20V1_CIFAR10()
        {
            var param = new ResNetBuilder();
            using (var cifar10 = new CIFAR10DataSet())
            using (var network = param.ResNet20V1_CIFAR10(cifar10))
            {
                network.FindBestLearningRate(cifar10.Training, 1e-7, 10.0, 128);
            }
        }
    }
}
