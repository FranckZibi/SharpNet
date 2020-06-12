using NUnit.Framework;
using SharpNet.Optimizers;

namespace SharpNetTests.Optimizers
{
    [TestFixture]
    public class OneCycleLearningRateSchedulerTests
    {
        [Test]
        public void TestLearningRates()
        {
            const double initialLearningRate = 1.0;
            var scheduler = new OneCycleLearningRateScheduler(initialLearningRate, 10, 0.2, 70);
            Assert.AreEqual(initialLearningRate/10, scheduler.LearningRate(1, 0), 1e-6);
            Assert.AreEqual(0.11607142857142858, scheduler.LearningRate(1, 0.5), 1e-6);
            Assert.AreEqual(0.73482142857142863, scheduler.LearningRate(20, 0.75), 1e-6);
            Assert.AreEqual(initialLearningRate, scheduler.LearningRate(29, 0), 1e-6);
            Assert.AreEqual(0.9598214285714286, scheduler.LearningRate(30, 0.25), 1e-6);
            Assert.AreEqual(initialLearningRate / 10, scheduler.LearningRate(57, 0), 1e-6);
            Assert.AreEqual(0.0080714285714285627, scheduler.LearningRate(70, 0), 1e-6);
            Assert.AreEqual(initialLearningRate/1000, scheduler.LearningRate(70, 1.0), 1e-6);
        }
    }
}
