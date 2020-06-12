using NUnit.Framework;
using SharpNet.Optimizers;

namespace SharpNetTests.Optimizers
{
    [TestFixture]
    public class CyclicCosineAnnealingLearningRateSchedulerTests
    {
        [Test]
        public void TestLearningRates()
        {
            const double initialLearningRate = 1.0;
            var scheduler = new CyclicCosineAnnealingLearningRateScheduler(0, initialLearningRate, 10, 2, 70);
            Assert.AreEqual(initialLearningRate, scheduler.LearningRate(1,0), 1e-6);
            Assert.AreEqual(initialLearningRate/2, scheduler.LearningRate(6,0), 1e-6);
            Assert.AreEqual(0.024471698166189548, scheduler.LearningRate(10,0), 1e-6);
            Assert.AreEqual(initialLearningRate, scheduler.LearningRate(11,0), 1e-6);
            Assert.AreEqual(initialLearningRate/2, scheduler.LearningRate(21,0), 1e-6);
            Assert.AreEqual(0.0061558180304185917, scheduler.LearningRate(30,0), 1e-6);
            Assert.AreEqual(initialLearningRate, scheduler.LearningRate(31,0), 1e-6);
            Assert.AreEqual(0.99961451810108981, scheduler.LearningRate(31,0.5), 1e-6);
            Assert.AreEqual(initialLearningRate/2, scheduler.LearningRate(51,0), 1e-6);
            Assert.AreEqual(0.0015413301293829562, scheduler.LearningRate(70,0), 1e-6);
            Assert.AreEqual(0.0, scheduler.LearningRate(70,1.0), 1e-6);
        }
    }
}
