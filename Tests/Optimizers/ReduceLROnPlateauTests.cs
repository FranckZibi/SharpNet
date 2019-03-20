using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.Data;
using SharpNet.Optimizers;

namespace SharpNetTests.Optimizers
{
    [TestFixture]
    public class ReduceLROnPlateauTests
    {
        private const double ValidationLoss = double.NaN;
        private const double ValidationAccuracy = double.NaN;

        [Test]
        public void NbConsecutiveEpochsWithoutProgressTest()
        {
            var epochs = new List<EpochData> { 
                    new EpochData(1, 1, 1, 0.5, 0.8, ValidationLoss, ValidationAccuracy),
                    new EpochData(2, 1, 1, 0.5, 0.81, ValidationLoss, ValidationAccuracy),
                    new EpochData(3, 1, 1, 0.5, 0.805, ValidationLoss, ValidationAccuracy),
                    new EpochData(4, 1, 1, 0.5, 0.79, ValidationLoss, ValidationAccuracy),
                    new EpochData(5, 1, 1, 0.5, 0.78, ValidationLoss, ValidationAccuracy)
            };
            Assert.AreEqual(3, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
            epochs.Add(new EpochData(6, 1, 1, 0.5, 0.81, ValidationLoss, ValidationAccuracy));
            Assert.AreEqual(0, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
        }

        [Test]
        public void NbConsecutiveEpochsWithSameMultiplicativeFactorTest()
        {
            var epochs = new List<EpochData>();
            Assert.AreEqual(0, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.5, 0.83, ValidationLoss, ValidationAccuracy));
            Assert.AreEqual(1, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.5, 0.83, ValidationLoss, ValidationAccuracy));
            Assert.AreEqual(2, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(2, 1, 0.9, 0.5, 0.81, ValidationLoss, ValidationAccuracy));
            Assert.AreEqual(1, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(3, 1, 0.9, 0.5, 0.805, ValidationLoss, ValidationAccuracy));
            Assert.AreEqual(2, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(4, 1, 1, 0.5, 0.805, ValidationLoss, ValidationAccuracy));
            Assert.AreEqual(1, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(5, 1, 1, 0.5, 0.831, ValidationLoss, ValidationAccuracy));
            Assert.AreEqual(2, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(6, 1, 1, 0.5, 0.82, ValidationLoss, ValidationAccuracy));
            Assert.AreEqual(3, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(7, 1, 1, 0.5, 0.82, ValidationLoss, ValidationAccuracy));
            Assert.AreEqual(4, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(8, 1, 1, 0.5, 0.825, ValidationLoss, ValidationAccuracy));
            Assert.AreEqual(5, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
        }

        [Test]
        public void ShouldReduceLrOnPlateauTest()
        {
            var reduce = new ReduceLROnPlateau(0.5, 2, 0);
            var epochs = new List<EpochData>();
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.5, 0.83, ValidationLoss, ValidationAccuracy));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(2, 1, 1, 0.5, 0.81, ValidationLoss, ValidationAccuracy));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(3, 1, 1, 0.5, 0.805, ValidationLoss, ValidationAccuracy));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(4, 1, 1, 0.5, 0.805, ValidationLoss, ValidationAccuracy));
            Assert.IsTrue(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(5, 1, 1, 0.5, 0.831, ValidationLoss, ValidationAccuracy));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(6, 1, 1, 0.5, 0.82, ValidationLoss, ValidationAccuracy));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(7, 1, 1, 0.5, 0.82, ValidationLoss, ValidationAccuracy));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(8, 1, 1, 0.5, 0.825, ValidationLoss, ValidationAccuracy));
            Assert.IsTrue(reduce.ShouldReduceLrOnPlateau(epochs));
        }
    }
}
