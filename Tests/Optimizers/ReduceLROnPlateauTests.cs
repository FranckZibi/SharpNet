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
        private const double TrainingAccuracy = double.NaN;
        private const double SecondsForEpoch = double.NaN;

        [Test]
        public void NbConsecutiveEpochsWithoutProgressTest()
        {
            var epochs = new List<EpochData>();
            Assert.AreEqual(0, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.50, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(0, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.51, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(1, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.52, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(2, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
            Assert.AreEqual(1, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs, 1));
            Assert.AreEqual(2, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs, 2));
            Assert.AreEqual(2, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs, 3));
            epochs.Add(new EpochData(1, 1, 1, 0.60, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(3, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.53, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(4, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
            Assert.AreEqual(1, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs, 1));
            Assert.AreEqual(2, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs, 2));
            Assert.AreEqual(3, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs, 3));
            Assert.AreEqual(4, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs, 4));
            Assert.AreEqual(4, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs, 5));
            epochs.Add(new EpochData(1, 1, 1, 0.4, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(0, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
            Assert.AreEqual(0, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs),1);
            epochs.Add(new EpochData(1, 1, 1, 0.45, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(1, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.39, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(0, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.70, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(1, ReduceLROnPlateau.NbConsecutiveEpochsWithoutProgress(epochs));
        }

        [Test]
        public void NbConsecutiveEpochsWithSameMultiplicativeFactorTest()
        {
            var epochs = new List<EpochData>();
            Assert.AreEqual(0, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(1, 1, 1, 1, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(1, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(1, 1, 1, 1, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(2, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(1, 1, 0.9, 1, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(1, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(1, 1, 0.9, 1, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(2, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(1, 1, 1, 1, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(1, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(1, 1, 1, 1, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(2, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(1, 1, 1, 1, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(3, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(1, 1, 1, 1, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(4, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
            epochs.Add(new EpochData(1, 1, 1, 1, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.AreEqual(5, ReduceLROnPlateau.NbConsecutiveEpochsWithSameMultiplicativeFactor(epochs));
        }

        [Test]
        public void ShouldReduceLrOnPlateauTestNoCoolDown()
        {
            var reduce = new ReduceLROnPlateau(0.5, 1, 0);
            var epochs = new List<EpochData>();
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.50, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.51, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.52, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsTrue(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.51, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsTrue(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.40, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.45, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.42, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsTrue(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.41, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsTrue(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.39, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));

            //always decreasing training loss
            epochs.Clear();
            for (int validationLoss = 100; validationLoss >= 0; --validationLoss)
            {
                epochs.Add(new EpochData(1, 1, 1, validationLoss, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
                Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            }
        }

        [Test]
        public void ShouldReduceLrOnPlateauTestWithCoolDown()
        {
            var reduce = new ReduceLROnPlateau(0.5, 1, 1);
            var epochs = new List<EpochData>();
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.50, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.51, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 1, 0.52, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsTrue(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 0.9, 0.51, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 0.9, 0.51, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsTrue(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 0.8, 0.40, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 0.8, 0.45, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 0.8, 0.42, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsTrue(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 0.7, 0.41, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            epochs.Add(new EpochData(1, 1, 0.7, 0.39, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
            Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));

            //always decreasing training loss
            epochs.Clear();
            for (int validationLoss = 100; validationLoss >= 0; --validationLoss)
            {
                epochs.Add(new EpochData(1, 1, 1, validationLoss, TrainingAccuracy, ValidationLoss, ValidationAccuracy, SecondsForEpoch));
                Assert.IsFalse(reduce.ShouldReduceLrOnPlateau(epochs));
            }
        }
    }
}
