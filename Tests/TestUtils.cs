using System;
using NUnit.Framework;
using SharpNet;

namespace SharpNetTests
{
    [TestFixture]
    public class TestUtils
    {
        [TestCase(100, 99, 20)]
        [TestCase(20, 19, 20)]
        [TestCase(20, 20, 20)]
        [TestCase(40, 21, 20)]
        [TestCase(3, 3, 1)]
        public void TestFirstMultipleOfAtomicValueAboveOrEqualToMinimum(int expected, int minimum, int atomicValue)
        {
            Assert.AreEqual(expected, Utils.FirstMultipleOfAtomicValueAboveOrEqualToMinimum(minimum, atomicValue));
        }

        [TestCase(0, null)]
        [TestCase(0, new int[0])]
        [TestCase(0, new int[0])]
        [TestCase(3, new[] {3})]
        [TestCase(3*5, new[] {3, 5})]
        [TestCase(3*5*7, new[] {3, 5, 7})]
        [TestCase(0, new[] {3, 5, 7, 0})]
        public void TestProduct(int expectedResult, int[] data)
        {
            Assert.AreEqual(expectedResult, Utils.Product(data));
        }

        [Test]
        public void TestBetaDistribution()
        {
            double sum = 0.0;
            double sumSquare = 0.0;
            var rand = new Random(0);
            const int count = 100000;
            for (int i = 0; i < count; ++i)
            {
                var val = Utils.BetaDistribution(1.0, 1.0, rand);
                sum += val;
                sumSquare += val * val;
            }
            var mean = sum / count;
            var meanOfSquare = sumSquare / count;
            var variance = Math.Max(0,meanOfSquare - mean * mean);
            const double epsilon = 0.01;
            Assert.AreEqual(0.5, mean, epsilon);
            Assert.AreEqual(1/12.0, variance, epsilon);
        }
        [Test]
        public void TestNewVersion()
        {
            Assert.AreEqual(new Version(7,6,5), Utils.NewVersion(7605));
            Assert.AreEqual(new Version(7,6,0), Utils.NewVersion(7600));
        }
        [Test]
        public void TestNewVersionXXYY0()
        {
            Assert.AreEqual(new Version(9, 2), Utils.NewVersionXXYY0(9020));
            Assert.AreEqual(new Version(10, 1), Utils.NewVersionXXYY0(10010));
        }
    }
}
