using NUnit.Framework;
using SharpNet.Networks;

namespace SharpNetTests
{
    [TestFixture]
    public class TestBlocksDescription
    {
        [TestCase(16, 16, 1.0f, 8)]
        [TestCase(24, 24, 1.0f, 8)]
        [TestCase(32, 32, 1.0f, 8)]
        [TestCase(40, 40, 1.0f, 8)]
        [TestCase(80, 80, 1.0f, 8)]
        [TestCase(112, 112, 1.0f, 8)]
        [TestCase(192, 192, 1.0f, 8)]
        [TestCase(320, 320, 1.0f, 8)]
        [TestCase(1280, 1280, 1.0f, 8)]
        [TestCase(16, 16, 1.1f, 8)]
        [TestCase(24, 24, 1.1f, 8)]
        [TestCase(32, 32, 1.1f, 8)]
        [TestCase(48, 40, 1.1f, 8)]
        [TestCase(88, 80, 1.1f, 8)]
        [TestCase(120, 112, 1.1f, 8)]
        [TestCase(208, 192, 1.1f, 8)]
        [TestCase(352, 320, 1.1f, 8)]
        [TestCase(1408, 1280, 1.1f, 8)]
        public void TestRoundFilters(int expectedValue, int outputFilters, float widthCoefficient, int depthDivisor)
        {
            Assert.AreEqual(expectedValue, MobileBlocksDescription.RoundFilters(outputFilters, widthCoefficient, depthDivisor));
        }

        [TestCase(1, 1, 1.0f)]
        [TestCase(2, 2, 1.0f)]
        [TestCase(3, 3, 1.0f)]
        [TestCase(4, 4, 1.0f)]
        [TestCase(2, 1, 1.2f)]
        [TestCase(3, 2, 1.2f)]
        [TestCase(4, 3, 1.2f)]
        [TestCase(5, 4, 1.2f)]
        public void TestRoundNumRepeat(int expectedValue, int numRepeat, float depthCoefficient)
        {
            Assert.AreEqual(expectedValue, MobileBlocksDescription.RoundNumRepeat(numRepeat, depthCoefficient));
        }
    }
}