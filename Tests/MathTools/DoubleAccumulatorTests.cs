using NUnit.Framework;
using SharpNet.MathTools;

namespace SharpNetTests.MathTools
{
    [TestFixture]
    public class DoubleAccumulatorTests
    {

        [Test]
        public void Test_0_element()
        {
            var accumulator = new DoubleAccumulator();
            Assert.AreEqual(0, accumulator.Count);
            Assert.AreEqual(0, accumulator.Average, 1e-6);
            Assert.AreEqual(0, accumulator.Volatility, 1e-6);
        }
        [Test]
        public void Test_1_element()
        {
            var accumulator = new DoubleAccumulator();
            accumulator.Add(52.21);
            Assert.AreEqual(1, accumulator.Count);
            Assert.AreEqual(52.21, accumulator.Average, 1e-6);
            Assert.AreEqual(0, accumulator.Volatility, 1e-6);
        }


        /// <summary>
        /// this test comes from https://en.wikipedia.org/wiki/Simple_linear_regression
        /// </summary>
        [Test]
        public void Test_15_elements()
        {
            var accumulator = new DoubleAccumulator();
            accumulator.Add(52.21);
            accumulator.Add(53.12);
            accumulator.Add(54.48);
            accumulator.Add(55.84);
            accumulator.Add(57.20);
            accumulator.Add(58.57);
            accumulator.Add(59.93);
            accumulator.Add(61.29);
            accumulator.Add(63.11);
            accumulator.Add(64.47);
            accumulator.Add(66.28);
            accumulator.Add(68.10);
            accumulator.Add(69.92);
            accumulator.Add(72.19);
            accumulator.Add(74.46);
            Assert.AreEqual(15, accumulator.Count);
            Assert.AreEqual(62.078, accumulator.Average, 1e-6);
            Assert.AreEqual(6.7988853988477986, accumulator.Volatility, 1e-6);
        }
    }
}
