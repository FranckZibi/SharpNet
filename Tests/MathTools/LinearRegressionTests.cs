using NUnit.Framework;
using SharpNet.MathTools;

namespace SharpNetTests.MathTools
{
    [TestFixture]
    public class LinearRegressionTests
    {
        [Test]
        public void Test_0_element()
        {
            var lr = new LinearRegression();
            Assert.AreEqual(0, lr.Beta, 1e-6);
            Assert.AreEqual(0, lr.Alpha, 1e-6);
            Assert.AreEqual(0, lr.Estimation(0.0), 1e-6);
            Assert.AreEqual(0, lr.Estimation(1.62), 1e-6);
        }

        [Test]
        public void Test_1_element()
        {
            var lr = new LinearRegression();
            Assert.AreEqual(0, lr.Beta, 1e-6);
            Assert.AreEqual(0, lr.Alpha, 1e-6);
            Assert.AreEqual(0, lr.Estimation(0.0), 1e-6);
            Assert.AreEqual(0, lr.Estimation(1.62), 1e-6);
        }

        /// <summary>
        /// this test comes from https://en.wikipedia.org/wiki/Simple_linear_regression
        /// </summary>
        [Test]
        public void Test_15_element()
        {
            var lr = new LinearRegression();
            lr.Add(1.47, 52.21);
            lr.Add(1.50, 53.12);
            lr.Add(1.52, 54.48);
            lr.Add(1.55, 55.84);
            lr.Add(1.57, 57.20);
            lr.Add(1.60, 58.57);
            lr.Add(1.63, 59.93);
            lr.Add(1.65, 61.29);
            lr.Add(1.68, 63.11);
            lr.Add(1.70, 64.47);
            lr.Add(1.73, 66.28);
            lr.Add(1.75, 68.10);
            lr.Add(1.78, 69.92);
            lr.Add(1.80, 72.19);
            lr.Add(1.83, 74.46);
            Assert.AreEqual(61.272186542107434, lr.Beta, 1e-6);
            Assert.AreEqual(-39.061955918838656, lr.Alpha, 1e-6);
            const double expectedPearsonCorrelationCoefficient = 0.99458379357687576;
            Assert.AreEqual(expectedPearsonCorrelationCoefficient, lr.PearsonCorrelationCoefficient, 1e-3);
            Assert.AreEqual(expectedPearsonCorrelationCoefficient* expectedPearsonCorrelationCoefficient, lr.RSquared, 1e-3);
            Assert.AreEqual(-39.061955918838656, lr.Estimation(0), 1e-6);
            Assert.AreEqual(60.198986279375397, lr.Estimation(1.62), 1e-6);
        }
    }
}