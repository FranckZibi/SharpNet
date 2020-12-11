using System;
using NUnit.Framework;
using SharpNet;
using SharpNetTests.Data;
using SharpNetTests.NonReg;

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

        //This test is coming from: https://en.wikipedia.org/wiki/QR_decomposition
        [TestCase("[[12,6,-4],[-51,167,24],[4,-68,-41]]", "[[0.857142866,0.428571433,-0.285714298],[-0.394285709,0.902857125,0.171428576],[-0.331428587,0.0342857353,-0.942857146]]")]
        //This test is coming from: https://rosettacode.org/wiki/QR_decomposition#C.23
        [TestCase("[[12,6,-4,-1,2],[-51,167,24,1,0],[4,-68,-41,0,3]]", "[[0.846414685,0.423207343,-0.282138228,-0.0705345571,0.141069114],[-0.391290814,0.904087186,0.170420542,0.0140406527,-0.0166555103],[-0.343124002,0.0292699095,-0.932856023,0.00109936972,0.105771616]]")]
        public void TestToOrthogonalMatrix(string input, string expectedOutput)
        {
            var inputMatrix = TestNetworkPropagation.FromNumpyArray(input);
            var expectedMatrix = TestNetworkPropagation.FromNumpyArray(expectedOutput);
            Utils.ToOrthogonalMatrix(inputMatrix.AsFloatCpuSpan, inputMatrix.Shape[0], inputMatrix.MultDim0);
            Assert.IsTrue(TestTensor.SameContent(expectedMatrix, inputMatrix, 1e-6));
        }

        [Test]
        public void TestNewVersionXXYY0()
        {
            Assert.AreEqual(new Version(9, 2), Utils.NewVersionXXYY0(9020));
            Assert.AreEqual(new Version(10, 1), Utils.NewVersionXXYY0(10010));
        }
    }
}
