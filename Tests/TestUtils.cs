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
    }
}
