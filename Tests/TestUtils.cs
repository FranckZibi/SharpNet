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

    }
}
