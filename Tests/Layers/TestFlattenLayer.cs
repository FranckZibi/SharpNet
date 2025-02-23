using NUnit.Framework;
using SharpNet.Layers;

namespace SharpNetTests.Layers
{
    [TestFixture]
    public class TestFlattenLayer
    {
        [Test]
        public void TestOutputShapeAfterFlatten()
        {
            Assert.AreEqual(new[] { 2, 3 * 4 * 5 }, Flatten.OutputShapeAfterFlatten(new[] { 2, 3, 4, 5 }, 1, -1));
            Assert.AreEqual(new[] { 2 * 3 * 4 * 5 }, Flatten.OutputShapeAfterFlatten(new[] { 2, 3, 4, 5 }, 0, -1));
            Assert.AreEqual(new[] { 2, 3, 4, 5 }, Flatten.OutputShapeAfterFlatten(new[] { 2, 3, 4, 5 }, 3, 3));
            Assert.AreEqual(new[] { 2, 3 * 4, 5 }, Flatten.OutputShapeAfterFlatten(new[] { 2, 3, 4, 5 }, 1, 2));
            Assert.AreEqual(new[] { 2* 3 * 4, 5 }, Flatten.OutputShapeAfterFlatten(new[] { 2, 3, 4, 5 }, 0, -2));
        }

    }
}