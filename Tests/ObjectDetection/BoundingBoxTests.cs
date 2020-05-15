using NUnit.Framework;
using SharpNet.ObjectDetection;

namespace SharpNetTests.ObjectDetection
{
    [TestFixture]
    public class BoundingBoxTests
    {
        private static readonly BoundingBox b1 = new BoundingBox(132.625, 76.742, 24.594, 41.12);
        private static readonly BoundingBox b2 = new BoundingBox(155.751, 55.233, 25.32, 43.186);
        private static readonly BoundingBox topLeft = new BoundingBox(0.25, 0.25, 0.5, 0.5);
        private static readonly BoundingBox center = new BoundingBox(0.5, 0.5, 0.5, 0.5);
        private static readonly BoundingBox bottomRight = new BoundingBox(0.75, 0.75, 0.5, 0.5);

        [Test]
        public void Test_Right_Left_Top_Bottom_Area()
        {
            var box = new BoundingBox(0.25, 0.25 + 0.25 / 2, 0.5, 0.25);

            Assert.AreEqual(0.0, box.Left, 1e-6);
            Assert.AreEqual(132.625 - 24.594 / 2, b1.Left, 1e-6);
            Assert.AreEqual(155.751- 25.32/2, b2.Left, 1e-6);

            Assert.AreEqual(0.5, box.Right, 1e-6);
            Assert.AreEqual(132.625+ 24.594/2, b1.Right, 1e-6);
            Assert.AreEqual(155.751 + 25.32 / 2, b2.Right, 1e-6);

            Assert.AreEqual(0.25, box.Top, 1e-6);
            Assert.AreEqual(76.742 - 41.12/2, b1.Top, 1e-6);
            Assert.AreEqual(55.233 - 43.186/2, b2.Top, 1e-6);

            Assert.AreEqual(0.5, box.Bottom, 1e-6);
            Assert.AreEqual(76.742 + 41.12 / 2, b1.Bottom, 1e-6);
            Assert.AreEqual(55.233 + 43.186 / 2, b2.Bottom, 1e-6);
        }

        [Test]
        public void Test_Area()
        {
            var box = new BoundingBox(0.25, 0.25 + 0.25 / 2, 0.5, 0.25);
            Assert.AreEqual(0.5 * 0.25, box.Area, 1e-6);
            Assert.AreEqual(1011.30528, b1.Area, 1e-6);
            Assert.AreEqual(1093.46952, b2.Area, 1e-6);
        }

        [Test]
        public void Test_Intersection()
        {
            Assert.AreEqual(0.0, topLeft.Intersection(bottomRight), 1e-6);
            Assert.AreEqual(0.25 * 0.25, topLeft.Intersection(center), 1e-6);
            Assert.AreEqual(0.25 * 0.25, center.Intersection(topLeft), 1e-6);
            Assert.AreEqual(0.25 * 0.25, bottomRight.Intersection(center), 1e-6);
            Assert.AreEqual(37.799164, b1.Intersection(b2), 1e-6);
            Assert.AreEqual(37.799164, b2.Intersection(b1), 1e-6);
        }

        [Test]
        public void Test_Union()
        {
            Assert.AreEqual(topLeft.Union(bottomRight), 0.5*0.5+0.5*0.5, 1e-6);
            Assert.AreEqual(topLeft.Union(center), 0.5 * 0.5 + 0.5 * 0.5-0.25*0.25, 1e-6);
            Assert.AreEqual(bottomRight.Union(center), 0.5 * 0.5 + 0.5 * 0.5-0.25*0.25, 1e-6);
            Assert.AreEqual(2066.975636, b1.Union(b2), 1e-6);
            Assert.AreEqual(2066.975636, b2.Union(b1), 1e-6);
        }

        [Test]
        public void Test_IoU()
        {
            Assert.AreEqual(topLeft.IoU(bottomRight), 0, 1e-6);
            Assert.AreEqual(topLeft.IoU(center), topLeft.Intersection(center)/ topLeft.Union(center), 1e-6);
            Assert.AreEqual(bottomRight.IoU(center), topLeft.Intersection(center) / topLeft.Union(center), 1e-6);
            Assert.AreEqual(0.018287184106, b1.IoU(b2), 1e-6);
            Assert.AreEqual(0.018287184106, b2.IoU(b1), 1e-6);
        }

        [Test]
        public void Test_UpSampling()
        {
            Assert.IsTrue(new BoundingBox(3*132.625, 2*76.742, 3*24.594, 2*41.12).Equals(b1.UpSampling(2.0, 3.0), 1e-6));
        }

    }
}
