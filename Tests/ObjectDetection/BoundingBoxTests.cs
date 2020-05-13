using NUnit.Framework;
using SharpNet.ObjectDetection;

namespace SharpNetTests.ObjectDetection
{
    [TestFixture]
    public class BoundingBoxTests
    {
        [Test]
        public void Test_Right_Left_Top_Bottom_Area()
        {
            var box =  new BoundingBox(0.25, 0.25+0.25/2, 0.5, 0.25);
            Assert.AreEqual(box.Left,0.0, 1e-6); 
            Assert.AreEqual(box.Right,0.5, 1e-6); 
            Assert.AreEqual(box.Top,0.25, 1e-6); 
            Assert.AreEqual(box.Bottom,0.5, 1e-6); 
            Assert.AreEqual(box.Area, 0.5*0.25, 1e-6); 
        }

        [Test]
        public void Test_Intersection()
        {
            var topLeft = new BoundingBox(0.25, 0.25, 0.5, 0.5);
            var center = new BoundingBox(0.5, 0.5, 0.5, 0.5);
            var bottomRight = new BoundingBox(0.75, 0.75, 0.5, 0.5);
            Assert.AreEqual(topLeft.Intersection(bottomRight), 0.0, 1e-6);
            Assert.AreEqual(topLeft.Intersection(center), 0.25*0.25, 1e-6);
            Assert.AreEqual(center.Intersection(topLeft), 0.25*0.25, 1e-6);
            Assert.AreEqual(bottomRight.Intersection(center), 0.25*0.25, 1e-6);
        }

        [Test]
        public void Test_Union()
        {
            var topLeft = new BoundingBox(0.25, 0.25, 0.5, 0.5);
            var center = new BoundingBox(0.5, 0.5, 0.5, 0.5);
            var bottomRight = new BoundingBox(0.75, 0.75, 0.5, 0.5);
            Assert.AreEqual(topLeft.Union(bottomRight), 0.5*0.5+0.5*0.5, 1e-6);
            Assert.AreEqual(topLeft.Union(center), 0.5 * 0.5 + 0.5 * 0.5-0.25*0.25, 1e-6);
            Assert.AreEqual(bottomRight.Union(center), 0.5 * 0.5 + 0.5 * 0.5-0.25*0.25, 1e-6);
        }

        [Test]
        public void Test_IoU()
        {
            var topLeft = new BoundingBox(0.25, 0.25, 0.5, 0.5);
            var center = new BoundingBox(0.5, 0.5, 0.5, 0.5);
            var bottomRight = new BoundingBox(0.75, 0.75, 0.5, 0.5);
            Assert.AreEqual(topLeft.IoU(bottomRight), 0, 1e-6);
            Assert.AreEqual(topLeft.IoU(center), topLeft.Intersection(center)/ topLeft.Union(center), 1e-6);
            Assert.AreEqual(bottomRight.IoU(center), topLeft.Intersection(center) / topLeft.Union(center), 1e-6);
        }

    }
}
