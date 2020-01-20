using System.Linq;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class RotateTests
    {
        [Test]
        public void TestRotate()
        {
            ////single element
            var input = new[] { 12f };
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = new[] { 12f };
            OperationTests.Check(new Rotate(60, inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            OperationTests.Check(new Rotate(150, inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 4x4 matrix
            input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            inputShape = new[] { 1, 1, 4, 4 };
            //360° rotation = no rotation
            expected = new[] { 0f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
            OperationTests.Check(new Rotate(360, inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //anti clockwise rotation of 90° (90.0)
            expected = new[] { 3f, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12 };
            OperationTests.Check(new Rotate(90, inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //rotation of 180° 
            expected = new[] { 15f, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
            OperationTests.Check(new Rotate(180, inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //rotation of -180° 
            expected = new[] { 15f, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
            OperationTests.Check(new Rotate(-180, inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //clockwise rotation of 90° (-90.0)
            expected = new[] { 12f, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3 };
            var operation = new Rotate(-90, inputShape[2], inputShape[3]);
            OperationTests.Check(operation, input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            Assert.IsTrue(operation.ChangeCoordinates());
        }
    }
}