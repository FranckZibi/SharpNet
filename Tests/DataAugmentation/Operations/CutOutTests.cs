using System.Linq;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class CutOutTests
    {
        [Test]
        public void TestCutOut()
        {
            // 4x4 matrix
            var input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            var inputShape = new[] { 1, 1, 4, 4 };

            //full cutout
            var expected = Enumerable.Repeat(0f, 16).ToArray();
            OperationTests.Check(new Cutout(0, 3, 0, 3), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);

            //single element cutout
            expected = new[] { 0f, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
            OperationTests.Check(new Cutout(1, 1, 2, 2), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);
            
            //single row cutout
            expected = new[] { 0f, 1, 2, 3, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15 };
            OperationTests.Check(new Cutout(1, 1, 0, 3), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);

            //single col cutout
            expected = new[] { 0f, 1, 0, 3, 4, 5, 0, 7, 8, 9, 0, 11, 12, 13, 0, 15 };
            OperationTests.Check(new Cutout(0, 3, 2, 2), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);

            //top right cutout
            expected = new[] { 0f, 1, 0, 0, 4, 5, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15 };
            var operation = new Cutout(0, 1, 2, 3);
            OperationTests.Check(operation, input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);

            Assert.IsFalse(operation.ChangeCoordinates());
        }
    }
}