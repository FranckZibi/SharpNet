using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class FlipTests
    {

        [Test]
        public void TestHorizontalFlip()
        {
            //single element
            var input = new[] { 12f };
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = new[] { 12f };
            OperationTests.Check(new HorizontalFlip(inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //single line
            input = new[] { 12f, 13, 14 };
            inputShape = new[] { 1, 1, 1, 3 };
            expected = new[] { 14f, 13, 12 };
            OperationTests.Check(new HorizontalFlip(inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //single column
            input = new[] { 12f, 13, 14 };
            inputShape = new[] { 1, 1, 3, 1 };
            expected = input;
            OperationTests.Check(new HorizontalFlip(inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 4x4 matrix
            input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            inputShape = new[] { 1, 1, 4, 4 };
            expected = new[] { 3f, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12 };
            OperationTests.Check(new HorizontalFlip(inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
        }

        [Test]
        public void TestVerticalFlip()
        {
            //single element
            var input = new[] { 12f};
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = new[] { 12f};
            OperationTests.Check(new VerticalFlip(inputShape[2]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //single line
            input = new[] { 12f,13, 14 };
            inputShape = new[] { 1, 1, 1, 3 };
            expected = input;
            OperationTests.Check(new VerticalFlip(inputShape[2]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //single column
            input = new[] { 12f, 13, 14 };
            inputShape = new[] { 1, 1, 3, 1 };
            expected = new[] { 14f, 13, 12 };
            OperationTests.Check(new VerticalFlip(inputShape[2]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 4x4 matrix
            input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            inputShape = new[] { 1, 1, 4, 4 };
            expected = new[] { 12f, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3 };
            OperationTests.Check(new VerticalFlip(inputShape[2]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
        }

        [Test]
        public void TestVerticalFlip_And_HorizontalFlip()
        {
            //single element
            var input = new[] { 12f };
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = new[] { 12f };
            OperationTests.CheckAllPermutations(new VerticalFlip(inputShape[2]), new HorizontalFlip(inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //single line
            input = new[] { 12f, 13, 14 };
            inputShape = new[] { 1, 1, 1, 3 };
            expected = new[] { 14f, 13, 12 };
            OperationTests.CheckAllPermutations(new VerticalFlip(inputShape[2]), new HorizontalFlip(inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //single column
            input = new[] { 12f, 13, 14 };
            inputShape = new[] { 1, 1, 3, 1 };
            expected = new[] { 14f, 13, 12 };
            OperationTests.CheckAllPermutations(new VerticalFlip(inputShape[2]), new HorizontalFlip(inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 4x4 matrix
            input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            inputShape = new[] { 1, 1, 4, 4 };
            expected = new[] { 15f, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
            var verticalFlip = new VerticalFlip(inputShape[2]);
            var horizontalFlip = new HorizontalFlip(inputShape[3]);
            OperationTests.CheckAllPermutations(verticalFlip, horizontalFlip, input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            Assert.IsTrue(verticalFlip.ChangeCoordinates());
            Assert.IsTrue(horizontalFlip.ChangeCoordinates());
        }


        [Test]
        public void TestRotate180Degrees()
        {
            //single element
            var input = new[] { 12f };
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = new[] { 12f };
            OperationTests.Check(new Rotate180Degrees(inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //single line
            input = new[] { 12f, 13, 14 };
            inputShape = new[] { 1, 1, 1, 3 };
            expected = new[] { 14f, 13, 12 };
            OperationTests.Check(new Rotate180Degrees(inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            //single column
            input = new[] { 12f, 13, 14 };
            inputShape = new[] { 1, 1, 3, 1 };
            expected = new[] { 14f, 13, 12 };
            OperationTests.Check(new Rotate180Degrees(inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 4x4 matrix
            input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            inputShape = new[] { 1, 1, 4, 4 };
            expected = (float[])input.Clone();
            Array.Reverse(expected);
            OperationTests.Check(new Rotate180Degrees(inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
        }

        [Test]
        public void TestRotate90Degrees()
        {
            //single element
            var input = new[] { 12f };
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = new[] { 12f };
            OperationTests.Check(new Rotate90Degrees(inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

          
            // 4x4 matrix
            input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            inputShape = new[] { 1, 1, 4, 4 };
            expected = new[] { 12f, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3 };
            Array.Reverse(expected);
            OperationTests.Check(new Rotate90Degrees(inputShape[2], inputShape[3]), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
        }
    }
}
