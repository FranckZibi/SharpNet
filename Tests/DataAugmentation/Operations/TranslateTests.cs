using System.Linq;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class TranslateTests
    {
        [Test]
        public void TestTranslateX()
        {
            //single element
            var input = new[] { 12f };
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = input;
            OperationTests.Check(new TranslateX(1), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            OperationTests.Check(new TranslateX(1), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Reflect);
            OperationTests.Check(new TranslateX(5), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            OperationTests.Check(new TranslateX(5), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Reflect);

            // 4x4 matrix
            input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            inputShape = new[] { 1, 1, 4, 4 };
            //shift with FillModeEnum.Nearest
            //1 to the right
            expected = new[] { 0f, 0, 1, 2, 4, 4, 5, 6, 8, 8, 9, 10, 12, 12, 13, 14 };
            OperationTests.Check(new TranslateX(1), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            //2 to the left
            expected = new[] { 2f, 3, 3, 3, 6, 7, 7, 7, 10, 11, 11, 11, 14, 15, 15, 15 };
            OperationTests.Check(new TranslateX(-2), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            //sift with FillModeEnum.Reflect
            //1 to the right
            expected = new[] { 0f, 0, 1, 2, 4, 4, 5, 6, 8, 8, 9, 10, 12, 12, 13, 14 };
            OperationTests.Check(new TranslateX(1), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Reflect);
            //2 to the left
            expected = new[] { 2f, 3, 3, 2, 6, 7, 7, 6, 10, 11, 11, 10, 14, 15, 15, 14 };
            var operation = new TranslateX(-2);
            OperationTests.Check(operation, input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Reflect);

            Assert.IsTrue(operation.ChangeCoordinates());
        }

        [Test]
        public void TestTranslateY()
        {
            //single element
            var input = new[] { 12f };
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = input;
            OperationTests.Check(new TranslateY(1), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            OperationTests.Check(new TranslateY(1), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Reflect);
            OperationTests.Check(new TranslateY(5), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            OperationTests.Check(new TranslateY(5), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Reflect);

            // 4x4 matrix
            input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            inputShape = new[] { 1, 1, 4, 4 };
            //shift with FillModeEnum.Nearest
            //3 to the top
            expected = new[] { 12f, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, };
            OperationTests.Check(new TranslateY(-3), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            //shift with FillModeEnum.Reflect
            //3 to the top
            expected = new[] { 12f, 13, 14, 15, 12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7 };
            var operation = new TranslateY(-3);
            OperationTests.Check(operation, input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Reflect);

            Assert.IsTrue(operation.ChangeCoordinates());

        }

        [Test]
        public void Test_TranslateX_and_TranslateY()
        {
            //single element
            var input = new[] { 12f };
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = input;
            OperationTests.CheckAllPermutations(new TranslateY(1), new TranslateX(3), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            OperationTests.CheckAllPermutations(new TranslateX(-2), new TranslateY(-4), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Reflect);
            OperationTests.CheckAllPermutations(new TranslateY(1), new TranslateX(3), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            OperationTests.CheckAllPermutations(new TranslateX(-2), new TranslateY(-4), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Reflect);

            // 4x4 matrix
            input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            inputShape = new[] { 1, 1, 4, 4 };
            //shift with FillModeEnum.Nearest
            //1 to the right & 1 to the bottom
            expected = new[] { 0f, 0, 1, 2, 0, 0, 1, 2, 4, 4, 5, 6, 8, 8, 9, 10 };
            OperationTests.CheckAllPermutations(new TranslateX(1), new TranslateY(1), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            //shift with FillModeEnum.Reflect
            //2 to the right & 2 to the bottom
            expected = new[] { 5f, 4, 4, 5, 1, 0, 0, 1, 1, 0, 0, 1, 5, 4, 4, 5 };
            OperationTests.CheckAllPermutations(new TranslateX(2), new TranslateY(2), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Reflect);
        }
    }
}
