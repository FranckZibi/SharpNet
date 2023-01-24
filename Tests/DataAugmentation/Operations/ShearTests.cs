﻿using System.Linq;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class ShearTests
    {
        [Test]
        public void  TestShearX()
        {
            //single element
            var input = new[] { 12f };
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = input;
            OperationTests.Check(new ShearX(0.0), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            OperationTests.Check(new ShearX(0.5), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            OperationTests.Check(new ShearX(-0.5), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 4x4 matrix
            input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            inputShape = new[] { 1, 1, 4, 4 };
            expected = new[] { 0f, 0, 1, 1, 4, 4, 5, 5, 8, 8, 9, 9, 12, 12, 13, 13 };

            var operation = new ShearX(2.0);
            OperationTests.Check(operation, input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            Assert.IsTrue(operation.ChangeCoordinates());
        }

        [Test]
        public void TestShearY()
        {
            //single element
            var input = new[] { 12f };
            var inputShape = new[] { 1, 1, 1, 1 };
            var expected = input;
            OperationTests.Check(new ShearY(0.0), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            OperationTests.Check(new ShearY(0.5), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
            OperationTests.Check(new ShearY(-0.5), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 4x4 matrix
            input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            inputShape = new[] { 1, 1, 4, 4 };
            expected = new[] { 0f, 1, 2, 3, 0f, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7 };

            var operation = new ShearY(2);
            OperationTests.Check(operation, input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            Assert.IsTrue(operation.ChangeCoordinates());
        }
    }
}