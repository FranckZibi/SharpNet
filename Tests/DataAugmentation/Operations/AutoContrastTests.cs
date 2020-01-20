using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class AutoContrastTests
    {
        [Test]
        public void TestAutoContrast()
        {
            // 1x1 matrix, 3 channels, no normalization
            var input = new[] { 250f, 150f, 50f };
            var inputShape = new[] { 1, 3, 1, 1 };
            var expected = new[] { 250f, 255f, 255f*(50f/200f) };
            var pixelThresholdByChannel = new List<Tuple<int, int>> {Tuple.Create(0,255), Tuple.Create(0, 125), Tuple.Create(0, 200) };
            var operation = new AutoContrast(pixelThresholdByChannel, null);
            OperationTests.Check(operation, input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 1x1 matrix, 3 channels, with normalization
            input = new[] { (250f - 10f) / 5f, (150f - 20f) / 10f, (50f - 40f) / 20f };
            var meanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(10f, 5f), Tuple.Create(20f, 10f), Tuple.Create(40f, 20f) };
            expected = new[] { (250f - 10f) / 5f, (255f - 20f) / 10f, (255f * (50f / 200f) - 40f) / 20f };
            operation = new AutoContrast(pixelThresholdByChannel, meanAndVolatilityForEachChannel);
            OperationTests.Check(operation, input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            Assert.IsFalse(operation.ChangeCoordinates());
        }
    }
}
