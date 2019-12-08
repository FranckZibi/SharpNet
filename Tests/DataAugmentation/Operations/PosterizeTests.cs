using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class PosterizeTests
    {
        [Test]
        public void TestPosterize()
        {
            // 2x2 matrix, 1 channel, no normalization, keeping all bits/pixels
            var input = new[] { 255f, 254f, 253f, 252f};
            var inputShape = new[] { 1, 1, 2, 2};
            OperationTests.Check(new Posterize(8, null), input, inputShape, input, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);

            // 2x2 matrix, 1 channel, no normalization, keeping 7 bits/pixels
            input = new[] { 255f, 254f, 253f, 252f };
            var expected = new[] { 254f, 254f, 252f, 252f };
            OperationTests.Check(new Posterize(7, null), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);

            // 2x2 matrix, 1 channel, no normalization, keeping 4 bits/pixels
            input = new[] { 255f, 200f, 150f, 50f };
            expected = new[] { 240f, 192f, 144f, 48f };
            OperationTests.Check(new Posterize(4, null), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);

            // 2x2 matrix, 1 channel, no normalization, keeping 1 bits/pixels
            input = new[] { 255f, 254f, 253f, 252f };
            expected = new[] { 128f, 128f, 128f, 128f };
            OperationTests.Check(new Posterize(1, null), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);

            // 1x1 matrix, 3 channels, with normalization, keeping 4 bits/pixels
            inputShape = new[] { 1, 3, 1, 1 };
            input = new[] { (255f - 10f) / 5f, (200f - 20f) / 10f, (150f - 40f) / 20f};
            var meanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(10f, 5f), Tuple.Create(20f, 10f), Tuple.Create(40f, 20f)};
            expected = new[] { (240f - 10f) / 5f, (192f - 20f) / 10f, (144f - 40f) / 20f};
            OperationTests.Check(new Posterize(4, meanAndVolatilityForEachChannel), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);
        }
    }
}