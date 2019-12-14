using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;
using SharpNet.Pictures;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class SolarizeTests
    {


        [Test]
        public void TestSolarize()
        {
            // 4x4 matrix, 1 channel, no normalization
            var input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            var inputShape = new[] { 1, 1, 4, 4 };
            var expected = new[] { 0f, 254f, 253f, 252f, 251f, 250f, 249f, 248f, 247f, 246f, 245f, 244f, 243f, 242f, 241f, 240f };
            OperationTests.Check(new Solarize(0, null), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);

            // 1x1 matrix, 3 channels, no normalization
            input = new[] { 250f, 150f, 50f };
            inputShape = new[] { 1, 3, 1, 1 };
            expected = new[] { 5f, 105f, 50f };
            OperationTests.Check(new Solarize(149, null), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);

            // 1x1 matrix, 3 channels, with normalization
            input = new[] { (253f - 10f) / 5f, (254f - 20f) / 10f, (255f - 40f) / 20f };
            var meanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(10f, 5f), Tuple.Create(20f, 10f), Tuple.Create(40f, 20f) };
            expected = new[] { (253f - 10f) / 5f, (1f - 20f) / 10f, (0f - 40f) / 20f };
            OperationTests.Check(new Solarize(253, meanAndVolatilityForEachChannel), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);
        }

        [Test, Explicit]
        public void TestOnRealPicture()
        {
            const string path = @"C:\download\b\srcimg12.jpg";
            var bmp = BitmapContent.ValueFomSingleRgbBitmap(path, path);
            var stats = ImageStatistic.ValueOf(bmp);
            OperationTests.ApplyToPicture(new List<Operation> { new Solarize(128, null) }, path, @"C:\download\b\srcimg12_Solarize.jpg");
        }

    }
}