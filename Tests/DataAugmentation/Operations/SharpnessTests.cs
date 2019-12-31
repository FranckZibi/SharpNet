using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;
using SharpNet.Pictures;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class SharpnessTests
    {

        [Test, Explicit]
        public void TestOnRealPicture()
        {
            const string path = @"C:\Download\ImageEnhance_Sharpness_100.jpg";
            var bmp = BitmapContent.ValueFomSingleRgbBitmap(path, path);
            var stats = ImageStatistic.ValueOf(bmp);
            OperationTests.ApplyToPicture(new List<Operation> { new Sharpness(null, 0f) }, path, @"C:\Download\ImageEnhance_Sharpness_000_observed.jpg");
            OperationTests.ApplyToPicture(new List<Operation> { new Sharpness(null, 1f) }, path, @"C:\Download\ImageEnhance_Sharpness_100_observed.jpg");
            OperationTests.ApplyToPicture(new List<Operation> { new Sharpness(null, 2f) }, path, @"C:\Download\ImageEnhance_Sharpness_200_observed.jpg");
        }


        //!D TODO
        [Test]
        public void TestSharpness()
        {
        }
    }
}
