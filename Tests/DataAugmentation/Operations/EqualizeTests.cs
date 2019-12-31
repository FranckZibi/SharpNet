using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;
using SharpNet.Pictures;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class EqualizeTests
    {
        [Test]
        public void TestGetOriginalPixelToEqualizedPixelByChannel()
        {
            var bmp = new BitmapContent(new [] {1, 8, 8},
                new byte[]
                {
                    52, 55, 61, 59, 79, 61, 76, 61, 62, 59, 55, 104, 94, 85, 59, 71, 63, 65, 66, 113, 144,
                    104, 63, 72, 64, 70, 70, 126, 154, 109, 71, 69, 67, 73, 68, 106, 122, 88, 68, 68, 68, 79,
                    60, 70, 77, 66, 58, 75, 69, 85, 64, 58, 55, 61, 65, 83, 70, 87, 69, 68, 65, 73, 78, 90
                }, "");
            var observed = Equalize.GetOriginalPixelToEqualizedPixelByChannel(ImageStatistic.ValueOf(bmp))[0];
            var expected = new[]{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,12,0,0,20,32,36,53,57,65,73,85,93,97,117,130,146,154,158,166,0,170,174,178,182,190,0,0,0,194,0,202,0,206,210,0,215,0,0,0,219,0,0,0,0,0,0,0,0,0,227,0,231,0,0,235,0,0,0,239,0,0,0,0,0,0,0,0,243,0,0,0,247,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,251,0,0,0,0,0,0,0,0,0,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            Assert.AreEqual(expected,observed);
        }

        [Test,Explicit]
        public void TestOnRealPicture()
        {
            const string path = @"C:\download\b\srcimg07.jpg";
            var bmp = BitmapContent.ValueFomSingleRgbBitmap(path, path);
            var stats = ImageStatistic.ValueOf(bmp);
            OperationTests.ApplyToPicture(new List<Operation> { new Equalize(Equalize.GetOriginalPixelToEqualizedPixelByChannel(stats), null) }, path, @"C:\download\b\srcimg07_Equalize2.jpg", false);
            var meanAndVolatilityForEachChannel = new CpuTensor<byte>(new[] { 1, bmp.Shape[0], bmp.Shape[1], bmp.Shape[2] }, bmp.Content, "").ComputeMeanAndVolatilityOfEachChannel(x => (float)x);
            OperationTests.ApplyToPicture(new List<Operation> { new Equalize(Equalize.GetOriginalPixelToEqualizedPixelByChannel(stats), meanAndVolatilityForEachChannel) }, path, @"C:\download\b\srcimg07_Equalize2_true.jpg", true);
        }

        [Test]
        public void TestEqualize()
        {
            // 1x1 matrix, 3 channels, no normalization
            var input = new[] { 250f, 150f, 50f };
            var originalPixelToEqualizedPixelByChannel = new List<int[]>();
            originalPixelToEqualizedPixelByChannel.Add(new int[256]);
            originalPixelToEqualizedPixelByChannel[0][250] = 25;
            originalPixelToEqualizedPixelByChannel.Add(new int[256]);
            originalPixelToEqualizedPixelByChannel[1][150] = 15;
            originalPixelToEqualizedPixelByChannel.Add(new int[256]);
            originalPixelToEqualizedPixelByChannel[2][50] = 5;

            var inputShape = new[] { 1, 3, 1, 1 };
            var expected = new[] { 25f, 15f, 5f };
            OperationTests.Check(new Equalize(originalPixelToEqualizedPixelByChannel, null), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 1x1 matrix, 3 channels, with normalization
            input = new[] { (250f - 10f) / 5f, (150f - 20f) / 10f, (50f - 40f) / 20f };
            var meanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(10f, 5f), Tuple.Create(20f, 10f), Tuple.Create(40f, 20f) };
            expected = new[] { (25f - 10f) / 5f, (15f - 20f) / 10f, (5f - 40f) / 20f };
            OperationTests.Check(new Equalize(originalPixelToEqualizedPixelByChannel, meanAndVolatilityForEachChannel), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
        }
    }
}
