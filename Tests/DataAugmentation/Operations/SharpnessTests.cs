using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class SharpnessTests
    {
        [Test]
        public void TestSharpness()
        {
            // 2x2 matrix, 3 channels
            var input = new float[] { 20, 30, 40,50, 120,130,140,150, 220,230,240,250 };
            var inputShape = new[] { 1, 3, 2, 2 };

            //enhancementFactor=1 => we should retrieve the original picture
            var expected = input;
            OperationTests.Check(new Sharpness(1.0f), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            expected = new [] { 27.5f, 32.5f, 37.5f, 42.5f, 127.5f, 132.5f, 137.5f, 142.5f, 227.5f, 232.5f, 237.5f, 242.5f };
            //TODO re enable test
            //OperationTests.Check(new Sharpness(0.0f), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
        }

        [Test, Explicit]
        public void TestOnRealPicture()
        {
            const string path = @"C:\Download\ImageEnhance_Sharpness_100.jpg";
            foreach (var normalized in new[] { false, true })
            { 
                foreach (var enhancementFactor in new[] { 0f, 0.5f, 1f, 1.5f, 2f })
                {
                    OperationTests.ApplyToPicture(new List<Operation> { new Sharpness(enhancementFactor) }, path, @"C:\Download\ImageEnhance_Sharpness_" + ((int)(100 * enhancementFactor)).ToString("D3") + "_observed_" + normalized + ".jpg", normalized);
                }
            }
        }
    }
}
