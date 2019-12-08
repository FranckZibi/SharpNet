using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class BrightnessTests
    {
        [Test]
        public void TestBrightness()
        {
            // 1x1 matrix, 3 channels, no normalization
            var input = new[] { 250f, 150f, 50f };
            var blackMean = 0f;

            var inputShape = new[] { 1, 3, 1, 1 };
            var expected = new[] { (input[0] + blackMean) / 2, (input[1] + blackMean) / 2, (input[2] + blackMean) / 2 };
            OperationTests.Check(new Brightness(0.5f, blackMean), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 1x1 matrix, 3 channels, with normalization
            input = new[] { (250f - 10f) / 5f, (150f - 20f) / 10f, (50f - 40f) / 20f };
            //var meanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(10f, 5f), Tuple.Create(20f, 10f), Tuple.Create(40f, 20f) };
            expected = new[] { (input[0] + blackMean) / 2, (input[1] + blackMean) / 2, (input[2] + blackMean) / 2 };
            OperationTests.Check(new Brightness(0.5f, blackMean), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
        }
    }
}