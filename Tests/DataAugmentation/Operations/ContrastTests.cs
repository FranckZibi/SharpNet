using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class ContrastTests
    {
        [Test]
        public void TestContrast()
        {
            var input = new[] { 250f, 150f, 50f };
            var greyMean = 191f;

            var inputShape = new[] { 1, 3, 1, 1 };
            var expected = new[] { (input[0] + greyMean) / 2, (input[1] + greyMean) / 2, (input[2] + greyMean) / 2 };
            OperationTests.Check(new Contrast(0.5f, greyMean), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 1x1 matrix, 3 channels, with normalization
            input = new[] { (250f - 10f) / 5f, (150f - 20f) / 10f, (50f - 40f) / 20f };
            //var meanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(10f, 5f), Tuple.Create(20f, 10f), Tuple.Create(40f, 20f) };
            expected = new[] { (input[0] + greyMean) / 2, (input[1] + greyMean) / 2, (input[2] + greyMean) / 2 };
            OperationTests.Check(new Contrast(0.5f, greyMean), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
        }
    }
}