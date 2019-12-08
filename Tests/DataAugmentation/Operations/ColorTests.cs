using NUnit.Framework;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class ColorTests
    {
        [Test]
        public void TestColor()
        {
            // 1x1 matrix, 3 channels, no normalization
            var input = new[] { 250f, 150f, 50f };
            var greyScale = Operation.GetGreyScale(input[0], input[1], input[2]);

            var inputShape = new[] { 1, 3, 1, 1 };
            var expected = new[] { (250f+ greyScale)/2, (150f+ greyScale)/2, (50f+greyScale)/2 };
            OperationTests.Check(new Color(0.5f), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);

            // 1x1 matrix, 3 channels, with normalization
            input = new[] { (250f - 10f) / 5f, (150f - 20f) / 10f, (50f - 40f) / 20f };
            greyScale = Operation.GetGreyScale(input[0], input[1], input[2]);
            //var meanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(10f, 5f), Tuple.Create(20f, 10f), Tuple.Create(40f, 20f) };
            expected = new[] { (input[0] + greyScale)/2, (input[1]+ greyScale)/2, (input[2] + greyScale)/2 };
            OperationTests.Check(new Color(0.5f), input, inputShape, expected, null, ImageDataGenerator.FillModeEnum.Nearest);
        }
    }
}
