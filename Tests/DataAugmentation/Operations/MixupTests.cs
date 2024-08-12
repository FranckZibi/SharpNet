using System.Linq;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class MixUpTests
    {
        [Test]
        public void TestMixUp()
        {
            //MixUp (mix 2nd picture into 1st picture)
            // 4x4 matrix
            var input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            var inputShape = new[] { 2, 1, 2, 4 };
            var yOriginal = new CpuTensor<float>(new []{ inputShape[0],5});
            yOriginal.Set(0,0, 1.0f); //1st picture categoryIndex = 0
            yOriginal.Set(1,4, 1.0f); //2nd picture categoryIndex = 4

            var xOriginalMiniBatch = new CpuTensor<float>(inputShape, input);
            //MixUp (mix 2nd picture into 1st picture)
            var expected = new[] { 4f, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0 };
            var yExpected = new CpuTensor<float>(new[] { inputShape[0], 5 });
            //1st picture categoryIndex = 0 (50%) + 4 (50%)
            yExpected.Set(0, 0, 0.5f);
            yExpected.Set(0, 4, 0.5f);
            //2nd picture categoryIndex = 4 (100%)
            yExpected.Set(1, 4, 1f);
            OperationTests.Check(new MixUp(0.5f, 1, xOriginalMiniBatch), input, inputShape, expected, (x=>x==0?0:4), ImageDataGenerator.FillModeEnum.Nearest, yOriginal, yExpected);

            expected = new[] { 0.25f * 0 + 0.75f * 8, 0.25f * 1 + 0.75f * 9, 0.25f * 2 + 0.75f * 10, 0.25f * 3 + 0.75f * 11, 0.25f * 4 + 0.75f * 12, 0.25f * 5 + 0.75f * 13, 0.25f * 6 + 0.75f * 14, 0.25f * 7 + 0.75f * 15, 0, 0, 0, 0, 0, 0, 0, 0 };
            yExpected = new CpuTensor<float>(new[] { inputShape[0], 5 });
            //1st picture categoryIndex = 0 (25%) + 4 (75%)
            yExpected.Set(0, 0, 0.25f);
            yExpected.Set(0, 4, 0.75f);
            //2nd picture categoryIndex = 4 (100%)
            yExpected.Set(1, 4, 1f);
            var operation = new MixUp(0.25f, 1, xOriginalMiniBatch);
            OperationTests.Check(operation, input, inputShape, expected, (x => x == 0 ? 0 : 4), ImageDataGenerator.FillModeEnum.Nearest, yOriginal, yExpected);

            Assert.IsFalse(operation.ChangeCoordinates());
        }
    }
}