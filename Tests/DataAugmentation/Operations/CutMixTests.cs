using System.Linq;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class CutMixTests
    {
        [Test]
        public void TestCutMix()
        {
            // 4x4 matrix
            var input = Enumerable.Range(0, 16).Select(x => (float)x).ToArray();
            var inputShape = new[] { 2, 1, 2, 4 };
            var yOriginal = new CpuTensor<float>(new[] { inputShape[0], 5 }, "yOriginal");
            yOriginal.Set(0, 0, 1.0f); //1st picture categoryIndex = 0
            yOriginal.Set(1, 4, 1.0f); //2nd picture categoryIndex = 4

            //right side of 2nd picture into right side of 1st picture
            var xOriginalMiniBatch = new CpuTensor<float>(inputShape, input, "xOriginalMiniBatch");
            var expected = new[] { 0f, 1, 10, 11, 4, 5, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0 };
            var yExpected = new CpuTensor<float>(new[] { inputShape[0], 5 }, "yExpected");
            //1st picture categoryIndex = 0 (50%) + 4 (50%)
            yExpected.Set(0, 0, 0.5f);
            yExpected.Set(0, 4, 0.5f);
            //2nd picture categoryIndex = 4 (100%)
            yExpected.Set(1, 4, 1f);
            OperationTests.Check(new CutMix(0, 1, 2, 3, 1, xOriginalMiniBatch), input, inputShape, expected, (x => x == 0 ? 0 : 4), ImageDataGenerator.FillModeEnum.Nearest, yOriginal, yExpected);

            //all of 2nd picture into 1st picture
            xOriginalMiniBatch = new CpuTensor<float>(inputShape, input, "xOriginalMiniBatch");
            expected = new[] { 8f, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0 };
            yExpected = new CpuTensor<float>(new[] { inputShape[0], 5 }, "yExpected");
            //1st picture categoryIndex = 4 (100%)
            yExpected.Set(0, 4, 1f);
            //2nd picture categoryIndex = 4 (100%)
            yExpected.Set(1, 4, 1f);
            OperationTests.Check(new CutMix(0, 1, 0, 3, 1, xOriginalMiniBatch), input, inputShape, expected, (x => x == 0 ? 0 : 4), ImageDataGenerator.FillModeEnum.Nearest, yOriginal, yExpected);

            //1 pixel of 2nd picture into 1st picture
            xOriginalMiniBatch = new CpuTensor<float>(inputShape, input, "xOriginalMiniBatch");
            expected = new[] { 0f, 9, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0 };
            yExpected = new CpuTensor<float>(new[] { inputShape[0], 5 }, "yExpected");
            //1st picture categoryIndex = 0 (7/8) + 4 (1/8)
            yExpected.Set(0, 0, 7/8f);
            yExpected.Set(0, 4, 1/8f);
            //2nd picture categoryIndex = 4 (100%)
            yExpected.Set(1, 4, 1f);
            OperationTests.Check(new CutMix(0, 0, 1, 1, 1, xOriginalMiniBatch), input, inputShape, expected, (x => x == 0 ? 0 : 4), ImageDataGenerator.FillModeEnum.Nearest, yOriginal, yExpected);
        }
    }
}