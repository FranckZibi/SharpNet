using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;
using SharpNet.Pictures;
using SharpNetTests.Data;

namespace SharpNetTests.DataAugmentation.Operations
{
    [TestFixture]
    public class OperationTests
    {
        public static void CheckAllPermutations(Operation op1, Operation op2, float[] input, int[] inputShape,
            float[] expectedOutput, Func<int, int> indexInMiniBatchToCategoryId,
            ImageDataGenerator.FillModeEnum _fillMode, CpuTensor<float> yOriginal = null,
            CpuTensor<float> yExpected = null)
        {
            CheckAllPermutations(new List<Operation> { op1, op2 }, input, inputShape, expectedOutput, indexInMiniBatchToCategoryId, _fillMode, yOriginal, yExpected);
        }


        public static void Check(Operation op, float[] input, int[] inputShape, float[] expectedOutput,
            Func<int, int> indexInMiniBatchToCategoryId, ImageDataGenerator.FillModeEnum _fillMode,
            CpuTensor<float> yOriginal = null, CpuTensor<float> yExpected = null)
        {
            Check(new List<Operation> {op}, input, inputShape, expectedOutput, indexInMiniBatchToCategoryId, null, _fillMode, yOriginal, yExpected);
        }

        private static void CheckAllPermutations(List<Operation> subPolicy, float[] input, int[] inputShape, float[] expectedOutput, Func<int, int> indexInMiniBatchToCategoryId, ImageDataGenerator.FillModeEnum _fillMode, CpuTensor<float> yOriginal, CpuTensor<float> yExpected)
        {
            foreach (var p in Utils.AllPermutations(subPolicy))
            {
                Check(p.ToList(), input, inputShape, expectedOutput, indexInMiniBatchToCategoryId, null, _fillMode, yOriginal, yExpected);
            }
        }

        public static void ApplyToPicture(List<Operation> subPolicy, string picturePath, string augmentedPicturePath)
        {
            var bmp = BitmapContent.ValueFomSingleRgbBitmap(picturePath, picturePath);

            var input = bmp.Content.Select(x => (float)x).ToArray();
            var inputShape = new []{1, bmp.Shape[0], bmp.Shape[1], bmp.Shape[2]};
            //Check(subPolicy, input, inputShape, bmp.Shape, null, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);
            var xOriginalMiniBatch = new CpuTensor<float>(inputShape, input, "xOriginalMiniBatch");
            var xDataAugmentedMiniBatch = new CpuTensor<float>(inputShape, "xDataAugmentedMiniBatch");
            SubPolicy.Apply(
                subPolicy,
                0,
                xOriginalMiniBatch,
                xDataAugmentedMiniBatch,
                null,
                null,
                null,
                ImageDataGenerator.FillModeEnum.Nearest);
            var t = new BitmapContent(bmp.Shape, xDataAugmentedMiniBatch.Content.Select(x => (byte)x).ToArray(), "description");
            t.Save(new List<string> { augmentedPicturePath });
        }

        private static void Check(List<Operation> subPolicy, float[] input, int[] inputShape, float[] expectedOutput, Func<int, int> indexInMiniBatchToCategoryId, List<Tuple<float, float>> meanAndVolatilityForEachChannel, ImageDataGenerator.FillModeEnum _fillMode, CpuTensor<float> yOriginal, CpuTensor<float> yExpected)
        {
            indexInMiniBatchToCategoryId = indexInMiniBatchToCategoryId ?? (x => x);

            var xOriginalMiniBatch = new CpuTensor<float>(inputShape, input, "xOriginalMiniBatch");
            var xDataAugmentedMiniBatch = new CpuTensor<float>(inputShape, "xDataAugmentedMiniBatch");
            yOriginal = (CpuTensor<float>) yOriginal?.Clone(null);
            SubPolicy.Apply(
                subPolicy,
                0,
                xOriginalMiniBatch,
                xDataAugmentedMiniBatch,
                yOriginal,
                indexInMiniBatchToCategoryId,
                meanAndVolatilityForEachChannel,
                _fillMode);

            var xExpectedDataAugmented = new CpuTensor<float>(inputShape, expectedOutput, "xExpectedDataAugmented");
            Assert.IsTrue(TestTensor.SameContent(xExpectedDataAugmented, xDataAugmentedMiniBatch, 1e-6));
            if (yOriginal != null)
            {
                Assert.IsTrue(TestTensor.SameContent(yExpected, yOriginal, 1e-6));
            }
        }
    }
}
