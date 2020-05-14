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
            float[] expectedOutput, Func<int, int> indexInMiniBatchToCategoryIndex,
            ImageDataGenerator.FillModeEnum _fillMode, CpuTensor<float> yOriginal = null,
            CpuTensor<float> yExpected = null)
        {
            CheckAllPermutations(new List<Operation> { op1, op2 }, input, inputShape, expectedOutput, indexInMiniBatchToCategoryIndex, _fillMode, yOriginal, yExpected);
        }


        public static void Check(Operation op, float[] input, int[] inputShape, float[] expectedOutput,
            Func<int, int> indexInMiniBatchToCategoryIndex, ImageDataGenerator.FillModeEnum _fillMode,
            CpuTensor<float> yOriginal = null, CpuTensor<float> yExpected = null)
        {
            Check(new List<Operation> {op}, input, inputShape, expectedOutput, indexInMiniBatchToCategoryIndex, _fillMode, yOriginal, yExpected);
        }

        private static void CheckAllPermutations(List<Operation> subPolicy, float[] input, int[] inputShape, float[] expectedOutput, Func<int, int> indexInMiniBatchToCategoryIndex, ImageDataGenerator.FillModeEnum _fillMode, CpuTensor<float> yOriginal, CpuTensor<float> yExpected)
        {
            foreach (var p in Utils.AllPermutations(subPolicy))
            {
                Check(p.ToList(), input, inputShape, expectedOutput, indexInMiniBatchToCategoryIndex, _fillMode, yOriginal, yExpected);
            }
        }

        public static void ApplyToPicture(List<Operation> subPolicy, string picturePath, string augmentedPicturePath, bool normalizeInput = false)
        {
            var bmp = BitmapContent.ValueFomSingleRgbBitmap(picturePath);

            var input = bmp.ReadonlyContent.Select(x => (float)x).ToArray();
            var inputShape = new []{1, bmp.Shape[0], bmp.Shape[1], bmp.Shape[2]};
            //Check(subPolicy, input, inputShape, bmp.Shape, null, null, ImageDataGenerator.FillModeEnum.Nearest, null, null);
            var xOriginalMiniBatch = new CpuTensor<float>(inputShape, input);

            List<Tuple<float, float>> meanAndVolatilityForEachChannel = null;
            if (normalizeInput)
            { 
                meanAndVolatilityForEachChannel = xOriginalMiniBatch.ComputeMeanAndVolatilityOfEachChannel(x => x);
                xOriginalMiniBatch = xOriginalMiniBatch.Select((n, c, val) => (float)((val - meanAndVolatilityForEachChannel[c].Item1) / Math.Max(meanAndVolatilityForEachChannel[c].Item2, 1e-9)));
            }

            var xDataAugmentedMiniBatch = new CpuTensor<float>(inputShape);
            var xBufferForDataAugmentedMiniBatch = new CpuTensor<float>(inputShape);
            SubPolicy.Apply(
                subPolicy,
                0,
                xOriginalMiniBatch,
                xDataAugmentedMiniBatch,
                null,
                null,
                ImageDataGenerator.FillModeEnum.Nearest,
                xBufferForDataAugmentedMiniBatch);

            if (normalizeInput)
            {
                xDataAugmentedMiniBatch = xDataAugmentedMiniBatch.Select((n, c, val) => val*meanAndVolatilityForEachChannel[c].Item2+meanAndVolatilityForEachChannel[c].Item1);
            }

            var t = new BitmapContent(bmp.Shape, xDataAugmentedMiniBatch.ReadonlyContent.Select(x => (byte)Math.Max(0,Math.Min(255,x))).ToArray());
            t.Save(new List<string> { augmentedPicturePath });
        }

        private static void Check(List<Operation> subPolicy, float[] input, int[] inputShape, float[] expectedOutput,
            Func<int, int> indexInMiniBatchToCategoryIndex, ImageDataGenerator.FillModeEnum _fillMode,
            CpuTensor<float> yOriginal, CpuTensor<float> yExpected)
        {
            indexInMiniBatchToCategoryIndex ??= (x => x);

            var xOriginalMiniBatch = new CpuTensor<float>(inputShape, input);
            var xDataAugmentedMiniBatch = new CpuTensor<float>(inputShape);
            var xBufferForDataAugmentedMiniBatch = new CpuTensor<float>(inputShape);

            yOriginal = (CpuTensor<float>)yOriginal?.Clone();
            SubPolicy.Apply(
                subPolicy,
                0,
                xOriginalMiniBatch,
                xDataAugmentedMiniBatch,
                yOriginal,
                indexInMiniBatchToCategoryIndex,
                _fillMode,
                xBufferForDataAugmentedMiniBatch);

            var xExpectedDataAugmented = new CpuTensor<float>(inputShape, expectedOutput);
            Assert.IsTrue(TestTensor.SameContent(xExpectedDataAugmented, xDataAugmentedMiniBatch, 1e-6));
            if (yOriginal != null)
            {
                Assert.IsTrue(TestTensor.SameContent(yExpected, yOriginal, 1e-6));
            }
        }
    }
}
