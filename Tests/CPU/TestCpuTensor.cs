using System;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNetTests.Data;

namespace SharpNetTests.CPU
{
    [TestFixture]
    public class TestCpuTensor
    {
        [Test]
        public void TestEquals()
        {
            var a = new CpuTensor<float>(new []{10, 5}, "a");
            var b = new CpuTensor<float>(new[] { 10, 5 }, "b");
            var c = new CpuTensor<float>(new[] { 11, 5 }, "c");
            Assert.IsTrue(TestTensor.SameContent(a, b, 0.0));
            Assert.IsFalse(TestTensor.SameContent(a, c, 10));
            b[0] += 1e-4f;
            Assert.IsTrue(TestTensor.SameContent(a, b, 1e-3));
            Assert.IsFalse(TestTensor.SameContent(a, c, 1e-5));
        }
        /*
         *  1 2 3      
         *  4 5 6
         *  7 8 9
         * 
         */
        [Test]
        public void TestConvolution()
        {
            const int f = 3;
            var input = new CpuTensor<float>(new[] { 1, 1, 3, 3 }, new float[]{1,2,3,4,5,6,7,8,9}, "input");
            var convolution = new CpuTensor<float>(new[] { 1, 1, f, f }, new float[] { 1, 0, -1, 1, 0, -1, 1, 0, -1 }, "convolution");

            var padding = 0;
            var stride = 1;
            var expectedOutput = new CpuTensor<float>(new[] { 1, 1, 1, 1 }, new float[] { -6 }, "expectedOutput");
            TestStandardConvolution(input, convolution, padding, padding, padding, padding, stride, expectedOutput);

            padding = f/2;
            stride = 1;
            expectedOutput = new CpuTensor<float>(new[] { 1, 1, 3, 3}, new float[] { -7,-4,7,-15,-6,15,-13,-4,13 }, "expectedOutput");
            TestStandardConvolution(input, convolution, padding, padding, padding, padding, stride, expectedOutput);

            input = new CpuTensor<float>(new[] { 3, 1, 3, 3 }, "input");
            for (int i = 1; i <= input.Count; ++i)
            {
                input[i - 1] = i;
            }
            input[0] = -333;
            padding = 0;
            stride = 1;
            expectedOutput = new CpuTensor<float>(new[] { 3, 1, 1, 1 }, new float[] {-340,-6, -6 }, "expectedOutput");
            TestStandardConvolution(input, convolution, padding, padding, padding, padding, stride, expectedOutput);

            padding = f / 2;
            stride = 1;
            expectedOutput = new CpuTensor<float>(new[] { 3, 1, 3, 3 }, new float[] { -7, -338, 7, -15, -340, 15, -13, -4, 13, -25, -4, 25, -42, -6, 42, -31, -4, 31, -43, -4, 43, -69, -6, 69, -49, -4, 49 }, "expectedOutput");
            TestStandardConvolution(input, convolution, padding, padding, padding, padding, stride, expectedOutput);
        }

        [Test]
        public void TestMultiplyTensor()
        {
            var rand = new Random(0);
            var shape = new[] {32, 1157, 7, 7};
            var maxValue = 10.0;
            var c = RandomFloatTensor(shape, rand, -maxValue, maxValue, "");
            var a = RandomFloatTensor(shape, rand, -maxValue, maxValue, "");
            var x = RandomFloatTensor(shape, rand, -maxValue, maxValue, "");
            var expected = new CpuTensor<float>(shape, null, "");
            for (int i = 0; i < expected.Count; ++i)
            {
                expected.Content[i] = a.Content[i]*x.Content[i];
            }
            c.MultiplyTensor(a, x);
            Assert.IsTrue(TestTensor.SameContent(expected, c, 1e-6));
            c = RandomFloatTensor(shape, rand, -maxValue, maxValue, "");
            a = RandomFloatTensor(shape, rand, -maxValue, maxValue, "");
            x = RandomFloatTensor(new[] { 32, 1157, 1, 1 }, rand, -maxValue, maxValue, "");
            expected = new CpuTensor<float>(shape, null, "");
            for (int i = 0; i < expected.Count; ++i)
            {
                expected.Content[i] = a.Content[i] * x.Content[i/(7*7)];
            }
            c.MultiplyTensor(a, x);
            Assert.IsTrue(TestTensor.SameContent(expected, c, 1e-6));
        }

        [Test]
        public void TestChangeAxis()
        {
            var t = new CpuTensor<int>(new [] {2, 2, 2, 2}, Enumerable.Range(0, 16).ToArray(), "");
            var t1 = (CpuTensor<int>)t.ChangeAxis(new[] {3, 2, 0, 1}).ChangeAxis(new[] {0, 1, 3, 2});
            var t2 = (CpuTensor<int>)t.ChangeAxis(new[] {3, 2, 1, 0});
            Assert.IsTrue(t1.Content.SequenceEqual(t2.Content));
            Assert.IsTrue(t1.Content.SequenceEqual(new []{ 0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15 }));
        }

        [Test]
        public void TestZeroPadding()
        {
            var rand = new Random(0);
            foreach (var shape in new[] {new [] {7, 3, 7, 8}, new[] { 4, 5, 12, 5 } })
            {
                var src = RandomFloatTensor(shape, rand, -100.0, 100.0, "");
                foreach (var top_pad in new[] {0, 1, 3})
                foreach (var bottom_pad in new[] {0, 1, 3})
                foreach (var left_pad in new[] {0, 1, 3})
                foreach (var right_pad in new[] {0, 1, 3})
                {
                    var destShape = new[] { shape[0], shape[1], top_pad + shape[2] + bottom_pad, left_pad + shape[3] + right_pad};
                    var observedDest = RandomFloatTensor(destShape, rand, -100.0, 100.0, "");
                    observedDest.ZeroPadding(src, top_pad, bottom_pad, left_pad, right_pad);

                    var expectedDest = new CpuTensor<float>(destShape, null, "");
                    expectedDest.ZeroMemory();
                    for (int n = 0; n < shape[0]; ++n)
                    for (int c = 0; c < shape[1]; ++c)
                    for (int h = 0; h < shape[2]; ++h)
                    for (int w = 0; w < shape[3]; ++w)
                    {
                        expectedDest.Set(n, c, h + top_pad, w + left_pad, src.Get(n, c, h, w));
                    }
                    Assert.IsTrue(TestTensor.SameContent(expectedDest, observedDest, 1e-6));
                }
            }
        }


        [Test]
        public void TestMultiplyEachRowIntoSingleValue()
        {
            var rand = new Random(0);
            var shape = new[] { 32, 1157, 7, 7 };
            var maxValue = 10.0;
            var a = RandomFloatTensor(shape, rand, -maxValue, maxValue, "");
            var b = RandomFloatTensor(shape, rand, -maxValue, maxValue, "");
            var result = new CpuTensor<float>(new[] { 32, 1157, 1, 1 }, null, "");
            result.MultiplyEachRowIntoSingleValue(a,b);
            var expected = new CpuTensor<float>(result.Shape, null, "");
            for (int i = 0; i < a.Count; ++i)
            {
                expected.Content[i/(7*7)] += a.Content[i] * b.Content[i];
            }
            Assert.IsTrue(TestTensor.SameContent(expected, result, 1e-6));
        }

        [TestCase(100000, 0.5, false, 0, 0)] //when not training, dropout is disabled
        [TestCase(100000, 0.0, true, 0, 0)] // no 0 if drop proba = 0%
        [TestCase(100000, 1.0, true, 100000-10, 100000)]  // only 0 if drop proba = 100%
        [TestCase(100000, 0.25, true, (int)(100000*0.2), (int)(100000 *0.3))]
        [TestCase(100000, 0.75, true, (int)(100000 *0.7), (int)(100000 *0.8))]
        public void TestDropoutForward(int nbRows, double dropProbability, bool isTraining, int minEqualToZeroAfterDropout, int maxEqualToZeroAfterDropout)
        {
            var rand = new Random(0);
            var x = RandomFloatTensor(new []{nbRows, 1}, rand, 10, 20, "x");
            var y = RandomFloatTensor(x.Shape, rand, 10, 20, "y");
            var dropoutMaskBuffer = RandomFloatTensor(x.Shape, rand, 10, 20, "dropoutMaskBuffer");
            x.DropoutForward(y, dropProbability, isTraining, rand, dropoutMaskBuffer);
            int nbObservedZeroAfterDropout = y.Content.Count(i => Math.Abs(i) < 1e-8);
            Assert.IsTrue(nbObservedZeroAfterDropout>=minEqualToZeroAfterDropout);
            Assert.IsTrue(nbObservedZeroAfterDropout<= maxEqualToZeroAfterDropout);
        }   

        [Test]
        public void TestMaxPooling()
        {
            const int poolingSize = 2;
            const int stride = 2;

            var input = new CpuTensor<float>(new[] { 1, 1, 3, 3 }, new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, "input");
            var output = new CpuTensor<float>(PoolingLayer.PoolingOutputShape(input.Shape, poolingSize, poolingSize, stride), "output");
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingSize, poolingSize, stride);
            var expectedOutput = new CpuTensor<float>(new[] { 1, 1, 1, 1 }, new float[] {5}, "expectedOutput");
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, output, 1e-6));

            input = new CpuTensor<float>(new[] { 3, 1, 4, 4 }, "input");
            for (int i = 1; i <= input.Count; ++i)
            {
                input[i - 1] = i;
            }
            input[0] = 333;
            output = new CpuTensor<float>(PoolingLayer.PoolingOutputShape(input.Shape, poolingSize, poolingSize, stride), "output");
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingSize, poolingSize, stride);
            expectedOutput = new CpuTensor<float>(new[] { 3, 1, 2, 2 }, new float[] { 333, 8, 14, 16, 22, 24, 30, 32, 38, 40, 46, 48 }, "expectedOutput");
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, output, 1e-6));
        }
        public static CpuTensor<float> RandomFloatTensor(int[] shape, Random rand, double minValue, double maxValue, string description)
        {
            var result = new CpuTensor<float>(shape, description);
            Utils.Randomize(result.Content, rand, minValue, maxValue);
            return result;
        }
        public static CpuTensor<byte> RandomByteTensor(int[] shape, Random rand, byte minValue, byte maxValue, string description)
        {
            var result = new CpuTensor<byte>(shape, description);
            Utils.Randomize(result.Content, rand, minValue, maxValue);
            return result;
        }
        private void TestStandardConvolution(CpuTensor<float> input, CpuTensor<float> convolution, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int stride, CpuTensor<float> expectedOutput)
        {
            var outputCPU = new CpuTensor<float>(expectedOutput.Shape, "output");
            input.Convolution(convolution, paddingTop, paddingBottom, paddingLeft, paddingRight, stride, outputCPU, false, GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM);
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, outputCPU, 1e-6));
        }
        public static CpuTensor<float> RandomOneHotTensor(int[] shape, Random rand, string description)
        {
            var result = new CpuTensor<float>(shape, description);
            for (int row = 0; row < result.Shape[0]; ++row)
            {
                result.Set(row, rand.Next(result.Shape[1]), 1f);
            }
            return result;
        }

        public static CpuTensor<int> RandomCategoryIndexTensor(int nbRows, int categoryCount, Random rand)
        {
            var data = new int[nbRows];
            for (int i = 0; i < nbRows; ++i)
            {
                data[i] = rand.Next(categoryCount);
            }
            return new CpuTensor<int>(new[]{nbRows}, data, "RandomOneHotTensorCategoryIndex");
        }


        //random tensor
        //in each row: only 2 elements with non zero value, the sum of the 2 elements is always = 1.0
        public static CpuTensor<float> RandomTwoHotTensor(int[] shape, Random rand, string description)
        {
            var result = new CpuTensor<float>(shape, description);
            int categoryCount = result.Shape[1];
            for (int row = 0; row < result.Shape[0]; ++row)
            {
                int indexFirstCategory = rand.Next(categoryCount);
                var expectedFirstCategory = (float)rand.NextDouble();
                result.Set(row, indexFirstCategory, expectedFirstCategory);
                int indexSecondCategory = (indexFirstCategory+7)%categoryCount;
                var expectedSecondCategory = 1f-expectedFirstCategory;
                result.Set(row, indexSecondCategory, expectedSecondCategory);
            }
            return result;
        }
    }
}
