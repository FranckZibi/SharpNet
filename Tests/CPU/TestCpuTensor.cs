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
            TestConvolution(input, convolution, padding, stride, expectedOutput);

            padding = f/2;
            stride = 1;
            expectedOutput = new CpuTensor<float>(new[] { 1, 1, 3, 3}, new float[] { -7,-4,7,-15,-6,15,-13,-4,13 }, "expectedOutput");
            TestConvolution(input, convolution, padding, stride, expectedOutput);

            input = new CpuTensor<float>(new[] { 3, 1, 3, 3 }, "input");
            for (int i = 1; i <= input.Count; ++i)
            {
                input[i - 1] = i;
            }
            input[0] = -333;
            padding = 0;
            stride = 1;
            expectedOutput = new CpuTensor<float>(new[] { 3, 1, 1, 1 }, new float[] {-340,-6, -6 }, "expectedOutput");
            TestConvolution(input, convolution, padding, stride, expectedOutput);

            padding = f / 2;
            stride = 1;
            expectedOutput = new CpuTensor<float>(new[] { 3, 1, 3, 3 }, new float[] { -7, -338, 7, -15, -340, 15, -13, -4, 13, -25, -4, 25, -42, -6, 42, -31, -4, 31, -43, -4, 43, -69, -6, 69, -49, -4, 49 }, "expectedOutput");
            TestConvolution(input, convolution, padding, stride, expectedOutput);
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
            var output = new CpuTensor<float>(PoolingLayer.PoolingOutputShape(input.Shape, poolingSize, stride), "output");
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingSize, stride);
            var expectedOutput = new CpuTensor<float>(new[] { 1, 1, 1, 1 }, new float[] {5}, "expectedOutput");
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, output, 1e-6));

            input = new CpuTensor<float>(new[] { 3, 1, 4, 4 }, "input");
            for (int i = 1; i <= input.Count; ++i)
            {
                input[i - 1] = i;
            }
            input[0] = 333;
            output = new CpuTensor<float>(PoolingLayer.PoolingOutputShape(input.Shape, poolingSize, stride), "output");
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingSize, stride);
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
        private void TestConvolution(CpuTensor<float> input, CpuTensor<float> convolution, int padding, int stride, CpuTensor<float> expectedOutput)
        {
            var outputCPU = new CpuTensor<float>(ConvolutionLayer.ConvolutionOutputShape(input.Shape, convolution.Shape, padding, stride), "output");
            input.Convolution(convolution, padding, stride, outputCPU);
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

        //random tensor
        //in each row: only 2 elements with non zero value, the sum of the 2 elements is always = 1.0
        public static CpuTensor<float> RandomTwoHotTensor(int[] shape, Random rand, string description)
        {
            var result = new CpuTensor<float>(shape, description);
            int nbCategories = result.Shape[1];
            for (int row = 0; row < result.Shape[0]; ++row)
            {
                int indexFirstCategory = rand.Next(nbCategories);
                var expectedFirstCategory = (float)rand.NextDouble();
                result.Set(row, indexFirstCategory, expectedFirstCategory);
                int indexSecondCategory = (indexFirstCategory+7)%nbCategories;
                var expectedSecondCategory = 1f-expectedFirstCategory;
                result.Set(row, indexSecondCategory, expectedSecondCategory);
            }
            return result;
        }




    }
}
