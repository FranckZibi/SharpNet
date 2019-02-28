using System;
using System.Linq;
using NUnit.Framework;
using NUnit.Framework.Constraints;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNetTests.Data;

namespace SharpNetTests.CPU
{
    [TestFixture]
    public class TestCpuTensor
    {
        private readonly GPUWrapper _gpuWrapper = GPUWrapper.Default;
        [Test]
        public void TestEquals()
        {
            var a = new CpuTensor<double>(new []{10, 5}, "a");
            var b = new CpuTensor<double>(new[] { 10, 5 }, "b");
            var c = new CpuTensor<double>(new[] { 11, 5 }, "c");
            Assert.IsTrue(TestTensor.SameContent(a, b, 0.0));
            Assert.IsFalse(TestTensor.SameContent(a, c, 10));
            b[0] += 1e-4;
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
            var input = new CpuTensor<double>(new[] { 1, 1, 3, 3 }, new double[]{1,2,3,4,5,6,7,8,9}, "input");
            var convolution = new CpuTensor<double>(new[] { 1, 1, f, f }, new double[] { 1, 0, -1, 1, 0, -1, 1, 0, -1 }, "convolution");

            var padding = 0;
            var stride = 1;
            var expectedOutput = new CpuTensor<double>(new[] { 1, 1, 1, 1 }, new double[] { -6 }, "expectedOutput");
            TestConvolution(input, convolution, padding, stride, expectedOutput);

            padding = f/2;
            stride = 1;
            expectedOutput = new CpuTensor<double>(new[] { 1, 1, 3, 3}, new double[] { -7,-4,7,-15,-6,15,-13,-4,13 }, "expectedOutput");
            TestConvolution(input, convolution, padding, stride, expectedOutput);

            input = new CpuTensor<double>(new[] { 3, 1, 3, 3 }, "input");
            for (int i = 1; i <= input.Count; ++i)
            {
                input[i - 1] = i;
            }
            input[0] = -333;
            padding = 0;
            stride = 1;
            expectedOutput = new CpuTensor<double>(new[] { 3, 1, 1, 1 }, new double[] {-340,-6, -6 }, "expectedOutput");
            TestConvolution(input, convolution, padding, stride, expectedOutput);

            padding = f / 2;
            stride = 1;
            expectedOutput = new CpuTensor<double>(new[] { 3, 1, 3, 3 }, new double[] { -7, -338, 7, -15, -340, 15, -13, -4, 13, -25, -4, 25, -42, -6, 42, -31, -4, 31, -43, -4, 43, -69, -6, 69, -49, -4, 49 }, "expectedOutput");
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
            var x = RandomDoubleTensor(new []{nbRows, 1}, rand, 10, 20, "x");
            var y = RandomDoubleTensor(x.Shape, rand, 10, 20, "y");
            var dropoutMaskBuffer = RandomDoubleTensor(x.Shape, rand, 10, 20, "dropoutMaskBuffer");
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

            var input = new CpuTensor<double>(new[] { 1, 1, 3, 3 }, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, "input");
            var output = new CpuTensor<double>(Tensor.PoolingOutputShape(input.Shape, poolingSize, stride), "output");
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingSize, stride);
            var expectedOutput = new CpuTensor<double>(new[] { 1, 1, 1, 1 }, new double[] {5}, "expectedOutput");
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, output, 1e-6));

            input = new CpuTensor<double>(new[] { 3, 1, 4, 4 }, "input");
            for (int i = 1; i <= input.Count; ++i)
            {
                input[i - 1] = i;
            }
            input[0] = 333;
            output = new CpuTensor<double>(Tensor.PoolingOutputShape(input.Shape, poolingSize, stride), "output");
            input.Pooling(output, cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, poolingSize, stride);
            expectedOutput = new CpuTensor<double>(new[] { 3, 1, 2, 2 }, new double[] { 333, 8, 14, 16, 22, 24, 30, 32, 38, 40, 46, 48 }, "expectedOutput");
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, output, 1e-6));
        }
        public static CpuTensor<double> RandomDoubleTensor(int[] shape, Random rand, double minValue, double maxValue, string description)
        {
            var result = new CpuTensor<double>(shape, description);
            Utils.Randomize(result.Content, rand, minValue, maxValue);
            return result;
        }
        public static CpuTensor<float> RandomFloatTensor(int[] shape, Random rand, double minValue, double maxValue, string description)
        {
            var result = new CpuTensor<float>(shape, description);
            Utils.Randomize(result.Content, rand, minValue, maxValue);
            return result;
        }
        private void TestConvolution(CpuTensor<double> input, CpuTensor<double> convolution, int padding, int stride, CpuTensor<double> expectedOutput)
        {
            var output = new CpuTensor<double>(Tensor.ConvolutionOutputShape(input.Shape, convolution.Shape, padding, stride), "output");
            input.Convolution(convolution, padding, stride, output);
            Assert.IsTrue(TestTensor.SameContent(expectedOutput, output, 1e-6));

            //We make the same test for GPU (CUDA)
            var convolutionGPU = convolution.ToGPU<double>(_gpuWrapper);
            var outputGPU = output.ToGPU<double>(_gpuWrapper);
            var inputGPU = input.ToGPU<double>(_gpuWrapper);
            inputGPU.Convolution(convolutionGPU, padding, stride, outputGPU);
            Assert.IsTrue(TestTensor.SameContent(expectedOutput,outputGPU, 1e-6));
        }



    }
}
