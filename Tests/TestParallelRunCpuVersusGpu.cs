using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.Data;
using SharpNet.GPU;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNetTests.CPU;
using SharpNetTests.Data;
using SharpNetTests.NonReg;

namespace SharpNetTests
{
    [TestFixture]
    public class TestParallelRunCpuVersusGpu
    {
        private const int BatchSize = 9;
        private const int FiltersCount = 8;
        private const int ChannelsCount = 3;
        private const int Height = 17;
        private const int Width = 32;
        private const int Nx = Height * Width * ChannelsCount;
	    private readonly Random _rand = new Random(0);
        private const GPUWrapper.ConvolutionAlgoPreference ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM;
        // ReSharper disable once MemberCanBeMadeStatic.Local
        private GPUWrapper GpuWrapper => GPUWrapper.FromDeviceId(0);

        [Test]
        public void TestConvolution()
        {
            foreach(ConvolutionLayer.PADDING_TYPE paddingType in Enum.GetValues(typeof(ConvolutionLayer.PADDING_TYPE)))
            foreach(NetworkConfig.CompatibilityModeEnum compatibilityMode in Enum.GetValues(typeof(NetworkConfig.CompatibilityModeEnum)))
            foreach(int stride in new[]{1,2})
            foreach (var isDepthwiseConvolution in new[] { true,false})
            {
                var channelsCount = 3;
                var height = 17;
                var width = 32;
                var f = 3;
                int filterCount = 128;
                var x = RandomTensor(new[] { BatchSize, channelsCount, height, width });
                var convolutionShape = isDepthwiseConvolution
                        ? new[] { 1, channelsCount, f, f }
                        : new[] { filterCount, channelsCount, f, f };
                var convolution = RandomTensor(convolutionShape);
	            var y = RandomTensor(ConvolutionLayer.OutputShape(x.Shape, convolution.Shape, paddingType, stride, isDepthwiseConvolution));
                ConvolutionLayer.Padding(x.Shape[2], f, stride, paddingType, compatibilityMode, out int paddingTop, out int paddingBottom);
                ConvolutionLayer.Padding(x.Shape[3], f, stride, paddingType, compatibilityMode, out int paddingLeft, out int paddingRight);
                var memoryPool =  new TensorMemoryPool(GpuWrapper, false);
                if (ConvolutionLayer.IsAsymmetricPadding(paddingTop, paddingBottom, paddingLeft, paddingRight))
                {
                    continue; //asymmetric padding is not supported by cuDNN
                }
                TestAll(new[] { x, convolution, y }, tensors => tensors[0].Convolution(tensors[1], paddingTop, paddingBottom, paddingLeft, paddingRight, stride, tensors[2], isDepthwiseConvolution, ConvolutionAlgoPreference, memoryPool));
            }
        }

        [Test]
        public void TestConvolutionBackwardBias()
        {
            var dxShape = new[] { BatchSize, FiltersCount, Height, Width };
            var biasShape = new[] { 1, FiltersCount, 1, 1 };
            var dx = RandomTensor(dxShape);
            var convolutionBackwardBias = new CpuTensor<float>(biasShape);
            TestAll(new[] { dx, convolutionBackwardBias }, tensors => tensors[0].ConvolutionBackwardBias(tensors[1]));
        }

        [Test]
        public void TestConvolutionGradient()
        {
            var memoryPool = new TensorMemoryPool(GpuWrapper, false);
            foreach (ConvolutionLayer.PADDING_TYPE paddingType in Enum.GetValues(typeof(ConvolutionLayer.PADDING_TYPE)))
            foreach (NetworkConfig.CompatibilityModeEnum compatibilityMode in Enum.GetValues(typeof(NetworkConfig.CompatibilityModeEnum)))
            foreach (int stride in new[] { 1, 2 })
            foreach (int f in new[] { 3, 5 })
            foreach (var isDepthwiseConvolution in new[] { true, false })
            {
                var channelsCount = 3;
                var height = 17;
                var width = 32;
                var x = RandomTensor(new[] { BatchSize, channelsCount, height, width });
                x = new CpuTensor<float>(x.Shape);
                var convolutionShape = isDepthwiseConvolution
                    ? new[] { 1, channelsCount, f, f }
                    : new[] { 9, channelsCount, f, f };
                var convolution = RandomTensor(convolutionShape);
                var dy = RandomTensor(ConvolutionLayer.OutputShape(x.Shape, convolution.Shape, paddingType, stride, isDepthwiseConvolution));
                //this will compute 'dx' && 'convolutionGradient'
                var dx = RandomTensor(x.Shape);
                var convolutionGradient = RandomTensor(convolution.Shape);
                ConvolutionLayer.Padding(x.Shape[2], f, stride, paddingType, compatibilityMode, out int paddingTop, out int paddingBottom);
                ConvolutionLayer.Padding(x.Shape[3], f, stride, paddingType, compatibilityMode, out int paddingLeft, out int paddingRight);

                if (ConvolutionLayer.IsAsymmetricPadding(paddingTop, paddingBottom, paddingLeft, paddingRight))
                {
                    var paddedXShape = new[] { x.Shape[0], x.Shape[1], paddingTop + x.Shape[2] + paddingBottom, paddingLeft + x.Shape[3] + paddingRight };
                    var padded_X = RandomTensor(paddedXShape);
                    padded_X.ZeroPadding(x, paddingTop, paddingBottom, paddingLeft, paddingRight);
                    var padded_dX = RandomTensor(paddedXShape);
                    TestAll(new[] { padded_X, convolution, dy, dx, padded_dX, convolutionGradient }, tensors => ConvolutionGradientAsymmetricPadding(
                        tensors[0], tensors[1], tensors[2], paddingTop, paddingBottom, paddingLeft, paddingRight, stride, tensors[3], tensors[4], tensors[5], isDepthwiseConvolution, memoryPool));
                }
                else
                {
                    TestAll(new[] { x, convolution, dy, dx, convolutionGradient }, tensors => tensors[0].ConvolutionGradient(tensors[1], tensors[2], paddingTop, paddingBottom, paddingLeft, paddingRight, stride, tensors[3], tensors[4], isDepthwiseConvolution, ConvolutionAlgoPreference, memoryPool));
                }
            }
        }

        private static void ConvolutionGradientAsymmetricPadding(Tensor padded_X, Tensor convolution, Tensor dy, int paddingTop,
            int paddingBottom, int paddingLeft, int paddingRight, int stride, Tensor dx, Tensor padded_dX, Tensor convGradient, bool isDepthwiseConvolution, TensorMemoryPool memoryPool)
        {
            padded_X.ConvolutionGradient(convolution, dy, 0, 0, 0, 0, stride, padded_dX, convGradient, isDepthwiseConvolution, ConvolutionAlgoPreference, memoryPool);
            dx.ZeroUnpadding(padded_dX, paddingTop, paddingBottom, paddingLeft, paddingRight);
        }

        [Test]
        public void TestBatchNormalization()
        {
            foreach (var mode in new []{ cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL, cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION })
            foreach (var batchSize in new[] { 1, 4, 9 })
            foreach (var momentum in new []{0.0, 0.5, 0.99, 1.0})
            foreach (var epsilon in new []{1e-3, 1e-5})
            foreach (var ignoreHW in new []{false, true})
            foreach (var isTraining in new[] { false, true })
            {
                var xShape = new[] { batchSize, ChannelsCount, Height, Width };
                var scaleAndBiasShape = (mode == cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION)
                    ? new[] { 1, ChannelsCount, Height, Width }
                    : new[] { 1, ChannelsCount, 1, 1 };
                if (ignoreHW)
                {
                    xShape = xShape.Take(2).ToArray();
                    scaleAndBiasShape = scaleAndBiasShape.Take(2).ToArray();
                }

                var x = RandomTensor(xShape);
                var y = RandomTensor(xShape);
                var scale = RandomTensor(scaleAndBiasShape);
                var bias = RandomTensor(scaleAndBiasShape);
                var runningInputMean = RandomTensor(scaleAndBiasShape);
                var runningInputVariance = TestCpuTensor.RandomFloatTensor(scaleAndBiasShape, _rand, 0.1, 2);
                var meanBuffer = RandomTensor(scaleAndBiasShape);
                var varianceBuffer = TestCpuTensor.RandomFloatTensor(scaleAndBiasShape, _rand, 0.1, 50);
                double exponentialAverageSmoothingFactor = 1 - momentum;
                TestAll(new[] { x, y, scale, bias, runningInputMean, runningInputVariance, meanBuffer, varianceBuffer }, tensors => tensors[0].BatchNormalization(tensors[1], tensors[2], tensors[3], exponentialAverageSmoothingFactor, tensors[4], tensors[5], mode, epsilon, tensors[6], tensors[7], isTraining));
            }
        }

        [Test]
        public void TestBatchNormalizationV2()
        {
            var tensorIdsToIgnore = new List<int> { 6, 7 };
            var mode = cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL;
            var xShape = new[] { 1, 32, 112, 112};
            var x = new CpuTensor<float>(xShape, null);
            x.ZeroMemory();
            var scale = TestNetworkPropagation.FromNumpyArray("[[[[2.82185745]],[[4.20555544]],[[4.44391775]],[[2.95071363]],[[0.901465356]],[[3.83799005]],[[2.20374274]],[[3.30325413]],[[3.38044739]],[[0.202515125]],[[2.14543128]],[[0.645111859]],[[3.29296565]],[[11.9912415]],[[0.810986161]],[[3.39099979]],[[2.6564517]],[[8.52717972]],[[2.52371788]],[[3.94317198]],[[2.74237108]],[[11.1155062]],[[4.08373785]],[[5.75315952]],[[0.335611582]],[[1.24477983]],[[3.90086651]],[[1.98501635]],[[0.818592787]],[[0.626930952]],[[6.75085163]],[[3.4190371]]]]");
            var bias = TestNetworkPropagation.FromNumpyArray("[[[[-3.74896479]],[[2.43146777]],[[2.31554103]],[[7.13698292]],[[-1.38208234]],[[8.66540337]],[[-2.95346022]],[[1.81856453]],[[0.995381236]],[[0.00296683772]],[[-2.85715914]],[[1.74939632]],[[0.599703848]],[[0.165816754]],[[1.90356266]],[[8.97630692]],[[2.26754451]],[[3.72180033]],[[2.572788]],[[1.96836185]],[[-3.36665225]],[[2.64624929]],[[10.5395947]],[[-10.4322577]],[[-1.63009882]],[[1.37903798]],[[9.95489788]],[[1.99438405]],[[0.159816369]],[[2.50823808]],[[-10.8555698]],[[2.08439994]]]]");
            var runningInputMean = TestNetworkPropagation.FromNumpyArray("[[[[-0.0474244691]],[[-0.00338064576]],[[0.00407501776]],[[0.0787407607]],[[0.0313696824]],[[0.0837314799]],[[-0.0393488146]],[[0.0694158077]],[[0.639113843]],[[-0.171755388]],[[-0.382961541]],[[0.0100561073]],[[0.606002986]],[[1.39727235]],[[0.420819908]],[[-0.0792663917]],[[0.00732345507]],[[-0.770392716]],[[0.00307485089]],[[-0.00288994168]],[[-0.0452340655]],[[-0.719747245]],[[-0.0934633166]],[[0.163005278]],[[0.121294215]],[[-0.00648898305]],[[-0.0706383437]],[[0.00286416081]],[[2.91242941E-09]],[[0.0120399296]],[[-0.063189812]],[[-0.00128901063]]]]");
            var runningInputVariance = TestNetworkPropagation.FromNumpyArray("[[[[7.25111055]],[[5.37058496]],[[6.66747379]],[[18.2757835]],[[5.69575691]],[[17.0573292]],[[6.76594353]],[[1.52835393]],[[18.0554256]],[[27.2328396]],[[10.9577389]],[[3.57627463]],[[12.896986]],[[39.5671387]],[[3.67913604]],[[13.6923494]],[[6.86120129]],[[19.7278404]],[[3.81912017]],[[9.09753227]],[[6.9455328]],[[23.5766983]],[[18.0286465]],[[18.6031551]],[[1.11303592]],[[6.78300667]],[[11.5361662]],[[6.32360983]],[[0]],[[1.08625805]],[[19.3687859]],[[5.1940341]]]]");
            var y = RandomTensor(xShape);
            var meanBuffer = TestNetworkPropagation.FromNumpyArray("[[[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]]]]");
            var varianceBuffer = TestNetworkPropagation.FromNumpyArray("[[[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]]]]");
            double exponentialAverageSmoothingFactor = 1 - 0.99;
            TestAll(new[] { x, y, scale, bias, runningInputMean, runningInputVariance, meanBuffer, varianceBuffer }, tensors => tensors[0].BatchNormalization(tensors[1], tensors[2], tensors[3], exponentialAverageSmoothingFactor, tensors[4], tensors[5], mode, 0.001, tensors[6], tensors[7], false), tensorIdsToIgnore);
        }
        [Test]
        public void TestBatchNormalizationBackward()
        {
            foreach (var mode in new[] { cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL, cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION })
            foreach (var batchSize in new[] { 1, 4, 9 })
            foreach (var exponentialAverageFactor in new[] { 0.5, 0.99, 1.0 })
            foreach (var epsilon in new[] { 1e-3, 1e-5 })
            foreach (var ignoreHW in new[] { false, true })
            {
                var xShape = new[] { batchSize, ChannelsCount, Height, Width };
                var scaleAndBiasShape = (mode == cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION)
                    ? new[] { 1, ChannelsCount, Height, Width }
                    : new[] { 1, ChannelsCount, 1, 1 };
                if (ignoreHW)
                {
                    xShape = xShape.Take(2).ToArray();
                    scaleAndBiasShape = scaleAndBiasShape.Take(2).ToArray();
                }
                var x = RandomTensor(xShape);
                var y = RandomTensor(xShape);
                var scale = RandomTensor(scaleAndBiasShape);
                var bias = RandomTensor(scaleAndBiasShape);
                var runningInputMean = RandomTensor(scaleAndBiasShape);
                var runningInputVariance = TestCpuTensor.RandomFloatTensor(scaleAndBiasShape, _rand, 0.1, 2);
                var meanBuffer = RandomTensor(scaleAndBiasShape);
                var invertOfUnbiasedVolatilityBuffer = RandomTensor(scaleAndBiasShape);
                x.BatchNormalization(y, scale, bias, exponentialAverageFactor, runningInputMean, runningInputVariance, mode, epsilon, meanBuffer, invertOfUnbiasedVolatilityBuffer, false);
                var dx = RandomTensor(xShape);
                var dy = RandomTensor(xShape);
                var scaleGradient = RandomTensor(scaleAndBiasShape);
                var biasGradient = RandomTensor(scaleAndBiasShape);
                TestAll(new[] { x, dy, dx, scale, scaleGradient, biasGradient, meanBuffer, invertOfUnbiasedVolatilityBuffer }, tensors => tensors[0].BatchNormalizationBackward(tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], mode, epsilon, tensors[6], tensors[7]));
            } 
        }
        [Test]
	    public void TestZeroMemory()
	    {
	        var a = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
	        TestAll(new[] { a }, tensors => tensors[0].ZeroMemory());
	    }

        [Test]
	    public void TestDot()
	    {
	        var a = RandomTensor(new[] { 8, 10});
	        var b = RandomTensor(new[] { a.Shape[1], 12});
	        var result = new CpuTensor<float>(new[] { a.Shape[0], b.Shape[1] });
	        TestAll(new[] { a, b, result}, tensors => tensors[2].Dot(tensors[0], false, tensors[1], false, 1, 0));
        }

        [Test]
        public void TestUpSampling()
        {
            const int rowFactor = 3;
            const int colFactor = 4;
            var shapeBefore = new[] { 5, 3, 8, 10};
            var tensorBeforeUpSampling = RandomTensor(shapeBefore);
            var shapeAfter = new[] { shapeBefore[0], shapeBefore[1], rowFactor * shapeBefore[2], colFactor * shapeBefore[3] };
            var tensorAfterUpSampling = RandomTensor(shapeAfter);
            TestAll(new[] { tensorAfterUpSampling, tensorBeforeUpSampling}, tensors => tensors[0].UpSampling2D(tensors[1], rowFactor, colFactor, UpSampling2DLayer.InterpolationEnum.Nearest));
        }

        [Test]
        public void TestDownSampling()
        {
            const int rowFactor = 3;
            const int colFactor = 4;
            var tensorAfterDownSampling = RandomTensor(new[] { 5, 3, 8, 10 });
            var tensorBeforeDownSampling = RandomTensor(new[] { 5, 3, 8 * rowFactor, 10 * colFactor });
            TestAll(new[] { tensorAfterDownSampling, tensorBeforeDownSampling }, tensors => tensors[0].DownSampling2D(tensors[1], rowFactor, colFactor));
        }

        [Test]
	    public void TestDotV2()
	    {
	        var a = RandomTensor(new[] { 8, 10 });
	        var b = RandomTensor(new[] { a.Shape[1], 12 });
	        var result = new CpuTensor<float>(new[] { a.Shape[0], b.Shape[1] });
            TestAll(new[] { a, b, result }, tensors => tensors[2].Dot(tensors[0], tensors[1]));
        }

        /// <summary>
        /// This test was written to reproduce the error: CUBLAS_STATUS_INTERNAL_ERROR (v 10.0)
        /// Currently this error is ignored
        /// </summary>
        [Test]
        public void TestDotV3()
        {
            var a = RandomTensor(new[] { 2, 20000 });
            var b = RandomTensor(new[] { a.Shape[1], 2 });
            var result = new CpuTensor<float>(new[] { a.Shape[0], b.Shape[1] });
            TestAll(new[] { a, b, result }, tensors => tensors[2].Dot(tensors[0], false, tensors[1], false, 1, 0));
        }

        [Test]
        public void TestMultiplyTensorSameShape()
        {
            var result = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            var a = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            var x = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            TestAll(new[] { result, a, x }, tensors => tensors[0].MultiplyTensor(tensors[1], tensors[2]));
        }

        [Test]
        public void TestMultiplyTensorDifferentShape()
        {
            var result = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            var a = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            var x = RandomTensor(new[] { BatchSize, ChannelsCount, 1, 1 });
            TestAll(new[] { result, a, x }, tensors => tensors[0].MultiplyTensor(tensors[1], tensors[2]));
        }

        [Test]
        public void TestMultiplyEachRowIntoSingleValue()
        {
            var result = RandomTensor(new[] { BatchSize, ChannelsCount, 1, 1 });
            var a = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            var b = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            TestAll(new[] { result, a, b }, tensors => tensors[0].MultiplyEachRowIntoSingleValue(tensors[1], tensors[2]));
        }

        [Test]
        public void TestZeroPadding()
        {
            foreach (var shape in new[] { new[] { 7, 3, 7, 8 }, new[] { 4, 5, 12, 5 } })
            {
                var src = RandomTensor(shape);
                foreach (var top_pad in new[] { 0, 1, 3 })
                foreach (var bottom_pad in new[] { 0, 1, 3 })
                foreach (var left_pad in new[] { 0, 1, 3 })
                foreach (var right_pad in new[] { 0, 1, 3 })
                {
                    var destShape = new[] { shape[0], shape[1], top_pad + shape[2] + bottom_pad, left_pad + shape[3] + right_pad };
                    var dest = RandomTensor(destShape);
                    TestAll(new[] { dest, src }, tensors => tensors[0].ZeroPadding(tensors[1], top_pad, bottom_pad, left_pad, right_pad));
                }
            }
        }

        [Test]
        public void TestZeroUnpadding()
        {
            foreach (var shape in new[] { new[] { 7, 3, 7, 8 }, new[] { 4, 5, 12, 5 } })
            {
                var unpaddedTensor = RandomTensor(shape);
                foreach (var top_pad in new[] { 0, 1, 3 })
                foreach (var bottom_pad in new[] { 0, 1, 3 })
                foreach (var left_pad in new[] { 0, 1, 3 })
                foreach (var right_pad in new[] { 0, 1, 3 })
                {
                    var paddedTensorShape = new[] { shape[0], shape[1], top_pad + shape[2] + bottom_pad, left_pad + shape[3] + right_pad };
                    var paddedTensor = RandomTensor(paddedTensorShape);
                    TestAll(new[] { unpaddedTensor , paddedTensor}, tensors => tensors[0].ZeroUnpadding(tensors[1], top_pad, bottom_pad, left_pad, right_pad));
                }
            }
        }

        [Test]
	    public void TestUpdate_Adding_Alpha_X()
	    {
	        var y = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width});
	        var x = RandomTensor(y.Shape);
	        TestAll(new[] { y, x}, tensors => tensors[0].Update_Adding_Alpha_X(0.5f, tensors[1]));
	    }

        [Test]
        public void TestAddTensor()
        {
            var y = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            var x = RandomTensor(y.Shape);
            TestAll(new[] { y, x }, tensors => tensors[0].AddTensor(0.5f, tensors[1], 0.75f));
        }

        [Test]
        public void TestConcatenate()
        {
            var x1 = RandomTensor(new[] { BatchSize, 17, Height, Width });
            var x2 = RandomTensor(new[] { BatchSize, 13, Height, Width });
            var concat = RandomTensor(new[] { BatchSize, 17 + 13, Height, Width });
            TestAll(new[] { concat, x1, x2 }, tensors => tensors[0].Concatenate(tensors[1], tensors[2]));
        }

        [Test]
        public void TestSplit()
        {
            var toSplit = RandomTensor(new[] { BatchSize, 17 + 13, Height, Width });
            var x1 = RandomTensor(new[] { BatchSize, 17, Height, Width });
            var x2 = RandomTensor(new[] { BatchSize, 13, Height, Width });
            TestAll(new[] { toSplit, x1, x2 }, tensors => tensors[0].Split(tensors[1], tensors[2]));
        }

        [Test]
        public void TestUpdate_Multiplying_By_Alpha()
        {
            var x = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            TestAll(new[] { x }, tensors => tensors[0].Update_Multiplying_By_Alpha(0.5f));
        }
        [Test]
	    public void TestCompute_BiasGradient_from_dy()
	    {
	        var dy = RandomTensor(new[] { BatchSize, 1, 1, Nx });
	        var db = RandomTensor(new[] { 1, 1, 1, Nx });
	        TestAll(new[] { dy, db }, tensors => tensors[0].Compute_BiasGradient_from_dy(tensors[1]));
	    }
	    [Test]
	    public void TestBroadcastConvolutionBiasToOutput()
	    {
            var convolutionBias = RandomTensor(new[] { 1, FiltersCount, 1, 1});
            var y = RandomTensor(new[] { BatchSize, FiltersCount, Height, Width});
	        TestAll(new[] { convolutionBias, y }, tensors => tensors[0].BroadcastConvolutionBiasToOutput(tensors[1]));
	    }
        [Test]
	    public void TestCopyTo()
	    {
	        var src = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width});
	        var dest = RandomTensor(src.Shape);
	        TestAll(new[] { src, dest}, tensors => tensors[0].CopyTo(tensors[1]));
	    }
        [Test]
        public void TestCopyTo_V2()
        {
            var src = RandomTensor(new[] { 10,5,2});
            var dest = RandomTensor(src.Shape);
            TestAll(new[] { src, dest }, tensors => tensors[0].CopyTo(20,tensors[1],60,40));
        }

        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_TANH, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU, 0.1)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU, 0.3)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_ELU, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH, 0)]
	    public void TestActivationForward(cudnnActivationMode_t activationMode, double alphaActivation)
	    {
	        var x = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
	        var y = RandomTensor(x.Shape);
	        TestAll(new[] { x, y }, tensors => tensors[0].ActivationForward(activationMode, alphaActivation, tensors[1]));
	    }

        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_TANH, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU, 0.1)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU, 0.3)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_ELU, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH, 0)]
	    public void TestActivationBackward(cudnnActivationMode_t activationMode, double alphaActivation)
	    {
            var y = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            var dy = RandomTensor(y.Shape);
            var x = RandomTensor(y.Shape);
            var dx = RandomTensor(y.Shape);
	        x.ActivationForward(activationMode, alphaActivation, y);
            TestAll(new[] { y, dy, x, dx }, tensors => tensors[0].ActivationBackward(tensors[1], tensors[2], activationMode, alphaActivation, tensors[3]));
	    }
        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, 1)]
        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC, 2)]
        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, 1)]
        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, 2)]
	    public void TestPooling(cudnnPoolingMode_t poolingMode, int stride)
	    {
            foreach(int poolingHeight in new[]{ 2,4})
            { 
                foreach(int poolingWidth in new[]{ 2,4})
                { 
	                var aBeforePooling = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
	                var aAfterPooling = RandomTensor(PoolingLayer.PoolingOutputShape(aBeforePooling.Shape, poolingHeight, poolingWidth, stride));
	                TestAll(new[] { aBeforePooling, aAfterPooling }, tensors => tensors[0].Pooling(tensors[1], poolingMode, poolingHeight, poolingWidth, stride));
                }
            }
	    }
        [Test]
        public void TestGlobalAveragePooling()
        {
            var poolingMode = cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            int poolingHeight = Height;
            int poolingWidth = Width;
            int poolingStride = Math.Max(Height, Width);
            var aBeforePooling = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            var aAfterPooling = RandomTensor(PoolingLayer.PoolingOutputShape(aBeforePooling.Shape, poolingHeight, poolingWidth, poolingStride));
            TestAll(new[] { aBeforePooling, aAfterPooling }, tensors => tensors[0].Pooling(tensors[1], poolingMode, poolingHeight, poolingWidth, poolingStride));
        }

        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC)]
        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)]
	    public void TestPoolingGradient(cudnnPoolingMode_t poolingMode)
        {
            foreach(var poolingSize in new[] {2,4})
            { 
                var shapeBeforePooling = new[] { BatchSize, ChannelsCount, Height, Width };
	            var shapeAfterPooling = PoolingLayer.PoolingOutputShape(shapeBeforePooling, poolingSize, poolingSize, poolingSize);
	            var dy = RandomTensor(shapeAfterPooling);
	            var x = RandomTensor(shapeBeforePooling);
	            var y = RandomTensor(shapeAfterPooling);
                x.Pooling(y, poolingMode, poolingSize, poolingSize, poolingSize);
                var dx = RandomTensor(shapeBeforePooling);
	            TestAll(new[] { dy, y, x, dx }, tensors => tensors[0].PoolingGradient(tensors[1], tensors[2], tensors[3], poolingMode, poolingSize, poolingSize, poolingSize));
	        }
        }

        [Test]
        public void TestGlobalAveragePoolingGradient()
        {
            var poolingMode = cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            int poolingHeight = Height;
            int poolingWidth = Width;
            int poolingStride = Math.Max(Height, Width);
            var shapeBeforePooling = new[] { BatchSize, ChannelsCount, Height, Width };
            var shapeAfterPooling = PoolingLayer.PoolingOutputShape(shapeBeforePooling, poolingHeight, poolingWidth, poolingStride);
            var dy = RandomTensor(shapeAfterPooling);
            var x = RandomTensor(shapeBeforePooling);
            var y = RandomTensor(shapeAfterPooling);
            x.Pooling(y, poolingMode, poolingHeight, poolingWidth, poolingStride);
            var dx = RandomTensor(shapeBeforePooling);
            TestAll(new[] { dy, y, x, dx }, tensors => tensors[0].PoolingGradient(tensors[1], tensors[2], tensors[3], poolingMode, poolingHeight, poolingWidth, poolingStride));
        }

        [Test]
	    public void TestBroadcastAddVectorToOutput()
	    {
            var b = RandomTensor(new []{1, Nx});
	        var y = RandomTensor(new []{BatchSize, Nx});
            TestAll(new[] {b,y}, tensors => tensors[0].BroadcastAddVectorToOutput(tensors[1]));
	    }
        [Test]
        public void TestUpdateAdamOptimizer()
        {
            var W = RandomTensor(new[] { Width, Height});
            var dW = RandomTensor(W.Shape);
            var adam_vW = RandomTensor(W.Shape);
            var adam_sW = RandomTensor(W.Shape);
            TestAll(new[] { W, dW, adam_vW, adam_sW}, tensors => tensors[0].UpdateAdamOptimizer(0.001, 0.9,0.999,1e-8, tensors[1], tensors[2], tensors[3], 5));
        }

        [TestCase(0.1, 0.9, 0.0, true)]
        [TestCase(0.1, 0.9, 0.0, false)]
        [TestCase(0.1, 0.9, 1e-4, true)]
        [TestCase(0.1, 0.9, 1e-4, false)]
        public void TestUpdateSGDOptimizer(double learningRate, double momentum, double decay, bool usenesterov)
        {
            var W = RandomTensor(new[] { Width, Height });
            var dW = RandomTensor(W.Shape);
            var velocity = RandomTensor(W.Shape);
            TestAll(new[] { W, dW, velocity }, tensors => tensors[0].UpdateSGDOptimizer(learningRate, momentum, usenesterov, tensors[1], tensors[2]));
        }

        
        [TestCase(1, NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)]
        [TestCase(1, NetworkConfig.LossFunctionEnum.BinaryCrossentropy)]
        [TestCase(2, NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)]
        [TestCase(2, NetworkConfig.LossFunctionEnum.BinaryCrossentropy)]
        [TestCase(10, NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)]
        [TestCase(10, NetworkConfig.LossFunctionEnum.BinaryCrossentropy)]
        public void TestComputeLoss(int categoryCount, NetworkConfig.LossFunctionEnum lossFunction)
        {
            var nbRows = 1000;
            var yPredicted = RandomTensor(new[] { nbRows, categoryCount });
            var yExpectedOneHot = TestCpuTensor.RandomOneHotTensor(yPredicted.Shape, _rand);
            var buffer = RandomTensor(new[] { nbRows });
            TestAllForReturnValue(new[] { yExpectedOneHot, yPredicted, buffer}, tensors => tensors[0].ComputeLoss(tensors[1], lossFunction, tensors[2]), new List<int>{2});
        }

        [TestCase(2, NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)]
        [TestCase(2, NetworkConfig.LossFunctionEnum.BinaryCrossentropy)]
        [TestCase(10, NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)]
        [TestCase(10, NetworkConfig.LossFunctionEnum.BinaryCrossentropy)]
        public void TestComputeLossFromCategoryIndexes(int categoryCount, NetworkConfig.LossFunctionEnum lossFunction)
        {
            const int nbRows = 10000;
            var yPredicted = RandomTensor(new[] { nbRows, categoryCount });
            var categoryIndexes = TestCpuTensor.RandomCategoryIndexTensor(nbRows, categoryCount, _rand);
            var buffer = RandomTensor(new[] { nbRows });
            TestAllForReturnValue(new[] { categoryIndexes }, new[] { yPredicted, buffer }, tensors => tensors[0].ComputeLossFromCategoryIndexes(tensors[1], lossFunction, tensors[2]), new List<int> { 1 });
        }

        [TestCase(10)]
        [TestCase(1)]
        public void TestComputeAccuracyOneHot(int categoryCount)
        {
            var nbRows = 10000;
            var yPredicted = RandomTensor(new[] { nbRows, categoryCount });
            var yExpectedOneHot = TestCpuTensor.RandomOneHotTensor(yPredicted.Shape, _rand);
            var buffer = RandomTensor(new[] { nbRows});
            TestAllForReturnValue(new[] { yExpectedOneHot, yPredicted, buffer }, tensors => tensors[0].ComputeAccuracy(tensors[1], tensors[2]), new List<int> { 2 });
        }

        [TestCase(10)]
        [TestCase(1)]
        public void TestComputeAccuracyTwoHot(int categoryCount)
        {
            var nbRows = 10000;
            var yPredicted = RandomTensor(new[] { nbRows, categoryCount });
            var yExpectedOneHot = TestCpuTensor.RandomTwoHotTensor(yPredicted.Shape, _rand);
            var buffer = RandomTensor(new[] { nbRows });
            TestAllForReturnValue(new[] { yExpectedOneHot, yPredicted, buffer }, tensors => tensors[0].ComputeAccuracy(tensors[1], tensors[2]), new List<int> { 2 });
        }

        [Test]
        public void TestComputeAccuracyFromCategoryIndexes()
        {
            const int nbRows = 10000;
            const int categoryCount = 10;
            var yPredicted = RandomTensor(new[] { nbRows, categoryCount });
            var categoryIndexes = TestCpuTensor.RandomCategoryIndexTensor(nbRows, categoryCount, _rand);
            var buffer = RandomTensor(new[] { nbRows });
            TestAllForReturnValue(new[] { categoryIndexes }, new[] { yPredicted, buffer }, tensors => tensors[0].ComputeAccuracyFromCategoryIndexes(tensors[1], tensors[2]), new List<int> { 1 });
        }

        private CpuTensor<float> RandomTensor(int[] shape)
	    {
	        return TestCpuTensor.RandomFloatTensor(shape, _rand, -1.5, +1.5);
	    }
        private static void AreEquals(CpuTensor<float> floatCpu, GPUTensor<float> floatGpu)
	    {
	        Assert.IsTrue(floatCpu.SameShape(floatGpu));
            Assert.IsTrue(TestTensor.SameContent(floatCpu, floatGpu, 1e-2), floatCpu + Environment.NewLine + floatGpu);
        }
	    [SuppressMessage("ReSharper", "CoVariantArrayConversion")]
	    private void TestAll(CpuTensor<float>[] data, Action<Tensor[]> work, List<int> tensorIdsToIgnore = null)
	    {
	        var cpuFloat = new List<CpuTensor<float>>();
	        cpuFloat.AddRange(data);
	        var gpuFloat = new List<GPUTensor<float>>();
	        gpuFloat.AddRange(cpuFloat.Select(x => CloneToGPU(x, GpuWrapper)));
	        work(cpuFloat.ToArray());
	        work(gpuFloat.ToArray());
            for (var i = 0; i < cpuFloat.Count; ++i)
            {
                if (tensorIdsToIgnore != null && tensorIdsToIgnore.Contains(i))
                {
                    continue;
                }
	            AreEquals(cpuFloat[i], gpuFloat[i]);
	        }
	    }

        private void TestAllForReturnValue(CpuTensor<float>[] cpuFloat, Func<Tensor[], double> work, List<int> tensorIdsToIgnore = null)
        {
            TestAllForReturnValue(new CpuTensor<int>[0], cpuFloat, work, tensorIdsToIgnore);
        }

        private void TestAllForReturnValue(CpuTensor<int>[] cpuInt, CpuTensor<float>[] cpuFloat, Func<Tensor[], double> work, List<int> tensorIdsToIgnore = null)
        {
            var gpuInt = cpuInt.Select(x => CloneToGPU(x, GpuWrapper)).ToList();
            var gpuFloat = cpuFloat.Select(x => CloneToGPU(x, GpuWrapper)).ToList();
            var cpu = cpuInt.Select(x => (Tensor) x).Union(cpuFloat.Select(x => (Tensor) x)).ToArray();
            var resultCpuFloat = work(cpu);
            var gpu = gpuInt.Select(x => (Tensor)x).Union(gpuFloat.Select(x => (Tensor)x)).ToArray();
            var resultGPUFloat = work(gpu);
            Assert.AreEqual(resultCpuFloat, resultGPUFloat, 1e-5, cpuFloat.Last().ReadonlyContent.Min() + " vs " + gpuFloat.Last().ContentAsFloatArray().Min());
            for (var i = 0; i < cpuFloat.Length; ++i)
            {
                if (tensorIdsToIgnore != null && tensorIdsToIgnore.Contains(i))
                {
                    continue;
                }
                AreEquals(cpuFloat[i], gpuFloat[i]);
            }
        }

        private static GPUTensor<T> CloneToGPU<T>(CpuTensor<T> cpuTensor, GPUWrapper gpuWrapper)
        {
            return new GPUTensor<T>(cpuTensor.Shape, cpuTensor.Content, gpuWrapper);
        }
    }
}
