using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.Data;
using SharpNet.GPU;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNetTests.CPU;
using SharpNetTests.Datasets;
using SharpNetTests.GPU;
using SharpNetTests.NonReg;

namespace SharpNetTests
{
    [TestFixture]
    public class TestParallelRunCpuVersusGpu
    {
        private const int BatchSize = 9;
        private const int OutChannels = 8;
        private const int ChannelsCount = 3;
        private const int Height = 17;
        private const int Width = 32;
        private const int Nx = Height * Width * ChannelsCount;
	    private readonly Random _rand = new (0);
        private const GPUWrapper.ConvolutionAlgoPreference ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM;
        // ReSharper disable once MemberCanBeMadeStatic.Local
        private GPUWrapper GpuWrapper => TestGPUTensor.GpuWrapper;

        [Test]
        public void TestConvolution()
        {
            foreach(ConvolutionLayer.PADDING_TYPE paddingType in Enum.GetValues(typeof(ConvolutionLayer.PADDING_TYPE)))
            foreach(NetworkSample.CompatibilityModeEnum compatibilityMode in Enum.GetValues(typeof(NetworkSample.CompatibilityModeEnum)))
            foreach(int stride in new[]{1,2})
            foreach (var isDepthwiseConvolution in new[] { true,false})
            {
                const int channelsCount = 3;
                const int height = 17;
                const int width = 32;
                const int kernelSize = 3;
                const int outChannels = 128;
                var x = RandomTensor(new[] { BatchSize, channelsCount, height, width });
                var convolutionShape = isDepthwiseConvolution
                        ? new[] { 1, channelsCount, kernelSize, kernelSize }
                        : new[] { outChannels, channelsCount, kernelSize, kernelSize };
                var convolution = RandomTensor(convolutionShape);
	            var y = RandomTensor(ConvolutionLayer.OutputShape(x.Shape, convolution.Shape, paddingType, stride, isDepthwiseConvolution));
                ConvolutionLayer.Padding(x.Shape[2], kernelSize, stride, paddingType, compatibilityMode, out int paddingTop, out int paddingBottom);
                ConvolutionLayer.Padding(x.Shape[3], kernelSize, stride, paddingType, compatibilityMode, out int paddingLeft, out int paddingRight);
                var memoryPool =  new TensorMemoryPool(GpuWrapper);
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
            var dxShape = new[] { BatchSize, OutChannels, Height, Width };
            var biasShape = new[] { 1, OutChannels, 1, 1 };
            var dx = RandomTensor(dxShape);
            var convolutionBackwardBias = new CpuTensor<float>(biasShape);
            TestAll(new[] { dx, convolutionBackwardBias }, tensors => tensors[0].ConvolutionBackwardBias(tensors[1]));
        }




        [Test]
        public void Test_Compute_Row_Mean_Variance()
        {
            foreach (var unbiasedVariance in new[] { true, false })
            {
                var xShape = new[] { BatchSize, OutChannels, Height, Width };
                var x = RandomTensor(xShape);
                foreach (var rows in new[] { BatchSize, BatchSize * OutChannels, BatchSize * OutChannels * Height })
                {
                    var mean = RandomTensor(new[] { rows, 1});
                    var variance = RandomTensor(mean.Shape);
                    TestAll(new[] { x, mean, variance }, tensors => tensors[0].Compute_Row_Mean_Variance(tensors[1], tensors[2], unbiasedVariance));
                }
            }
        }

        [Test]
        public void Test_numpy_sum()
        {
            foreach (var axis in new[] { 0, 1})
            {
                var xShape = new[] { BatchSize, OutChannels, Height, Width };
                foreach (var rows in new[] { BatchSize, BatchSize * OutChannels, BatchSize * OutChannels * Height })
                {
                    var a = RandomTensor(xShape);
                    var cols = Utils.Product(xShape) / rows;
                    var sum_shape = (axis == 1) ? new[] { rows, 1 } : new[] { 1, cols };
                    var sum_buffer = RandomTensor(sum_shape);
                    TestAll(new[] { a, sum_buffer }, tensors => tensors[0].numpy_sum(tensors[1], axis));
                }
            }
        }

        [Test]
        public void Test_StandardizeInPlace()
        {
            foreach (var unbiasedVariance in new[] { true, false })
            foreach (var epsilon in new[] {0.1f, 0.001f, 1e-8f})
            {
                var xShape = new[] { BatchSize, OutChannels, Height, Width };
                var x = RandomTensor(xShape);
                foreach (var rows in new[] { BatchSize, BatchSize * OutChannels, BatchSize * OutChannels * Height })
                {
                    var row_mean = RandomTensor(new[] { rows, 1 });
                    var row_variance = RandomTensor(row_mean.Shape);
                    x.Compute_Row_Mean_Variance(row_mean, row_variance, unbiasedVariance);
                    TestAll(new[] { x, row_mean, row_variance }, tensors => tensors[0].StandardizeInPlace(tensors[1], tensors[2], 1, epsilon));
                }
            }
        }


        [Test]
        public void TestStandardizeRowsInPlaceBroadcastGammasBetas()
        {
            foreach (var unbiasedVariance in new[] { true, false })
            foreach (var epsilon in new[] { 0.1f, 0.001f, 1e-8f })
            {
                var xShape = new[] { BatchSize, OutChannels, Height, Width };
                var x = RandomTensor(xShape);
                foreach (var rows in new[] { BatchSize, BatchSize * OutChannels, BatchSize * OutChannels * Height })
                {
                    var cols = x.Count / rows;

                    var row_mean = RandomTensor(new[] { rows, 1 });
                    var row_variance = RandomTensor(row_mean.Shape);

                    var col_gammas = RandomTensor(new[] { 1, cols});
                    var col_betas = RandomTensor(col_gammas.Shape);

                    x.Compute_Row_Mean_Variance(row_mean, row_variance, unbiasedVariance);
                    TestAll(new[] { x, row_mean, row_variance, col_gammas, col_betas}, tensors => tensors[0].StandardizeRowsInPlaceBroadcastGammasBetas(tensors[1], tensors[2], epsilon, tensors[3], tensors[4]));
                }
            }
        }

        [Test]
        public void TestConvolutionGradient()
        {
            var memoryPool = new TensorMemoryPool(GpuWrapper);
            foreach (ConvolutionLayer.PADDING_TYPE paddingType in Enum.GetValues(typeof(ConvolutionLayer.PADDING_TYPE)))
            foreach (NetworkSample.CompatibilityModeEnum compatibilityMode in Enum.GetValues(typeof(NetworkSample.CompatibilityModeEnum)))
            foreach (int stride in new[] { 1, 2 })
            foreach (int kernelSize in new[] { 3, 5 })
            foreach (var isDepthwiseConvolution in new[] { true, false })
            {
                const int channelsCount = 3;
                const int height = 17;
                const int width = 32;
                var x = RandomTensor(new[] { BatchSize, channelsCount, height, width });
                x = new CpuTensor<float>(x.Shape);
                var convolutionShape = isDepthwiseConvolution
                    ? new[] { 1, channelsCount, kernelSize, kernelSize }
                    : new[] { 9, channelsCount, kernelSize, kernelSize };
                var convolution = RandomTensor(convolutionShape);
                var dy = RandomTensor(ConvolutionLayer.OutputShape(x.Shape, convolution.Shape, paddingType, stride, isDepthwiseConvolution));
                //this will compute 'dx' && 'convolutionGradient'
                var dx = RandomTensor(x.Shape);
                var convolutionGradient = RandomTensor(convolution.Shape);
                ConvolutionLayer.Padding(x.Shape[2], kernelSize, stride, paddingType, compatibilityMode, out int paddingTop, out int paddingBottom);
                ConvolutionLayer.Padding(x.Shape[3], kernelSize, stride, paddingType, compatibilityMode, out int paddingLeft, out int paddingRight);

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
            foreach (var batchSize in new[] {/* 1,*/ 4, 9 })                //TODO: enable batchSize = 1
            foreach (var momentum in new []{0.0, 0.5, 0.99 /*, 1.0*/ })     //TODO: enable momentum = 1.0
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
            const cudnnBatchNormMode_t mode = cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL;
            var xShape = new[] { 1, 32, 112, 112};
            var x = new CpuTensor<float>(xShape);
            x.ZeroMemory();
            var scale = TestNetworkPropagation.FromNumpyArray("[[[[2.82185745]],[[4.20555544]],[[4.44391775]],[[2.95071363]],[[0.901465356]],[[3.83799005]],[[2.20374274]],[[3.30325413]],[[3.38044739]],[[0.202515125]],[[2.14543128]],[[0.645111859]],[[3.29296565]],[[11.9912415]],[[0.810986161]],[[3.39099979]],[[2.6564517]],[[8.52717972]],[[2.52371788]],[[3.94317198]],[[2.74237108]],[[11.1155062]],[[4.08373785]],[[5.75315952]],[[0.335611582]],[[1.24477983]],[[3.90086651]],[[1.98501635]],[[0.818592787]],[[0.626930952]],[[6.75085163]],[[3.4190371]]]]");
            var bias = TestNetworkPropagation.FromNumpyArray("[[[[-3.74896479]],[[2.43146777]],[[2.31554103]],[[7.13698292]],[[-1.38208234]],[[8.66540337]],[[-2.95346022]],[[1.81856453]],[[0.995381236]],[[0.00296683772]],[[-2.85715914]],[[1.74939632]],[[0.599703848]],[[0.165816754]],[[1.90356266]],[[8.97630692]],[[2.26754451]],[[3.72180033]],[[2.572788]],[[1.96836185]],[[-3.36665225]],[[2.64624929]],[[10.5395947]],[[-10.4322577]],[[-1.63009882]],[[1.37903798]],[[9.95489788]],[[1.99438405]],[[0.159816369]],[[2.50823808]],[[-10.8555698]],[[2.08439994]]]]");
            var runningInputMean = TestNetworkPropagation.FromNumpyArray("[[[[-0.0474244691]],[[-0.00338064576]],[[0.00407501776]],[[0.0787407607]],[[0.0313696824]],[[0.0837314799]],[[-0.0393488146]],[[0.0694158077]],[[0.639113843]],[[-0.171755388]],[[-0.382961541]],[[0.0100561073]],[[0.606002986]],[[1.39727235]],[[0.420819908]],[[-0.0792663917]],[[0.00732345507]],[[-0.770392716]],[[0.00307485089]],[[-0.00288994168]],[[-0.0452340655]],[[-0.719747245]],[[-0.0934633166]],[[0.163005278]],[[0.121294215]],[[-0.00648898305]],[[-0.0706383437]],[[0.00286416081]],[[2.91242941E-09]],[[0.0120399296]],[[-0.063189812]],[[-0.00128901063]]]]");
            var runningInputVariance = TestNetworkPropagation.FromNumpyArray("[[[[7.25111055]],[[5.37058496]],[[6.66747379]],[[18.2757835]],[[5.69575691]],[[17.0573292]],[[6.76594353]],[[1.52835393]],[[18.0554256]],[[27.2328396]],[[10.9577389]],[[3.57627463]],[[12.896986]],[[39.5671387]],[[3.67913604]],[[13.6923494]],[[6.86120129]],[[19.7278404]],[[3.81912017]],[[9.09753227]],[[6.9455328]],[[23.5766983]],[[18.0286465]],[[18.6031551]],[[1.11303592]],[[6.78300667]],[[11.5361662]],[[6.32360983]],[[0]],[[1.08625805]],[[19.3687859]],[[5.1940341]]]]");
            var y = RandomTensor(xShape);
            var meanBuffer = TestNetworkPropagation.FromNumpyArray("[[[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]]]]");
            var varianceBuffer = TestNetworkPropagation.FromNumpyArray("[[[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]],[[31.622776]]]]");
            const double exponentialAverageSmoothingFactor = 1 - 0.99;
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
        public void TestLayerNormalization()
        {
            foreach (var unbiasedVariance in new[] { true, false })
            foreach (var epsilon in new[] { 1e-3f, 1e-5f })
            foreach (var batchSize in new[] { 1, 4, 9 })
            {
                var xShape = new[] { batchSize, ChannelsCount, Height, Width };
                foreach (var cols in new[] { Width, Width * Height, Width * Height * ChannelsCount })
                {
                    var rows = Utils.Product(xShape) / cols;
                    var x = RandomTensor(xShape);
                    var y = RandomTensor(xShape);
                    var gammas = RandomTensor(new[] { 1, cols });
                    var betas = RandomTensor(gammas.Shape);
                    var mean = RandomTensor(new[] { rows, 1 });
                    var variance = RandomTensor(mean.Shape);
                    x.Compute_Row_Mean_Variance(mean, variance, unbiasedVariance);
                    TestAll(new[] { x, y, gammas, betas, mean, variance }, tensors => tensors[0].LayerNormalization(tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], epsilon));
                }
            }
        }

        [Test]
        public void Test_RMSNorm()
        {
            foreach (var epsilon in new[] { 1e-3f, 1e-5f })
            foreach (var batchSize in new[] { 1, 4, 9 })
            {
                var xShape = new[] { batchSize, ChannelsCount, Height, Width };
                foreach (var cols in new[] { Width, Width * Height, Width * Height * ChannelsCount })
                {
                    var rows = Utils.Product(xShape) / cols;
                    var x = RandomTensor(xShape);
                    var y = RandomTensor(xShape);
                    var gammas = RandomTensor(new[] { 1, cols });
                    var mean_squares = RandomTensor(new[] { rows, 1 });
                    x.Compute_Mean_Squares_Buffer(mean_squares);
                    TestAll(new[] { x, y, gammas, mean_squares }, tensors => tensors[0].RMSNormalization(tensors[1], tensors[2], tensors[3], epsilon));
                }
            }
        }


        [Test]
        public void Test_LayerNormalizationBackward()
        {
            foreach (var unbiasedVariance in new[] { true, false })
            foreach (var epsilon in new[] { 1e-3f, 1e-5f })
            foreach (var batchSize in new[] { 1, 4, 9 })
            {
                var xShape = new[] { batchSize, ChannelsCount, Height, Width };
                foreach (var cols in new[] { Width, Width * Height, Width * Height * ChannelsCount })
                {
                    var rows = Utils.Product(xShape) / cols;
                    var x = RandomTensor(xShape);
                    var dy = RandomTensor(xShape);
                    var dx = RandomTensor(xShape);
                    var gammas = RandomTensor(new[] { 1, cols });
                    var mean = RandomTensor(new[] { rows, 1 });
                    var variance = RandomTensor(mean.Shape);

                    var dmean = RandomTensor(mean.Shape);
                    var dvariance = RandomTensor(mean.Shape);

                    x.Compute_Row_Mean_Variance(mean, variance, unbiasedVariance);
                    TestAll(new[] { x, dy, dx, gammas, mean, variance, dmean, dvariance }, tensors => tensors[0].LayerNormalizationBackward(tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], epsilon, tensors[6], tensors[7]), tensorIdsToIgnore: new List<int> { 6, 7 });
                }
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
        public void TestBatchMatrixMultiplication()
        {
            const int batchSize = 3;
            const int maxLength = 12;
            const int dim = 7;
            
            var weights_gradients_buffer = RandomTensor(new[] { batchSize, maxLength, maxLength });
            var V = RandomTensor(new[] { batchSize, maxLength, dim });
            var dy = RandomTensor(V.Shape);
            const float scaling = 0.25f;
            //weights_gradients_buffer.BatchMatrixMultiplication(dy, false, V, true, scaling, 0.0f);
            TestAll(new[] { weights_gradients_buffer, dy, V}, tensors => tensors[0].BatchMatrixMultiplication(tensors[1], false, tensors[2], true, scaling, 0));

            var dQ = RandomTensor(V.Shape);
            var scores_gradients_buffer = RandomTensor(weights_gradients_buffer.Shape);
            var K = RandomTensor(V.Shape);
            //dQ.BatchMatrixMultiplication(scores_gradients_buffer, false, K, false, 1, 0.0f);
            TestAll(new[] { dQ, scores_gradients_buffer, K }, tensors => tensors[0].BatchMatrixMultiplication(tensors[1], false, tensors[2], false, 1, 0));

            var dK = RandomTensor(V.Shape);
            var Q = RandomTensor(V.Shape);
            //dK.BatchMatrixMultiplication(scores_gradients_buffer, true, Q, false, 1, 0.0f);
            TestAll(new[] { dK, scores_gradients_buffer, Q }, tensors => tensors[0].BatchMatrixMultiplication(tensors[1], true, tensors[2], false, 1, 0));
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
        public void TestYOLOV3Forward()
        {
            var anchors = new[] { 116,90, 156, 198, 373, 326};
            var x = RandomTensor(new []{3,255,13,13});
            var y = RandomTensor(new []{x.Shape[0],3* x.Shape[2]* x.Shape[3], x.Shape[1]/3 });
            TestAll(new[] { y, x }, tensors => tensors[0].YOLOV3Forward(tensors[1], 416, 416, anchors));
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
        public void TestWordEmbeddingForwardPropagation()
        {
            const int num_embeddings = 3;
            foreach(var timeSteps in new []{1,100})
            foreach(var inputSize in new[] {1,7})
            foreach(var embedding_dim in new[] {0, 1, 10})
            {
                for (int indexInLastDimensionToUse = 0; indexInLastDimensionToUse<inputSize;++indexInLastDimensionToUse)
                { 
                    var x = XWithRandomWordIndexes3D(BatchSize, timeSteps, num_embeddings, inputSize, indexInLastDimensionToUse);
                    var y = RandomTensor(new[] { BatchSize, timeSteps, inputSize+embedding_dim-1 });
                    var wordEmbedding = RandomTensor(new[] { num_embeddings, embedding_dim });
                    var indexInLastDimensionToUseLocal = indexInLastDimensionToUse;
                    TestAll(new[] { y, x, wordEmbedding }, tensors => tensors[0].WordEmbeddingForwardPropagation(tensors[1], tensors[2], indexInLastDimensionToUseLocal, indexInLastDimensionToUseLocal, indexInLastDimensionToUseLocal, x.Shape[2]-indexInLastDimensionToUseLocal-1));
                }
            }
        }

        [Test]
        public void TestWordEmbeddingBackwardPropagation()
        {
            const int num_embeddings = 50;
            foreach (var timeSteps in new[] { 1, 100 })
            foreach (var inputSize in new[] { 1, 7 })
            foreach (var embedding_dim in new[] { 0, 1, 10 })
            {
                for (int indexInLastDimensionToUse = 0; indexInLastDimensionToUse < inputSize; ++indexInLastDimensionToUse)
                {
                    var dW = RandomTensor(new[] {num_embeddings, embedding_dim});
                    var x = XWithRandomWordIndexes3D(BatchSize, timeSteps, num_embeddings, inputSize, indexInLastDimensionToUse);
                    var dx = RandomTensor(x.Shape);
                    var dy = RandomTensor(new[] { BatchSize, timeSteps, inputSize + embedding_dim - 1 });
                    var indexInLastDimensionToUseCopy = indexInLastDimensionToUse;
                    TestAll(new[] { dW, x, dx, dy }, tensors => tensors[0].WordEmbeddingBackwardPropagation(tensors[1], tensors[2], tensors[3], indexInLastDimensionToUseCopy, indexInLastDimensionToUseCopy, indexInLastDimensionToUseCopy, x.Shape[2]-indexInLastDimensionToUseCopy-1));
                }
            }
        }
        private CpuTensor<float> XWithRandomWordIndexes3D(int batchSize, int timeSteps, int num_embeddings, int inputSize, int indexInLastDimensionToUse)
        {
            var x = RandomTensor(new[] { batchSize, timeSteps, inputSize });
            var xSpan = x.AsFloatCpuSpan;
            var r = new Random(0);
            for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
            {
                for (int timeStep = 0; timeStep < timeSteps; ++timeStep)
                {
                    xSpan[x.Idx(batchIndex, timeStep, indexInLastDimensionToUse)] = 1f + r.Next(num_embeddings - 1);
                }
            }
            return x;
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
        public void TestTranspose()
        {
            foreach (int height in new[] { 1, 2, 5, 10 })
            {
                foreach (int weight in new[] { 1, 2, 5, 10 })
                {
                    var a = RandomTensor(new[] { height, weight});
                    var transposed = RandomTensor(new[] { weight , height });
                    TestAll(new[] { a, transposed }, tensors => tensors[0].Transpose(tensors[1]));
                }
            }
        }


        [Test]
        public void TestSetToZeroAllElementsBelowMainDiagonal()
        {
            foreach (int rows in new[] { 1, 2, 5, 10 })
            {
                foreach (int cols in new[] { 1, 2, 5, 10 })
                {
                    var a = RandomTensor(new[] { rows, cols });
                    TestAll(new[] { a }, tensors => tensors[0].SetToZeroAllElementsBelowMainDiagonal());
                }
            }
        }

        [Test]
        public void TestSetAllElementsAboveMainDiagonal_2D()
        {
            foreach (int rows in new[] { 1, 2, 5, 10 })
            foreach (int cols in new[] { 1, 2, 5, 10 })
            foreach (var valueForElementsAboveMainDiagonal in new[] { 0f, 1.12345f, -1000000000 })
            {
                var a = RandomTensor(new[] { rows, cols });
                TestAll(new[] { a }, tensors => tensors[0].SetAllElementsAboveMainDiagonal(valueForElementsAboveMainDiagonal));
            }
        }
        
        [Test]
        public void TestSetAllElementsAboveMainDiagonal_3D()
        {
            foreach (int matrices_count in new[] { 1, 2, 5, 10 })
            foreach (int rows in new[] { 1, 2, 5, 10 })
            foreach (int cols in new[] { 1, 2, 5, 10 })
            foreach (var valueForElementsAboveMainDiagonal in new[] { 0f, 1.12345f, -1000000000})
            {
                var a = RandomTensor(new[] { matrices_count, rows, cols });
                TestAll(new[] { a }, tensors => tensors[0].SetAllElementsAboveMainDiagonal(valueForElementsAboveMainDiagonal));
            }
        }

        [Test]
        public void TestSetIdentityMatrix()
        {
            foreach (int height in new[] { 1, 2, 5, 10 })
            {
                var a = RandomTensor(new[] { height, height});
                TestAll(new[] { a }, tensors => tensors[0].SetIdentityMatrix());
            }
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

        [TestCase(new[] { 1 })]
        [TestCase(new[] { 10000 })]
        [TestCase(new[] { BatchSize})]
        [TestCase(new[] { 1024,1024})]
        [TestCase(new[] { 1024,1})]
        [TestCase(new[] { 1,1024})]
        [TestCase(new[] { BatchSize, ChannelsCount, 1, 1 })]
        public void TestClip(int[] shape)
        {
            var result = TestCpuTensor.RandomFloatTensor(shape, _rand, -1.5, +1.5);
            TestAll(new[] { result}, tensors => tensors[0].Clip(-0.5f, 0.7f));
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
        public void TestLinearFunction()
        {
            var y = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            var x = RandomTensor(y.Shape);
            TestAll(new[] { y, x }, tensors => tensors[0].LinearFunction(0.5f, tensors[1], 0.75f));
        }

        [Test]
        public void TestLinearFunctionWithTensors()
        {
            var y = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width });
            var x = RandomTensor(y.Shape);
            var beta = RandomTensor(y.Shape);
            var alpha = RandomTensor(y.Shape);
            TestAll(new[] { y, beta, x, alpha }, tensors => tensors[0].LinearFunction(tensors[1], tensors[2], tensors[3]));
        }


        [Test]
        public void TestConcatenate2Tensors()
        {
            var x1 = RandomTensor(new[] { BatchSize, 17, Height, Width });
            var x2 = RandomTensor(new[] { BatchSize, 13, Height, Width });
            var concat = RandomTensor(new[] { BatchSize, 17 + 13, Height, Width });
            TestAll(new[] { concat, x1, x2 }, tensors => tensors[0].Concatenate(new[]{ tensors[1], tensors[2]}));
        }

        [Test]
        public void TestConcatenate3Tensors()
        {
            var x1 = RandomTensor(new[] { BatchSize, 17, Height, Width });
            var x2 = RandomTensor(new[] { BatchSize, 13, Height, Width });
            var x3 = RandomTensor(new[] { BatchSize, 10, Height, Width });
            var concat = RandomTensor(new[] { BatchSize, 17+13+10, Height, Width });
            TestAll(new[] { concat, x1, x2, x3 }, tensors => tensors[0].Concatenate(new[] { tensors[1], tensors[2], tensors[3] }));
        }

        [Test]
        public void TestSplit2Tensors()
        {
            var toSplit = RandomTensor(new[] { BatchSize, 17 + 13, Height, Width });
            var x1 = RandomTensor(new[] { BatchSize, 17, Height, Width });
            var x2 = RandomTensor(new[] { BatchSize, 13, Height, Width });
            TestAll(new[] { toSplit, x1, x2 }, tensors => tensors[0].Split( new[]{tensors[1], tensors[2]}));
        }

        [Test]
        public void TestSplit3Tensors()
        {
            var toSplit = RandomTensor(new[] { BatchSize, 17 + 13 + 10, Height, Width });
            var x1 = RandomTensor(new[] { BatchSize, 17, Height, Width });
            var x2 = RandomTensor(new[] { BatchSize, 13, Height, Width });
            var x3 = RandomTensor(new[] { BatchSize, 10, Height, Width });
            TestAll(new[] { toSplit, x1, x2, x3 }, tensors => tensors[0].Split(new[] { tensors[1], tensors[2], tensors[3] }));
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
	        var dy = RandomTensor(new[] { BatchSize, Nx });
	        var dBias_1D = RandomTensor(new[] { Nx});
	        TestAll(new[] { dy, dBias_1D }, tensors => tensors[0].Compute_BiasGradient_from_dy(tensors[1]));
	    }
	    [Test]
	    public void TestBroadcastConvolutionBiasToOutput()
	    {
            var convolutionBias = RandomTensor(new[] { 1, OutChannels, 1, 1});
            var y = RandomTensor(new[] { BatchSize, OutChannels, Height, Width});
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
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_LN, 0)]
	    public void TestActivationForward(cudnnActivationMode_t activationMode, double alphaActivation)
	    {
            foreach (var shape in new[]
                     {
                         new[] { BatchSize, ChannelsCount, Height, Width },
                         new[] { BatchSize, 1, 1, Width },
                         new[] { BatchSize, ChannelsCount, Height},
                         new[] { BatchSize, 1, Height},
                         new[] { BatchSize, ChannelsCount },
                         new[] { BatchSize, 1 }
                     })
            {
                var x = RandomTensor(shape);
	            var y = RandomTensor(shape);
                var alphaActivationAsTensor = Tensor.SingleFloat((float) alphaActivation);
                TestAll(new[] { x, alphaActivationAsTensor, y }, tensors => tensors[0].ActivationForward(activationMode, tensors[1], tensors[2]));
	        }
	    }



        [Test]
        public void TestActivationForwardSoftmaxWithHierarchyActivation()
        {
            var x = TestCpuTensor.GetPredictedCategoricalCrossentropyWithHierarchy();
            var rootPrediction = TestCategoryHierarchy.StarExample().RootPrediction();
            var activationTensor = new CpuTensor<float>(new[] { rootPrediction.Length }, rootPrediction);
            var y = RandomTensor(x.Shape);
            TestAll(new[] { x, activationTensor, y }, tensors => tensors[0].ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY, tensors[1], tensors[2]));
        }

        [Test, Explicit]
        public void TestActivationForwardSoftmaxWithHierarchyActivationV2()
        {
            var x = (CpuTensor<float>)TensorExtensions.FromNumpyArray(File.ReadAllText( Path.Combine(NetworkSample.DefaultDataDirectory, "NonReg", "TestActivationForwardSoftmaxWithHierarchyActivationV2_X.txt")));
            var activationTensor = (CpuTensor<float>)TensorExtensions.FromNumpyArray(File.ReadAllText( Path.Combine(NetworkSample.DefaultDataDirectory, "NonReg", "TestActivationForwardSoftmaxWithHierarchyActivationV2_ActivationTensor.txt")));
            var y = RandomTensor(x.Shape);
            TestAll(new[] { x, activationTensor, y }, tensors => tensors[0].ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY, tensors[1], tensors[2]));
        }

        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_TANH, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU, 0.1)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU, 0.3)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_ELU, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH, 0)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_LN, 0)]
	    public void TestActivationBackward(cudnnActivationMode_t activationMode, double alphaActivation)
	    {
            foreach(var shape in new[]
                    {
                        new[] { BatchSize, ChannelsCount, Height, Width },
                        new[] { BatchSize, ChannelsCount, Height},
                        new[] { BatchSize, ChannelsCount }
                    })
            {
                var y = RandomTensor(shape);
                var dy = RandomTensor(shape);
                var x = RandomTensor(shape);
                var dx = RandomTensor(shape);
                var activationParameter = Tensor.SingleFloat((float)alphaActivation);
                x.ActivationForward(activationMode, activationParameter, y);
                TestAll(new[] { dx, dy, x, y }, tensors => tensors[0].ActivationBackward(activationMode, activationParameter, tensors[1], tensors[2], tensors[3]));
	        }
	    }

        [Test]
        public void TestSoftmaxActivationBackward()
        {
            var y = RandomTensor(new[] { 1, 3});
            var dy = RandomTensor(y.Shape);
            //dy = new CpuTensor<float>(dy.Shape, new[] { 0, 100, 10f });
            var x = RandomTensor(y.Shape);
            var dx = RandomTensor(y.Shape);
            const cudnnActivationMode_t activationMode = cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX;
            x.ActivationForward(activationMode, null, y);
            TestAll(new[] { dx, dy,x,  y }, tensors => tensors[0].ActivationBackward(activationMode, null, tensors[1], tensors[2], tensors[3]));
        }

        [Test]
        public void TestActivationBackwardSoftmaxWithHierarchyActivation()
        {
            var x = TestCpuTensor.GetPredictedCategoricalCrossentropyWithHierarchy();
            var rootPrediction = TestCategoryHierarchy.StarExample().RootPrediction();
            var activationParameter = new CpuTensor<float>(new[] { rootPrediction.Length }, rootPrediction);
            var y = new CpuTensor<float>(x.Shape);
            x.ActivationForward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY, activationParameter, y);
            var dy = RandomTensor(y.Shape);
            var dx = RandomTensor(y.Shape);
            TestAll(new[] { dx, activationParameter, dy, x, y }, tensors => tensors[0].ActivationBackward(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY, tensors[1], tensors[2], tensors[3], tensors[4]));
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
                    foreach (var shapeBeforePooling in new[] { new[] { BatchSize, 1, Height, Width }, new[] { BatchSize, ChannelsCount, Height, Width } })
                    { 
	                    var aBeforePooling = RandomTensor(shapeBeforePooling);
	                    var aAfterPooling = RandomTensor(PoolingLayer.PoolingOutputShape(aBeforePooling.Shape, poolingHeight, poolingWidth, stride, stride));
	                    TestAll(new[] { aBeforePooling, aAfterPooling }, tensors => tensors[0].Pooling(tensors[1], poolingMode, poolingHeight, poolingWidth, stride, stride));
                    }
                }
            }
	    }

        [Test]
        public void TestGlobalAveragePooling()
        {
            const cudnnPoolingMode_t poolingMode = cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            const int poolingHeight = Height;
            const int poolingWidth = Width;
            const int verticalStride = Height;
            const int horizontalStride = Width;

            foreach (var shapeBeforePooling in new[] { new[] { BatchSize, 1, Height, Width }, new[] { BatchSize, ChannelsCount, Height, Width } })
            {
                var aBeforePooling = RandomTensor(shapeBeforePooling);
                var aAfterPooling = RandomTensor(PoolingLayer.PoolingOutputShape(aBeforePooling.Shape, poolingHeight, poolingWidth, verticalStride, horizontalStride));
                TestAll(new[] { aBeforePooling, aAfterPooling }, tensors => tensors[0].Pooling(tensors[1], poolingMode, poolingHeight, poolingWidth, verticalStride, horizontalStride));
            }
        }

        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC)]
        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)]
	    public void TestPoolingGradient(cudnnPoolingMode_t poolingMode)
        {
            foreach (var poolingSize in new[] {2,4})
            { 
                foreach(var shapeBeforePooling in new[] { new[] { BatchSize, 1, Height, Width }, new[] { BatchSize, ChannelsCount, Height, Width } })
                { 
	                var shapeAfterPooling = PoolingLayer.PoolingOutputShape(shapeBeforePooling, poolingSize, poolingSize, poolingSize, poolingSize);
	                var dy = RandomTensor(shapeAfterPooling);
	                var x = RandomTensor(shapeBeforePooling);
	                var y = RandomTensor(shapeAfterPooling);
                    x.Pooling(y, poolingMode, poolingSize, poolingSize, poolingSize, poolingSize);
                    var dx = RandomTensor(shapeBeforePooling);
	                TestAll(new[] { dy, y, x, dx }, tensors => tensors[0].PoolingGradient(tensors[1], tensors[2], tensors[3], poolingMode, poolingSize, poolingSize, poolingSize, poolingSize));
	            }
            }
        }

        [Test]
        public void TestGlobalAveragePoolingGradient()
        {
            const cudnnPoolingMode_t poolingMode = cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
            const int poolingHeight = Height;
            const int poolingWidth = Width;
            int poolingStride = Math.Max(Height, Width);
            foreach (var shapeBeforePooling in new[] { new[] { BatchSize, 1, Height, Width }, new[] { BatchSize, ChannelsCount, Height, Width } })
            { 
                var shapeAfterPooling = PoolingLayer.PoolingOutputShape(shapeBeforePooling, poolingHeight, poolingWidth, poolingStride, poolingStride);
                var dy = RandomTensor(shapeAfterPooling);
                var x = RandomTensor(shapeBeforePooling);
                var y = RandomTensor(shapeAfterPooling);
                x.Pooling(y, poolingMode, poolingHeight, poolingWidth, poolingStride, poolingStride);
                var dx = RandomTensor(shapeBeforePooling);
                TestAll(new[] { dy, y, x, dx }, tensors => tensors[0].PoolingGradient(tensors[1], tensors[2], tensors[3], poolingMode, poolingHeight, poolingWidth, poolingStride, poolingStride));
            }
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
            TestAll(new[] { W, dW, adam_vW, adam_sW}, tensors => tensors[0].UpdateAdamOptimizer(0.001, 0.9, 0.999,1e-8, 0.0, tensors[1], tensors[2], tensors[3], 5));
        }

        [Test]
        public void TestUpdateAdamWOptimizer()
        {
            const double weight_decay = 0.0005;
            var W = RandomTensor(new[] { Width, Height });
            var dW = RandomTensor(W.Shape);
            var adam_vW = RandomTensor(W.Shape);
            var adam_sW = RandomTensor(W.Shape);
            TestAll(new[] { W, dW, adam_vW, adam_sW }, tensors => tensors[0].UpdateAdamOptimizer(0.001, 0.9, 0.999, 1e-8, weight_decay, tensors[1], tensors[2], tensors[3], 5));
        }

        [TestCase(0.1, 0.9, 0.0, true)]
        [TestCase(0.1, 0.9, 0.0, false)]
        [TestCase(0.1, 0.9, 1e-4, true)]
        [TestCase(0.1, 0.9, 1e-4, false)]
        public void TestUpdateSGDOptimizer(double learningRate, double momentum, double decay, bool nesterov)
        {
            var W = RandomTensor(new[] { Width, Height });
            var dW = RandomTensor(W.Shape);
            var velocity = RandomTensor(W.Shape);
            TestAll(new[] { W, dW, velocity }, tensors => tensors[0].UpdateSGDOptimizer(learningRate, momentum, nesterov, tensors[1], tensors[2]));
        }

        [TestCase(1, EvaluationMetricEnum.Huber)]
        [TestCase(1, EvaluationMetricEnum.Mse)]
        [TestCase(1, EvaluationMetricEnum.Rmse)]
        [TestCase(1, EvaluationMetricEnum.Mae)]
        [TestCase(2, EvaluationMetricEnum.Huber)]
        [TestCase(2, EvaluationMetricEnum.Mse)]
        [TestCase(2, EvaluationMetricEnum.Rmse)]
        [TestCase(2, EvaluationMetricEnum.Mae)]
        [TestCase(10, EvaluationMetricEnum.Huber)]
        [TestCase(10, EvaluationMetricEnum.Mse)]
        [TestCase(10, EvaluationMetricEnum.Rmse)]
        [TestCase(10, EvaluationMetricEnum.Mae)]
        public void TestComputeEvaluationMetricRegression(int numClass, EvaluationMetricEnum evaluationMetric)
        {
            const int batchSize = 1000;
            var buffer = RandomTensor(new[] { batchSize });
            var yPredicted = RandomTensor(new[] { batchSize, numClass });
            var yExpected = RandomTensor(yPredicted.Shape);
            var metricConfig  = new TestMetricConfig(huber_Delta:1.0f);
            TestAllForReturnValue(new[] { buffer, yExpected, yPredicted}, tensors => tensors[0].ComputeEvaluationMetric(tensors[1], tensors[2], evaluationMetric, metricConfig));
        }

        [TestCase(1, EvaluationMetricEnum.CategoricalCrossentropy)]
        [TestCase(1, EvaluationMetricEnum.BinaryCrossentropy)]
        [TestCase(1, EvaluationMetricEnum.BCEContinuousY)]
        [TestCase(1, EvaluationMetricEnum.BCEWithFocalLoss)]
        [TestCase(2, EvaluationMetricEnum.CategoricalCrossentropy)]
        [TestCase(2, EvaluationMetricEnum.BinaryCrossentropy)]
        [TestCase(2, EvaluationMetricEnum.BCEContinuousY)]
        [TestCase(2, EvaluationMetricEnum.BCEWithFocalLoss)]
        [TestCase(10, EvaluationMetricEnum.CategoricalCrossentropy)]
        public void TestComputeEvaluationMetricCategorical(int numClass, EvaluationMetricEnum evaluationMetric)
        {
            const int rows = 317;
            var metricConfig = new TestMetricConfig(bceWithFocalLossPercentageInTrueClass: 0.43f, bceWithFocalLoss_Gamma: 1.5f);

            var buffer = RandomTensor(new[] { rows });
            var yExpected = TestCpuTensor.RandomOneHotTensor(new[] { rows, numClass }, _rand);
            var yPredicted = TestCpuTensor.RandomFloatTensor(yExpected.Shape, _rand, 0, 1);
            TestAllForReturnValue(new[] { buffer, yExpected, yPredicted }, tensors => tensors[0].ComputeEvaluationMetric(tensors[1], tensors[2], evaluationMetric, metricConfig));

            buffer = RandomTensor(new[] { rows });
            yExpected = TestCpuTensor.RandomOneHotTensor(new[] { rows, numClass }, _rand);
            yPredicted = TestCpuTensor.RandomFloatTensor(yExpected.Shape, _rand, 0, 1);
            TestAll(new[] { buffer, yExpected, yPredicted }, tensors => tensors[0].ComputeLossBufferForEvaluationMetric(tensors[1], tensors[2], evaluationMetric, metricConfig));

            yExpected = TestCpuTensor.RandomOneHotTensor(new[] { rows, numClass }, _rand);
            yPredicted = TestCpuTensor.RandomFloatTensor(yExpected.Shape, _rand, 0, 1);
            var gradients = RandomTensor(yExpected.Shape);
            TestAll(new[] { gradients, yExpected, yPredicted }, tensors => tensors[0].ComputeGradientForEvaluationMetric(tensors[1], tensors[2], evaluationMetric, metricConfig));
        }


        [TestCase(1, EvaluationMetricEnum.BCEContinuousY)]
        [TestCase(2, EvaluationMetricEnum.BCEContinuousY)]
        [TestCase(1, EvaluationMetricEnum.BCEWithFocalLoss)]
        [TestCase(2, EvaluationMetricEnum.BCEWithFocalLoss)]
        public void TestComputeEvaluationMetricContinuousYBetween0And1(int numClass, EvaluationMetricEnum evaluationMetric)
        {
            const int rows = 317;
            var metricConfig = new TestMetricConfig(bceWithFocalLossPercentageInTrueClass: 0.43f, bceWithFocalLoss_Gamma: 1.5f);

            var buffer = RandomTensor(new[] { rows });
            var yExpected = TestCpuTensor.RandomFloatTensor(new [] { rows, numClass }, _rand, 0, 1);
            var yPredicted = TestCpuTensor.RandomFloatTensor(yExpected.Shape, _rand, 0, 1);
            TestAllForReturnValue(new[] { buffer, yExpected, yPredicted }, tensors => tensors[0].ComputeEvaluationMetric(tensors[1], tensors[2], evaluationMetric, metricConfig));

            buffer = RandomTensor(new[] { rows });
            yExpected = TestCpuTensor.RandomFloatTensor(new[] { rows, numClass }, _rand, 0, 1);
            yPredicted = TestCpuTensor.RandomFloatTensor(yExpected.Shape, _rand, 0, 1);
            TestAll(new[] { buffer, yExpected, yPredicted }, tensors => tensors[0].ComputeLossBufferForEvaluationMetric(tensors[1], tensors[2], evaluationMetric, metricConfig));

            yExpected = TestCpuTensor.RandomOneHotTensor(new[] { rows, numClass }, _rand);
            yPredicted = TestCpuTensor.RandomFloatTensor(yExpected.Shape, _rand, 0, 1);
            var gradients = RandomTensor(yExpected.Shape);
            TestAll(new[] { gradients, yExpected, yPredicted }, tensors => tensors[0].ComputeGradientForEvaluationMetric(tensors[1], tensors[2], evaluationMetric, metricConfig));
        }


        [Test]
        public void TestComputeCategoricalCrossentropyWithHierarchyEvaluationMetric()
        {
            var expected = TestCpuTensor.GetExpectedCategoricalCrossentropyWithHierarchy();
            var predicted = TestCpuTensor.GetPredictedCategoricalCrossentropyWithHierarchy();
            var buffer = RandomTensor(new[] { expected.Shape[0] });
            TestAllForReturnValue(new[] { buffer, expected, predicted}, tensors => tensors[0].ComputeEvaluationMetric(tensors[1], tensors[2], EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy, null));
        }

        [Test]
        public void TestCategoricalCrossentropyWithHierarchyGradient()
        {
            var expected = TestCpuTensor.GetExpectedCategoricalCrossentropyWithHierarchy();
            var predicted = TestCpuTensor.GetPredictedCategoricalCrossentropyWithHierarchy();
            var loss = RandomTensor(expected.Shape);
            TestAll(new[] { loss, expected, predicted}, tensors => tensors[0].CategoricalCrossentropyWithHierarchyGradient(tensors[1], tensors[2]));
        }


        [TestCase(1, 1)]
        [TestCase(50, 1)]
        [TestCase(1, 50)]
        [TestCase(50, 504)]
        public void TestCosineSimilarityLoss(int itemCount, int timeSeriesLength)
        {
            var shape = new [] { itemCount * timeSeriesLength };
            var predicted = RandomTensor(shape);
            var expected = RandomTensor(predicted.Shape);
            var cosineSimilarityLoss = RandomTensor(new[] { timeSeriesLength });
            TestAll(new[] { cosineSimilarityLoss, expected, predicted }, tensors => tensors[0].CosineSimilarityLossBuffer(tensors[1], tensors[2], timeSeriesLength));
        }

        [TestCase(1, 1)]
        [TestCase(50, 1)]
        [TestCase(1, 50)]
        [TestCase(50, 504)]

        public void TestCosineSimilarityGradient(int itemCount, int timeSeriesLength)
        {
            var shape = new [] { itemCount * timeSeriesLength };
            var predicted = RandomTensor(shape);
            var expected = RandomTensor(predicted.Shape);
            var cosineSimilarityGradient = RandomTensor(expected.Shape);
            TestAll(new[] { cosineSimilarityGradient, expected, predicted }, tensors => tensors[0].CosineSimilarityGradient(tensors[1], tensors[2], timeSeriesLength));
        }


        [TestCase(new[] { 10000, 10 })]
        [TestCase(new[] { 10000, 1 })]
        [TestCase(new[] { 5000, 3, 10 })]
        [TestCase(new[] { 5000, 3, 1 })]
        [TestCase(new[] { 3000, 3, 2, 10 })]
        [TestCase(new[] { 3000, 3, 2, 1 })]
        public void TestHuberLoss(int[] shape)
        {
            var predicted = RandomTensor(shape);
            var expected = RandomTensor(predicted.Shape);
            var batchSize = shape[0];
            var huberLoss = RandomTensor(new []{ batchSize });
            const float huberDelta= 1.0f;
            TestAll(new[] { huberLoss, expected, predicted }, tensors => tensors[0].HuberLossBuffer(tensors[1], tensors[2], huberDelta));
        }

        [TestCase(new[]{10000,10})]
        [TestCase(new[]{10000,1})]
        [TestCase(new[] { 5000, 3, 10 })]
        [TestCase(new[] { 5000, 3, 1 })]
        [TestCase(new[]{3000,3,2,10})]
        [TestCase(new[]{3000,3,2,1})]
        public void TestHuberGradient(int[] shape)
        {
            var predicted = RandomTensor(shape);
            var expected = RandomTensor(predicted.Shape);
            var huberGradient = RandomTensor(expected.Shape);
            const float huberDelta = 1.0f;
            TestAll(new[] { huberGradient, expected, predicted}, tensors => tensors[0].HuberGradient(tensors[1], tensors[2], huberDelta));
        }

        //[TestCase(new[] { 10000, 1 })]
        //[TestCase(new[] { 10000, 10 })]
        //[TestCase(new[] { 5000, 3, 10 })]
        //[TestCase(new[] { 5000, 3, 1 })]
        //[TestCase(new[] { 3000, 3, 2, 10 })]
        //[TestCase(new[] { 3000, 3, 2, 1 })]
        //public void TestMseLoss(int[] shape)
        //{
        //    var predicted = RandomTensor(shape);
        //    var expected = RandomTensor(predicted.Shape);
        //    var batchSize = shape[0];
        //    var huberLoss = RandomTensor(new[] { batchSize });
        //    TestAll(new[] { huberLoss, expected, predicted }, tensors => tensors[0].MseLoss(tensors[1], tensors[2]));
        //}

        [TestCase(new[] { 10000, 1 })]
        [TestCase(new[] { 10000, 10 })]
        [TestCase(new[] { 5000, 3, 10 })]
        [TestCase(new[] { 5000, 3, 1 })]
        [TestCase(new[] { 3000, 3, 2, 10 })]
        [TestCase(new[] { 3000, 3, 2, 1 })]
        public void TestMseGradient(int[] shape)
        {
            var predicted = RandomTensor(shape);
            var expected = RandomTensor(predicted.Shape);
            var mseGradient = RandomTensor(expected.Shape);
            TestAll(new[] { mseGradient, expected, predicted }, tensors => tensors[0].MseGradient(tensors[1], tensors[2]));
        }

        [Test]
        public void TestComputeSparseAccuracy_2D()
        {
            foreach (var batchSize in new[] { 1, 32 })
            foreach (var numClass in new[] { 1, 10 })
            {
                var yPredicted = TestCpuTensor.RandomFloatTensor(new[] { batchSize, numClass }, _rand, 0.0, 1.0);
                var yExpectedSparse = RandomTensorWithIndexes(new[] { batchSize, 1 }, 0, numClass - 1);
                var buffer = RandomTensor(new[] { yExpectedSparse.Count });
                TestAllForReturnValue(new[] { buffer, yExpectedSparse, yPredicted }, tensors => tensors[0].ComputeSparseAccuracy(tensors[1], tensors[2]));
            }
        }

        [Test]
        public void TestAUC()
        {
            var y_true = TestCpuTensor.RandomIntValuesTensor(new[] { 128, 1 }, _rand, 0, 2);
            var y_pred = TestCpuTensor.RandomFloatTensor(y_true.Shape, _rand, 0, 1.0);
            var buffer = RandomTensor(new[] {1});
            TestAllForReturnValue(new[] { buffer, y_true, y_pred }, tensors => tensors[0].ComputeAUC(tensors[1], tensors[2]));
        }


        [Test]
        public void TestAveragePrecisionScore()
        {
            var y_true = TestCpuTensor.RandomIntValuesTensor(new[] { 128, 1 }, _rand, 0, 2);
            var y_pred = TestCpuTensor.RandomFloatTensor(y_true.Shape, _rand, 0, 1.0);
            var buffer = RandomTensor(new[] { 1 });
            TestAllForReturnValue(new[] { buffer, y_true, y_pred }, tensors => tensors[0].ComputeAveragePrecisionScore(tensors[1], tensors[2]));
        }


        [Test]
        public void TestComputeSparseAccuracy_3D()
        {
            foreach (var batchSize in new[] { 1, 32 })
            foreach (var timeSteps in new[] { 1, 10 })
            foreach (var numClass in new[] { 1, 10 })
            {
               var yPredicted = TestCpuTensor.RandomFloatTensor(new[] { batchSize, timeSteps, numClass }, _rand, 0.0, 1.0);
                var yExpectedSparse = RandomTensorWithIndexes(new[] { batchSize, timeSteps }, 0, numClass - 1);
                var buffer = RandomTensor(new[] { yExpectedSparse.Count });
                TestAllForReturnValue(new[] { buffer, yExpectedSparse, yPredicted }, tensors => tensors[0].ComputeSparseAccuracy(tensors[1], tensors[2]));
            }
        }

        [Test]
        public void TestArgMax()
        {
            foreach (var shape in new[]
                     {
                         new[]{1, 10 },
                         new[]{7, 5 },
                         new[]{7, 5, 1 },
                         new[]{7, 5, 3 },
                         new[]{2, 5, 3,11 },
                     })
            {
                var yPredicted = RandomTensor(shape);
                var bufferShape = (int[])shape.Clone();
                bufferShape[^1] = 1;
                var buffer = RandomTensor(bufferShape);
                TestAll(new[] { yPredicted, buffer }, tensors => tensors[0].ArgMax(tensors[1]));
            }
        }

        [Test]
        public void TestComputeSparseCategoricalCrossentropyLoss_2D()
        {
            foreach (var batchSize in new[] { 1, 32 })
            foreach (var numClass in new[] { 1, 10 })
            {
                var buffer = RandomTensor(new[] { batchSize });
                var yExpectedSparse = RandomTensorWithIndexes(new[] { batchSize, 1 }, 0, numClass - 1);
                var yPredicted = TestCpuTensor.RandomFloatTensor(new[] { batchSize, numClass }, _rand, 0.0, 1.0);
                TestAllForReturnValue(new[] { buffer, yExpectedSparse, yPredicted }, tensors => tensors[0].SparseCategoricalCrossentropyLoss(tensors[1], tensors[2]));
            }
        }


        [Test]
        public void TestComputeSparseCategoricalCrossentropyLoss_3D()
        {
            foreach (var batchSize in new[] { 1, 16 })
            foreach (var timeSteps in new[] { 1, 10 })
            foreach (var numClass in new[] { 1, 5 })
            {
                var buffer = RandomTensor(new[] { batchSize * timeSteps });
                var yExpectedSparse = RandomTensorWithIndexes(new[] { batchSize, timeSteps }, 0, numClass - 1);
                var yPredicted = TestCpuTensor.RandomFloatTensor(new[] { batchSize, timeSteps, numClass }, _rand, 0.0, 1.0);
                TestAllForReturnValue(new[] { buffer, yExpectedSparse, yPredicted }, tensors => tensors[0].SparseCategoricalCrossentropyLoss(tensors[1], tensors[2]));
            }
        }


        [Test]
        public void TestSparseCategoricalCrossentropyGradient_2D()
        {
            foreach(var batchSize in new[]{1,32})
            foreach(var numClass in new[]{1,10})
            {
                var predicted = RandomTensor(new []{batchSize,numClass});
                var gradients = RandomTensor(predicted.Shape);
                var expectedSparse = RandomTensorWithIndexes(new[] { batchSize, 1 }, 0, numClass-1);
                TestAll(new[] { gradients, expectedSparse, predicted }, tensors => tensors[0].SparseCategoricalCrossentropyGradient(tensors[1], tensors[2]));
            }
        }
        [Test]
        public void TestSparseCategoricalCrossentropyGradient_3D()
        {
            foreach (var batchSize in new[] { 1, 16 })
            foreach (var timeSteps in new[] { 1, 10})
            foreach (var numClass in new[] { 1, 5})
            {
                var predicted = RandomTensor(new[] { batchSize, timeSteps, numClass });
                var gradients = RandomTensor(predicted.Shape);
                var expectedSparse = RandomTensorWithIndexes(new[] { batchSize, timeSteps }, 0, numClass - 1);
                TestAll(new[] { gradients, expectedSparse, predicted }, tensors => tensors[0].SparseCategoricalCrossentropyGradient(tensors[1], tensors[2]));
            }
        }

        [TestCase(new[] { 10000, 1 })]
        [TestCase(new[] { 10000, 10 })]
        [TestCase(new[] { 5000, 3, 10 })]
        [TestCase(new[] { 5000, 3, 1 })]
        [TestCase(new[] { 3000, 3, 2, 10 })]
        [TestCase(new[] { 3000, 3, 2, 1 })]
        public void TestMseOfLogLoss(int[] shape)
        {
            var predicted = RandomTensor(shape);
            //all expected values must be strictly > 0
            var expected = TestCpuTensor.RandomFloatTensor(predicted.Shape, _rand, 0.00001, +1.5);
            var batchSize = shape[0];
            var mseOfLogLoss = RandomTensor(new[] { batchSize });
            var metricConfig = new TestMetricConfig(mseOfLog_Epsilon: 0.0123f);
            TestAll(new[] { mseOfLogLoss, expected, predicted }, tensors => tensors[0].ComputeEvaluationMetric(tensors[1], tensors[2], EvaluationMetricEnum.MseOfLog, metricConfig));
        }

        [TestCase(new[] { 10000, 1 })]
        [TestCase(new[] { 10000, 10 })]
        [TestCase(new[] { 5000, 3, 10 })]
        [TestCase(new[] { 5000, 3, 1 })]
        [TestCase(new[] { 3000, 3, 2, 10 })]
        [TestCase(new[] { 3000, 3, 2, 1 })]
        public void TestMseOfLogGradient(int[] shape)
        {
            var predicted = RandomTensor(shape);
            //all expected values must be strictly > 0
            var expected = TestCpuTensor.RandomFloatTensor(predicted.Shape, _rand, 0.00001, +1.5);
            var mseOfLogGradient = RandomTensor(expected.Shape);
            const float epsilon = 0.001f;
            TestAll(new[] { mseOfLogGradient, expected, predicted }, tensors => tensors[0].MseOfLogGradient(tensors[1], tensors[2], epsilon));
        }

        [TestCase(new[] { 10000, 1 })]
        [TestCase(new[] { 10000, 10 })]
        [TestCase(new[] { 5000, 3, 10 })]
        [TestCase(new[] { 5000, 3, 1 })]
        [TestCase(new[] { 3000, 3, 2, 10 })]
        [TestCase(new[] { 3000, 3, 2, 1 })]
        public void TestMaeGradient(int[] shape)
        {
            var predicted = RandomTensor(shape);
            var expected = RandomTensor(predicted.Shape);
            var huberGradient = RandomTensor(expected.Shape);
            TestAll(new[] { huberGradient, expected, predicted }, tensors => tensors[0].MaeGradient(tensors[1], tensors[2]));
        }

        [TestCase(new[] { 1000, 10 })]
        [TestCase(new[] { 1000, 1 })]
        [TestCase(new[] { 500, 3, 10 })]
        [TestCase(new[] { 500, 3, 1 })]
        [TestCase(new[] { 300, 3, 2, 10 })]
        [TestCase(new[] { 300, 3, 2, 1 })]
        public void TesSwitch_First_2_axis(int[] shape)
        {
            var src = RandomTensor(shape);
            var target = RandomTensor(shape);
            TestAll(new[] {src, target}, tensors => tensors[0].Switch_First_2_axis(tensors[1]));
        }


        [TestCase(new[] { 2, 3,7}, new[]{ 2, 7, 3 })]
        [TestCase(new[] { 2, 3,7,1}, new[]{ 2, 7, 3 })]
        [TestCase(new[] { 2, 3,7,1}, new[]{ 2,7,3, 1})]
        [TestCase(new[] { 2, 3,7}, new[]{2,7,3,1})]
        public void TestSwitchSecondAndThirdDimension(int[] srcShape, int[] targetShape)
        {
            var src = RandomTensor(srcShape);
            var target = RandomTensor(targetShape);
            TestAll(new[] { src, target }, tensors => tensors[0].SwitchSecondAndThirdDimension(tensors[1]));
        }

        [Test]
        public void Test_UpdateWithPositionalEncoding_AttnIsAllYouNeed()
        {
            foreach (var n in new[] { 1, 100, 10000 })
            {
                var src = RandomTensor(new[] { 7, 13, 17 });
                TestAll(new[] { src}, tensors => tensors[0].UpdateWithPositionalEncoding_AttnIsAllYouNeed(n));
            }
        }

        [Test]
        public void TestTransposeSecondAndThirdDimension()
        {
            foreach(var a in new[]{1,2,3})
            foreach(var b in new[]{1,2,3})
            foreach(var c in new[]{1,2,3})
            foreach(var d in new[]{1,2,3})
            {
                foreach (var length in new[] { 3, 4})
                {
                    var srcShape = new [] { a, b, c, d}.Take(length).ToArray();
                    var src = RandomTensor(srcShape);
                    var target = RandomTensor(srcShape);
                    TestAll(new[] { src, target }, tensors => tensors[0].TransposeSecondAndThirdDimension(tensors[1]));
                }
            }
        }
        
      
 
        [TestCase(10)]
        [TestCase(1)]
        public void TestComputeAccuracyOneHot(int numClass)
        {
            const int rows = 10000;
            var buffer = RandomTensor(new[] { rows });
            var yPredicted = RandomTensor(new[] { rows, numClass });
            var yExpectedOneHot = TestCpuTensor.RandomOneHotTensor(yPredicted.Shape, _rand);
            TestAllForReturnValue(new[] { buffer, yExpectedOneHot, yPredicted}, tensors => tensors[0].ComputeAccuracy(tensors[1], tensors[2]));
        }

        [TestCase(10)]
        [TestCase(1)]
        public void TestComputeAccuracyTwoHot(int numClass)
        {
            const int rows = 10000;
            var buffer = RandomTensor(new[] { rows });
            var yExpectedOneHot = TestCpuTensor.RandomTwoHotTensor(new[] { rows, numClass }, _rand);
            var yPredicted = RandomTensor(yExpectedOneHot.Shape);
            TestAllForReturnValue(new[] { buffer, yExpectedOneHot, yPredicted}, tensors => tensors[0].ComputeAccuracy(tensors[1], tensors[2]));
        }

        [Test]
        public void TestComputeAccuracyCategoricalCrossentropyWithHierarchy()
        {
            var expected = TestCpuTensor.GetExpectedCategoricalCrossentropyWithHierarchy();
            var predicted = TestCpuTensor.GetPredictedCategoricalCrossentropyWithHierarchy();
            var buffer = RandomTensor(new[] { expected.Shape[0] });
            TestAllForReturnValue(new[] { buffer, expected, predicted}, tensors => tensors[0].ComputeAccuracyCategoricalCrossentropyWithHierarchy(tensors[1], tensors[2]));
        }


        private CpuTensor<float> RandomTensor(int[] shape)
        {
            return TestCpuTensor.RandomFloatTensor(shape, _rand, -1.5, +1.5);
        }
        /// <summary>
        /// return a tensor of shape 'shape' with random index value sin range [minIndex, maxIndex]
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="minIndex"></param>
        /// <param name="maxIndex"></param>
        /// <returns></returns>
        private CpuTensor<float> RandomTensorWithIndexes(int[] shape, int minIndex, int maxIndex)
        {
            var contentExpectedSparse = new int[Utils.Product(shape)];
            Utils.UniformDistribution(contentExpectedSparse, _rand, minIndex, maxIndex);
            return new CpuTensor<float>(shape, contentExpectedSparse.Select(x => (float)x).ToArray());
        }


        private static void AreEquals(CpuTensor<float> floatCpu, GPUTensor<float> floatGpu)
	    {
	        Assert.IsTrue(floatCpu.SameShape(floatGpu));
            if (!TensorExtensions.SameFloatContent(floatCpu, floatGpu, 1e-2, out var difference))
            {
                Assert.IsTrue(false, floatCpu + Environment.NewLine + floatGpu+Environment.NewLine+ difference);
            }
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
            for (var i = 0; i < cpuFloat.Length; ++i)
            {
                if (tensorIdsToIgnore != null && tensorIdsToIgnore.Contains(i))
                {
                    continue;
                }
                AreEquals(cpuFloat[i], gpuFloat[i]);
            }
            Assert.AreEqual(resultCpuFloat, resultGPUFloat, 1e-5, cpuFloat.Last().ReadonlyContent.Min() + " vs " + gpuFloat.Last().ContentAsFloatArray().Min());
        }

        private static GPUTensor<T> CloneToGPU<T>(CpuTensor<T> cpuTensor, GPUWrapper gpuWrapper)
        {
            return new GPUTensor<T>((int[])cpuTensor.Shape.Clone(), cpuTensor.Content, gpuWrapper);
        }
    }
}
