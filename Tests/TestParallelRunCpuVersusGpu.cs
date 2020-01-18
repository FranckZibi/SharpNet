using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.Data;
using SharpNet.GPU;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using SharpNet.CPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNetTests.CPU;
using SharpNetTests.Data;

namespace SharpNetTests
{
    [TestFixture]
    public class TestParallelRunCpuVersusGpu
    {
        private const int BatchSize = 9;
        private const int FiltersCount = 8;
        private const int ChannelsCount = 3;
        private const int Height = 17;
        private const int Width = 33;
        private const int Nx = Height * Width * ChannelsCount;
        private const int ConvolutionF = 5;
	    private const int ConvolutionPadding = 2;
	    private const int ConvolutionStride = 1;
        private const int PoolingSize = 2;
        private const int PoolingStride = 2;
	    private readonly Random _rand = new Random(0);
        private GPUWrapper GpuWrapper => GPUWrapper.FromDeviceId(0);

        [Test]
	    public void TestConvolutionBackwardBias()
	    {
	        var dxShape = new [] { BatchSize, FiltersCount, Height, Width };
            var biasShape = new [] { 1, FiltersCount, 1, 1};
            var dx = RandomTensor(dxShape, "dx");
	        var convolutionBackwardBias = new CpuTensor<float>(biasShape, "convolutionBackwardBias");
	        TestAll(new[] {dx, convolutionBackwardBias}, tensors => tensors[0].ConvolutionBackwardBias(tensors[1]));
	    }
	    [Test]
	    public void TestConvolution()
	    {
	        var x = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width }, "x");
	        var convolution = RandomTensor(new[] { BatchSize, ChannelsCount, ConvolutionF, ConvolutionF }, "convolution");
	        var y = RandomTensor(ConvolutionLayer.ConvolutionOutputShape(x.Shape, convolution.Shape, ConvolutionPadding, ConvolutionStride), "y");
	        TestAll(new[] { x, convolution, y }, tensors => tensors[0].Convolution(tensors[1], ConvolutionPadding, ConvolutionStride, tensors[2]));
	    }

        [Test]
        public void TestConvolutionGradient()
        {
            var x = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width }, "x");
            x = new CpuTensor<float>(x.Shape, "x");
            var convolution = RandomTensor(new[] { BatchSize, ChannelsCount, ConvolutionF, ConvolutionF }, "convolution");
            var dy = RandomTensor(ConvolutionLayer.ConvolutionOutputShape(x.Shape, convolution.Shape, ConvolutionPadding, ConvolutionStride), "dy");
            //this will compute 'dx' && 'convolutionGradient'
            var dx = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width }, "dx");
            var convolutionGradient = RandomTensor(convolution.Shape, "convolutionGradient");
            TestAll(new[] { x, convolution, dy, dx, convolutionGradient}, tensors => tensors[0].ConvolutionGradient(tensors[1], tensors[2], ConvolutionPadding, ConvolutionStride, tensors[3], tensors[4]));
        }

        [TestCase(cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL, BatchSize)]
        [TestCase(cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL, 1)]
        [TestCase(cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION, BatchSize)]
        [TestCase(cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION, 1)]
        public void TestBatchNormalization(cudnnBatchNormMode_t mode, int batchSize)
        {
            foreach(bool ignoreHW in new []{false, true})
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

                var x = RandomTensor(xShape, "x");
                var y = RandomTensor(xShape, "y");
                y.ZeroMemory();
                var bnScale = RandomTensor(scaleAndBiasShape, "bnScale");
                var bnBias = RandomTensor(scaleAndBiasShape, "bnBias");
                var resultRunningMean = RandomTensor(scaleAndBiasShape, "resultRunningMean");
                resultRunningMean.ZeroMemory();
                var resultRunningVariance = RandomTensor(scaleAndBiasShape, "resultRunningVariance");
                resultRunningVariance.ZeroMemory();
                var resultSaveMean = RandomTensor(scaleAndBiasShape, "resultSaveMean");
                resultSaveMean.ZeroMemory();
                var resultSaveVariance = RandomTensor(scaleAndBiasShape, "resultSaveVariance");
                resultSaveVariance.ZeroMemory();

                var epsilon = 1e-5;
                //exponentialAverageFactor = 1.0;
                //isTraining = true;
                TestAll(new[] { x, y, bnScale, bnBias, resultRunningMean, resultRunningVariance, resultSaveMean, resultSaveVariance }, tensors => tensors[0].BatchNormalization(tensors[1], tensors[2], tensors[3], 1.0, tensors[4], tensors[5], mode, epsilon, tensors[6], tensors[7], true));
                //exponentialAverageFactor = 1.0;
                //isTraining = false; 
                TestAll(new[] { x, y, bnScale, bnBias, resultRunningMean, resultRunningVariance, resultSaveMean, resultSaveVariance }, tensors => tensors[0].BatchNormalization(tensors[1], tensors[2], tensors[3], 1.0, tensors[4], tensors[5], mode, epsilon, tensors[6], tensors[7], false));

                //exponentialAverageFactor = 0.5;
                //isTraining = true;
                TestAll(new[] { x, y, bnScale, bnBias, resultRunningMean, resultRunningVariance, resultSaveMean, resultSaveVariance }, tensors => tensors[0].BatchNormalization(tensors[1], tensors[2], tensors[3], 0.5, tensors[4], tensors[5], mode, epsilon, tensors[6], tensors[7], true));
                //exponentialAverageFactor = 0.5;
                //isTraining = false;
                TestAll(new[] { x, y, bnScale, bnBias, resultRunningMean, resultRunningVariance, resultSaveMean, resultSaveVariance }, tensors => tensors[0].BatchNormalization(tensors[1], tensors[2], tensors[3], 0.5, tensors[4], tensors[5], mode, epsilon, tensors[6], tensors[7], false));
            }
        }

        [TestCase(cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL, BatchSize)]
        [TestCase(cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL, 1)]
        [TestCase(cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION, BatchSize)]
        [TestCase(cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION, 1)]
        public void TestBatchNormalizationBackward(cudnnBatchNormMode_t mode, int batchSize)
        {
            foreach (bool ignoreHW in new[] { false, true })
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
                var x = RandomTensor(xShape, "x");
                var y = RandomTensor(xShape, "y");
                y.ZeroMemory();
                var bnScale = RandomTensor(scaleAndBiasShape, "bnScale");
                var bnBias = RandomTensor(scaleAndBiasShape, "bnBias");
                var resultRunningMean = RandomTensor(scaleAndBiasShape, "resultRunningMean");
                resultRunningMean.ZeroMemory();
                var resultRunningVariance = RandomTensor(scaleAndBiasShape, "resultRunningVariance");
                resultRunningVariance.ZeroMemory();
                var resultSaveMean = RandomTensor(scaleAndBiasShape, "resultSaveMean");
                resultSaveMean.ZeroMemory();
                var resultSaveVariance = RandomTensor(scaleAndBiasShape, "resultSaveVariance");
                resultSaveVariance.ZeroMemory();
                const double exponentialAverageFactor = 1.0;
                const double epsilon = 1e-5;
                x.BatchNormalization(y, bnScale, bnBias, exponentialAverageFactor, resultRunningMean, resultRunningVariance, mode, epsilon, resultSaveMean, resultSaveVariance, false);
                var dx = RandomTensor(xShape, "dx");
                var dy = RandomTensor(xShape, "dy");
                var resultBnScaleDiff = RandomTensor(scaleAndBiasShape, "resultBnScaleDiff");
                var resultBnBiasDiff = RandomTensor(scaleAndBiasShape, "resultBnBiasDiff");
                TestAll(new[] { x, dy, dx, bnScale, resultBnScaleDiff, resultBnBiasDiff, resultSaveMean, resultSaveVariance }, tensors => tensors[0].BatchNormalizationBackward(tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], mode, epsilon, tensors[6], tensors[7]));
            } 
        }
        [Test]
	    public void TestZeroMemory()
	    {
	        var a = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width }, "a");
	        TestAll(new[] { a }, tensors => tensors[0].ZeroMemory());
	    }

        [Test]
	    public void TestDot()
	    {
	        var a = RandomTensor(new[] { 8, 10}, "a");
	        var b = RandomTensor(new[] { a.Shape[1], 12}, "b");
	        var result = new CpuTensor<float>(new[] { a.Shape[0], b.Shape[1] }, "result");
	        TestAll(new[] { a, b, result}, tensors => tensors[2].Dot(tensors[0], false, tensors[1], false, 1, 0));
        }

	    [Test]
	    public void TestDotV2()
	    {
	        var a = RandomTensor(new[] { 8, 10 }, "a");
	        var b = RandomTensor(new[] { a.Shape[1], 12 }, "b");
	        var result = new CpuTensor<float>(new[] { a.Shape[0], b.Shape[1] }, "result");
            TestAll(new[] { a, b, result }, tensors => tensors[2].Dot(tensors[0], tensors[1]));
        }

        /// <summary>
        /// This test was written to reproduce the error: CUBLAS_STATUS_INTERNAL_ERROR (v 10.0)
        /// Currently this error is ignored
        /// </summary>
        [Test]
        public void TestDotV3()
        {
            var a = RandomTensor(new[] { 2, 20000 }, "a");
            var b = RandomTensor(new[] { a.Shape[1], 2 }, "b");
            var result = new CpuTensor<float>(new[] { a.Shape[0], b.Shape[1] }, "result");
            TestAll(new[] { a, b, result }, tensors => tensors[2].Dot(tensors[0], false, tensors[1], false, 1, 0));
        }

        [Test]
	    public void TestUpdate_Adding_Alpha_X()
	    {
	        var y = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width}, "y");
	        var x = RandomTensor(y.Shape, "x");
	        TestAll(new[] { y, x}, tensors => tensors[0].Update_Adding_Alpha_X(0.5f, tensors[1]));
	    }

        [Test]
        public void TestAddTensor()
        {
            var y = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width }, "y");
            var x = RandomTensor(y.Shape, "x");
            TestAll(new[] { y, x }, tensors => tensors[0].AddTensor(0.5f, tensors[1], 0.75f));
        }

        [Test]
        public void TestConcatenate()
        {
            var x1 = RandomTensor(new[] { BatchSize, 17, Height, Width }, "x1");
            var x2 = RandomTensor(new[] { BatchSize, 13, Height, Width }, "x2");
            var concat = RandomTensor(new[] { BatchSize, 17 + 13, Height, Width }, "concat");
            TestAll(new[] { concat, x1, x2 }, tensors => tensors[0].Concatenate(tensors[1], tensors[2]));
        }

        [Test]
        public void TestSplit()
        {
            var toSplit = RandomTensor(new[] { BatchSize, 17 + 13, Height, Width }, "toSplit");
            var x1 = RandomTensor(new[] { BatchSize, 17, Height, Width }, "x1");
            var x2 = RandomTensor(new[] { BatchSize, 13, Height, Width }, "x2");
            TestAll(new[] { toSplit, x1, x2 }, tensors => tensors[0].Split(tensors[1], tensors[2]));
        }

        [Test]
        public void TestUpdate_Multiplying_By_Alpha()
        {
            var x = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width }, "x");
            TestAll(new[] { x }, tensors => tensors[0].Update_Multiplying_By_Alpha(0.5f));
        }
        [Test]
	    public void TestCompute_BiasGradient_from_dy()
	    {
	        var dy = RandomTensor(new[] { BatchSize, 1, 1, Nx }, "dy");
	        var db = RandomTensor(new[] { 1, 1, 1, Nx }, "db");
	        TestAll(new[] { dy, db }, tensors => tensors[0].Compute_BiasGradient_from_dy(tensors[1]));
	    }
	    [Test]
	    public void TestBroadcastConvolutionBiasToOutput()
	    {
            var convolutionBias = RandomTensor(new[] { 1, FiltersCount, 1, 1}, "convolutionBias");
            var y = RandomTensor(new[] { BatchSize, FiltersCount, Height, Width}, "y");
	        TestAll(new[] { convolutionBias, y }, tensors => tensors[0].BroadcastConvolutionBiasToOutput(tensors[1]));
	    }
        [Test]
	    public void TestCopyTo()
	    {
	        var src = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width}, "src");
	        var dest = RandomTensor(src.Shape, "dest");
	        TestAll(new[] { src, dest}, tensors => tensors[0].CopyTo(tensors[1]));
	    }
        [Test]
        public void TestCopyTo_V2()
        {
            var src = RandomTensor(new[] { 10,5,2}, "src");
            var dest = RandomTensor(src.Shape, "dest");
            TestAll(new[] { src, dest }, tensors => tensors[0].CopyTo(20,tensors[1],60,40));
        }

        //TODO [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_TANH)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_ELU)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)]
	    public void TestActivationForward(cudnnActivationMode_t activationMode)
	    {
	        var x = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width }, "x");
	        var y = RandomTensor(x.Shape, "y");
	        TestAll(new[] { x, y }, tensors => tensors[0].ActivationForward(activationMode, tensors[1]));
	    }

        //TODO [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_TANH)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_ELU)]
        [TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)]
        //!D TODO : re enable test 
        //[TestCase(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)]
	    public void TestActivationBackward(cudnnActivationMode_t activationMode)
	    {
            var y = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width }, "y");
            var dy = RandomTensor(y.Shape, "dy");
            var x = RandomTensor(y.Shape, "x");
            var dx = RandomTensor(y.Shape, "dx");
	        x.ActivationForward(activationMode, y);
            TestAll(new[] { y, dy, x, dx }, tensors => tensors[0].ActivationBackward(tensors[1], tensors[2], activationMode, tensors[3]));
	    }
        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC)]
        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)]
	    public void TestPooling(cudnnPoolingMode_t poolingMode)
	    {
	        var aBeforePooling = RandomTensor(new[] { BatchSize, ChannelsCount, Height, Width }, "aBeforePooling");
	        var aAfterPooling = RandomTensor(PoolingLayer.PoolingOutputShape(aBeforePooling.Shape, PoolingSize, PoolingStride), "aAfterPooling");
	        TestAll(new[] { aBeforePooling, aAfterPooling }, tensors => tensors[0].Pooling(tensors[1], poolingMode, PoolingSize, PoolingStride));
	    }
        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_MAX_DETERMINISTIC)]
        [TestCase(cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)]
	    public void TestPoolingGradient(cudnnPoolingMode_t poolingMode)
        {
	        var shapeBeforePooling = new[] { BatchSize, ChannelsCount, Height, Width };
	        var shapeAfterPooling = PoolingLayer.PoolingOutputShape(shapeBeforePooling, PoolingSize, PoolingStride);
	        var dy = RandomTensor(shapeAfterPooling, "dy");
	        var x = RandomTensor(shapeBeforePooling, "x");
	        var y = RandomTensor(shapeAfterPooling, "y");
            x.Pooling(y, poolingMode, PoolingSize, PoolingStride);
            var dx = RandomTensor(shapeBeforePooling, "dx");
	        TestAll(new[] { dy, y, x, dx }, tensors => tensors[0].PoolingGradient(tensors[1], tensors[2], tensors[3], poolingMode, PoolingSize, PoolingStride));
	    }
        [Test]
	    public void TestBroadcastAddVectorToOutput()
	    {
            var b = RandomTensor(new []{1, Nx}, "bias");
	        var y = RandomTensor(new []{BatchSize, Nx}, "y");
            TestAll(new[] {b,y}, tensors => tensors[0].BroadcastAddVectorToOutput(tensors[1]));
	    }
        [Test]
        public void TestUpdateAdamOptimizer()
        {
            var W = RandomTensor(new[] { Width, Height}, "Weights");
            var dW = RandomTensor(W.Shape, "dW");
            var adam_vW = RandomTensor(W.Shape, "adam_vW");
            var adam_sW = RandomTensor(W.Shape, "adam_sW");
            TestAll(new[] { W, dW, adam_vW, adam_sW}, tensors => tensors[0].UpdateAdamOptimizer(0.001, 0.9,0.999,1e-8, tensors[1], tensors[2], tensors[3], 5));
        }

        [TestCase(0.1, 0.9, 0.0, true)]
        [TestCase(0.1, 0.9, 0.0, false)]
        [TestCase(0.1, 0.9, 1e-4, true)]
        [TestCase(0.1, 0.9, 1e-4, false)]
        public void TestUpdateSGDOptimizer(double learningRate, double momentum, double decay, bool usenesterov)
        {
            var W = RandomTensor(new[] { Width, Height }, "W");
            var dW = RandomTensor(W.Shape, "dW");
            var velocity = RandomTensor(W.Shape, "velocity");
            TestAll(new[] { W, dW, velocity }, tensors => tensors[0].UpdateSGDOptimizer(learningRate, momentum, usenesterov, tensors[1], tensors[2]));
        }

        
        [TestCase(1, NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)]
        [TestCase(1, NetworkConfig.LossFunctionEnum.BinaryCrossentropy)]
        [TestCase(2, NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)]
        [TestCase(2, NetworkConfig.LossFunctionEnum.BinaryCrossentropy)]
        [TestCase(10, NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)]
        [TestCase(10, NetworkConfig.LossFunctionEnum.BinaryCrossentropy)]
        public void TestComputeLoss(int nbCategories, NetworkConfig.LossFunctionEnum lossFunction)
        {
            var nbRows = 1000;
            var yPredicted = RandomTensor(new[] { nbRows, nbCategories }, "yPredicted");
            var yExpectedOneHot = TestCpuTensor.RandomOneHotTensor(yPredicted.Shape, _rand, "yExpectedOneHot");
            var buffer = RandomTensor(new[] { nbRows }, "buffer");
            TestAllForReturnValue(new[] { yExpectedOneHot, yPredicted, buffer}, tensors => tensors[0].ComputeLoss(tensors[1], lossFunction, tensors[2]), new List<int>{2});
        }

        [TestCase(2, NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)]
        [TestCase(2, NetworkConfig.LossFunctionEnum.BinaryCrossentropy)]
        [TestCase(10, NetworkConfig.LossFunctionEnum.CategoricalCrossentropy)]
        [TestCase(10, NetworkConfig.LossFunctionEnum.BinaryCrossentropy)]
        public void TestComputeLossFromCategoryIndexes(int nbCategories, NetworkConfig.LossFunctionEnum lossFunction)
        {
            const int nbRows = 10000;
            var yPredicted = RandomTensor(new[] { nbRows, nbCategories }, "yPredicted");
            var categoryIndexes = TestCpuTensor.RandomCategoryIndexTensor(nbRows, nbCategories, _rand);
            var buffer = RandomTensor(new[] { nbRows }, "buffer");
            TestAllForReturnValue(new[] { categoryIndexes }, new[] { yPredicted, buffer }, tensors => tensors[0].ComputeLossFromCategoryIndexes(tensors[1], lossFunction, tensors[2]), new List<int> { 1 });
        }

        [TestCase(10)]
        [TestCase(1)]
        public void TestComputeAccuracyOneHot(int nbCategories)
        {
            var nbRows = 10000;
            var yPredicted = RandomTensor(new[] { nbRows, nbCategories }, "yPredicted");
            var yExpectedOneHot = TestCpuTensor.RandomOneHotTensor(yPredicted.Shape, _rand, "yExpectedOneHot");
            var buffer = RandomTensor(new[] { nbRows}, "buffer");
            TestAllForReturnValue(new[] { yExpectedOneHot, yPredicted, buffer }, tensors => tensors[0].ComputeAccuracy(tensors[1], tensors[2]), new List<int> { 2 });
        }

        [TestCase(10)]
        [TestCase(1)]
        public void TestComputeAccuracyTwoHot(int nbCategories)
        {
            var nbRows = 10000;
            var yPredicted = RandomTensor(new[] { nbRows, nbCategories }, "yPredicted");
            var yExpectedOneHot = TestCpuTensor.RandomTwoHotTensor(yPredicted.Shape, _rand, "yExpectedTwoHot");
            var buffer = RandomTensor(new[] { nbRows }, "buffer");
            TestAllForReturnValue(new[] { yExpectedOneHot, yPredicted, buffer }, tensors => tensors[0].ComputeAccuracy(tensors[1], tensors[2]), new List<int> { 2 });
        }

        [Test]
        public void TestComputeAccuracyFromCategoryIndexes()
        {
            const int nbRows = 10000;
            const int nbCategories = 10;
            var yPredicted = RandomTensor(new[] { nbRows, nbCategories }, "yPredicted");
            var categoryIndexes = TestCpuTensor.RandomCategoryIndexTensor(nbRows, nbCategories, _rand);
            var buffer = RandomTensor(new[] { nbRows }, "buffer");
            TestAllForReturnValue(new[] { categoryIndexes }, new[] { yPredicted, buffer }, tensors => tensors[0].ComputeAccuracyFromCategoryIndexes(tensors[1], tensors[2]), new List<int> { 1 });
        }

        private CpuTensor<float> RandomTensor(int[] shape, string description)
	    {
	        return TestCpuTensor.RandomFloatTensor(shape, _rand, -1.5, +1.5, description);
	    }
        private static void AreEquals(CpuTensor<float> floatCpu, GPUTensor<float> floatGpu)
	    {
	        Assert.IsTrue(floatCpu.SameShape(floatGpu));
            Assert.IsTrue(TestTensor.SameContent(floatCpu, floatGpu, 1e-2), floatCpu + Environment.NewLine + floatGpu);
        }
	    [SuppressMessage("ReSharper", "CoVariantArrayConversion")]
	    private void TestAll(CpuTensor<float>[] data, Action<Tensor[]> work)
	    {
	        var cpuFloat = new List<CpuTensor<float>>();
	        cpuFloat.AddRange(data);
	        var gpuFloat = new List<GPUTensor<float>>();
	        gpuFloat.AddRange(cpuFloat.Select(x => CloneToGPU(x, GpuWrapper)));
	        work(cpuFloat.ToArray());
	        work(gpuFloat.ToArray());
            for (var i = 0; i < cpuFloat.Count; ++i)
            {
	            AreEquals(cpuFloat[i], gpuFloat[i]);
	        }
	    }

        private void TestAllForReturnValue(CpuTensor<float>[] cpuFloat, Func<Tensor[], double> work, List<int> tensorIdsToIgnore = null)
        {
            TestAllForReturnValue(new CpuTensor<int>[0], cpuFloat, work, tensorIdsToIgnore);
        }

        private void TestAllForReturnValue(CpuTensor<int>[] cpuInt, CpuTensor<float>[] cpuFloat, Func<Tensor[], double> work, List<int> tensorIdsToIgnore = null)
        {
            var gpuInt = cpuInt.Select(x => CloneToGPU(x, GpuWrapper)).ToList(); ;
            var gpuFloat = cpuFloat.Select(x => CloneToGPU(x, GpuWrapper)).ToList();
            var cpu = cpuInt.Select(x => (Tensor) x).Union(cpuFloat.Select(x => (Tensor) x)).ToArray();
            var resultCpuFloat = work(cpu);
            var gpu = gpuInt.Select(x => (Tensor)x).Union(gpuFloat.Select(x => (Tensor)x)).ToArray();
            var resultGPUFloat = work(gpu);
            Assert.AreEqual(resultCpuFloat, resultGPUFloat, 1e-5, cpuFloat.Last().Content.Min() + " vs " + gpuFloat.Last().ContentAsFloatArray().Min());
            for (var i = 0; i < cpuFloat.Length; ++i)
            {
                if (tensorIdsToIgnore != null && tensorIdsToIgnore.Contains(i))
                {
                    continue;
                }
                AreEquals(cpuFloat[i], gpuFloat[i]);
            }
        }

        private static GPUTensor<T> CloneToGPU<T>(CpuTensor<T> cpuTensor, GPUWrapper gpuWrapper) where T : struct
        {
            var result =  new GPUTensor<T>(cpuTensor.Shape, cpuTensor.Description, gpuWrapper);
            result.CopyToDevice(cpuTensor.Content);
            return result;
        }
    }
}
