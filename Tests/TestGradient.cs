using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.Data;
using SharpNet.GPU;
using System.Collections.Generic;
using SharpNet.CPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNetTests.NonReg;

namespace SharpNetTests
{
    [TestFixture]
    public class TestGradient
    {
	    private readonly Random _rand = new Random(0);

        [Test]
        public void TestGradientForDenseLayer()
        {
            var X = GetX();
            var Y = GetY();
            var network = GetNetwork();
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            network.Predict(X, true);
            var denseLayer = (DenseLayer)network.Layers[1];
            //We test the gradients computation for weights
            CompareExpectedVsObservedGradients(network, (CpuTensor<float>)denseLayer.Weights, (CpuTensor<float>)denseLayer.WeightGradients, X, Y, _rand);
            //We test the gradients computation for bias
            CompareExpectedVsObservedGradients(network, (CpuTensor<float>)denseLayer.Bias, (CpuTensor<float>)denseLayer.BiasGradients, X, Y, _rand);
        }

        [Test]
        public void TestGradientForConvolutionLayer()
        {
            var X = GetX();
            var Y = GetY();
            var network = GetNetwork();
            network.Config.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM;
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(3, 3,1,ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            network.Predict(X, true);
            var convLayer = (ConvolutionLayer)network.Layers[1];
            //We test the gradients computation for weights
            CompareExpectedVsObservedGradients(network, convLayer.Convolution as CpuTensor<float>, convLayer.ConvolutionGradients as CpuTensor<float>, X, Y, _rand);
            //We test the gradients computation for bias
            CompareExpectedVsObservedGradients(network, convLayer.ConvolutionBias as CpuTensor<float>,convLayer.ConvolutionBiasGradients as CpuTensor<float>, X, Y, _rand);
        }

        [Test]
        public void TestGradientForDepthwiseConvolutionLayer()
        {
            var X = GetX();
            var Y = GetY();
            var network = GetNetwork();
            network.Config.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM;
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .DepthwiseConvolution(3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 1, 0.0, true )
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            network.Predict(X, true);
            var convLayer = (ConvolutionLayer)network.Layers[1];
            //We test the gradients computation for weights
            CompareExpectedVsObservedGradients(network, convLayer.Convolution as CpuTensor<float>, convLayer.ConvolutionGradients as CpuTensor<float>, X, Y, _rand);
            //We test the gradients computation for bias
            CompareExpectedVsObservedGradients(network, convLayer.ConvolutionBias as CpuTensor<float>, convLayer.ConvolutionBiasGradients as CpuTensor<float>, X, Y, _rand);
        }

        [Test, Ignore("fail to compute the expected gradients")]
        public void TestGradientForBatchNormLayer()
        {
            var X = GetX();
            var Y = GetY();
            var network = GetNetwork();
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .BatchNorm(1.0, 1e-5)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_ELU);
            network.Predict(X, true);
            //n.SaveLayers();
            var batchNormLayer = (BatchNormalizationLayer)network.Layers[1];
            //We test the gradients computation for weights
            CompareExpectedVsObservedGradients(network, batchNormLayer.Scale as CpuTensor<float>, batchNormLayer.ScaleGradients as CpuTensor<float>, X, Y, _rand, 1);
            //We test the gradients computation for bias
            CompareExpectedVsObservedGradients(network, batchNormLayer.Bias as CpuTensor<float>, batchNormLayer.BiasGradients as CpuTensor<float>, X, Y, _rand, 1);
        }

        private static CpuTensor<float> FromNumpyArray(string s, string description)
        {
            var tensor = TensorExtensions.FromNumpyArray(s, description);
            return tensor.AsFloatCpu;
        }
        private static double ComputeGradientAndReturnLoss(Network network, Tensor X, CpuTensor<float> yExpected)
        {
            var yPredicted = (CpuTensor<float>)network.Predict(X, true);
            //we compute gradients
            network.PropagationManager.Backward(yExpected, yPredicted);
            double result = 0;
            for (int i = 0; i < yExpected.Count; ++i)
            {
                double err = Math.Pow(yExpected[i] - yPredicted[i], 2);
                result += err / 2.0;
            }
            return result;
        }
        private static void AbsWeightsWithMinimum(Network n, float min)
        {
            foreach (var l in n.TensorsIndependentOfBatchSize.OfType<CpuTensor<float>>())
            {
                for (int i = 0; i < l.Count; ++i)
                {
                    l[i] = Math.Max(min, Math.Abs(l[i]));
                }
            }
        }
        private static void CompareExpectedVsObservedGradients(Network n, CpuTensor<float> w, CpuTensor<float> dW, CpuTensor<float> X, CpuTensor<float> Y, Random r, int nbTests = 100)
        {
            AbsWeightsWithMinimum(n, 0);
            ComputeGradientAndReturnLoss(n, X, Y);
            float epsilon = 1e-2f;
            var observedDifferences =  new List<double>();
            for (int testIndex = 0; testIndex < nbTests; ++testIndex)
            {
                int weightIndexToUpdate = r.Next(w.Count);
                ComputeGradientAndReturnLoss(n, X, Y);
                var initialWeightValue = w[weightIndexToUpdate];
                var expectedGradient = dW[weightIndexToUpdate];

                w[weightIndexToUpdate] = initialWeightValue + epsilon;
                var lossUp = ComputeGradientAndReturnLoss(n, X, Y);
                //n.SaveLayers();
                w[weightIndexToUpdate] = initialWeightValue - epsilon;
                var lossDown = ComputeGradientAndReturnLoss(n, X, Y);
                //n.SaveLayers();
                w[weightIndexToUpdate] = initialWeightValue;

                var observedGradient = (lossUp - lossDown) / (2 * epsilon);
                var maxGradient = Math.Max(Math.Abs(observedGradient), Math.Abs(expectedGradient));
                if (maxGradient <= 0)
                {
                    continue;
                }

                var differenceObservedVsExpected = Math.Abs(expectedGradient - observedGradient) / maxGradient;
                if (differenceObservedVsExpected > 0.001)
                {
                    ComputeGradientAndReturnLoss(n, X, Y);
                    //n.SaveLayers();
                    observedDifferences.Add(differenceObservedVsExpected);
                }
            }
            if (observedDifferences.Count != 0)
            {
                var errorMsg = "found "+observedDifferences.Count+" differences, max = "+observedDifferences.Max();
                throw new Exception(errorMsg);
            }
        }
        private static Network GetNetwork()
        {
            var gpuDeviceId = -1;
            return new Network(new NetworkConfig{LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy, CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1 }, gpuDeviceId);
        }
        private static CpuTensor<float> GetY()
        {
            return FromNumpyArray(TestNetworkPropagation.Y_2_3, "Y");
        }

        private static CpuTensor<float> GetX()
        {
            return FromNumpyArray(TestNetworkPropagation.X_2_1_4_4, "X");
        }

    }
}
