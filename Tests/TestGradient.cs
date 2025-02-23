using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.Data;
using SharpNet.GPU;
using System.Collections.Generic;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNetTests.NonReg;

namespace SharpNetTests
{
    [TestFixture]
    public class TestGradient
    {
	    private readonly Random _rand = new (0);

        [Test]
        public void TestGradientForLinearLayer()
        {
            var X = GetX();
            var Y = GetY();
            var network = GetNetwork();
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Linear(Y.Shape[1], true, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            network.Predict(X, true);
            var linearLayer = (Linear)network.Layers[1];
            //We test the gradients computation for weights
            CompareExpectedVsObservedGradients(network, linearLayer, false, X, Y, _rand);
            //We test the gradients computation for bias
            CompareExpectedVsObservedGradients(network, linearLayer, true, X, Y, _rand);
        }

        [Test]
        public void TestGradientForConvolutionLayer()
        {
            var X = GetX();
            var Y = GetY();
            var network = GetNetwork();
            network.Sample.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM;
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(3, 3,1,ConvolutionLayer.PADDING_TYPE.SAME, true)
                .Linear(Y.Shape[1], true, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            network.Predict(X, true);
            var convLayer = (ConvolutionLayer)network.Layers[1];
            //We test the gradients computation for weights
            CompareExpectedVsObservedGradients(network, convLayer, false, X, Y, _rand);
            //We test the gradients computation for bias
            CompareExpectedVsObservedGradients(network, convLayer, true, X, Y, _rand);
        }

        [Test]
        public void TestGradientForDepthwiseConvolutionLayer()
        {
            var X = GetX();
            var Y = GetY();
            var network = GetNetwork();
            network.Sample.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM;
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .DepthwiseConvolution(3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 1, true)
                .Linear(Y.Shape[1], true, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            network.Predict(X, true);
            var convLayer = (ConvolutionLayer)network.Layers[1];
            //We test the gradients computation for weights
            CompareExpectedVsObservedGradients(network, convLayer, false, X, Y, _rand);
            //We test the gradients computation for bias
            CompareExpectedVsObservedGradients(network, convLayer, true, X, Y, _rand);
        }

        [Test, Ignore("fail to compute the expected gradients")]
        public void TestGradientForBatchNormLayer()
        {
            var X = GetX();
            var Y = GetY();
            var network = GetNetwork();
            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .BatchNorm(1.0, 1e-5).Linear(Y.Shape[1], true, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_ELU);
            network.Predict(X, true);
            //n.SaveLayers();
            var batchNormLayer = (BatchNormalizationLayer)network.Layers[1];
            //We test the gradients computation for weights
            CompareExpectedVsObservedGradients(network, batchNormLayer, false, X, Y, _rand, 1);
            //We test the gradients computation for bias
            CompareExpectedVsObservedGradients(network, batchNormLayer, true, X, Y, _rand, 1);
        }

        private static CpuTensor<float> FromNumpyArray(string s)
        {
            var tensor = TensorExtensions.FromNumpyArray(s);
            return tensor.AsFloatCpu;
        }
        private static double ComputeGradientAndReturnLoss(Network network, Tensor X, CpuTensor<float> yExpected)
        {
            var yPredicted = (CpuTensor<float>)network.Predict(X, true);
            var metricConfig = new TestMetricConfig(lossMetric: EvaluationMetricEnum.CategoricalCrossentropy);
            //we compute gradients
            network.PropagationManager.Backward(yExpected, yPredicted, metricConfig);
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
            foreach (var l in Parameters(n).OfType<CpuTensor<float>>())
            {
                for (int i = 0; i < l.Count; ++i)
                {
                    l[i] = Math.Max(min, Math.Abs(l[i]));
                }
            }
        }
        private static List<Tensor> Parameters(Network network)
        {
            return network.Layers.SelectMany(x => x.Parameters).Select(t=>t.Item1).ToList();
        }

        private static void CompareExpectedVsObservedGradients(Network n, Layer layer, bool isBias, CpuTensor<float> X, CpuTensor<float> Y, Random r, int nbTests = 100)
        {
            AbsWeightsWithMinimum(n, 0);
            ComputeGradientAndReturnLoss(n, X, Y);
            const float epsilon = 1e-2f;
            var w = (CpuTensor<float>)(isBias?layer.Bias:layer.Weights);
            var dW = (CpuTensor<float>)(isBias?layer.BiasGradients:layer.WeightGradients);
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
            var resourceIds = new List<int> {-1};
            return TestNetwork.NewForTests(new NetworkSample
                {
                    LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                    CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
                    ResourceIds = resourceIds.ToList()
                },
                NetworkSample.DefaultWorkingDirectory,
                "ModelName"
                );
        }
        private static CpuTensor<float> GetY()
        {
            return FromNumpyArray(TestNetworkPropagation.Y_2_3);
        }

        private static CpuTensor<float> GetX()
        {
            return FromNumpyArray(TestNetworkPropagation.X_2_1_4_4);
        }

    }
}
