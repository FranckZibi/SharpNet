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
using SharpNet.Pictures;
using SharpNetTests.NonReg;

namespace SharpNetTests
{
    [TestFixture]
    [SuppressMessage("ReSharper", "PossibleNullReferenceException")]
    public class TestGradient
    {
	    private readonly Random _rand = new Random(0);

        [TestCase(true, false)]
        [TestCase(false, true)]
        public void TestGradientForDenseLayer(bool testWeights, bool testBias)
        {
            var X = FromNumpyArray(TestNetworkPropagation.X_2_1_4_4, "X");
            var Y = FromNumpyArray(TestNetworkPropagation.Y_2_1_4_4, "Y");
            var n = GetNetwork();
            n
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            n.Predict(X, false);
            var denseLayer = (DenseLayer)n.Layers[1];
            if (testWeights)
            {
                CompareExpectedVsObservedGradients(n, (CpuTensor<float>)denseLayer.Weights, (CpuTensor<float>)denseLayer.WeightGradients, X, Y, _rand);
            }
            if (testBias)
            {
                CompareExpectedVsObservedGradients(n, (CpuTensor<float>)denseLayer.Bias, (CpuTensor<float>)denseLayer.BiasGradients, X, Y, _rand);
            }
        }

        [TestCase(true, false)]
        [TestCase(false, true)]
        public void TestGradientForConvolutionLayer(bool testWeights, bool testBias)
        {

            var X = FromNumpyArray(TestNetworkPropagation.X_2_1_4_4, "X");
            var Y = FromNumpyArray(TestNetworkPropagation.Y_2_1_4_4, "Y");
            var n = GetNetwork();
            n
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(3, 3,1,1, 0.0, true)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            n.Predict(X, false);
            var convLayer = (ConvolutionLayer)n.Layers[1];
            if (testWeights)
            {
                CompareExpectedVsObservedGradients(n, convLayer.Convolution as CpuTensor<float>, convLayer.ConvolutionGradients as CpuTensor<float>, X, Y, _rand);
            }
            if (testBias)
            {
                CompareExpectedVsObservedGradients(n, convLayer.ConvolutionBias as CpuTensor<float>,convLayer.ConvolutionBiasGradients as CpuTensor<float>, X, Y, _rand);
                
            }
        }

        [TestCase(true, false), Ignore]
        [TestCase(false, true)]
        public void TestGradientForBatchNormLayer(bool testWeights, bool testBias)
        {
            var X = FromNumpyArray(TestNetworkPropagation.X_2_1_4_4, "X");
            var Y = FromNumpyArray(TestNetworkPropagation.Y_2_1_4_4, "Y");
            var n = GetNetwork();
            n
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .BatchNorm(1.0)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_ELU);
            n.Predict(X, false);
            //n.SaveLayers();
            var batchNormLayer = (BatchNormalizationLayer)n.Layers[1];
            if (testWeights)
            {
                CompareExpectedVsObservedGradients(n, batchNormLayer.Weights as CpuTensor<float>, batchNormLayer.WeightGradients as CpuTensor<float>, X, Y, _rand, 1);
            }
            if (testBias)
            {
                CompareExpectedVsObservedGradients(n, batchNormLayer.Bias as CpuTensor<float>, batchNormLayer.BiasGradients as CpuTensor<float>, X, Y, _rand, 1);
            }
        }

        private static CpuTensor<float> FromNumpyArray(string s, string description)
        {
            var X0 = TensorExtensions.FromNumpyArray(s, description);
            return X0.AsFloatCpu;
        }
        private static double ComputeGradientAndReturnLoss(Network n, CpuTensor<float> X, CpuTensor<float> yExpected, bool isTraining)
        {
            var yPredicted = n.Predict(X, isTraining) as CpuTensor<float>;
            n.BackwardPropagation(yExpected); //we compute gradients
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
            foreach (var l in n.TensorsIndependantOfBatchSize.OfType<CpuTensor<float>>())
            {
                for (int i = 0; i < l.Content.Length; ++i)
                {
                    l[i] = Math.Max(min, Math.Abs(l[i]));
                }
            }
        }
        private static void CompareExpectedVsObservedGradients(Network n, CpuTensor<float> w, CpuTensor<float> dW, CpuTensor<float> X, CpuTensor<float> Y, Random r, int nbTests = 100)
        {
            AbsWeightsWithMinimum(n, 0);
            ComputeGradientAndReturnLoss(n, X, Y, true);
            float epsilon = 1e-2f;
            var observedDifferences =  new List<double>();
            for (int testIndex = 0; testIndex < nbTests; ++testIndex)
            {
                int weightIndexToUpdate = r.Next(w.Count);
                ComputeGradientAndReturnLoss(n, X, Y, false);
                var initialWeightValue = w[weightIndexToUpdate];
                var expectedGradient = dW[weightIndexToUpdate];

                w[weightIndexToUpdate] = initialWeightValue + epsilon;
                var lossUp = ComputeGradientAndReturnLoss(n, X, Y, false);
                //n.SaveLayers();
                w[weightIndexToUpdate] = initialWeightValue - epsilon;
                var lossDown = ComputeGradientAndReturnLoss(n, X, Y, false);
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
                    ComputeGradientAndReturnLoss(n, X, Y, false);
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
            return new Network(new NetworkConfig{LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy, ForceTensorflowCompatibilityMode = true},ImageDataGenerator.NoDataAugmentation, gpuDeviceId);
        }
    }
}
