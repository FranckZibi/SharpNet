using System;
using System.Linq;
using NUnit.Framework;
using SharpNet.Data;
using SharpNet.GPU;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using SharpNet;
using SharpNet.CPU;
using SharpNetTests.NonReg;

namespace SharpNetTests
{
    [TestFixture]
    [SuppressMessage("ReSharper", "PossibleNullReferenceException")]
    public class TestGradient
    {
	    private readonly Random _rand = new Random(0);
        private static readonly string logFileName = Utils.ConcatenatePathWithFileName(@"c:\temp\ML\", "TestGradient" + "_" + Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
        private static readonly Logger logger = new Logger(logFileName, true);

        [TestCase(true, false)]
        [TestCase(false, true)]
        public void TestGradientForDenseLayer(bool testWeights, bool testBias)
        {
            var X = FromNumpyArray(TestNetworkPropagation.X_2_1_4_4, "X");
            var Y = FromNumpyArray(TestNetworkPropagation.Y_2_1_4_4, "Y");
            var n = GetNetwork();
            n
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            n.Predict(X, false);
            var denseLayer = (DenseLayer)n.Layers[1];
            if (testWeights)
            {
                CompareExpectedVsObservedGradients(n, denseLayer.Weights as CpuTensor<double>, denseLayer.WeightGradients as CpuTensor<double>, X, Y, _rand);
            }
            if (testBias)
            {
                CompareExpectedVsObservedGradients(n, denseLayer.Bias as CpuTensor<double>,denseLayer.BiasGradients as CpuTensor<double>, X, Y, _rand);
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
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .AddConvolution(3, 3,1,1, 0.0)
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_RELU);
            n.Predict(X, false);
            var convLayer = (ConvolutionLayer)n.Layers[1];
            if (testWeights)
            {
                CompareExpectedVsObservedGradients(n, convLayer.Convolution as CpuTensor<double>, convLayer.ConvolutionGradients as CpuTensor<double>, X, Y, _rand);
            }
            if (testBias)
            {
                CompareExpectedVsObservedGradients(n, convLayer.ConvolutionBias as CpuTensor<double>,convLayer.ConvolutionBiasGradients as CpuTensor<double>, X, Y, _rand);
                
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
                .AddInput(X.Shape[1], X.Shape[2], X.Shape[3])
                .AddBatchNorm(1.0)
                .AddOutput(Y.Shape[1], cudnnActivationMode_t.CUDNN_ACTIVATION_ELU);
            n.Predict(X, false);
            //n.SaveLayers();
            var batchNormLayer = (BatchNormalizationLayer)n.Layers[1];
            if (testWeights)
            {
                CompareExpectedVsObservedGradients(n, batchNormLayer.Weights as CpuTensor<double>, batchNormLayer.WeightGradients as CpuTensor<double>, X, Y, _rand, 1);
            }
            if (testBias)
            {
                CompareExpectedVsObservedGradients(n, batchNormLayer.Bias as CpuTensor<double>, batchNormLayer.BiasGradients as CpuTensor<double>, X, Y, _rand, 1);
            }
        }

        private static CpuTensor<double> FromNumpyArray(string s, string description)
        {
            var X0 = TensorExtensions.FromNumpyArray(s, description);
            if (X0.UseSinglePrecision)
            {
                return X0.AsFloatCpu.ToDoublePrecision();
            }
            return X0 as CpuTensor<double>;
        }
        private static double ComputeGradientAndReturnLoss(Network n, CpuTensor<double> X, CpuTensor<double> yExpected, bool isTraining)
        {
            var yPredicted = n.Predict(X, isTraining) as CpuTensor<double>;
            n.BackwardPropagation(yExpected); //we compute gradients
            double result = 0;
            for (int i = 0; i < yExpected.Count; ++i)
            {
                double err = Math.Pow(yExpected[i] - yPredicted[i], 2);
                result += err / 2.0;
            }
            return result;
        }
        private static void AbsWeigthsWithMinimum(Network n, double min)
        {
            foreach (var l in n.TensorsIndependantOfBatchSize.OfType<CpuTensor<double>>())
            {
                for (int i = 0; i < l.Content.Length; ++i)
                {
                    l[i] = Math.Max(min, Math.Abs(l[i]));
                }
            }
        }
        private static void CompareExpectedVsObservedGradients(Network n, CpuTensor<double> w, CpuTensor<double> dW, CpuTensor<double> X, CpuTensor<double> Y, Random r, int nbTests = 100)
        {
            AbsWeigthsWithMinimum(n, 0.0);
            ComputeGradientAndReturnLoss(n, X, Y, true);
            double epsilon = 1e-8;
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
                logger.Info(errorMsg);
                throw new Exception(errorMsg);
            }
        }
        private static Network GetNetwork()
        {
            return new Network(new NetworkConfig(false) { Logger = logger, UseDoublePrecision = true, LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy });
        }
    }
}
