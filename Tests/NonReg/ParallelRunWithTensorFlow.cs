using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNetTests.Data;

namespace SharpNetTests.NonReg
{
    /// <summary>
    /// Sand Box to make // run with TensorFlow on several kind of networks
    /// </summary>
    [TestFixture]
    public class ParallelRunWithTensorFlow
    {
        [Test]
        public void TestParallelRunWithTensorFlow_Efficientnet_Inference()
        {
            var xFileName = Path.Combine(NetworkConfig.DefaultDataDirectory, "NonReg", "X_1_224_224_3.txt");
            var yExpectedFileName = Path.Combine(NetworkConfig.DefaultDataDirectory, "NonReg", "YExpected_1_224_224_3.txt");
            if (!File.Exists(xFileName) || !File.Exists(yExpectedFileName))
            {
                Console.WriteLine("ignoring test "+nameof(TestParallelRunWithTensorFlow_Efficientnet_Inference)+" because some files are missing");
                return;
            }

            var X = TestNetworkPropagation.FromNumpyArray(File.ReadAllText(xFileName));
            X = (CpuTensor<float>)X.ChangeAxis(new[] { 0, 3, 1, 2 });
            var yExpectedFromKeras = TestNetworkPropagation.FromNumpyArray(File.ReadAllText(yExpectedFileName));

            //we ensure that the network prediction is the same as in Keras
            var networkBuilder = EfficientNetBuilder.CIFAR10();
            networkBuilder.SetResourceId(0);
            var network = networkBuilder.EfficientNetB0(true, "imagenet", new[] {3, 224, 224});
            var yPredicted = network.Predict(X, false);
            Assert.IsTrue(TestTensor.SameContent(yExpectedFromKeras, yPredicted, 1e-5));

            //we save the network
            var savedModelFile = Path.Combine(NetworkConfig.DefaultLogDirectory, "test_EfficientNetB0.txt");
            var saveParametersFile = Network.ModelFilePath2ParameterFilePath(savedModelFile);
            network.SaveModelAndParameters(savedModelFile, saveParametersFile);
            network.Dispose();

            //we ensure that the saved version of the network behave the same as the original one
            var networkFromSavedFile = Network.ValueOf(savedModelFile, saveParametersFile);
            var yPredictedFromSavedFile = networkFromSavedFile.Predict(X, false);
            Assert.IsTrue(TestTensor.SameContent(yExpectedFromKeras, yPredictedFromSavedFile, 1e-5));

            File.Delete(savedModelFile);
            File.Delete(saveParametersFile);
        }
        
        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Efficientnet()
        {
            const int numEpochs = 1;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;

            var networkBuilder = EfficientNetBuilder.CIFAR10();
            networkBuilder.SetResourceId(0);

            //int defaultHeight = 32;
            int defaultHeight = 224;

            var network = networkBuilder.EfficientNetB0(true, "imagenet", new[] { 3, defaultHeight, defaultHeight });
            //network.Save();
            //var logFileName = Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory, "Efficientnet_" + System.Diagnostics.Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            //var logger = new Logger(logFileName, true);

            //var xShape = new[] { 1, 3, defaultHeight, defaultHeight };
            var X = TestNetworkPropagation.FromNumpyArray(Path.Combine(NetworkConfig.DefaultDataDirectory, "NonReg", "X_1_224_224_3.txt"));
            X = (CpuTensor<float>)X.ChangeAxis(new[] { 0, 3, 1, 2 });
            //for (int i = 0; i < X.Count; ++i)
            //{
            //    X.Content[i] = 0;
            //}


            //var X = new CpuTensor<float>(xShape, null, "input_1");
            //X.Content[0] = 1;
            //Utils.RandomizeNormalDistribution(X.Content, new Random(), 0, 1);
            int batchSize = X.Shape[0];
            var Y = new CpuTensor<float>(new[] { batchSize, 1000 }, null);
            Y.SpanContent[388] = 1; //panda

            network.Info("x_train" + Environment.NewLine + X.ToNumpy());
            network.Info("y_train" + Environment.NewLine + Y.ToNumpy());


            network.Info(network.Summary() + Environment.NewLine);

            var predict_before_tensor = network.Predict(X, false);
            var predict_before = PredictionToString(predict_before_tensor, "C# prediction_before");

            //network.LogContent();

            var trainingDataSet = new InMemoryDataSet(X, Y, new int[batchSize], "", null);
            var lossAccuracyBefore = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            network.Info("-");
            network.Info("--------------------------------------------------------------------");
            network.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);
            //network.LogContent();

            var predict_after_tensor = network.Predict(X, false);
            var predict_after = PredictionToString(predict_after_tensor, "C# prediction_after");

            //network.LogContent();

            var lossAccuracyAfter = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            network.Info("C# numEpochs= " + numEpochs);
            network.Info("C# learningRate= " + learningRate);
            network.Info("C# l2regularizer= " + lambdaL2Regularization);
            network.Info("C# momentum= " + momentum);
            network.Info(predict_before);
            network.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            network.Info(predict_after);
            network.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
        }

        private static string PredictionToString(Tensor prediction, string description)
        {
            var tmp = prediction.ToCpuFloat().ReadonlyContent;

            string result = description + " " + Tensor.ShapeToString(prediction.Shape) + Environment.NewLine;
            int idxMax = tmp.IndexOf(tmp.Max());
            result += description + "[" + idxMax + "]=" + tmp[idxMax] + Environment.NewLine;
            result += prediction.ToNumpy();
            return result;
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Convolution()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var logFileName = Utils.ConcatenatePathWithFileName(NetworkConfig.DefaultLogDirectory, "NetworkPropagation" + "_" + System.Diagnostics.Process.GetCurrentProcess().Id + "_" + System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            var logger = new Logger(logFileName, true);

            var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_2_3_4_5);
            var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_2_2);

            int batchSize = X.Shape[0];
            //var gpuDeviceId = -1;
            var gpuDeviceId = 0;
            var network = new Network(new NetworkConfig
            {
                Logger = logger,
                LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                RandomizeOrder = false,
                CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1
            }
                       .WithSGD(momentum, false),
                        new List<int> {gpuDeviceId}
                );

            network.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true)
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true)
                .GlobalAvgPooling()
                .MultiplyLayer(1, 3)
                .Flatten()
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);


            logger.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromConvNumpyArray("[[[[-0.4878799319267273, -0.6471760272979736], [-0.11215460300445557, 0.24113142490386963], [-0.5400518774986267, -0.8205036520957947]]]]").CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromConvNumpyArray("[[[[-0.7247111797332764, -0.3986714482307434], [-0.4940018653869629, 0.04389345645904541]]]]").CopyTo(((ConvolutionLayer)network.Layers[2]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.029460519552230835, 0.1628669798374176], [-0.28001704812049866, -0.23855498433113098], [0.07715305685997009, 0.11627233028411865], [0.32925912737846375, 0.011087954044342041], [0.12424156069755554, -0.05900973081588745], [-0.2703372836112976, 0.12233385443687439], [-0.08240920305252075, 0.006095200777053833], [-0.023135006427764893, 0.08786126971244812], [-0.2075882852077484, -0.3384675085544586], [0.10181871056556702, -0.08105111122131348], [0.04287368059158325, -0.014433145523071289], [-0.050517499446868896, 0.19285127520561218], [0.16756221652030945, -0.06256869435310364], [-0.1878374218940735, -0.17477598786354065], [0.3118181526660919, 0.36103251576423645], [0.16790542006492615, 0.27620890736579895], [0.21295377612113953, -0.15440134704113007], [0.03934970498085022, -0.35186851024627686], [-0.19449061155319214, -0.2855254113674164], [-0.08950188755989075, 0.2891680896282196], [-0.37375181913375854, 0.18617329001426697], [0.07124421000480652, 0.28268447518348694], [0.041756272315979004, 0.13584479689598083], [0.12497344613075256, 0.151188462972641], [0.3146173655986786, -0.22298070788383484], [-0.22048203647136688, -0.30460700392723083], [0.12072917819023132, -0.2646358907222748], [-0.15740737318992615, 0.17554828524589539], [0.13976749777793884, -0.357845664024353], [-0.365357369184494, -0.15716126561164856], [0.14519938826560974, 0.22951403260231018], [0.03488221764564514, 0.1870688498020172], [0.28289076685905457, 0.14199396967887878], [0.31583401560783386, 0.08595579862594604], [0.005727171897888184, 0.2800586521625519], [0.013508498668670654, 0.3192369043827057], [-0.14768590033054352, -0.05077126622200012], [-0.28260645270347595, -0.3034713864326477], [-0.05905658006668091, -0.3151003122329712], [-0.12471392750740051, -0.2689373791217804]]").CopyTo(((DenseLayer)network.Layers[6]).Weights);

            var predict_before = network.Predict(X, false).ToNumpy();
            network.LogContent();

            var trainingDataSet = new InMemoryDataSet(X, Y, new int[X.Shape[0]], "", null);
            var lossAccuracyBefore = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            logger.Info("-");
            logger.Info("--------------------------------------------------------------------");
            logger.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            network.LogContent();
            var lossAccuracyAfter = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            logger.Info("C# numEpochs= " + numEpochs);
            logger.Info("C# learningRate= " + learningRate);
            logger.Info("C# l2regularizer= " + lambdaL2Regularization);
            logger.Info("C# momentum= " + momentum);
            logger.Info("C# prediction_before= " + predict_before);
            logger.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            logger.Info("C# prediction_after= " + predict_after);
            logger.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
        }
    }
}