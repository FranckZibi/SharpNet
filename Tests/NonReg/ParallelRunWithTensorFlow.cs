﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using log4net;
using Newtonsoft.Json;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNet.Pictures;
using SharpNet.TextPreprocessing;
using SharpNetTests.Data;
// ReSharper disable AccessToDisposedClosure

namespace SharpNetTests.NonReg
{
    /// <summary>
    /// Sand Box to make // run with TensorFlow on several kind of networks
    /// </summary>
    [TestFixture]
    public class ParallelRunWithTensorFlow
    {
        private static readonly ILog Log = LogManager.GetLogger(typeof(ParallelRunWithTensorFlow));

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
            networkBuilder.Config.LogDirectory = "";
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
            var networkFromSavedFile = Network.ValueOf(savedModelFile);
            var yPredictedFromSavedFile = networkFromSavedFile.Predict(X, false);
            Assert.IsTrue(TestTensor.SameContent(yExpectedFromKeras, yPredictedFromSavedFile, 1e-5));

            File.Delete(savedModelFile);
            File.Delete(saveParametersFile);
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_YOLOV3()
        {
            var weightPath = Path.Combine(NetworkConfig.DefaultDataDirectory, "YOLO", "yolov3_weights.h5");
            if (!File.Exists(weightPath))
            {
                Console.WriteLine("ignoring test " + nameof(TestParallelRunWithTensorFlow_YOLOV3) + " because weight file is missing");
                return;
            }

            var networkBuilder = new YOLOV3NetBuilder();
            var network = networkBuilder.Value(new List<int> { 0 });
            //network.PropagationManager.LogPropagation = true; 
            network.LoadParametersFromH5File(weightPath, NetworkConfig.CompatibilityModeEnum.TensorFlow2);

            //var imagePaths = new DirectoryInfo(@"C:\Franck\Photos\2019\Madagascar").GetFiles("*.jpg").Select(f => f.FullName).ToList();
            var imagePaths = new List<string>{ @"C:\Projects\YOLOv3_TF2\data\images\test.jpg"};
            foreach(var imagePath in imagePaths)
            {
                Log.Info("processing "+imagePath);

                var imageSize = PictureTools.ImageSize(imagePath);
                using var originalBmp = new Bitmap(imagePath);
                PreferredResizedSizeForYoloV3(imageSize.Height, imageSize.Width, out var resizedHeight,out var resizedWidth);
                var resizedOriginalBitmap = PictureTools.ResizeImage(originalBmp, resizedWidth, resizedHeight, InterpolationMode.Bicubic);
                var content = BitmapContent.ValueFomSingleRgbBitmap(resizedOriginalBitmap);
                var X = content.Select((x, y, b) => b / 255f);
                X.Reshape(new []{1, content.Shape[0], content.Shape[1], content.Shape[2]});
                var yPredicted = network.Predict(X, false);
                var predictions = NonMaxSuppressionLayer.ExtractSelectedAfterNonMaxSuppression(yPredicted.ToCpuFloat(), 0, int.MaxValue, int.MaxValue, 0.5, 0.5);
                predictions.ForEach(p=>p.Box.UpSampling(imageSize.Height/ (double)resizedHeight, imageSize.Width/ (double)resizedWidth).Draw(originalBmp, p.CaptionFor(COCODataSet.CategoryIndexToDescription)));
                originalBmp.Save(imagePath+"_"+DateTime.Now.Ticks+"_output.jpg");
                
                Log.Info("finished processing of " + imagePath);
            }
        }

        /// <summary>
        /// the width and height of the processed image must be a multiple of '32' in YOLO V3
        /// </summary>
        /// <param name="originalHeight"></param>
        /// <param name="originalWidth"></param>
        /// <param name="resizedHeight"></param>
        /// <param name="resizedWidth"></param>
        private static void PreferredResizedSizeForYoloV3(int originalHeight, int originalWidth, out int resizedHeight, out int resizedWidth)
        {
            double capacity = 608 * 608;
            double originalCount = originalHeight * originalWidth;

            resizedHeight = originalHeight;
            resizedWidth = originalWidth;

            if (originalCount > capacity)
            {
                double coeff = Math.Sqrt(originalCount / capacity);
                resizedHeight = (int)(resizedHeight / coeff);
                resizedWidth = (int)(resizedWidth / coeff);
            }

            const int forcedSizeMultiple = 32;
            resizedHeight = forcedSizeMultiple * ((resizedHeight + forcedSizeMultiple - 1) / forcedSizeMultiple);
            resizedWidth = forcedSizeMultiple * ((resizedWidth + forcedSizeMultiple - 1) / forcedSizeMultiple);
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

            Log.Info("x_train" + Environment.NewLine + X.ToNumpy());
            Log.Info("y_train" + Environment.NewLine + Y.ToNumpy());


            Log.Info(network.Summary() + Environment.NewLine);

            var predict_before_tensor = network.Predict(X, false);
            var predict_before = PredictionToString(predict_before_tensor, "C# prediction_before");

            //network.LogContent();

            using var trainingDataSet = new InMemoryDataSet(X, Y, "", null, ImageNetDataSet._CategoryIndexToDescription);
            var lossAccuracyBefore = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);
            //network.LogContent();

            var predict_after_tensor = network.Predict(X, false);
            var predict_after = PredictionToString(predict_after_tensor, "C# prediction_after");

            //network.LogContent();

            var lossAccuracyAfter = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Network.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Network.MetricsToString(lossAccuracyAfter, ""));
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
            var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_2_3_4_5);
            var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_2_2);

            int batchSize = X.Shape[0];
            //var gpuDeviceId = -1;
            var gpuDeviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            LogFile = "TestParallelRunWithTensorFlow_Convolution",
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


            Log.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromConvNumpyArray("[[[[-0.4878799319267273, -0.6471760272979736], [-0.11215460300445557, 0.24113142490386963], [-0.5400518774986267, -0.8205036520957947]]]]").CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromConvNumpyArray("[[[[-0.7247111797332764, -0.3986714482307434], [-0.4940018653869629, 0.04389345645904541]]]]").CopyTo(((ConvolutionLayer)network.Layers[2]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.029460519552230835, 0.1628669798374176], [-0.28001704812049866, -0.23855498433113098], [0.07715305685997009, 0.11627233028411865], [0.32925912737846375, 0.011087954044342041], [0.12424156069755554, -0.05900973081588745], [-0.2703372836112976, 0.12233385443687439], [-0.08240920305252075, 0.006095200777053833], [-0.023135006427764893, 0.08786126971244812], [-0.2075882852077484, -0.3384675085544586], [0.10181871056556702, -0.08105111122131348], [0.04287368059158325, -0.014433145523071289], [-0.050517499446868896, 0.19285127520561218], [0.16756221652030945, -0.06256869435310364], [-0.1878374218940735, -0.17477598786354065], [0.3118181526660919, 0.36103251576423645], [0.16790542006492615, 0.27620890736579895], [0.21295377612113953, -0.15440134704113007], [0.03934970498085022, -0.35186851024627686], [-0.19449061155319214, -0.2855254113674164], [-0.08950188755989075, 0.2891680896282196], [-0.37375181913375854, 0.18617329001426697], [0.07124421000480652, 0.28268447518348694], [0.041756272315979004, 0.13584479689598083], [0.12497344613075256, 0.151188462972641], [0.3146173655986786, -0.22298070788383484], [-0.22048203647136688, -0.30460700392723083], [0.12072917819023132, -0.2646358907222748], [-0.15740737318992615, 0.17554828524589539], [0.13976749777793884, -0.357845664024353], [-0.365357369184494, -0.15716126561164856], [0.14519938826560974, 0.22951403260231018], [0.03488221764564514, 0.1870688498020172], [0.28289076685905457, 0.14199396967887878], [0.31583401560783386, 0.08595579862594604], [0.005727171897888184, 0.2800586521625519], [0.013508498668670654, 0.3192369043827057], [-0.14768590033054352, -0.05077126622200012], [-0.28260645270347595, -0.3034713864326477], [-0.05905658006668091, -0.3151003122329712], [-0.12471392750740051, -0.2689373791217804]]").CopyTo(((DenseLayer)network.Layers[6]).Weights);

            network.PropagationManager.LogPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);
            var lossAccuracyBefore = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Network.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Network.MetricsToString(lossAccuracyAfter, ""));
        }


        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Conv1D()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_3_4_5);
            var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_3_3);

            int batchSize = X.Shape[0];
            //var gpuDeviceId = -1;
            var gpuDeviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            LogFile = "TestParallelRunWithTensorFlow_Convolution",
                            LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1
                        }
                       .WithSGD(momentum, false),
                        new List<int> { gpuDeviceId }
                );

            network.Input(X.Shape[1], X.Shape[2], -1)
                .Conv1D(2, 3, 1, ConvolutionLayer.PADDING_TYPE.VALID, lambdaL2Regularization, true)
                .Conv1D(2, 3, 2, ConvolutionLayer.PADDING_TYPE.CAUSAL, lambdaL2Regularization, true)
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, lambdaL2Regularization, true)
                .Flatten()
                .Output(Y.Shape[1], lambdaL2Regularization, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            Log.Info(network.Summary() + Environment.NewLine);

            //network.Layers[1].Weights.ZeroMemory();
            TestNetworkPropagation.FromConvNumpyArray("[[[0.09934896230697632, -0.11215364933013916], [0.3982505798339844, 0.342079758644104], [-0.06867659091949463, -0.46536481380462646], [0.2547714114189148, -0.08702009916305542]], [[-0.5021747350692749, -0.1221388578414917], [-0.3608691096305847, 0.3861338496208191], [0.10946327447891235, -0.052802085876464844], [-0.016413629055023193, 0.3857215642929077]], [[0.4184006452560425, -0.2657143771648407], [0.296006977558136, -0.28657031059265137], [-0.016508877277374268, -0.2890245020389557], [0.1388271450996399, 0.02789127826690674]]]").CopyTo(network.Layers[1].Weights);
            //network.Layers[2].Weights.ZeroMemory();
            TestNetworkPropagation.FromConvNumpyArray("[[[0.39741700887680054, 0.5679424405097961], [0.103904128074646, 0.46203213930130005]], [[0.5664966702461243, -0.5104600191116333], [-0.4302336871623993, 0.2359222173690796]], [[0.1441558599472046, -0.3472554683685303], [0.3229832053184509, -0.13790547847747803]]]").CopyTo(network.Layers[2].Weights);
            TestNetworkPropagation.FromConvNumpyArray("[[[-1.0770841836929321, 0.557166576385498], [0.405431866645813, -0.2015085220336914]]]").CopyTo(network.Layers[3].Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.38363194465637207, 0.2582963705062866, 0.15701913833618164], [0.5796942710876465, -0.42992860078811646, 0.28377270698547363], [-0.34947991371154785, 0.8033483028411865, -0.22690773010253906], [0.8054455518722534, 0.22870910167694092, -0.36302077770233154]]").CopyTo(network.Layers[5].Weights);
            //TestNetworkPropagation.FromConvNumpyArray("[[[0.4353485107421875, 0.6221498250961304, 0.11382126808166504], [0.5061308145523071, 0.6205660104751587, -0.5591809749603271]], [[-0.47129738330841064, 0.25843989849090576, 0.1579148769378662], [-0.38039931654930115, 0.3538104295730591, -0.15106791257858276]]]").CopyTo(network.Layers[2].Weights);
            //TestNetworkPropagation.FromNumpyArray("[[0.39741700887680054, 0.5679424405097961, 0.103904128074646], [0.46203213930130005, 0.5664966702461243, -0.5104600191116333], [-0.4302336871623993, 0.2359222173690796, 0.1441558599472046], [-0.3472554683685303, 0.3229832053184509, -0.13790547847747803], [0.31644493341445923, -0.011439502239227295, -0.4673982560634613], [-0.6368072032928467, 0.23920577764511108, 0.265876829624176], [0.4810141921043396, 0.5506053566932678, -0.6353087425231934], [-0.13411635160446167, 0.4802754521369934, 0.136569082736969], [-0.1479487419128418, 0.23149830102920532, 0.14344310760498047]]").CopyTo(network.Layers[4].Weights);

            network.PropagationManager.LogPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);
            var lossAccuracyBefore = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Network.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Network.MetricsToString(lossAccuracyAfter, ""));
        }
        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Embedding()
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            int vocabularySize = 3;
            int embeddingDim = 5;
            int maxWordsBySentence = 4;

            var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1, 2, 1, 1], [2, 2, 1, 1]], numpy.float)");
            var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1], [0]], numpy.float)");


            int batchSize = X.Shape[0];
            var deviceId = -1;
            //var deviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            LogFile = "Embedding",
                            LossFunction = NetworkConfig.LossFunctionEnum.BinaryCrossentropy,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2
                        }
                       .WithSGD(momentum, false),
                        new List<int> { deviceId }
                );

            network
                .InputAndEmbedding(maxWordsBySentence, vocabularySize, embeddingDim, 0.0)
                .Flatten()
                .Dense(1, 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);


            Log.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromNumpyArray("[[-0.020802486687898636, -0.02934335544705391, 0.0035390742123126984, 0.006125748157501221, -0.008332550525665283], [0.0307827927172184, -0.0006774887442588806, 0.0498129241168499, 0.019673515111207962, -0.037462640553712845],[0.020981673151254654, 0.016241561621427536, 0.007225655019283295, -0.013524651527404785, -0.007948171347379684]]")
                .CopyTo(((EmbeddingLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.05924016237258911], [-0.2979503273963928], [0.39012110233306885], [0.2964285612106323], [0.15513628721237183], [0.032458603382110596], [-0.5190843939781189], [0.3992980718612671], [-0.03236877918243408], [-0.12109190225601196], [0.4128159284591675], [0.14623379707336426], [-0.5325161814689636], [0.38246530294418335], [-0.4191945493221283], [0.4918263554573059], [-0.30854684114456177], [0.1737397313117981], [-0.40517792105674744], [-0.3750319480895996]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);

            network.PropagationManager.LogPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Network.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Network.MetricsToString(lossAccuracyAfter, ""));
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Embedding_GlobalPooling()
        {
            const int numEpochs = 5;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            int batchSize = 2;
            var deviceId = -1;
            //var deviceId = 0;
            int vocabularySize = 3;
            int embeddingDim = 5;
            int maxWordsBySentence = 4;

            var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1, 1, 1, 2], [2, 2, 2, 2], [1, 2, 2, 2],[1, 1, 1, 1]], numpy.float)");
            var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1], [0], [0], [1]], numpy.float)");


            var networkConfig = new NetworkConfig
            {
                LogFile = "Embedding_GlobalPooling",
                LossFunction = NetworkConfig.LossFunctionEnum.BinaryCrossentropy,
                RandomizeOrder = false,
                CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
            };
            networkConfig.DataAugmentation = DataAugmentationConfig.NoDataAugmentation;


            var network = new Network(
                        networkConfig
                       .WithAdam(0.9, 0.999, 1e-7),
                       //.WithSGD(momentum, false),
                        new List<int> { deviceId }
                );
            network.PropagationManager.LogPropagation = true;

            network
                .InputAndEmbedding(maxWordsBySentence, vocabularySize, embeddingDim, 0.0)
                .GlobalAvgPooling()
                .Dense(4, 0.0, false).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(1, 0.0, false).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);


            Log.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromNumpyArray("[[-0.020802486687898636, -0.02934335544705391, 0.0035390742123126984, 0.006125748157501221, -0.008332550525665283], [0.0307827927172184, -0.0006774887442588806, 0.0498129241168499, 0.019673515111207962, -0.037462640553712845], [0.020981673151254654, 0.016241561621427536, 0.007225655019283295, -0.013524651527404785, -0.007948171347379684]]")
                .CopyTo(((EmbeddingLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.09049081802368164, -0.45512667298316956, 0.5959198474884033, 0.4528021812438965], [0.2369745969772339, 0.04958134889602661, -0.7929145097732544, 0.6099379062652588], [-0.04944407939910889, -0.18497097492218018, 0.6305867433547974, 0.22337579727172852], [-0.813431978225708, 0.5842254161834717, -0.6403303146362305, 0.7512772083282471], [-0.47131311893463135, 0.26539182662963867, -0.6189195513725281, -0.5728708505630493]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.6677531003952026], [0.5261931419372559], [-0.026724934577941895], [0.8222856521606445]]")
                .CopyTo(((DenseLayer)network.Layers[5]).Weights);


            network.PropagationManager.LogPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);


            var lossAccuracyBefore = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("- Fit -------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, trainingDataSet, learningRate, numEpochs, batchSize, null);

            Log.Info("-");
            Log.Info("- Using Trained Network -------------------------------------------------------------------");
            Log.Info("-");

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# batchSize= " + batchSize);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Network.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Network.MetricsToString(lossAccuracyAfter, ""));
        }


        private class SarcasmEntry
        {
            [JsonProperty("article_link")]
            public string ArticleLink { get; set; }
            [JsonProperty("headline")]
            public string Headline { get; set; }
            [JsonProperty("is_sarcastic")]
            public bool IsSarcastic { get; set; }
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Sarcasm()
        {
            const int numEpochs = 30;
            const double learningRate = 0.001;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            int batchSize = 128;
            //var deviceId = -1;
            var deviceId = 0;
            int vocab_size = 10000;
            int embedding_dim = 16;
            int max_length = 100;
            string oov_tok = "<OOV>";
            int training_size = 20000;

            var jsonText = File.ReadAllText(@"C:\Download\sarcasm.json");
            List<SarcasmEntry> allEntries = JsonConvert.DeserializeObject<List< SarcasmEntry>>(jsonText);

            var trainingEntries = allEntries.Take(training_size).ToList();
            var trainingHeadlines = trainingEntries.Select(e => e.Headline).ToList();
            var tokenizer = new Tokenizer(vocab_size, oov_tok);
            tokenizer.FitOnTexts(trainingHeadlines);
            //var word_index = tokenizer.WordIndex;

            var training_sequences = tokenizer.TextsToSequences(trainingHeadlines);
            var X  = PadSequenceTools.PadSequence(training_sequences, max_length, false, false).Select(x=>(float)x);
            var Y  = new CpuTensor<float>(new[]{X.Shape[0],1}, trainingEntries.Select(e => e.IsSarcastic?1f:0f).ToArray());

            var networkConfig = new NetworkConfig
            {
                LogFile = "TestParallelRunWithTensorFlow_Sarcasm",
                LossFunction = NetworkConfig.LossFunctionEnum.BinaryCrossentropy,
                RandomizeOrder = true,
                CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
            };
            networkConfig.DataAugmentation = DataAugmentationConfig.NoDataAugmentation;

            var network = new Network(
                        networkConfig
                        .WithAdam(0.9, 0.999, 1e-7)
                        //.WithSGD()
                        ,
                        new List<int> { deviceId }
                );

            network
                .InputAndEmbedding(max_length, vocab_size, embedding_dim, 0.0)
                .GlobalAvgPooling()
                .Dense(24, 0.0, false).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(1, 0.0, false).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

        
            //Log.Info(network.Summary() + Environment.NewLine);
            //network.PropagationManager.LogPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);


            var validationEntries = allEntries.Skip(training_size).ToList();
            var validationHeadlines = validationEntries.Select(e => e.Headline).ToList();
            var validation_sequences = tokenizer.TextsToSequences(validationHeadlines);
            var X_val = PadSequenceTools.PadSequence(validation_sequences, max_length, false, false).Select(x => (float)x);
            var Y_val = new CpuTensor<float>(new[] { X_val.Shape[0], 1 }, validationEntries.Select(e => e.IsSarcastic ? 1f : 0f).ToArray());
            using InMemoryDataSet validationDataSet = new InMemoryDataSet(X_val, Y_val);

            var lossAccuracyBefore = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            //Log.Info("-");
            //Log.Info("--------------------------------------------------------------------");
            //Log.Info("-");

            TestNetwork.Fit(network, trainingDataSet, learningRate, numEpochs, batchSize, validationDataSet);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Network.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Network.MetricsToString(lossAccuracyAfter, ""));
        }


        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_DownSampling2D()
        {
            const int numEpochs = 10;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_2_3_4_5);
            var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_2_3);

            int batchSize = X.Shape[0];
            var gpuDeviceId = -1;
            //var gpuDeviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            LogFile = "TestParallelRunWithTensorFlow_DownSampling2D",
                            LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1,
                            ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM
                        }
                       .WithSGD(momentum, false),
                        new List<int> { gpuDeviceId }
                );


            network.PropagationManager.LogPropagation = true;

            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(4,1,1,ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true)
                .UpSampling2D(3,2,UpSampling2DLayer.InterpolationEnum.Nearest)
                .Convolution(1, 3, 2, ConvolutionLayer.PADDING_TYPE.SAME, 0.0, true)
                .Output(Y.Shape[1], 0.0, cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            network.PropagationManager.LogPropagation = true;

            Log.Info(network.Summary() + Environment.NewLine);


            TestNetworkPropagation.FromConvNumpyArray("[[[[-0.41233378648757935, -0.5469635725021362, -0.09478795528411865, 0.20379328727722168], [-0.45642712712287903, -0.6934521198272705, 0.7060458660125732, 0.6550993919372559], [-0.40876543521881104, 0.5751461982727051, 0.0005752444267272949, 0.8542157411575317]]]]").CopyTo(((ConvolutionLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromConvNumpyArray("[[[[-0.1615283042192459], [-0.0656551718711853], [0.1326923966407776], [0.21013426780700684]], [[0.23147475719451904], [0.15308880805969238], [0.0008010268211364746], [0.2704615592956543]], [[-0.2763732671737671], [-0.11263367533683777], [-0.3622085750102997], [0.03678843379020691]]], [[[-0.1616799682378769], [0.029316306114196777], [-0.15289030969142914], [-0.21387864649295807]], [[0.032195329666137695], [-0.013419240713119507], [0.10481679439544678], [-0.18447379767894745]], [[-0.15118040144443512], [0.052129119634628296], [0.07085898518562317], [-0.08211708068847656]]], [[[-0.02411407232284546], [0.17931300401687622], [-0.2963199317455292], [-0.019487440586090088]], [[-0.2584547698497772], [0.23713970184326172], [-0.351848304271698], [0.3424469232559204]], [[0.22793227434158325], [0.13822901248931885], [-0.12481275200843811], [-0.32772859930992126]]]]").CopyTo(((ConvolutionLayer)network.Layers[3]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.07366013526916504, 0.3170207142829895, -0.1550242304801941], [0.420951247215271, -0.4191424548625946, 0.3381590247154236], [0.11008310317993164, 0.0986890196800232, 0.31357908248901367], [0.41440945863723755, 0.30317842960357666, 0.3536931872367859], [-0.010290741920471191, -0.21904385089874268, -0.020769357681274414], [-0.2869524359703064, -0.3439455032348633, 0.2285328507423401], [-0.022606879472732544, -0.1754196584224701, -0.12093043327331543], [-0.19505150616168976, 0.32367968559265137, 0.27787232398986816], [0.1375676393508911, -0.1417226493358612, 0.33683180809020996], [-0.36117273569107056, 0.001855224370956421, 0.24049299955368042], [-0.02008679509162903, 0.22243833541870117, -0.27483871579170227], [-0.20811842381954193, -0.17607355117797852, -0.1847764253616333], [-0.41185829043388367, 0.14473176002502441, 0.10743755102157593], [0.3232056498527527, -0.2687329947948456, 0.041926443576812744], [-0.07551324367523193, 0.23673099279403687, -0.4212562143802643], [-0.32285287976264954, -0.20976179838180542, 0.35986894369125366], [-0.42236655950546265, 0.06221747398376465, 0.19280701875686646], [-0.1036037802696228, 0.22280341386795044, 0.2663360834121704], [-0.278300404548645, 0.3701552152633667, -0.3987610638141632], [-0.2845539450645447, 0.08112376928329468, -0.06442150473594666], [0.13321810960769653, 0.39671868085861206, -0.34261322021484375], [-0.23947212100028992, -0.10445082187652588, -0.36301395297050476], [0.20646917819976807, 0.11567127704620361, 0.15597444772720337], [-0.3057088851928711, 0.39422833919525146, -0.23814217746257782], [0.1633470058441162, 0.12872058153152466, 0.2478216290473938], [-0.3868710696697235, -0.335817813873291, 0.42601829767227173], [-0.3151834011077881, 0.30162113904953003, -0.06157597899436951], [-0.19710223376750946, 0.0573333203792572, 0.2074006199836731], [-0.28093406558036804, 0.2030026912689209, 0.4050601124763489], [0.29869991540908813, -0.31979823112487793, 0.41144388914108276]]").CopyTo(((DenseLayer)network.Layers[4]).Weights);


            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);
            var lossAccuracyBefore = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Network.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Network.MetricsToString(lossAccuracyAfter, ""));
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Huber()
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const int batchSize = 2;

            var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_2_3_4_5);
            var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_2_3);

            var deviceId = -1;
            //var deviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            LogFile = "Huber",
                            LossFunction = NetworkConfig.LossFunctionEnum.Huber,
                            //LossFunction = NetworkConfig.LossFunctionEnum.BinaryCrossentropy,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2
                        }
                       .WithSGD(momentum, false),
                        new List<int> { deviceId }
                );

            network
                .Input(X.Shape[1], 1, -1)
                .Dense(3, 0.0, false)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(1, 0.0, false)
                ;


            Log.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromNumpyArray("[[0.17207741737365723, -0.19425582885742188, 0.6897902488708496], [0.5924994945526123, -0.11895132064819336, -0.8060355186462402], [0.44127702713012695, -0.15072321891784668, -0.8697922229766846]]")
                .CopyTo(((DenseLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.6883463859558105], [0.9837051630020142], [0.17996716499328613]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);

            network.PropagationManager.LogPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# batchSize= " + batchSize);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Network.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Network.MetricsToString(lossAccuracyAfter, ""));
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Mse()
        {
            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[0,1,2],[3,4,5]], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[0],[5]], numpy.float)");

            var X = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.X_2_3_4_5);
            var Y = TestNetworkPropagation.FromNumpyArray(TestNetworkPropagation.Y_2_3);


            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            int batchSize = X.Shape[0];


            var deviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            LogFile = "Mse",
                            LossFunction = NetworkConfig.LossFunctionEnum.Mse,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                            Metrics = new List<NetworkConfig.Metric> {NetworkConfig.Metric.Loss, NetworkConfig.Metric.Mae, NetworkConfig.Metric.Mse}
                        }
                       .WithSGD(momentum, false),
                        new List<int> { deviceId }
                );

            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Dense(3, 0.0, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(3, 0.0, false)
                ;


            Log.Info(network.Summary() + Environment.NewLine);
            TestNetworkPropagation.FromNumpyArray("[[0.14902335405349731, -0.16823047399520874, 0.5973758101463318], [0.513119637966156, -0.10301488637924194, -0.6980472207069397], [0.3821571469306946, -0.13053011894226074, -0.7532621622085571], [-0.18320828676223755, -0.5413036346435547, 0.579200804233551], [0.16419488191604614, -0.07920318841934204, -0.024620473384857178]]")
                .CopyTo(((DenseLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.22044727206230164, 0.3150377571582794, 0.05763563513755798], [0.2562893331050873, 0.31423577666282654, -0.2831522822380066], [-0.23865070939064026, 0.13086608052253723, 0.07996329665184021], [-0.19262267649173737, 0.17915889620780945, -0.07649621367454529], [0.17553207278251648, -0.006345480680465698, -0.2592658996582031], [-0.3532370626926422, 0.13268747925758362, 0.14748194813728333], [0.26681867241859436, 0.3054209053516388, -0.3524059057235718], [-0.0743943452835083, 0.26640889048576355, 0.07575491070747375], [-0.08206719160079956, 0.1284121572971344, 0.07956790924072266], [0.2841203510761261, 0.012592524290084839, 0.15496674180030823], [-0.17980638146400452, -0.2484310269355774, 0.04503124952316284], [-0.2442535012960434, 0.24186745285987854, 0.12160143256187439], [-0.27119147777557373, -0.3446856439113617, -0.30974677205085754], [-0.3750991225242615, 0.23870810866355896, -0.28597140312194824], [0.15718379616737366, -0.0744141936302185, 0.09016254544258118], [-0.2230645716190338, 0.31610485911369324, -0.18952983617782593], [-0.37350326776504517, -0.3442955017089844, 0.3457939922809601], [0.209515780210495, 0.05385535955429077, 0.107828289270401], [0.27271541953086853, 0.13648775219917297, 0.30335161089897156], [0.19686904549598694, -0.3310122787952423, 0.27363476157188416], [-0.010377317667007446, 0.33995702862739563, 0.004759669303894043], [0.0042154788970947266, -0.11687412858009338, -0.06438267230987549], [0.371217280626297, 0.04651784896850586, -0.1674891859292984], [0.3474888503551483, 0.28037092089653015, -0.04222455620765686], [0.0916936993598938, 0.2884680926799774, 0.0825285017490387], [0.30377236008644104, -0.11384806036949158, -0.3356134295463562], [-0.37017637491226196, -0.10759112238883972, 0.0320684015750885], [-0.029354870319366455, 0.2205376923084259, 0.10602417588233948], [-0.049274712800979614, 0.3876277506351471, -0.28544583916664124], [-0.37114545702934265, -0.233595073223114, -0.23805370926856995], [0.045278966426849365, 0.16116967797279358, -0.01359209418296814], [0.28720858693122864, -0.10025259852409363, -0.09117457270622253], [-0.22608521580696106, -0.06644889712333679, 0.20117756724357605], [-0.0758948028087616, -0.3437873423099518, 0.3798452317714691], [-0.15284530818462372, -0.3742479383945465, 0.13099047541618347], [-0.14322248101234436, -0.18771331012248993, 0.3071592152118683]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);

            network.PropagationManager.LogPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeMetricsForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# batchSize= " + batchSize);
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Network.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Network.MetricsToString(lossAccuracyAfter, ""));
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Recurrent()
        {

            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1.7],[2.5],[3.8]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [5.2] ], numpy.float)");

            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1.7],[2.5],[3.8]] , [[2.5],[3.8],[5.2]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[5.2],[6.6]], numpy.float)");
            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[10.0],[20.0]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [30.0] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[20.0],[30.0]] ], numpy.float)");


            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1.0],[2.0],[3]] , [[2.0],[3.0],[4]] , [[3.0],[4.0],[5]]  , [[4.0],[5.0],[6]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[4]] , [[5]] , [[6]] , [[7]] ], numpy.float)");


            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1.0],[2.0]] , [[2.0],[3.0]] , [[3.0],[4.0]]  , [[4.0],[5.0]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[3.0]] , [[4.0]] , [[5.0]] , [[6.0]] ], numpy.float)");

            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1.0],[2.0]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[3.0]] ], numpy.float)");
            ////var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[2.0],[3.0]] ], numpy.float)");

            var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1.0],[2.0]] , [[2.0],[3.0]] , [[3.0],[4.0]]  , [[4.0],[5.0]] ], numpy.float)");
            var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[2.0],[3.0]] , [[3.0],[4.0]] , [[4.0],[5.0]] , [[5.0],[6.0]] ], numpy.float)");
            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1.0],[2.0]] , [[2.0],[3.0]]  ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[2.0],[3.0]] , [[3.0],[4.0]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[3.0]] , [[4.0]] ], numpy.float)");
            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1.0],[2.0]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[2.0],[3.0]]  ], numpy.float)");

            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1]], [[2]], [[3]], [[4]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[2]] , [[3]], [[4]], [[5]] ], numpy.float)");

            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1],[2]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[3]] ], numpy.float)");

            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1],[2]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[2],[3]] ], numpy.float)");

            //var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[1]] ], numpy.float)");
            //var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([ [[2]]] ], numpy.float)");

            const int numEpochs = 15;
            const double learningRate = 0.1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            int batchSize = X.Shape[0];
            int timeSteps = X.Shape[1];     //number of words in each sentence
            int inputSize = X.Shape[2];     //number of distinct words in the dictionary 
            var returnSequences = Y.Shape[1] != 1;
            var deviceId = 0;
            int hiddenSize = 2;

            var network = new Network(
                        new NetworkConfig
                        {
                            LogFile = "GRU",
                            LossFunction = NetworkConfig.LossFunctionEnum.Huber,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                            Metrics = new List<NetworkConfig.Metric> { NetworkConfig.Metric.Loss, NetworkConfig.Metric.Mae}
                        }
                       .WithSGD(momentum, false),
                        new List<int> { deviceId }
                );


            network
                .Input(timeSteps, inputSize, -1)
                .SimpleRNN(hiddenSize, returnSequences, true, false)
                .Dense(1, 0.0, true)
                ;


            Log.Info(network.Summary() + Environment.NewLine);

            //1 unit
            //network.Layers[1].Weights.ZeroMemory();
            //TestNetworkPropagation.FromNumpyArray("[[0.9734686613082886]]").CopyTo(network.Layers[2].Weights);

            //1 unit bidirectional
            //network.Layers[1].Weights.ZeroMemory();
            //TestNetworkPropagation.FromNumpyArray("[[-1.243709683418274], [0.6433604955673218]]").CopyTo(network.Layers[2].Weights);

            //2 units
            //network.Layers[1].Weights.ZeroMemory();
            //TestNetworkPropagation.FromNumpyArray("[[-1.243709683418274], [0.6433604955673218]]").CopyTo(((DenseLayer)network.Layers[2]).Weights);

            //2 units bidirectional
            network.Layers[1].Weights.ZeroMemory();
            TestNetworkPropagation.FromNumpyArray("[[0.18850135803222656], [-0.2127966284751892], [0.7556273937225342], [0.6490507125854492]]").CopyTo(network.Layers[2].Weights);

            network.PropagationManager.LogPropagation = true;
            var predict_before = network.Predict(X, false);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# batchSize= " + batchSize);
            Log.Info("C# hiddenSize= " + hiddenSize);
            Log.Info("C# return_sequences= " + returnSequences);
            Log.Info(predict_before.ToShapeAndNumpy());
            Log.Info("C# metrics_before= " + Network.MetricsToString(network.EpochData[0].TrainingMetrics, ""));
            Log.Info(predict_after.ToShapeAndNumpy());
            Log.Info("C# metrics_after= " + Network.MetricsToString(network.EpochData.Last().TrainingMetrics, ""));
        }
    }
}
