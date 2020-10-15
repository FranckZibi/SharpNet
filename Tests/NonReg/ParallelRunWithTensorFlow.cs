using System;
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
            var lossAccuracyBefore = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);
            //network.LogContent();

            var predict_after_tensor = network.Predict(X, false);
            var predict_after = PredictionToString(predict_after_tensor, "C# prediction_after");

            //network.LogContent();

            var lossAccuracyAfter = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info(predict_before);
            Log.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            Log.Info(predict_after);
            Log.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
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
            var lossAccuracyBefore = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# prediction_before= " + predict_before);
            Log.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            Log.Info("C# prediction_after= " + predict_after);
            Log.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
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
                .Dense(1, 0.0)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);


            Log.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromNumpyArray("[[-0.020802486687898636, -0.02934335544705391, 0.0035390742123126984, 0.006125748157501221, -0.008332550525665283], [0.0307827927172184, -0.0006774887442588806, 0.0498129241168499, 0.019673515111207962, -0.037462640553712845],[0.020981673151254654, 0.016241561621427536, 0.007225655019283295, -0.013524651527404785, -0.007948171347379684]]")
                .CopyTo(((EmbeddingLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.05924016237258911], [-0.2979503273963928], [0.39012110233306885], [0.2964285612106323], [0.15513628721237183], [0.032458603382110596], [-0.5190843939781189], [0.3992980718612671], [-0.03236877918243408], [-0.12109190225601196], [0.4128159284591675], [0.14623379707336426], [-0.5325161814689636], [0.38246530294418335], [-0.4191945493221283], [0.4918263554573059], [-0.30854684114456177], [0.1737397313117981], [-0.40517792105674744], [-0.3750319480895996]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);

            network.PropagationManager.LogPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# prediction_before= " + predict_before);
            Log.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            Log.Info("C# prediction_after= " + predict_after);
            Log.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
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
                .Dense(4, 0.0).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(1, 0.0).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);


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


            var lossAccuracyBefore = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("- Fit -------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, trainingDataSet, learningRate, numEpochs, batchSize, null);

            Log.Info("-");
            Log.Info("- Using Trained Network -------------------------------------------------------------------");
            Log.Info("-");

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# batchSize= " + batchSize);
            Log.Info("C# prediction_before= " + predict_before);
            Log.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            Log.Info("C# prediction_after= " + predict_after);
            Log.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
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
                .Dense(24, 0.0).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(1, 0.0).Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);

        
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

            var lossAccuracyBefore = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            //Log.Info("-");
            //Log.Info("--------------------------------------------------------------------");
            //Log.Info("-");

            TestNetwork.Fit(network, trainingDataSet, learningRate, numEpochs, batchSize, validationDataSet);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# prediction_before= " + predict_before);
            Log.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            Log.Info("C# prediction_after= " + predict_after);
            Log.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
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
            var lossAccuracyBefore = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# prediction_before= " + predict_before);
            Log.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            Log.Info("C# prediction_after= " + predict_after);
            Log.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_SimpleRNN()
        {
            const int numEpochs = 1;
            const double learningRate = 1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            

            var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[0,1,2,3]], numpy.float)");
            var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[4]], numpy.float)");


            int batchSize = X.Shape[0];
            var deviceId = -1;
            //var deviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            LogFile = "SimpleRNN",
                            LossFunction = NetworkConfig.LossFunctionEnum.BinaryCrossentropy,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2
                        }
                       .WithSGD(momentum, false),
                        new List<int> { deviceId }
                );

            network
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Dense(1, 0.0)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);


            Log.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromNumpyArray("[[-0.020802486687898636, -0.02934335544705391, 0.0035390742123126984, 0.006125748157501221, -0.008332550525665283], [0.0307827927172184, -0.0006774887442588806, 0.0498129241168499, 0.019673515111207962, -0.037462640553712845],[0.020981673151254654, 0.016241561621427536, 0.007225655019283295, -0.013524651527404785, -0.007948171347379684]]")
                .CopyTo(((EmbeddingLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.05924016237258911], [-0.2979503273963928], [0.39012110233306885], [0.2964285612106323], [0.15513628721237183], [0.032458603382110596], [-0.5190843939781189], [0.3992980718612671], [-0.03236877918243408], [-0.12109190225601196], [0.4128159284591675], [0.14623379707336426], [-0.5325161814689636], [0.38246530294418335], [-0.4191945493221283], [0.4918263554573059], [-0.30854684114456177], [0.1737397313117981], [-0.40517792105674744], [-0.3750319480895996]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);

            network.PropagationManager.LogPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# prediction_before= " + predict_before);
            Log.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            Log.Info("C# prediction_after= " + predict_after);
            Log.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Huber()
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const int batchSize = 2;

            var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[0,1,2],[3,4,5]], numpy.float)");
            var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[0],[5]], numpy.float)");

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
                .Dense(3, 0.0)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dense(1, 0.0)
                ;


            Log.Info(network.Summary() + Environment.NewLine);

            TestNetworkPropagation.FromNumpyArray("[[0.17207741737365723, -0.19425582885742188, 0.6897902488708496], [0.5924994945526123, -0.11895132064819336, -0.8060355186462402], [0.44127702713012695, -0.15072321891784668, -0.8697922229766846]]")
                .CopyTo(((DenseLayer)network.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.6883463859558105], [0.9837051630020142], [0.17996716499328613]]")
                .CopyTo(((DenseLayer)network.Layers[3]).Weights);

            network.PropagationManager.LogPropagation = true;
            var predict_before = network.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, X, Y, learningRate, numEpochs, batchSize);

            var predict_after = network.Predict(X, false).ToNumpy();
            var lossAccuracyAfter = network.ComputeLossAndAccuracyForTestDataSet(batchSize, trainingDataSet);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# batchSize= " + batchSize);
            Log.Info("C# prediction_before= " + predict_before);
            Log.Info("C# loss_before= " + lossAccuracyBefore.Item1 + " , accuracy_before= " + lossAccuracyBefore.Item2);
            Log.Info("C# prediction_after= " + predict_after);
            Log.Info("C# loss_after= " + lossAccuracyAfter.Item1 + " , accuracy_after= " + lossAccuracyAfter.Item2);
        }


    }
}
