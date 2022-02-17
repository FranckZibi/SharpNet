using System;
using System.Collections.Generic;
using System.Diagnostics;
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
using SharpNet.HyperParameters;
using SharpNet.Layers;
using SharpNet.Models;
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
            var networkBuilder = EfficientNetSample.CIFAR10();
            networkBuilder.Config.WorkingDirectory = NetworkConfig.DefaultWorkingDirectory;
            networkBuilder.Config.SetResourceId(0);
            var network = networkBuilder.EfficientNetB0(true, "imagenet", new[] {3, 224, 224});
            var yPredicted = network.Predict(X, false);
            Assert.IsTrue(TestTensor.SameContent(yExpectedFromKeras, yPredicted, 1e-5));

            //we save the network
            network.Save(network.WorkingDirectory, network.DynamicModelName);
            network.Dispose();

            //we ensure that the saved version of the network behave the same as the original one
            var networkFromSavedFile = Network.ValueOf(network.WorkingDirectory, network.DynamicModelName);
            var yPredictedFromSavedFile = networkFromSavedFile.Predict(X, false);
            Assert.IsTrue(TestTensor.SameContent(yExpectedFromKeras, yPredictedFromSavedFile, 1e-5));

            var savedModelFile = Network.ToModelFilePath(network.WorkingDirectory, network.DynamicModelName);
            File.Delete(savedModelFile);
            var saveParametersFile = Network.ToParameterFilePath(network.WorkingDirectory, network.DynamicModelName);
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

            var networkBuilder = Yolov3NetSample.ValueOf(new List<int> { 0 });
            var network = networkBuilder.Build();
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
                var X = content.Select((_, _, b) => b / 255f);
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
            const double capacity = 608 * 608;
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

            var hp = EfficientNetSample.CIFAR10();
            hp.Config.SetResourceId(0);

            //int defaultHeight = 32;
            const int defaultHeight = 224;

            var network = hp.EfficientNetB0(true, "imagenet", new[] { 3, defaultHeight, defaultHeight });
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

            using var trainingDataSet = new InMemoryDataSet(X, Y, "", Objective_enum.Classification, null, ImageNetDataSet._CategoryIndexToDescription);
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
            Log.Info("C# metrics_before= " + IModel.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + IModel.MetricsToString(lossAccuracyAfter, ""));
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
            const int gpuDeviceId = 0;

            var sample = new NetworkSample(new ISample[]
            {
                new NetworkConfig
                    {
                        ModelName = "TestParallelRunWithTensorFlow_Convolution",
                        LossFunction = LossFunctionEnum.CategoricalCrossentropy,
                        RandomizeOrder = false,
                        CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1,
                        ResourceIds = new List<int> { gpuDeviceId }
                    }
                    .WithSGD(momentum, false),
                new DataAugmentationSample()
            });

            var network = new Network(sample);

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
            Log.Info("C# metrics_before= " + IModel.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + IModel.MetricsToString(lossAccuracyAfter, ""));
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
            //const int  gpuDeviceId = -1;
            const int gpuDeviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            ModelName = "TestParallelRunWithTensorFlow_Convolution",
                            LossFunction = LossFunctionEnum.CategoricalCrossentropy,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1,
                            ResourceIds = new List<int> { gpuDeviceId }
                        }
                       .WithSGD(momentum, false),
                        new DataAugmentationSample()
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
            Log.Info("C# metrics_before= " + IModel.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + IModel.MetricsToString(lossAccuracyAfter, ""));
        }
        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Embedding()
        {
            const int numEpochs = 10;
            const double learningRate = 0.1;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const int vocabularySize = 3;
            const int embeddingDim = 5;
            const int maxWordsBySentence = 4;

            var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1, 2, 1, 1], [2, 2, 1, 1]], numpy.float)");
            var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1], [0]], numpy.float)");


            int batchSize = X.Shape[0];
            const int deviceId = -1;
            //var deviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            ModelName = "Embedding",
                            LossFunction = LossFunctionEnum.BinaryCrossentropy,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                            ResourceIds = new List<int> { deviceId }
                        }
                       .WithSGD(momentum, false),
                        new DataAugmentationSample()
                );

            network
                .InputAndEmbedding(maxWordsBySentence, vocabularySize, embeddingDim, -1, 0.0)
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
            Log.Info("C# metrics_before= " + IModel.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + IModel.MetricsToString(lossAccuracyAfter, ""));
        }

        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_Embedding_GlobalPooling()
        {
            const int numEpochs = 5;
            const double learningRate = 0.01;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const int batchSize = 2;
            const int deviceId = -1;
            //var deviceId = 0;
            const int vocabularySize = 3;
            const int embeddingDim = 5;
            const int maxWordsBySentence = 4;

            var X = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1, 1, 1, 2], [2, 2, 2, 2], [1, 2, 2, 2],[1, 1, 1, 1]], numpy.float)");
            var Y = TestNetworkPropagation.FromNumpyArray(@"numpy.array([[1], [0], [0], [1]], numpy.float)");


            var networkConfig = new NetworkConfig
            {
                ModelName = "Embedding_GlobalPooling",
                LossFunction = LossFunctionEnum.BinaryCrossentropy,
                RandomizeOrder = false,
                CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                ResourceIds = new List<int> { deviceId }
            };

            var network = new Network(
                        networkConfig
                       .WithAdam(0.9, 0.999, 1e-7),
                        //.WithSGD(momentum, false),
                        new DataAugmentationSample()
                );
            network.PropagationManager.LogPropagation = true;

            network
                .InputAndEmbedding(maxWordsBySentence, vocabularySize, embeddingDim, -1, 0.0)
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
            Log.Info("C# metrics_before= " + IModel.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + IModel.MetricsToString(lossAccuracyAfter, ""));
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
            const int batchSize = 128;
            //var deviceId = -1;
            const int deviceId = 0;
            const int vocab_size = 10000;
            const int embedding_dim = 16;
            const int max_length = 100;
            const string oov_tok = "<OOV>";
            const int training_size = 20000;

            var jsonText = File.ReadAllText(Path.Combine(NetworkConfig.DefaultDataDirectory, "Sarcasm", "sarcasm.json"));
            var allEntries = JsonConvert.DeserializeObject<List< SarcasmEntry>>(jsonText);

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
                ModelName = "TestParallelRunWithTensorFlow_Sarcasm",
                LossFunction = LossFunctionEnum.BinaryCrossentropy,
                RandomizeOrder = true,
                CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                ResourceIds = new List<int> { deviceId }
            };

            var network = new Network(
                        networkConfig
                        .WithAdam(0.9, 0.999, 1e-7),
                        new DataAugmentationSample()
                );

            network
                .InputAndEmbedding(max_length, vocab_size, embedding_dim, -1, 0.0)
                .GlobalAvgPoolingOnHeight()
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
            using var validationDataSet = new InMemoryDataSet(X_val, Y_val);

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
            Log.Info("C# metrics_before= " + IModel.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + IModel.MetricsToString(lossAccuracyAfter, ""));
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
            const int gpuDeviceId = -1;
            //var gpuDeviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            ModelName = "TestParallelRunWithTensorFlow_DownSampling2D",
                            LossFunction = LossFunctionEnum.CategoricalCrossentropy,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1,
                            ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                            ResourceIds = new List<int> { gpuDeviceId }
                        }
                       .WithSGD(momentum, false),
                        new DataAugmentationSample()
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
            Log.Info("C# metrics_before= " + IModel.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + IModel.MetricsToString(lossAccuracyAfter, ""));
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

            const int deviceId = -1;
            //var deviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            ModelName = "Huber",
                            LossFunction = LossFunctionEnum.Huber,
                            //LossFunction = LossFunctionEnum.BinaryCrossentropy,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                            ResourceIds = new List<int> { deviceId }
                        }
                       .WithSGD(momentum, false),
                        new DataAugmentationSample()
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
            Log.Info("C# metrics_before= " + IModel.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + IModel.MetricsToString(lossAccuracyAfter, ""));
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


            const int deviceId = 0;
            var network = new Network(
                        new NetworkConfig
                        {
                            ModelName = "Mse",
                            LossFunction = LossFunctionEnum.Mse,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                            Metrics = new List<MetricEnum> {MetricEnum.Loss, MetricEnum.Mae, MetricEnum.Mse},
                            ResourceIds = new List<int> { deviceId }
                        }
                       .WithSGD(momentum, false),
                        new DataAugmentationSample()
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
            Log.Info("C# metrics_before= " + IModel.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + IModel.MetricsToString(lossAccuracyAfter, ""));
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
            const int deviceId = 0;
            const int hiddenSize = 2;

            var network = new Network(
                        new NetworkConfig
                        {
                            ModelName = "GRU",
                            LossFunction = LossFunctionEnum.Huber,
                            RandomizeOrder = false,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                            Metrics = new List<MetricEnum> { MetricEnum.Loss, MetricEnum.Mae},
                            ResourceIds = new List<int> { deviceId }
                        }
                       .WithSGD(momentum, false),
                        new DataAugmentationSample()
                );


            network
                .Input(timeSteps, inputSize, -1)
                .SimpleRNN(hiddenSize, returnSequences, true)
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
            Log.Info("C# metrics_before= " + IModel.MetricsToString(network.EpochData[0].TrainingMetrics, ""));
            Log.Info(predict_after.ToShapeAndNumpy());
            Log.Info("C# metrics_after= " + IModel.MetricsToString(network.EpochData.Last().TrainingMetrics, ""));
        }


        // Mae: 4.87 - val_Mae: 6.24        DEFAULT learningRate = 5*1e-7;
        // Mae: 4.87 - val_Loss: 71.559 - val_Mae: 6.2359
        // Mae: 5.78 - val_Mae: 9.15        Huber Loss
        // Mae: 5.12 - val_Mae: 5.02        Normalization (0 => 1) of input
        // Mae: 5.31 - val_Mae: 7.29        Input with Mean=0 and Vol=1
        // Mae: 4.86 - val_Mae: 6.24        WithCyclicCosineAnnealingLearningRateScheduler(2, 2)
        //-0.16     hiddenSize = 64
        //-0.09     no shuffle
        //-0.04     timeSteps = 10
        //+0.03     timeSteps = 50
        //+0.03     learningRate = 5*1e-8;
        //+0.04     learningRate = 1e-7;
        //+0.15     OLD DEFAULT (Weights_aa.Orthogonal / lr=1e-6 / IsBidirectional / LSTM)
        //+0.17     Weights_aa.NormalDistribution
        //+0.35     Weights_aa.GlorotUniform
        //+0.36     GRU
        //+0.59     Constant LearningRate
        //+1.50     learningRate = 1e-8;
        //+3.00     !IsBidrectional
        //+6.00     learningRate = 1e-5;



        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_UnivariateTimeSeries()
        {
            const int numEpochs = 300;
            const double learningRate = 5*1e-7;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const int batchSize = 32;
            const int trainingDataSetCount = 31*batchSize;
            const int timeSteps = 20; //number of words in each sentence
            const int inputSize = 1; //number of distinct words in the dictionary 
            const bool returnSequences = false;
            const int deviceId = 0;
            const int hiddenSize = 32;

            const bool shuffle = true;
            Debug.Assert(trainingDataSetCount % batchSize == 0);

            var series = TestNetworkPropagation.FromNumpyArray(@"numpy.array([52.48357009887695, 49.35274887084961, 53.314735412597656, 57.711822509765625, 48.934444427490234, 48.931243896484375, 57.98289489746094, 53.897125244140625, 47.67393112182617, 52.68370819091797, 47.59171676635742, 47.50637435913086, 50.959415435791016, 40.086177825927734, 40.91941452026367, 46.61247253417969, 44.228206634521484, 50.72064208984375, 44.45498275756836, 41.76799011230469, 55.98093795776367, 47.33583450317383, 48.60329818725586, 40.93183898925781, 45.1126594543457, 48.157318115234375, 41.605098724365234, 48.985233306884766, 43.83963394165039, 45.10993576049805, 43.27504348754883, 55.24983215332031, 45.6156120300293, 40.079105377197266, 49.15495681762695, 38.60270309448242, 45.40616226196289, 34.20876693725586, 37.001930236816406, 44.253509521484375, 46.57859420776367, 43.351009368896484, 41.51481246948242, 40.17753219604492, 33.8716926574707, 37.23771286010742, 38.09790802001953, 45.24254608154297, 41.22268295288086, 30.229015350341797, 40.19637680053711, 36.174564361572266, 34.231815338134766, 40.183837890625, 41.782230377197266, 40.778324127197266, 31.41360855102539, 33.54463577270508, 36.221431732177734, 38.91083526611328, 31.098981857299805, 32.02223587036133, 26.86863136291504, 25.863332748413086, 35.345516204833984, 37.497249603271484, 29.783985137939453, 34.584693908691406, 30.79339599609375, 25.173189163208008, 29.61490821838379, 34.90300750732422, 26.434551239013672, 33.83389663696289, 12.305289268493652, 28.90338706970215, 24.615705490112305, 22.068950653076172, 23.40342903137207, 12.38474178314209, 20.599699020385742, 22.856821060180664, 27.831846237182617, 17.220260620117188, 15.136700630187988, 16.036466598510742, 22.48707389831543, 18.917560577392578, 13.987879753112793, 18.56522560119629, 15.846029281616211, 19.565366744995117, 10.573323249816895, 11.806878089904785, 10.846607208251953, 4.852085590362549, 13.013545036315918, 12.202471733093262, 10.288180351257324, 8.456475257873535, 1.9209299087524414, 6.264708518981934, 6.026490211486816, 3.102989673614502, 5.684642314910889, 7.890594482421875, 14.68332290649414, 5.510314464569092, 5.31337833404541, 3.0449743270874023, -6.781511306762695, 2.0786733627319336, 1.9152562618255615, 13.337308883666992, -0.5293726325035095, 1.3560261726379395, -0.9047056436538696, -7.149067401885986, 3.838982582092285, 1.3203569650650024, 0.9571147561073303, -8.098090171813965, 2.915529489517212, -11.649012565612793, -2.240624189376831, 5.248558044433594, -11.178715705871582, -9.573151588439941, -6.752202987670898, -10.26962661743164, -16.000173568725586, -8.391325950622559, -14.52544116973877, -7.318101406097412, -14.747515678405762, -2.857105255126953, -14.971320152282715, -13.10537338256836, -7.859101295471191, -18.504087448120117, -11.626856803894043, -6.634021282196045, -21.603775024414062, -13.030718803405762, -13.03278923034668, -10.792065620422363, 23.163013458251953, 22.696868896484375, 31.861047744750977, 30.689258575439453, 30.410579681396484, 30.84493064880371, 25.667919158935547, 30.185440063476562, 30.446434020996094, 25.366971969604492, 38.22601318359375, 31.225473403930664, 22.859697341918945, 32.05963134765625, 23.86482810974121, 32.63576126098633, 34.456138610839844, 24.523286819458008, 33.40782165527344, 30.619787216186523, 32.6318244934082, 37.97182083129883, 27.227935791015625, 24.653888702392578, 23.94336700439453, 24.280925750732422, 27.944168090820312, 30.005796432495117, 29.65450096130371, 32.37862014770508, 28.280010223388672, 35.455604553222656, 26.838214874267578, 41.736534118652344, 31.238834381103516, 23.80013656616211, 22.707500457763672, 30.450965881347656, 26.898534774780273, 31.563684463500977, 30.338298797607422, 27.58698272705078, 23.69675636291504, 20.336660385131836, 25.659061431884766, 32.15494155883789, 28.925289154052734, 21.608558654785156, 28.686138153076172, 29.730350494384766, 23.368555068969727, 28.54108238220215, 28.048654556274414, 22.028438568115234, 29.518434524536133, 30.52014923095703, 33.11873245239258, 32.96024703979492, 20.791162490844727, 22.97916030883789, 30.232738494873047, 30.21626853942871, 30.212846755981445, 46.89202117919922, 30.47405433654785, 33.289154052734375, 32.37353515625, 30.8531551361084, 26.01299476623535, 31.377792358398438, 23.712888717651367, 26.387451171875, 25.139707565307617, 27.97132682800293, 39.131126403808594, 18.21782684326172, 30.982215881347656, 19.48453140258789, 25.186080932617188, 32.988548278808594, 27.863677978515625, 22.1524600982666, 23.963987350463867, 30.938232421875, 23.88855743408203, 28.62323760986328, 27.76976776123047, 24.28527069091797, 38.26475143432617, 30.716781616210938, 17.42401885986328, 28.484941482543945, 24.247058868408203, 31.82185935974121, 23.60116958618164, 26.994548797607422, 30.097999572753906, 31.907039642333984, 21.58234977722168, 25.917253494262695, 25.221323013305664, 24.336055755615234, 36.43697738647461, 29.641969680786133, 21.32034683227539, 32.22212600708008, 38.25199508666992, 32.81227493286133, 20.062171936035156, 25.247257232666016, 34.01272201538086, 24.149890899658203, 29.917734146118164, 31.58252716064453, 23.085744857788086, 27.434131622314453, 11.537099838256836, 22.63348960876465, 26.50489044189453, 21.541425704956055, 35.95531463623047, 20.655773162841797, 25.619779586791992, 28.487524032592773, 35.05430221557617, 20.68303680419922, 33.692867279052734, 27.943204879760742, 22.999773025512695, 30.233394622802734, 28.93401527404785, 24.95375633239746, 28.320253372192383, 26.061349868774414, 28.572450637817383, 31.332735061645508, 35.96965408325195, 21.86824607849121, 38.74050521850586, 18.33317756652832, 27.353227615356445, 31.07253646850586, 29.554960250854492, 25.055810928344727, 27.148250579833984, 25.743661880493164, 25.28189468383789, 32.49702453613281, 30.05463409423828, 24.825788497924805, 32.809356689453125, 29.86911392211914, 32.41842269897461, 31.52397918701172, 24.252817153930664, 25.61907386779785, 32.178863525390625, 31.516889572143555, 28.383394241333008, 29.097625732421875, 34.9226188659668, 25.599964141845703, 31.317052841186523, 27.594558715820312, 27.54128646850586, 34.14795684814453, 32.805747985839844, 32.77101516723633, 35.255863189697266, 28.858699798583984, 32.18885803222656, 27.253374099731445, 30.451356887817383, 28.205821990966797, 29.36772346496582, 31.884931564331055, 24.844640731811523, 39.424468994140625, 23.959423065185547, 22.94573211669922, 34.83458709716797, 33.02988052368164, 32.21989059448242, 32.268924713134766, 29.09404754638672, 24.697277069091797, 29.591012954711914, 25.854799270629883, 34.14500427246094, 28.563085556030273, 25.200027465820312, 27.749893188476562, 31.450960159301758, 26.59733009338379, 25.334665298461914, 30.694183349609375, 30.730724334716797, 27.001482009887695, 27.21147346496582, 30.757543563842773, 22.387659072875977, 22.6217041015625, 26.097902297973633, 28.654142379760742, 31.307323455810547, 37.16112518310547, 34.10435485839844, 29.0482234954834, 29.78484344482422, 24.899433135986328, 29.85181999206543, 28.533538818359375, 31.62301254272461, 25.906002044677734, 32.67176055908203, 37.7717399597168, 29.59739875793457, 32.183048248291016, 33.65864181518555, 66.243896484375, 69.41453552246094, 68.3892593383789, 68.83505249023438, 64.49015808105469, 68.47447967529297, 70.82682037353516, 75.56566619873047, 73.06765747070312, 78.98682403564453, 64.32206726074219, 72.44662475585938, 68.91631317138672, 78.85159301757812, 63.75251388549805, 63.475303649902344, 64.54540252685547, 56.77992630004883, 64.61632537841797, 63.283851623535156, 67.6546630859375, 68.42350006103516, 75.8965072631836, 71.05770111083984, 63.20005416870117, 61.360633850097656, 68.06966400146484, 58.755577087402344, 74.2501220703125, 70.71560668945312, 62.18769836425781, 55.672767639160156, 70.70246124267578, 63.044960021972656, 69.48131561279297, 54.98478317260742, 59.61497116088867, 62.2833366394043, 62.12776184082031, 59.26887512207031, 64.2505111694336, 55.406070709228516, 59.63115692138672, 60.53453063964844, 62.08649444580078, 62.6450080871582, 53.027889251708984, 50.5363655090332, 64.14297485351562, 58.955787658691406, 53.083526611328125, 64.1057357788086, 56.44479751586914, 61.271942138671875, 55.21482849121094, 64.67566680908203, 62.63639831542969, 52.09587860107422, 57.67296600341797, 55.509986877441406, 58.5880126953125, 46.37591552734375, 54.080562591552734, 55.38648986816406, 40.739192962646484, 43.04975509643555, 38.1978759765625, 46.46999740600586, 50.82292556762695, 54.16057205200195, 46.42840576171875, 53.60590744018555, 37.963172912597656, 35.74376678466797, 43.376277923583984, 44.964202880859375, 42.266998291015625, 31.476778030395508, 40.74903106689453, 34.05023956298828, 43.29642105102539, 41.15425109863281, 33.99297332763672, 35.492279052734375, 32.13310241699219, 36.48185348510742, 40.93577575683594, 30.595172882080078, 37.40691375732422, 31.59760093688965, 29.646276473999023, 32.43699264526367, 27.15737533569336, 28.926944732666016, 25.067758560180664, 40.2432861328125, 29.959264755249023, 25.64856719970703, 29.5825138092041, 27.31777000427246, 26.142934799194336, 29.688772201538086, 29.777599334716797, 22.711872100830078, 21.86197853088379, 22.745080947875977, 11.992790222167969, 15.311469078063965, 29.10999870300293, 29.89204216003418, 19.817163467407227, 23.344030380249023, 21.42035484313965, 34.665504455566406, 24.280305862426758, 17.458702087402344, 12.741153717041016, 8.912089347839355, 17.392189025878906, 12.028938293457031, 8.14068603515625, 11.465986251831055, 8.743816375732422, 22.04595375061035, 17.48328971862793, 12.506416320800781, 19.423690795898438, 11.895176887512207, 6.693121433258057, 18.113370895385742, 12.697693824768066, 4.329632759094238, 8.084382057189941, 4.18584680557251, 1.1856069564819336, 12.274110794067383, 16.742033004760742, -0.23790547251701355, 9.138160705566406, 2.6470212936401367, 3.0502166748046875, 2.1182992458343506, 0.363685667514801, 4.538722038269043, -0.2369534969329834, 4.90110445022583, 47.346580505371094, 46.354408264160156, 42.963523864746094, 44.57047653198242, 51.1850700378418, 49.867279052734375, 42.430267333984375, 47.770835876464844, 50.988006591796875, 38.84170150756836, 49.863948822021484, 43.79318618774414, 49.91920852661133, 43.210567474365234, 37.96382522583008, 38.81262969970703, 47.153587341308594, 48.17530822753906, 42.319358825683594, 49.99884796142578, 38.46392822265625, 46.4074592590332, 40.64979553222656, 43.413387298583984, 46.8779296875, 42.30791473388672, 44.65689468383789, 51.58150100708008, 43.63658905029297, 50.67116165161133, 40.816471099853516, 49.086952209472656, 53.61934280395508, 34.0274658203125, 42.37601852416992, 49.221290588378906, 45.296730041503906, 48.14433288574219, 43.24592208862305, 46.676631927490234, 45.443729400634766, 52.040035247802734, 47.45282745361328, 47.84891128540039, 44.082252502441406, 43.68492126464844, 43.9420280456543, 48.05950927734375, 43.96531295776367, 47.50263977050781, 56.41484832763672, 50.37807846069336, 44.37749099731445, 51.999359130859375, 43.93912124633789, 35.77560806274414, 40.91304397583008, 36.5872802734375, 44.17194366455078, 46.0103759765625, 54.28975296020508, 47.5319709777832, 44.79210662841797, 50.025390625, 34.813926696777344, 47.03940200805664, 49.707847595214844, 38.45326614379883, 51.558109283447266, 47.525428771972656, 43.750579833984375, 48.985450744628906, 57.16999053955078, 46.72128677368164, 47.0489387512207, 43.50735092163086, 41.55168914794922, 49.949790954589844, 41.51531982421875, 46.15162658691406, 43.403995513916016, 48.18607711791992, 47.45881652832031, 50.977943420410156, 43.24030685424805, 44.44157409667969, 40.89809036254883, 43.571807861328125, 47.68153381347656, 49.582130432128906, 41.188907623291016, 50.15069580078125, 52.58418273925781, 47.87686538696289, 55.19775390625, 41.94929122924805, 39.59978485107422, 36.934661865234375, 53.31405258178711, 49.11158752441406, 45.56813049316406, 47.252540588378906, 40.232261657714844, 58.095821380615234, 46.5208740234375, 46.42979431152344, 49.52004623413086, 48.30499267578125, 47.028438568115234, 41.9660530090332, 48.2855110168457, 55.348358154296875, 52.67573547363281, 53.925289154052734, 43.4143180847168, 41.03373718261719, 45.364501953125, 46.284053802490234, 51.48868942260742, 37.56801986694336, 53.69101333618164, 45.26643753051758, 43.93559265136719, 41.02330017089844, 37.82365417480469, 50.22820281982422, 46.49363327026367, 39.69223403930664, 39.681922912597656, 44.49394989013672, 54.5338249206543, 44.906883239746094, 38.705528259277344, 45.00920104980469, 44.8912467956543, 32.78765106201172, 46.018096923828125, 45.15264892578125, 49.806373596191406, 55.58839797973633, 51.99497604370117, 45.03650665283203, 40.86737060546875, 59.28610610961914, 46.734954833984375, 46.5283088684082, 46.358097076416016, 47.48944091796875, 45.79775619506836, 43.672027587890625, 43.82706069946289, 46.418853759765625, 43.886985778808594, 43.061607360839844, 47.1799430847168, 45.39509201049805, 54.212364196777344, 33.46018981933594, 52.1954345703125, 52.99141311645508, 36.41734313964844, 45.09438705444336, 44.974361419677734, 39.817962646484375, 42.99060821533203, 41.351192474365234, 55.69001770019531, 51.63185501098633, 53.336246490478516, 50.61204147338867, 41.38383483886719, 44.43210983276367, 49.52739715576172, 40.99589920043945, 50.69773483276367, 45.95751953125, 45.311641693115234, 50.767330169677734, 49.4608268737793, 45.46184158325195, 53.09067916870117, 41.916255950927734, 50.42897033691406, 50.34270477294922, 45.857547760009766, 49.064212799072266, 41.2064208984375, 52.110748291015625, 46.59489440917969, 44.93475341796875, 52.82255554199219, 44.08510971069336, 40.59400177001953, 39.882808685302734, 50.72581481933594, 41.32360076904297, 56.52986526489258, 37.37655258178711, 56.298946380615234, 48.902381896972656, 47.39451217651367, 45.18443298339844, 49.93580627441406, 47.783203125, 53.51929473876953, 48.6054801940918, 48.817569732666016, 46.27985382080078, 47.84519577026367, 49.70109176635742, 39.64353942871094, 41.485904693603516, 51.97574234008789, 49.146484375, 47.405113220214844, 48.45021438598633, 50.12910842895508, 45.72569274902344, 44.5663948059082, 87.4792251586914, 81.65220642089844, 88.61756134033203, 78.0837631225586, 91.75098419189453, 88.96491241455078, 87.86698150634766, 91.47340393066406, 94.84867095947266, 91.54276275634766, 77.20443725585938, 79.9371337890625, 83.1255111694336, 86.28303527832031, 88.6323013305664, 82.29519653320312, 86.72618865966797, 81.87249755859375, 82.43751525878906, 78.29621124267578, 80.53652954101562, 78.206298828125, 79.88629150390625, 89.82379913330078, 79.58757781982422, 97.26461791992188, 86.32665252685547, 84.53092956542969, 79.05104064941406, 86.56995391845703, 79.90538787841797, 83.0984878540039, 94.98352813720703, 81.38735961914062, 87.28860473632812, 77.6910400390625, 80.68690490722656, 89.36112976074219, 77.0080337524414, 88.8314437866211, 82.92501831054688, 76.18183898925781, 81.75509643554688, 83.04582977294922, 80.87334442138672, 69.48580932617188, 73.26541137695312, 75.21934509277344, 75.6324234008789, 78.6475830078125, 75.96446228027344, 67.92325592041016, 76.01741027832031, 76.67838287353516, 75.92618560791016, 78.02582550048828, 76.27930450439453, 73.8865966796875, 70.71428680419922, 62.22830581665039, 72.14295196533203, 70.48896789550781, 70.2582015991211, 61.96062469482422, 62.37760925292969, 72.4818115234375, 66.44625854492188, 69.47453308105469, 65.6268081665039, 65.04756927490234, 68.99935150146484, 61.13260269165039, 63.59428787231445, 60.19930648803711, 59.73153305053711, 59.748016357421875, 61.7911376953125, 57.67024612426758, 65.7234115600586, 54.34954833984375, 57.263702392578125, 55.37260437011719, 64.17726135253906, 57.29438781738281, 60.8383903503418, 47.61744689941406, 55.74531173706055, 58.221961975097656, 53.548099517822266, 57.826290130615234, 49.274200439453125, 58.26887893676758, 62.07808303833008, 48.13099670410156, 47.07963562011719, 55.93658447265625, 55.93080520629883, 44.78289794921875, 44.66168212890625, 44.72048568725586, 38.77553176879883, 40.27467727661133, 39.219364166259766, 39.775390625, 42.81764602661133, 43.541412353515625, 49.5048942565918, 36.14565658569336, 45.44723892211914, 38.84726333618164, 39.065025329589844, 42.0853385925293, 32.50049591064453, 39.43314743041992, 37.76469421386719, 38.810546875, 37.21469497680664, 47.4708251953125, 31.43617057800293, 31.40570640563965, 30.386249542236328, 30.171459197998047, 29.214622497558594, 37.805328369140625, 38.4276123046875, 27.942548751831055, 26.1121883392334, 32.11541748046875, 26.488426208496094, 31.912410736083984, 29.267759323120117, 20.1871395111084, 35.02360534667969, 35.79332733154297, 23.285661697387695, 23.954715728759766, 26.8742733001709, 26.67721939086914, 27.86603546142578, 34.201255798339844, 22.851106643676758, 19.33877944946289, 16.037046432495117, 18.891464233398438, 22.00216293334961, 30.771610260009766, 63.259708404541016, 66.91809844970703, 65.66922760009766, 71.64629364013672, 78.29277801513672, 62.958343505859375, 63.120849609375, 70.7449722290039, 68.8905258178711, 74.6722640991211, 68.31678009033203, 63.55984878540039, 68.26949310302734, 70.82038116455078, 69.34064483642578, 67.73670959472656, 70.49653625488281, 70.97317504882812, 72.00173950195312, 68.2994384765625, 64.18592834472656, 65.72142791748047, 70.9874267578125, 60.837890625, 66.73430633544922, 62.89328384399414, 64.97340393066406, 71.19229888916016, 65.72654724121094, 64.97489166259766, 57.915714263916016, 68.41920471191406, 67.88892364501953, 75.45196533203125, 63.07160568237305, 65.68167114257812, 65.80888366699219, 72.42587280273438, 64.03936767578125, 65.8887939453125, 67.51158905029297, 65.38417053222656, 62.19855880737305, 65.38134765625, 69.75979614257812, 59.24037170410156, 65.01966857910156, 60.836647033691406, 70.29547119140625, 56.687828063964844, 61.49323272705078, 66.15852355957031, 72.0852279663086, 63.91453552246094, 61.4535026550293, 73.62200927734375, 56.96340560913086, 53.197208404541016, 66.37958526611328, 61.65800857543945, 59.05139923095703, 67.68911743164062, 65.35660552978516, 61.3079719543457, 57.718082427978516, 68.47361755371094, 67.35453796386719, 63.600318908691406, 73.32252502441406, 58.732521057128906, 56.44939422607422, 60.61200714111328, 63.83859634399414, 65.27865600585938, 62.85165023803711, 65.8144302368164, 57.79322052001953, 71.26693725585938, 63.63498306274414, 69.63027954101562, 65.75590515136719, 66.324951171875, 66.88934326171875, 66.2787857055664, 67.2540054321289, 70.68670654296875, 65.0245132446289, 67.58829498291016, 63.596351623535156, 71.24777221679688, 60.66777420043945, 73.057373046875, 63.855201721191406, 56.90581512451172, 64.70429992675781, 60.66297149658203, 68.27628326416016, 60.81514358520508, 61.85291290283203, 54.64205551147461, 61.83452224731445, 51.98331069946289, 56.190189361572266, 67.91913604736328, 68.05376434326172, 66.26010131835938, 59.306331634521484, 63.91139221191406, 64.14100646972656, 58.37660217285156, 71.69515991210938, 68.5750503540039, 63.09381866455078, 64.34378814697266, 65.26231384277344, 54.0230827331543, 63.00755310058594, 60.84551239013672, 59.2596321105957, 62.874839782714844, 73.28169250488281, 67.51069641113281, 61.46410369873047, 67.19673919677734, 71.34471893310547, 68.98551177978516, 64.67520141601562, 61.1573600769043, 67.8984375, 66.39030456542969, 68.91468811035156, 67.63069915771484, 69.71900939941406, 61.81174087524414, 71.09183502197266, 65.51007843017578, 74.91587829589844, 61.11138153076172, 73.25515747070312, 65.58316802978516, 61.35506057739258, 62.21152114868164, 63.04826354980469, 66.79013061523438, 67.30303192138672, 61.84016418457031, 64.60694885253906, 75.46036529541016, 73.40727233886719, 66.97195434570312, 65.00137329101562, 65.43277740478516, 67.92169952392578, 59.76187515258789, 63.610904693603516, 56.57705307006836, 66.93851470947266, 68.20101165771484, 62.57196807861328, 72.88092041015625, 58.90546417236328, 57.7359504699707, 66.20382690429688, 70.34101104736328, 73.54933166503906, 62.85965347290039, 70.57206726074219, 65.01092529296875, 64.3653335571289, 69.6719741821289, 68.54070281982422, 57.42274856567383, 72.71322631835938, 72.25699615478516, 62.25492858886719, 67.38816833496094, 67.90589904785156, 66.76589965820312, 62.73798370361328, 62.15856170654297, 65.416259765625, 71.43521881103516, 68.31729125976562, 63.77412414550781, 69.5137710571289, 51.440834045410156, 71.45581817626953, 57.04203796386719, 63.95719528198242, 60.20001983642578, 59.35410690307617, 71.66095733642578, 63.54780197143555, 67.64847564697266, 65.71116638183594, 68.36094665527344, 66.39000701904297, 59.621238708496094, 71.0479965209961, 63.62850570678711, 58.34517288208008, 64.01844787597656, 73.69392395019531, 70.47248077392578, 64.509521484375, 64.53805541992188, 64.7078857421875, 76.73165893554688, 68.28959655761719, 68.56228637695312, 71.59580993652344, 67.67078399658203, 65.2142105102539, 65.5604019165039, 66.2170181274414, 66.42193603515625, 70.27935028076172, 66.93421936035156, 70.3711166381836, 104.34642028808594, 105.18724822998047, 94.83529663085938, 109.4283218383789, 106.58765411376953, 109.84197998046875, 90.35556030273438, 115.25182342529297, 104.07335662841797, 110.2618179321289, 99.45927429199219, 107.64889526367188, 99.23253631591797, 101.28373718261719, 113.8641586303711, 103.22049713134766, 105.12953186035156, 108.249755859375, 106.2235107421875, 104.33160400390625, 105.22750091552734, 115.23179626464844, 102.72756958007812, 103.81108093261719, 107.83783721923828, 107.88033294677734, 108.04521942138672, 105.05039978027344, 95.87779998779297, 109.48556518554688, 95.30184936523438, 102.25162506103516, 96.66172790527344, 99.79696655273438, 101.43604278564453, 101.06371307373047, 101.22145080566406, 106.82568359375, 100.66053771972656, 96.79841613769531, 102.45669555664062, 103.19152069091797, 90.70502471923828, 99.42005920410156, 99.5201644897461, 94.09911346435547, 102.02963256835938, 93.9566650390625, 94.88247680664062, 92.9288558959961, 93.40385437011719, 87.36859893798828, 85.16616821289062, 99.84798431396484, 87.14241790771484, 85.91496276855469, 79.59274291992188, 86.6458969116211, 82.69966888427734, 96.99317932128906, 93.2939453125, 84.25978088378906, 98.41249084472656, 91.5031967163086, 84.40872955322266, 72.9690170288086, 96.3487548828125, 77.36917114257812, 75.50821685791016, 88.26164245605469, 94.7566909790039, 88.88419342041016, 84.1832275390625, 83.73445129394531, 84.42108154296875, 83.33851623535156, 80.33643341064453, 78.83499908447266, 77.38166046142578, 73.30276489257812, 75.04468536376953, 67.35647583007812, 74.70068359375, 69.61865997314453, 68.41121673583984, 74.19471740722656, 79.62007904052734, 76.61539459228516, 63.534175872802734, 65.80074310302734, 74.81449890136719, 64.55970764160156, 67.7104263305664, 70.94544982910156, 62.715423583984375, 67.44654083251953, 59.612815856933594, 62.64036178588867, 66.61151885986328, 56.4144401550293, 65.95015716552734, 63.01974868774414, 65.25251770019531, 62.983951568603516, 68.061767578125, 61.2464599609375, 57.855369567871094, 59.99892044067383, 61.49211883544922, 58.4114990234375, 57.76530075073242, 53.45128631591797, 53.04961013793945, 48.75807189941406, 63.9303092956543, 48.37897491455078, 50.55432891845703, 49.852291107177734, 57.349491119384766, 50.835941314697266, 57.99066162109375, 47.831398010253906, 52.04653549194336, 45.933509826660156, 60.30083465576172, 43.1082878112793, 50.07307434082031, 51.17722702026367, 49.568538665771484, 46.071311950683594, 45.85403823852539, 46.234920501708984, 44.80107498168945, 49.88333511352539, 55.652217864990234, 41.35576248168945, 36.8459358215332, 42.81352996826172, 55.72186279296875, 38.381858825683594, 50.181427001953125, 49.968772888183594, 38.415706634521484, 43.641029357910156, 48.559776306152344, 38.15318298339844, 83.07987213134766, 81.14075469970703, 78.92755889892578, 80.70793914794922, 77.78842163085938, 84.03311157226562, 79.96817779541016, 84.94309997558594, 75.95158386230469, 85.34313201904297, 87.8147964477539, 73.63763427734375, 85.4365005493164, 89.66519927978516, 77.44003295898438, 91.81320190429688, 85.50826263427734, 79.85163116455078, 83.06208801269531, 86.09751892089844, 83.65155792236328, 85.931640625, 78.60150909423828, 84.01937103271484, 76.0723648071289, 82.55384826660156, 78.56014251708984, 79.37239074707031, 89.20150756835938, 88.44925689697266, 86.01069641113281, 77.47637939453125, 81.32946014404297, 88.951171875, 83.56908416748047, 94.43257904052734, 84.77855682373047, 83.74884796142578, 81.22026062011719, 83.41138458251953, 81.95976257324219, 86.24166870117188, 87.4642333984375, 78.73094940185547, 75.9854736328125, 73.4419174194336, 85.144775390625, 77.07041931152344, 71.8057861328125, 84.49665832519531, 95.00283813476562, 82.49209594726562, 86.70005798339844, 82.9024429321289, 81.98505401611328, 87.06161499023438, 81.00210571289062, 83.7781982421875, 84.03800201416016, 79.07782745361328, 87.3677749633789, 81.52253723144531, 78.60887908935547, 85.0609130859375, 77.87726593017578, 82.50223541259766, 82.30792999267578, 87.77568054199219, 84.71282958984375, 82.20780944824219, 86.41584777832031, 89.27257537841797, 85.10558319091797, 82.36372375488281, 75.74864959716797, 76.97858428955078, 80.77478790283203, 79.25054931640625, 81.36088562011719, 82.5770492553711, 84.94074249267578, 81.93869018554688, 84.72301483154297, 82.61261749267578, 72.41305541992188, 77.59426879882812, 81.57147216796875, 76.24479675292969, 85.294677734375, 89.95094299316406, 88.39354705810547, 81.2354507446289, 89.7596206665039, 83.05303955078125, 80.62834167480469, 79.251220703125, 80.81070709228516, 80.38737487792969, 83.1859130859375, 83.14263153076172, 82.36128234863281, 84.53739166259766, 88.31293487548828, 87.11483764648438, 74.95027160644531, 69.61321258544922, 87.06281280517578, 75.5655517578125, 81.28520202636719, 76.56786346435547, 73.41826629638672, 85.14555358886719, 86.24441528320312, 79.5768051147461, 69.51519012451172, 79.75054168701172, 84.45246124267578, 75.11087799072266, 83.43453979492188, 82.45379638671875, 85.43971252441406, 83.15438079833984, 77.70465850830078, 88.56668090820312, 81.80529022216797, 82.475830078125, 77.96070861816406, 80.4256362915039, 78.23330688476562, 81.80814361572266, 91.24726104736328, 75.8453369140625, 74.65343475341797, 90.0937728881836, 81.708251953125, 79.42671966552734, 87.98908996582031, 79.77924346923828, 91.95539093017578, 86.23323822021484, 80.42259979248047, 93.66749572753906, 79.87142944335938, 86.62979125976562, 84.43531799316406, 89.46737670898438, 90.78627014160156, 83.1590347290039, 79.25247192382812, 85.34019470214844, 79.67278289794922, 93.14954376220703, 83.78678894042969, 81.29923248291016, 84.0711898803711, 76.4343490600586, 78.33432006835938, 89.21710205078125, 79.95343780517578, 78.02642822265625, 85.96755981445312, 89.23633575439453, 86.92633056640625, 88.33576202392578, 79.59571838378906, 76.2950210571289, 90.93533325195312, 81.84007263183594, 82.22431182861328, 90.14464569091797, 86.31024932861328, 85.83415222167969, 94.4055404663086, 80.38894653320312, 88.27194213867188, 83.9442138671875, 85.0287094116211, 91.35487365722656, 86.27869415283203, 86.45816040039062, 89.15656280517578, 81.9968032836914, 79.65324401855469, 78.65315246582031, 74.07350158691406, 94.21458435058594, 78.4459457397461, 82.88433837890625, 82.63534545898438, 85.58540344238281, 88.15620422363281, 88.40918731689453, 81.22091674804688, 83.3303451538086, 85.60867309570312, 82.9822998046875, 92.29261779785156, 86.74107360839844, 87.9910659790039, 87.66169738769531, 90.24545288085938, 85.31412506103516, 77.95596313476562, 86.46981811523438, 81.24600219726562, 81.89126586914062, 87.49787902832031, 90.7893295288086, 84.73628997802734, 86.20624542236328, 93.2054672241211, 85.93059539794922, 97.76783752441406, 87.61970520019531, 76.0212173461914, 88.624755859375, 86.79698944091797, 91.37325286865234, 88.3238296508789, 122.3077163696289], numpy.float)");

            //we normalize input with value in [0, 1]
            // x => (x-min) / (max-min) = (1/(max-min)) x - (min/(max-min))
            //var dataPoints = series.ContentAsFloatArray();
            //var maxValue = dataPoints.Min();
            //var minValue = dataPoints.Max();
            //var aNormalization = 1f / (maxValue - minValue);
            //var bNormalization = -minValue / (maxValue - minValue);

            //we normalize input with mean=0 and volatility=1
            //var (mean, volatility) = Utils.MeanAndVolatility(series.Content.Span);
            //var aNormalization = 1f / volatility;
            //var bNormalization = -mean / (volatility);


            using var fullDataSet = new UnivariateTimeSeriesDataSet(series.Content, timeSteps, 1);
            using var trainAndTestDataSet = fullDataSet.SplitIntoTrainingAndValidation(trainingDataSetCount / (double)fullDataSet.Count);

            var network = new Network(
                new NetworkConfig
                    {
                        ModelName = "TimeSeries",
                        LossFunction = LossFunctionEnum.Mse,
                        RandomizeOrder = shuffle,
                        CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                        Metrics = new List<MetricEnum> { MetricEnum.Loss, MetricEnum.Mae },
                        ResourceIds = new List<int> { deviceId }
                    }
                    .WithSGD(momentum, false)
                    .WithCyclicCosineAnnealingLearningRateScheduler(10, 2),
                new DataAugmentationSample());

            const bool isBidirectional = true;
            network
                .Input(timeSteps, inputSize, -1)
                //.Linear(aNormalization, bNormalization)
                .LSTM(hiddenSize, true, isBidirectional, 1, 0.0, false)
                .LSTM(hiddenSize, false, isBidirectional, 1, 0.0, false)
                .Dense(1, 0.0, true)
                //.Linear(1f/aNormalization, -bNormalization/aNormalization)
                .Linear(100f, 0)
                ;


            Log.Info(network.Summary() + Environment.NewLine);

            //looking for best learning rate
            //using var resizedDataSet = MappedDataSet.Resize(dataSet, batchSize * 1000, true);
            //var res = network.FindBestLearningRate(resizedDataSet, 1e-9, 1, batchSize); return;

            //network.PropagationManager.LogPropagation = true;
            //var predict_before = network.Predict(trainAndTestDataSet.Training, batchSize);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, trainAndTestDataSet.Training, learningRate, numEpochs, batchSize, trainAndTestDataSet.Test);

            //var predict_after = network.Predict(dataSet, batchSize);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# trainingDataSetCount= " + trainAndTestDataSet.Training.Count);
            Log.Info("C# testDataSetCount= " + trainAndTestDataSet.Test.Count);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# batchSize= " + batchSize);
            Log.Info("C# hiddenSize= " + hiddenSize);
            Log.Info("C# timeSteps= " + timeSteps);
            Log.Info("C# return_sequences= " + returnSequences);
            Log.Info("C# shuffle= " + shuffle);
            //Log.Info(predict_before.ToShapeAndNumpy());
            Log.Info("C# metrics_before= " + IModel.TrainingAndValidationMetricsToString(network.EpochData[0].TrainingMetrics, network.EpochData[0].ValidationMetrics));
            //Log.Info(predict_after.ToShapeAndNumpy());
            Log.Info("C# metrics_after= " + IModel.TrainingAndValidationMetricsToString(network.EpochData.Last().TrainingMetrics, network.EpochData.Last().ValidationMetrics));
        }



        [Test, Explicit]
        public void TestParallelRunWithTensorFlow_IMDB()
        {
            const int numEpochs = 10;
            const double learningRate = 0.001;
            const double lambdaL2Regularization = 0.00;
            const double momentum = 0.9;
            const int batchSize = 32;
            const int deviceId = 0;

            const int maxWordsBySentence = 250;
            const int vocabularySize = 2000;
            const int embeddingDim = 32;
            

            const bool shuffle = true;


            using var fullTrainingAndTestDataSet = new IMDBTrainingAndTestDataSet();
            using var trainAndTestDataSet = fullTrainingAndTestDataSet.Training.SplitIntoTrainingAndValidation(0.8);

            var network = new Network(
                        new NetworkConfig
                        {
                            ModelName = "IMDB",
                            LossFunction = LossFunctionEnum.BinaryCrossentropy,
                            RandomizeOrder = shuffle,
                            CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow2,
                            Metrics = new List<MetricEnum> { MetricEnum.Loss, MetricEnum.Accuracy},
                            ResourceIds = new List<int> { deviceId }
                        }
                       //.WithSGD(momentum, false),
                       .WithAdam(),
                        //.WithCyclicCosineAnnealingLearningRateScheduler(10, 2)
                        new DataAugmentationSample()
                );

            const bool isBidirectional = false;
            network
                //.Input(timeSteps, inputSize, -1)
                .InputAndEmbedding(maxWordsBySentence, vocabularySize, embeddingDim, -1, lambdaL2Regularization)
                .Dropout(0.2)
                .LSTM(32, false, isBidirectional, 1, 0.0, false)
                .Dense_Activation(256, lambdaL2Regularization, true, cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Dropout(0.2)
                .Dense_Activation(1, lambdaL2Regularization, true, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)
                ;

            Log.Info(network.Summary() + Environment.NewLine);

            //looking for best learning rate
            //using var resizedDataSet = MappedDataSet.Resize(fullDataSet.Training, batchSize * 1000, true);
            //var res = network.FindBestLearningRate(resizedDataSet, 1e-9, 1, batchSize); return;

            //network.PropagationManager.LogPropagation = true;
            //var predict_before = network.Predict(trainAndTestDataSet.Training.Resize(batchSize, false), batchSize);

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            TestNetwork.Fit(network, trainAndTestDataSet.Training, learningRate, numEpochs, batchSize, trainAndTestDataSet.Test);

            //var predict_after = network.Predict(dataSet, batchSize);

            Log.Info("C# numEpochs= " + numEpochs);
            Log.Info("C# trainingDataSetCount= " + trainAndTestDataSet.Training.Count);
            Log.Info("C# testDataSetCount= " + trainAndTestDataSet.Test.Count);
            Log.Info("C# learningRate= " + learningRate);
            Log.Info("C# l2regularizer= " + lambdaL2Regularization);
            Log.Info("C# momentum= " + momentum);
            Log.Info("C# batchSize= " + batchSize);
            Log.Info("C# shuffle= " + shuffle);
            //Log.Info(predict_before.ToShapeAndNumpy());
            Log.Info("C# metrics_before= " + IModel.TrainingAndValidationMetricsToString(network.EpochData[0].TrainingMetrics, network.EpochData[0].ValidationMetrics));
            //Log.Info(predict_after.ToShapeAndNumpy());
            Log.Info("C# metrics_after= " + IModel.TrainingAndValidationMetricsToString(network.EpochData.Last().TrainingMetrics, network.EpochData.Last().ValidationMetrics));
        }
    }
}
