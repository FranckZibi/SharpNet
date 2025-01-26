using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Models;
using SharpNet.Networks;
using log4net;
using System.IO;
using System.Linq;
using System.Reflection;
using SharpNet.CPU;
using SharpNet.Optimizers;

namespace SharpNetTests.NonReg
{
    /// <summary>
    /// sandbox to make // run with PyTorch on several kind of networks
    /// </summary>
    //[TestFixture]
    public class ParallelRunWithPyTorch
    {
        private static readonly ILog Log = LogManager.GetLogger(typeof(ParallelRunWithPyTorch));


        private static string GetDefaultWorkingDirectory()
        {
            // ReSharper disable once AssignNullToNotNullAttribute
            return Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), "../../../NonReg");
        }
        static ParallelRunWithPyTorch()
        {
            var log_directory = GetDefaultWorkingDirectory();
            Utils.TryDelete(Path.Join(log_directory, "ParallelRunWithPyTorch.log"));
            // ReSharper disable once PossibleNullReferenceException
            Console.WriteLine(Path.GetDirectoryName(Assembly.GetEntryAssembly().Location));
            Utils.ConfigureGlobalLog4netProperties(log_directory, "ParallelRunWithPyTorch");
            Utils.ConfigureThreadLog4netProperties(log_directory, "ParallelRunWithPyTorch");
        }
        private static (float loss_before, float loss_after) Train(Network model, CpuTensor<float> X, CpuTensor<float> Y)
        {
            if (model.Sample.BatchSize<= 0)
            {
                model.Sample.BatchSize = X.Shape[0];
            }

            Log.Info(model.Summary() + Environment.NewLine);
            Log.Info(model.ToPytorchModule(X.Shape[0]) + Environment.NewLine);

            var predict_before = model.Predict(X, false).ToNumpy();

            using var trainingDataSet = new InMemoryDataSet(X, Y);

            var lossAccuracyBefore = model.ComputeMetricsForValidationDataSet(model.Sample.BatchSize, trainingDataSet);
            var loss_before = (float)lossAccuracyBefore.First(t => t.Key == model.NetworkSample.LossFunction).Value;

            Log.Info("-");
            Log.Info("--------------------------------------------------------------------");
            Log.Info("-");

            model.Sample.DisableReduceLROnPlateau = true;
            model.Fit(trainingDataSet, null);

            var predict_after = model.Predict(X, false).ToNumpy();
            List<KeyValuePair<EvaluationMetricEnum, double>> lossAccuracyAfter = model.ComputeMetricsForValidationDataSet(model.Sample.BatchSize, trainingDataSet);
            var loss_after = (float)lossAccuracyAfter.First(t => t.Key == model.NetworkSample.LossFunction).Value;

            Log.Info("C# num_epochs= " + model.Sample.num_epochs);
            Log.Info("C# learningRate= " + model.Sample.InitialLearningRate);
            Log.Info("C# momentum= " + GetMomentum(model));
            if (model.Sample.SGD_usenesterov)
            {
                Log.Info("C# nesterov = " + model.Sample.SGD_usenesterov);
            }
            Log.Info("C# batch_size= " + model.Sample.BatchSize);
            Log.Info("C# lambdaL2Regularization= " + Get_lambdaL2Regularization(model));
            Log.Info(predict_before);
            Log.Info("C# metrics_before= " + Model.MetricsToString(lossAccuracyBefore, ""));
            Log.Info(predict_after);
            Log.Info("C# metrics_after= " + Model.MetricsToString(lossAccuracyAfter, ""));
            return (loss_before, loss_after);
        }


        private static double GetMomentum(Network model)
        {
            switch (model.Sample.OptimizerType)
            {
                case Optimizer.OptimizationEnum.SGD:
                    return model.Sample.SGD_momentum;
                default:
                    return 0;
            }
        }


        private static double Get_lambdaL2Regularization(Network model)
        {
            switch (model.Sample.OptimizerType)
            {
                case Optimizer.OptimizationEnum.AdamW:
                    return model.Sample.AdamW_L2Regularization;
                default:
                    return model.Sample.lambdaL2Regularization;
            }
        }

        [Test, Explicit]
        public void TestParallelRunWithPyTorch_Mse()
        {
            var X = TestNetworkPropagation.numpy_array_for_tests(2, 3, 4, 5);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(2, 3);
            const double momentum = 0.9;
            const int deviceId = 0;

            var sample = new NetworkSample
                {
                    LossFunction = EvaluationMetricEnum.Mse,
                    ShuffleDatasetBeforeEachEpoch = false,
                    AutoSaveIntervalInMinutes = -1,
                    CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                    LogNetworkPropagation = true,
                    InitialLearningRate = 0.1,
                    num_epochs = 10,
                    BatchSize = X.Shape[0],
                    lambdaL2Regularization = 0,
                    ResourceIds = new List<int> { deviceId }
                }
                .WithSGD(momentum, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestParallelRunWithPyTorch_Mse), false);
            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Linear(3, true, 0.0, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Flatten()
                .Linear(3, true, 0.0, true)
                ;

            TestNetworkPropagation.FromNumpyArray("[[-0.0033482015132904053, 0.23990488052368164, -0.36807698011398315, -0.3291219472885132, -0.1722462773323059], [0.11992359161376953, -0.008860737085342407, 0.3545909523963928, -0.039687544107437134, 0.11833834648132324], [-0.13515380024909973, -0.08790671825408936, -0.4272448420524597, -0.2961815595626831, -0.18435177206993103]]").CopyTo(((LinearLayer)model.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.11299018561840057, -0.07257714122533798, 0.06053619086742401, 0.13839800655841827, -0.034300029277801514, 0.12471862137317657, -0.026863887906074524, 0.017635688185691833, 0.15091271698474884, -0.154611736536026, -0.10492299497127533, -0.04219420999288559, -0.06496666371822357, 0.1440001279115677, -0.10802994668483734, -0.0767221450805664, -0.11644008010625839, -0.1560935080051422, -0.09729008376598358, 0.14326633512973785, 0.07436972856521606, 0.08077876269817352, 0.008765265345573425, -0.08544725179672241, 0.028197452425956726, -0.15561579167842865, -0.12042771279811859, -0.08592166751623154, 0.1051563173532486, 0.09772022068500519, -0.07391583919525146, -0.006013736128807068, 0.10659344494342804, 0.16568885743618011, 0.06614702939987183, 0.022515475749969482], [0.11174772679805756, -0.09813372790813446, 0.031057342886924744, -0.12921759486198425, -0.1155143603682518, -0.08609726279973984, 0.07541216909885406, 0.06702673435211182, -0.09872542321681976, 0.05035118758678436, 0.09149535000324249, -0.021036222577095032, 0.006363585591316223, 0.03861744701862335, 0.10339610278606415, 0.16003234684467316, -0.12843726575374603, -0.06107829511165619, 0.06550164520740509, 0.138091579079628, 0.1450345665216446, 0.14705945551395416, 0.03316909074783325, -0.14493045210838318, 0.015332087874412537, -0.10426755994558334, -0.15532569587230682, 0.14808209240436554, 0.1267266422510147, -0.1662546694278717, 0.031195342540740967, -0.028076663613319397, -0.02742685377597809, -0.07629281282424927, 0.06409269571304321, -0.09871725738048553], [0.06109856069087982, 0.08428467810153961, 0.11931194365024567, 0.06231851875782013, -0.16495588421821594, -0.10811614990234375, 0.08321917057037354, 0.03488355875015259, -0.13001400232315063, -0.09596991539001465, 0.156791552901268, 0.11230297386646271, -0.07267086207866669, -0.04194746911525726, -0.15876635909080505, -0.0029956847429275513, -0.1255098283290863, -0.12855945527553558, -0.00918327271938324, 0.0250241756439209, -0.06825505197048187, 0.09889627993106842, -0.10142318904399872, 0.15122835338115692, 0.1142166405916214, -0.14054715633392334, -0.04148072004318237, 0.007520437240600586, 0.024316847324371338, 0.039529040455818176, 0.06540471315383911, 0.009983360767364502, -0.08132146298885345, 0.07886482775211334, -0.159874826669693, -0.0987844467163086]]").CopyTo(((LinearLayer)model.Layers[4]).Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);
            Assert.AreEqual(0.297953724861145, loss_before, 1e-6);
            Assert.AreEqual(0.09050967544317245, loss_after, 1e-6);
        }


        [Test, Explicit]
        public void TestParallelRunWithPyTorch_Mse_AdamW()
        {
            const int rows = 1;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 1, 1, 1);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 2);
            const int deviceId = -1;

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.Mse,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                LogNetworkPropagation = true,
                InitialLearningRate = 1,
                num_epochs = 1,
                BatchSize = X.Shape[0],
                lambdaL2Regularization = 0,
                ResourceIds = new List<int> { deviceId }
            }
                .WithAdamW(0.1, 0.9, 0.999, 1e-8);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestParallelRunWithPyTorch_Mse_AdamW), false);
            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Linear(2, true, 0.0, false)
                //.Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                //.Linear(3, 0.0, false)
                ;

            TestNetworkPropagation.FromNumpyArray("[[-0.007486820220947266], [0.5364435911178589]]").CopyTo(((LinearLayer)model.Layers[1]).Weights);
            //TestNetworkPropagation.FromNumpyArray("[[-0.11299018561840057, -0.07257714122533798, 0.06053619086742401, 0.13839800655841827, -0.034300029277801514, 0.12471862137317657, -0.026863887906074524, 0.017635688185691833, 0.15091271698474884, -0.154611736536026, -0.10492299497127533, -0.04219420999288559, -0.06496666371822357, 0.1440001279115677, -0.10802994668483734, -0.0767221450805664, -0.11644008010625839, -0.1560935080051422, -0.09729008376598358, 0.14326633512973785, 0.07436972856521606, 0.08077876269817352, 0.008765265345573425, -0.08544725179672241, 0.028197452425956726, -0.15561579167842865, -0.12042771279811859, -0.08592166751623154, 0.1051563173532486, 0.09772022068500519, -0.07391583919525146, -0.006013736128807068, 0.10659344494342804, 0.16568885743618011, 0.06614702939987183, 0.022515475749969482], [0.11174772679805756, -0.09813372790813446, 0.031057342886924744, -0.12921759486198425, -0.1155143603682518, -0.08609726279973984, 0.07541216909885406, 0.06702673435211182, -0.09872542321681976, 0.05035118758678436, 0.09149535000324249, -0.021036222577095032, 0.006363585591316223, 0.03861744701862335, 0.10339610278606415, 0.16003234684467316, -0.12843726575374603, -0.06107829511165619, 0.06550164520740509, 0.138091579079628, 0.1450345665216446, 0.14705945551395416, 0.03316909074783325, -0.14493045210838318, 0.015332087874412537, -0.10426755994558334, -0.15532569587230682, 0.14808209240436554, 0.1267266422510147, -0.1662546694278717, 0.031195342540740967, -0.028076663613319397, -0.02742685377597809, -0.07629281282424927, 0.06409269571304321, -0.09871725738048553], [0.06109856069087982, 0.08428467810153961, 0.11931194365024567, 0.06231851875782013, -0.16495588421821594, -0.10811614990234375, 0.08321917057037354, 0.03488355875015259, -0.13001400232315063, -0.09596991539001465, 0.156791552901268, 0.11230297386646271, -0.07267086207866669, -0.04194746911525726, -0.15876635909080505, -0.0029956847429275513, -0.1255098283290863, -0.12855945527553558, -0.00918327271938324, 0.0250241756439209, -0.06825505197048187, 0.09889627993106842, -0.10142318904399872, 0.15122835338115692, 0.1142166405916214, -0.14054715633392334, -0.04148072004318237, 0.007520437240600586, 0.024316847324371338, 0.039529040455818176, 0.06540471315383911, 0.009983360767364502, -0.08132146298885345, 0.07886482775211334, -0.159874826669693, -0.0987844467163086]]").CopyTo(((LinearLayer)model.Layers[3]).Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);
            Assert.AreEqual(0.6364270448684692, loss_before, 1e-6);
            Assert.AreEqual(1.6577095985412598, loss_after, 1e-5); //1.6577413082122803d , 1.6577074527740479d
        }


        [Test, Explicit]
        public void TestReluActivation_NCHW_2_1_4_4()
        {
            var X = TestNetworkPropagation.numpy_array_for_tests(2, 1, 4, 4);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(2, 3);

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.BinaryCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.1,
                num_epochs = 10,
                BatchSize = X.Shape[0],
                lambdaL2Regularization = 0.03,
                ResourceIds = new List<int> { -1 }
            }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestReluActivation_NCHW_2_1_4_4), false);
            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Flatten()
                .Linear(3, true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);
            TestNetworkPropagation.FromNumpyArray("[[-0.0018717050552368164, 0.13411089777946472, -0.20576128363609314, -0.18398475646972656, -0.09628859162330627, 0.06703934073448181, -0.004953294992446899, 0.19822236895561218, -0.02218601107597351, 0.0661531388759613, -0.07555326819419861, -0.04914134740829468, -0.2388371229171753, -0.16557052731513977, -0.10305577516555786, 0.009260892868041992], [0.09883379936218262, 0.15000569820404053, -0.16948527097702026, -0.10886570811271667, 0.09080427885055542, 0.20759698748588562, -0.05145004391670227, 0.18707793951034546, -0.04029583930969238, 0.026453524827957153, 0.22636905312538147, -0.2319175899028778, -0.1573844850063324, -0.06329131126403809, -0.09744998812675476, 0.21600019931793213], [-0.16204491257667542, -0.11508321762084961, -0.1746601164340973, -0.23414024710655212, -0.14593511819839478, 0.21489951014518738, 0.1115545928478241, 0.12116813659667969, 0.013147890567779541, -0.12817087769508362, 0.04229617118835449, -0.23342368006706238, -0.1806415617465973, -0.128882497549057, 0.1577344834804535, 0.146580308675766]]").CopyTo(((LinearLayer)model.Layers[2]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.5739630460739136, 0.229140043258667, 0.0779958963394165], [0.38710546493530273, -0.3399451971054077, 0.10758578777313232], [-0.4476228356361389, -0.4001534581184387, -0.29824966192245483]]").CopyTo(((LinearLayer)model.Layers[4]).Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);
            Assert.AreEqual(0.6726791262626648, loss_before, 1e-6);
            Assert.AreEqual(0.5357139110565186, loss_after, 1e-6);

            //test_save_sharpnet(model, X);
        }


        [Test, Explicit]
        public void TestParallelRunWithPyTorch_Convolution()
        {
            var X = TestNetworkPropagation.numpy_array_for_tests(2, 3, 4, 5);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(2, 2);
            const int deviceId = 0;

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.01,
                num_epochs = 10,
                BatchSize = X.Shape[0],
                lambdaL2Regularization = 0,
                ResourceIds = new List<int> { deviceId }
            }
                    .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestParallelRunWithPyTorch_Convolution), false);
            model.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true)
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true)
                .GlobalAvgPooling()
                .MultiplyLayer(1, 3)
                .Flatten()
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            TestNetworkPropagation.FromNumpyArray("[[[[-0.004322528839111328]], [[0.3097158670425415]], [[-0.4751853346824646]]], [[[-0.4248945713043213]], [[-0.222368985414505]], [[0.1548207402229309]]]]").CopyTo(((ConvolutionLayer)model.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[[[-0.06275153160095215]], [[0.1871093511581421]]], [[[-0.2136968970298767]], [[-0.13899272680282593]]]]").CopyTo(((ConvolutionLayer)model.Layers[2]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.06517819315195084, 0.005857110023498535, 0.0625079870223999, 0.09487192332744598, -0.10719189792871475, -0.06885271519422531, 0.05742967128753662, 0.13129587471485138, -0.03253985941410065, 0.11831848323345184, -0.025485321879386902, 0.01673068106174469, 0.1431683450937271, -0.14667756855487823, -0.09953868389129639, -0.04002893716096878, -0.061632782220840454, 0.13661052286624908, -0.10248620063066483, -0.07278501987457275, -0.11046475172042847, -0.14808329939842224, -0.09229747205972672, 0.1359143704175949, 0.07055331766605377, 0.07663345336914062, 0.00831545889377594, -0.08106237649917603, 0.02675044536590576, -0.14763009548187256, -0.11424775421619415, -0.081512451171875, 0.099760040640831, 0.09270553290843964, -0.07012271881103516, -0.005705133080482483, 0.10112343728542328, 0.15718625485897064, 0.06275257468223572, 0.02136005461215973], [0.10601319372653961, -0.09309782087802887, 0.029463574290275574, -0.12258656322956085, -0.10958653688430786, -0.08167903125286102, 0.07154226303100586, 0.06358714401721954, -0.09365915507078171, 0.04776732623577118, 0.08680009841918945, -0.019956722855567932, 0.006037026643753052, 0.03663572669029236, 0.09809015691280365, 0.15182001888751984, -0.1218462884426117, -0.057943955063819885, 0.062140315771102905, 0.1310051530599594, 0.13759185373783112, 0.13951285183429718, 0.03146696090698242, -0.1374930888414383, 0.014545291662216187, -0.09891688823699951, -0.1473548859357834, 0.14048300683498383, 0.12022344768047333, -0.15772302448749542, 0.02959449589252472, -0.026635870337486267, -0.02601940929889679, -0.07237771898508072, 0.06080366671085358, -0.09365140646696091, 0.05796317756175995, 0.07995946705341339, 0.11318923532962799, 0.0591205358505249]]").CopyTo(((LinearLayer)model.Layers[6]).Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);

            Assert.AreEqual(0.6931346654891968, loss_before, 1e-6);
            Assert.AreEqual(0.6924928426742554, loss_after, 1e-6);
        }


        //TODO: make it work with PyTorch
        [Test, Explicit]
        public void TestParallelRunWithPyTorch_Convolution_AdamW()
        {
            var X = TestNetworkPropagation.numpy_array_for_tests(2, 3, 4, 5);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(2, 2);
            const int deviceId = 0;

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.01,
                num_epochs = 10,
                BatchSize = X.Shape[0],
                lambdaL2Regularization = 0,
                ResourceIds = new List<int> { deviceId }
            }
                    .WithAdamW(0.005, 0.9, 0.999, 1e-8);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestParallelRunWithPyTorch_Convolution_AdamW), false);
            model.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true)
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true)
                .GlobalAvgPooling()
                .MultiplyLayer(1, 3)
                .Flatten()
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            TestNetworkPropagation.FromNumpyArray("[[[[-0.004322528839111328]], [[0.3097158670425415]], [[-0.4751853346824646]]], [[[-0.4248945713043213]], [[-0.222368985414505]], [[0.1548207402229309]]]]").CopyTo(((ConvolutionLayer)model.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[[[-0.06275153160095215]], [[0.1871093511581421]]], [[[-0.2136968970298767]], [[-0.13899272680282593]]]]").CopyTo(((ConvolutionLayer)model.Layers[2]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.06517819315195084, 0.005857110023498535, 0.0625079870223999, 0.09487192332744598, -0.10719189792871475, -0.06885271519422531, 0.05742967128753662, 0.13129587471485138, -0.03253985941410065, 0.11831848323345184, -0.025485321879386902, 0.01673068106174469, 0.1431683450937271, -0.14667756855487823, -0.09953868389129639, -0.04002893716096878, -0.061632782220840454, 0.13661052286624908, -0.10248620063066483, -0.07278501987457275, -0.11046475172042847, -0.14808329939842224, -0.09229747205972672, 0.1359143704175949, 0.07055331766605377, 0.07663345336914062, 0.00831545889377594, -0.08106237649917603, 0.02675044536590576, -0.14763009548187256, -0.11424775421619415, -0.081512451171875, 0.099760040640831, 0.09270553290843964, -0.07012271881103516, -0.005705133080482483, 0.10112343728542328, 0.15718625485897064, 0.06275257468223572, 0.02136005461215973], [0.10601319372653961, -0.09309782087802887, 0.029463574290275574, -0.12258656322956085, -0.10958653688430786, -0.08167903125286102, 0.07154226303100586, 0.06358714401721954, -0.09365915507078171, 0.04776732623577118, 0.08680009841918945, -0.019956722855567932, 0.006037026643753052, 0.03663572669029236, 0.09809015691280365, 0.15182001888751984, -0.1218462884426117, -0.057943955063819885, 0.062140315771102905, 0.1310051530599594, 0.13759185373783112, 0.13951285183429718, 0.03146696090698242, -0.1374930888414383, 0.014545291662216187, -0.09891688823699951, -0.1473548859357834, 0.14048300683498383, 0.12022344768047333, -0.15772302448749542, 0.02959449589252472, -0.026635870337486267, -0.02601940929889679, -0.07237771898508072, 0.06080366671085358, -0.09365140646696091, 0.05796317756175995, 0.07995946705341339, 0.11318923532962799, 0.0591205358505249]]").CopyTo(((LinearLayer)model.Layers[6]).Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);

            Assert.AreEqual(0.6931346654891968, loss_before, 1e-6);
            //!D Assert.AreEqual(0.6222155690193176, loss_after, 1e-6);
            Assert.AreEqual(0.62760639190673828d, loss_after, 1e-6);
        }

        //TODO: make it work with PyTorch
        [Test, Explicit]
        public void Test_Convolution_With_Asymmetric_Padding()
        {
            var X = TestNetworkPropagation.numpy_array_for_tests(2, 3, 4, 5);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(2, 2);

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.1,
                num_epochs = 10,
                BatchSize = X.Shape[0],
                lambdaL2Regularization = 0.15,
                ResourceIds = new List<int> { -1 }
            }
                    .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(Test_Convolution_With_Asymmetric_Padding), false);
            model.Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 3, 2, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, false)
                .Flatten()
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);


            TestNetworkPropagation.FromNumpyArray("[[[[-0.0014408379793167114, 0.10323862731456757, -0.15839511156082153], [-0.14163152873516083, -0.07412299513816833, 0.051606908440589905], [-0.003813043236732483, 0.15259166061878204, -0.01707880198955536]], [[0.05092470347881317, -0.0581609308719635, -0.03782902657985687], [-0.18385690450668335, -0.12745624780654907, -0.0793323740363121], [0.007129043340682983, 0.07608230412006378, 0.11547444760799408]], [[-0.13046982884407043, -0.08380486071109772, 0.06990115344524384], [0.1598082333803177, -0.039606258273124695, 0.1440126746892929], [-0.031019747257232666, 0.020363926887512207, 0.1742589920759201]]]]").CopyTo(((ConvolutionLayer)model.Layers[1]).Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.37871986627578735, -0.25700777769088745, -0.10335427522659302, -0.15913516283035278, 0.35272687673568726, -0.26461824774742126], [-0.18793010711669922, -0.2852187752723694, -0.38234943151474, -0.2383110523223877, 0.3509294390678406, 0.1821678876876831]]").CopyTo(((LinearLayer)model.Layers[3]).Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);

            Assert.AreEqual(0.6809013485908508, loss_before, 1e-6);
            Assert.AreEqual(0.45675480365753174, loss_after, 1e-6);
            //test_save_sharpnet(model, X);
        }

        [Test, Explicit]
        public void TestParallelRunWithPyTorch_Conv1D()
        {
            var X = TestNetworkPropagation.numpy_array_for_tests(3, 4, 5);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(3, 3);
            const int deviceId = 0;

            var sample = new NetworkSample
                {
                    LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                    ShuffleDatasetBeforeEachEpoch = false,
                    AutoSaveIntervalInMinutes = -1,
                    CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                    LogNetworkPropagation = true,
                    InitialLearningRate = 0.1,
                    num_epochs = 10,
                    BatchSize = X.Shape[0],
                    lambdaL2Regularization = 0,
                    ResourceIds = new List<int> { deviceId }
                }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestParallelRunWithPyTorch_Conv1D), false);
            model.Input(X.Shape[1], X.Shape[2], -1)
                .Conv1D(2, 3, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true)
                .Conv1D(2, 3, 2, ConvolutionLayer.PADDING_TYPE.VALID, sample.lambdaL2Regularization, true)
                .Flatten()
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);


            TestNetworkPropagation.FromNumpyArray("[[[-0.002161264419555664, 0.15485793352127075, -0.2375926673412323], [-0.21244728565216064, -0.1111844927072525, 0.07741037011146545], [-0.005719572305679321, 0.22888749837875366, -0.02561819553375244], [0.07638707756996155, -0.08724139630794525, -0.05674353241920471]], [[-0.275785356760025, -0.1911843717098236, -0.11899855732917786], [0.010693550109863281, 0.11412343382835388, 0.17321166396141052], [-0.19570472836494446, -0.12570728361606598, 0.10485175251960754], [0.23971235752105713, -0.05940939486026764, 0.21601897478103638]]]").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[0.3696591258049011, -0.37871986627578735, -0.25700777769088745], [-0.10335427522659302, -0.15913516283035278, 0.35272687673568726]], [[-0.26461824774742126, -0.18793010711669922, -0.2852187752723694], [-0.38234943151474, -0.2383110523223877, 0.3509294390678406]]]").CopyTo(model.Layers[2].Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.026295781135559082, -0.25634175539016724, 0.08459234237670898, -0.46684736013412476], [-0.3612831234931946, -0.257764995098114, 0.315468966960907, 0.293160617351532], [-0.2217475175857544, -0.01804119348526001, 0.3197803497314453, 0.49706655740737915]]").CopyTo(model.Layers[4].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);

            Assert.AreEqual(1.0818352699279785, loss_before, 1e-6);
            Assert.AreEqual(0.7389676570892334, loss_after, 1e-6);
        }


        [Test, Explicit]
        public void TestParallelRunWithPyTorch_LayerNormalizationNchw()
        {
            var X = TestNetworkPropagation.numpy_array_for_tests(2, 3, 4, 5);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(2, 2);
            const int deviceId = 0;

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.001,
                num_epochs = 10,
                BatchSize = X.Shape[0],
                lambdaL2Regularization = 0,
                ResourceIds = new List<int> { deviceId }
            }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestParallelRunWithPyTorch_LayerNormalizationNchw), false);

            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 5, 2, ConvolutionLayer.PADDING_TYPE.SAME, 0.00, false)
                .LayerNorm(1, 0.001)
                .Flatten()
                .Linear(Y.Shape[1], true, 0.00, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            TestNetworkPropagation.FromNumpyArray("[[[[-0.0008645057678222656, 0.0619431734085083, -0.0950370654463768, -0.08497891575098038, -0.044473797082901], [0.03096415102481842, -0.002287827432155609, 0.09155498445034027, -0.010247282683849335, 0.03055483102798462], [-0.03489656001329422, -0.022697418928146362, -0.11031413823366165, -0.07647375017404556, -0.047599419951438904], [0.004277423024177551, 0.04564937949180603, 0.06928466260433197, -0.07828189432621002, -0.05028291791677475], [0.041940703988075256, 0.09588493406772614, -0.023763753473758698, 0.0864076018333435, -0.0186118483543396]], [[0.012218356132507324, 0.10455538332462311, -0.1071181446313858, -0.07269278168678284, -0.029233001172542572], [-0.04501022398471832, 0.09976620972156525, -0.0748453363776207, -0.053154658526182175, -0.08067205548286438], [-0.10814474523067474, -0.06740453839302063, 0.09925782680511475, 0.05152486264705658, 0.05596517026424408], [0.0060727521777153015, -0.05919959023594856, 0.019535765051841736, -0.1078137755393982, -0.08343476057052612], [-0.059528276324272156, 0.0728544294834137, 0.06770254671573639, -0.051210395991802216, -0.004166431725025177]], [[0.0738501101732254, 0.114792600274086, 0.04582799971103668, 0.015599176287651062, 0.07742108404636383], [-0.0679890364408493, 0.021517157554626465, -0.08952456712722778, -0.08003069460391998, -0.05964992940425873], [0.05224709212779999, 0.04643748700618744, -0.06839897483587265, 0.034884318709373474, 0.06338982284069061], [-0.014574326574802399, 0.0044088214635849, 0.026754960417747498, 0.0716349184513092, 0.11087366938591003], [-0.0889839455485344, -0.04231628030538559, 0.04538087546825409, 0.09567263722419739, 0.10048288106918335]]], [[[0.10188578069210052, 0.022980213165283203, -0.10041075199842453, 0.010622382164001465, -0.07223868370056152], [-0.10761279612779617, 0.1025942713022232, 0.08779877424240112, -0.11518460512161255, 0.02161276340484619], [-0.019452087581157684, -0.019001886248588562, -0.05285721272230148, 0.04440471529960632, -0.06839331984519958], [0.04233032464981079, 0.05839413404464722, 0.08266173303127289, 0.043175533413887024, -0.11428478360176086], [-0.07490506768226624, 0.057655930519104004, 0.024168044328689575, -0.09007634222507477, -0.06648990511894226]], [[0.10862836241722107, 0.0778057873249054, -0.050347842276096344, -0.02906205505132675, -0.10999654978513718], [-0.002075470983982086, -0.08695575594902039, -0.08906859904527664, -0.006362356245517731, 0.017337262630462646], [-0.04728848487138748, 0.06851734220981598, -0.07026804238557816, 0.10477407276630402, 0.07913161814212799], [-0.09737392514944077, -0.028738684952259064, 0.005210310220718384, 0.01684720814228058, 0.027386531233787537], [0.04531371593475342, 0.006916671991348267, -0.056341156363487244, 0.05463916063308716, -0.1107645183801651]], [[-0.06843987107276917, -0.028905600309371948, -0.05624700337648392, -0.04039527475833893, -0.09464175999164581], [-0.024562232196331024, 0.024682462215423584, -0.07522478699684143, -0.005925849080085754, 0.08265933394432068], [-0.011870346963405609, 0.0032091662287712097, -0.00996147096157074, 0.02336898446083069, 0.07342042028903961], [0.10937856137752533, 0.07332994043827057, 0.109628826379776, -0.008350983262062073, -0.1037292331457138], [-0.05474172160029411, 0.07862415909767151, -0.0007485300302505493, -0.05739396810531616, -0.08848606050014496]]]]").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.27015721797943115, -0.24364417791366577, -0.058553919196128845, 0.15831118822097778, 0.15606963634490967, -0.278407484292984, 0.18007037043571472, -0.22589102387428284, -0.061028897762298584, -0.11704987287521362, -0.055603235960006714, -0.05667924880981445], [-0.2590426206588745, -0.249253049492836, -0.04517173767089844, 0.0037331879138946533, -0.13113786280155182, 0.10874369740486145, -0.2598244845867157, -0.019481897354125977, 0.2538664937019348, -0.11774827539920807, 0.26067453622817993, 0.10454478859901428]]").CopyTo(model.Layers[4].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);

            Assert.AreEqual(0.8300741910934448, loss_before, 1e-6);
            Assert.AreEqual(0.5169422626495361, loss_after, 1e-6);
            //test_save_sharpnet(model, X);
        }

        [Test, Explicit]
        public void TestParallelRunWithPyTorch_RMSNormNchw()
        {
            var X = TestNetworkPropagation.numpy_array_for_tests(2, 3, 4, 5);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(2, 2);
            const int deviceId = -1;

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.001,
                num_epochs = 10,
                BatchSize = X.Shape[0],
                lambdaL2Regularization = 0,
                ResourceIds = new List<int> { deviceId }
            }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestParallelRunWithPyTorch_RMSNormNchw), false);

            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 5, 2, ConvolutionLayer.PADDING_TYPE.SAME, 0.00, false)
                .RMSNorm(1, 0.001)
                .Flatten()
                .Linear(Y.Shape[1], true, 0.00, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            TestNetworkPropagation.FromNumpyArray("[[[[-0.0008645057678222656, 0.0619431734085083, -0.0950370654463768, -0.08497891575098038, -0.044473797082901], [0.03096415102481842, -0.002287827432155609, 0.09155498445034027, -0.010247282683849335, 0.03055483102798462], [-0.03489656001329422, -0.022697418928146362, -0.11031413823366165, -0.07647375017404556, -0.047599419951438904], [0.004277423024177551, 0.04564937949180603, 0.06928466260433197, -0.07828189432621002, -0.05028291791677475], [0.041940703988075256, 0.09588493406772614, -0.023763753473758698, 0.0864076018333435, -0.0186118483543396]], [[0.012218356132507324, 0.10455538332462311, -0.1071181446313858, -0.07269278168678284, -0.029233001172542572], [-0.04501022398471832, 0.09976620972156525, -0.0748453363776207, -0.053154658526182175, -0.08067205548286438], [-0.10814474523067474, -0.06740453839302063, 0.09925782680511475, 0.05152486264705658, 0.05596517026424408], [0.0060727521777153015, -0.05919959023594856, 0.019535765051841736, -0.1078137755393982, -0.08343476057052612], [-0.059528276324272156, 0.0728544294834137, 0.06770254671573639, -0.051210395991802216, -0.004166431725025177]], [[0.0738501101732254, 0.114792600274086, 0.04582799971103668, 0.015599176287651062, 0.07742108404636383], [-0.0679890364408493, 0.021517157554626465, -0.08952456712722778, -0.08003069460391998, -0.05964992940425873], [0.05224709212779999, 0.04643748700618744, -0.06839897483587265, 0.034884318709373474, 0.06338982284069061], [-0.014574326574802399, 0.0044088214635849, 0.026754960417747498, 0.0716349184513092, 0.11087366938591003], [-0.0889839455485344, -0.04231628030538559, 0.04538087546825409, 0.09567263722419739, 0.10048288106918335]]], [[[0.10188578069210052, 0.022980213165283203, -0.10041075199842453, 0.010622382164001465, -0.07223868370056152], [-0.10761279612779617, 0.1025942713022232, 0.08779877424240112, -0.11518460512161255, 0.02161276340484619], [-0.019452087581157684, -0.019001886248588562, -0.05285721272230148, 0.04440471529960632, -0.06839331984519958], [0.04233032464981079, 0.05839413404464722, 0.08266173303127289, 0.043175533413887024, -0.11428478360176086], [-0.07490506768226624, 0.057655930519104004, 0.024168044328689575, -0.09007634222507477, -0.06648990511894226]], [[0.10862836241722107, 0.0778057873249054, -0.050347842276096344, -0.02906205505132675, -0.10999654978513718], [-0.002075470983982086, -0.08695575594902039, -0.08906859904527664, -0.006362356245517731, 0.017337262630462646], [-0.04728848487138748, 0.06851734220981598, -0.07026804238557816, 0.10477407276630402, 0.07913161814212799], [-0.09737392514944077, -0.028738684952259064, 0.005210310220718384, 0.01684720814228058, 0.027386531233787537], [0.04531371593475342, 0.006916671991348267, -0.056341156363487244, 0.05463916063308716, -0.1107645183801651]], [[-0.06843987107276917, -0.028905600309371948, -0.05624700337648392, -0.04039527475833893, -0.09464175999164581], [-0.024562232196331024, 0.024682462215423584, -0.07522478699684143, -0.005925849080085754, 0.08265933394432068], [-0.011870346963405609, 0.0032091662287712097, -0.00996147096157074, 0.02336898446083069, 0.07342042028903961], [0.10937856137752533, 0.07332994043827057, 0.109628826379776, -0.008350983262062073, -0.1037292331457138], [-0.05474172160029411, 0.07862415909767151, -0.0007485300302505493, -0.05739396810531616, -0.08848606050014496]]]]").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.27015721797943115, -0.24364417791366577, -0.058553919196128845, 0.15831118822097778, 0.15606963634490967, -0.278407484292984, 0.18007037043571472, -0.22589102387428284, -0.061028897762298584, -0.11704987287521362, -0.055603235960006714, -0.05667924880981445], [-0.2590426206588745, -0.249253049492836, -0.04517173767089844, 0.0037331879138946533, -0.13113786280155182, 0.10874369740486145, -0.2598244845867157, -0.019481897354125977, 0.2538664937019348, -0.11774827539920807, 0.26067453622817993, 0.10454478859901428]]").CopyTo(model.Layers[4].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);

            Assert.AreEqual(0.8518396019935608, loss_before, 1e-6);
            Assert.AreEqual(0.6556870937347412, loss_after, 1e-6);
            //test_save_sharpnet(model, X);
        }


        [Test, Explicit]
        public void TestParallelRunWithPyTorch_BatchNormalizationNchw2345()
        {
            const int rows = 10;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 3, 4, 5);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 60);

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.Mae,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = false,
                InitialLearningRate = 0.1,
                num_epochs = 10,
                BatchSize = 2,
                lambdaL2Regularization = 0,
                ResourceIds = new List<int> {-1}
            }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestParallelRunWithPyTorch_BatchNormalizationNchw2345), false);

            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .BatchNorm(0.99, 0.001)
                .Flatten()
                ;
            (float loss_before, float loss_after) = Train(model, X, Y);

            Assert.AreEqual(0.2588789165019989, loss_before, 1e-6);
            Assert.AreEqual(0.035503968596458435, loss_after, 1e-6);

            //Assert.AreEqual(0.26209956407546997, loss_before, 1e-6);
            //Assert.AreEqual(0.15799619257450104, loss_after, 1e-3);
            //test_save_sharpnet(model, X);
        }


        [Test, Explicit]
        public void TestResNet_Shortcut_Same_Dimension_NCHW_2_1_4_4()
        {
            var X = TestNetworkPropagation.numpy_array_for_tests(2, 1, 4, 4);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(2, 3);
            const int deviceId = -1;

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.1,
                num_epochs = 10,
                BatchSize = X.Shape[0],
                lambdaL2Regularization = 0,
                ResourceIds = new List<int> { deviceId }
            }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestResNet_Shortcut_Same_Dimension_NCHW_2_1_4_4), false);

            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true)
                .AddLayer(2, 1)
                .Flatten()
                .Linear(Y.Shape[1], true, 0.0, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            
            TestNetworkPropagation.FromNumpyArray("[[[[-0.007486820220947266]]]]").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[[-0.8230451345443726]]]]").CopyTo(model.Layers[2].Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.09628859162330627, 0.06703934073448181, -0.004953294992446899, 0.19822236895561218, -0.02218601107597351, 0.0661531388759613, -0.07555326819419861, -0.04914134740829468, -0.2388371229171753, -0.16557052731513977, -0.10305577516555786, 0.009260892868041992, 0.09883379936218262, 0.15000569820404053, -0.16948527097702026, -0.10886570811271667], [0.09080427885055542, 0.20759698748588562, -0.05145004391670227, 0.18707793951034546, -0.04029583930969238, 0.026453524827957153, 0.22636905312538147, -0.2319175899028778, -0.1573844850063324, -0.06329131126403809, -0.09744998812675476, 0.21600019931793213, -0.16204491257667542, -0.11508321762084961, -0.1746601164340973, -0.23414024710655212], [-0.14593511819839478, 0.21489951014518738, 0.1115545928478241, 0.12116813659667969, 0.013147890567779541, -0.12817087769508362, 0.04229617118835449, -0.23342368006706238, -0.1806415617465973, -0.128882497549057, 0.1577344834804535, 0.146580308675766, -0.1108737587928772, -0.009020596742630005, 0.15989017486572266, 0.24853327870368958]]").CopyTo(model.Layers[5].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);

            Assert.AreEqual(1.0986199378967285, loss_before, 1e-6);
            Assert.AreEqual(0.7018476724624634, loss_after, 1e-6);


        }


        [Test, Explicit]
        public void TestResNet_Shortcut_Different_Dimension_With_Conv_1x1_to_change_Dimension_NCHW_2_1_4_4()
        {
            var X = TestNetworkPropagation.numpy_array_for_tests(2, 1, 4, 4);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(2, 3);
            const int deviceId = -1;

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.1,
                num_epochs = 10,
                BatchSize = X.Shape[0],
                lambdaL2Regularization = 0,
                ResourceIds = new List<int> { deviceId }
            }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestResNet_Shortcut_Different_Dimension_With_Conv_1x1_to_change_Dimension_NCHW_2_1_4_4), false);

            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true) //left
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, 0.0, true, 1) //right (identity shortcut)
                .AddLayer(3, 2)
                .Flatten()
                .Linear(Y.Shape[1], true, 0.0, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX)
                ;

            TestNetworkPropagation.FromNumpyArray("[[[[-0.007486820220947266]]]]").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[[-0.8230451345443726]]]]").CopyTo(model.Layers[2].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[[-0.3851543664932251]]]]").CopyTo(model.Layers[3].Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.004953294992446899, 0.19822236895561218, -0.02218601107597351, 0.0661531388759613, -0.07555326819419861, -0.04914134740829468, -0.2388371229171753, -0.16557052731513977, -0.10305577516555786, 0.009260892868041992, 0.09883379936218262, 0.15000569820404053, -0.16948527097702026, -0.10886570811271667, 0.09080427885055542, 0.20759698748588562], [-0.05145004391670227, 0.18707793951034546, -0.04029583930969238, 0.026453524827957153, 0.22636905312538147, -0.2319175899028778, -0.1573844850063324, -0.06329131126403809, -0.09744998812675476, 0.21600019931793213, -0.16204491257667542, -0.11508321762084961, -0.1746601164340973, -0.23414024710655212, -0.14593511819839478, 0.21489951014518738], [0.1115545928478241, 0.12116813659667969, 0.013147890567779541, -0.12817087769508362, 0.04229617118835449, -0.23342368006706238, -0.1806415617465973, -0.128882497549057, 0.1577344834804535, 0.146580308675766, -0.1108737587928772, -0.009020596742630005, 0.15989017486572266, 0.24853327870368958, 0.09922054409980774, 0.033773213624954224]]").CopyTo(model.Layers[6].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);
            Assert.AreEqual(1.0974769592285156, loss_before, 1e-6);
            Assert.AreEqual(0.5784467458724976, loss_after, 1e-6);

        }


        [Test, Explicit]
        public void TestL2Regularization_ConvolutionLayer_SGDVanilla_NCHW_2_1_4_4()
        {
            const int rows = 2;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 1, 4, 4);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 3);

            var sample = new NetworkSample
             {
                 LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                 ShuffleDatasetBeforeEachEpoch = false,
                 AutoSaveIntervalInMinutes = -1,
                 CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                 ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                 LogNetworkPropagation = true,
                 InitialLearningRate = 0.1,
                 num_epochs = 10,
                 BatchSize = X.Shape[0],
                 lambdaL2Regularization = 0.9,
                 ResourceIds = new List<int> { -1 }
             }
            .WithSGD(0.6, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestL2Regularization_ConvolutionLayer_SGDVanilla_NCHW_2_1_4_4), false);

            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, sample.lambdaL2Regularization, true)
                .Flatten()
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            TestNetworkPropagation.FromNumpyArray("[[[[-0.007486820220947266]]]]").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.20576128363609314, -0.18398475646972656, -0.09628859162330627, 0.06703934073448181, -0.004953294992446899, 0.19822236895561218, -0.02218601107597351, 0.0661531388759613, -0.07555326819419861, -0.04914134740829468, -0.2388371229171753, -0.16557052731513977, -0.10305577516555786, 0.009260892868041992, 0.09883379936218262, 0.15000569820404053], [-0.16948527097702026, -0.10886570811271667, 0.09080427885055542, 0.20759698748588562, -0.05145004391670227, 0.18707793951034546, -0.04029583930969238, 0.026453524827957153, 0.22636905312538147, -0.2319175899028778, -0.1573844850063324, -0.06329131126403809, -0.09744998812675476, 0.21600019931793213, -0.16204491257667542, -0.11508321762084961], [-0.1746601164340973, -0.23414024710655212, -0.14593511819839478, 0.21489951014518738, 0.1115545928478241, 0.12116813659667969, 0.013147890567779541, -0.12817087769508362, 0.04229617118835449, -0.23342368006706238, -0.1806415617465973, -0.128882497549057, 0.1577344834804535, 0.146580308675766, -0.1108737587928772, -0.009020596742630005]]").CopyTo(model.Layers[3].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);
            Assert.AreEqual(1.099147081375122, loss_before, 1e-6);
            Assert.AreEqual(0.9695441722869873, loss_after, 1e-6);

        }



        [Test, Explicit]
        public void Test_Huber()
        {
            const int rows = 4;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 3, 2, 2);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 1);

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.Huber,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                Huber_Delta = 0.5f,  
                LogNetworkPropagation = true,
                InitialLearningRate = 0.1,
                num_epochs = 10,
                BatchSize = 2,
                lambdaL2Regularization = 0.05,
                ResourceIds = new List<int> { -1 }
            }
            .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(Test_Huber), false);

            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Flatten()
                .Linear(3, true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_RELU)
                .Linear(1, true, sample.lambdaL2Regularization, true)
                ;

            TestNetworkPropagation.FromNumpyArray("[[-0.002161264419555664, 0.15485793352127075, -0.2375926673412323, -0.21244728565216064, -0.1111844927072525, 0.07741037011146545, -0.005719572305679321, 0.22888749837875366, -0.02561819553375244, 0.07638707756996155, -0.08724139630794525, -0.05674353241920471], [-0.275785356760025, -0.1911843717098236, -0.11899855732917786, 0.010693550109863281, 0.11412343382835388, 0.17321166396141052, -0.19570472836494446, -0.12570728361606598, 0.10485175251960754, 0.23971235752105713, -0.05940939486026764, 0.21601897478103638], [-0.046529620885849, 0.03054589033126831, 0.26138848066329956, -0.2677953541278839, -0.1817319393157959, -0.07308250665664673, -0.11252555251121521, 0.24941551685333252, -0.18711334466934204, -0.13288664817810059, -0.20168012380599976, -0.27036187052726746]]").CopyTo(model.Layers[2].Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.2798258066177368, 0.030363738536834717, -0.2959979474544525]]").CopyTo(model.Layers[4].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);
            Assert.AreEqual(0.32030075788497925, loss_before, 1e-6);
            Assert.AreEqual(0.0027381146792322397, loss_after, 1e-6);
            //test_save_sharpnet(model, X);
        }




        [Test, Explicit]
        public void Test_DepthwiseConvolution()
        {
            const int rows = 12;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 2, 2, 2);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 2);

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.01,
                num_epochs = 10,
                BatchSize = 4,
                lambdaL2Regularization = 0.01,
                ResourceIds = new List<int> { -1 }
            }
            .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(Test_DepthwiseConvolution), false);

            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .DepthwiseConvolution(3, 1, ConvolutionLayer.PADDING_TYPE.SAME, 1, sample.lambdaL2Regularization, true)
                .Flatten()
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            TestNetworkPropagation.FromNumpyArray("[[[[-0.0024956166744232178, 0.17881456017494202, -0.2743483781814575], [-0.24531301856040955, -0.1283847987651825, 0.08938577771186829], [-0.006604403257369995, 0.2642965018749237, -0.029581338167190552]]], [[[0.08820417523384094, -0.10073769092559814, -0.06552180647850037], [-0.3184494972229004, -0.22076070308685303, -0.13740770518779755], [0.01234784722328186, 0.13177838921546936, 0.2000075876712799]]]]\r\n").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.12841662764549255, 0.29358646273612976, -0.07276135683059692, 0.26456817984580994, -0.05698692798614502, 0.03741094470024109, 0.3201341927051544, -0.32798099517822266], [-0.22257526218891144, -0.08950743079185486, -0.13781508803367615, 0.30547037720680237, -0.22916612029075623, -0.16275224089622498, -0.24700669944286346, -0.33112430572509766]]").CopyTo(model.Layers[3].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);
            Assert.AreEqual(0.700163722038269, loss_before, 1e-6);
            Assert.AreEqual(0.6944987177848816, loss_after, 1e-6);
            //test_save_sharpnet(model, X);
        }



        //[Test, Explicit]
        //public void TestParallelRunWithPyTorch_Efficientnet_Inference()
        //{
        //    var xFileName = Path.Combine(NetworkSample.DefaultDataDirectory, "NonReg", "X_1_224_224_3.txt");
        //    var yExpectedFileName = Path.Combine(NetworkSample.DefaultDataDirectory, "NonReg", "YExpected_1_224_224_3.txt");
        //    if (!File.Exists(xFileName) || !File.Exists(yExpectedFileName))
        //    {
        //        Console.WriteLine("ignoring test " + nameof(TestParallelRunWithPyTorch_Efficientnet_Inference) + " because some files are missing");
        //        return;
        //    }

        //    var X = TestNetworkPropagation.FromNumpyArray(File.ReadAllText(xFileName));
        //    X = (CpuTensor<float>)X.ChangeAxis(new[] { 0, 3, 1, 2 });
        //    var Y = TestNetworkPropagation.FromNumpyArray(File.ReadAllText(yExpectedFileName));

        //    //we ensure that the network prediction is the same as in Keras
        //    EfficientNetNetworkSample sample = EfficientNetNetworkSample.CIFAR10();
        //    sample.SetResourceId(0);
        //    sample.ConvolutionAlgoPreference = ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM;
        //    var model = sample.EfficientNetB0(NetworkSample.DefaultWorkingDirectory, true, null, new[] { 3, 224, 224 });


        //    (float loss_before, float loss_after) = Train(model, X, Y);
        //    //Assert.AreEqual(0.3986191749572754, loss_before, 1e-6);
        //    //Assert.AreEqual(0.0003985593211837113, loss_after, 1e-6);

        //    /*

        //    var yPredicted = model.Predict(X, false);
        //    Assert.IsTrue(TensorExtensions.SameFloatContent(Y, yPredicted, 1e-5));

        //    //we save the network
        //    model.Save(model.WorkingDirectory, model.ModelName);
        //    model.Dispose();

        //    //we ensure that the saved version of the network behave the same as the original one
        //    var networkFromSavedFile = Network.LoadTrainedNetworkModel(model.WorkingDirectory, model.ModelName);
        //    var yPredictedFromSavedFile = networkFromSavedFile.Predict(X, false);
        //    Assert.IsTrue(TensorExtensions.SameFloatContent(Y, yPredictedFromSavedFile, 1e-5));

        //    var savedModelFile = Network.ToModelFilePath(model.WorkingDirectory, model.ModelName);
        //    File.Delete(savedModelFile);
        //    var saveParametersFile = Network.ToParameterFilePath(model.WorkingDirectory, model.ModelName);
        //    File.Delete(saveParametersFile);
        //    */
        //}

        [Test, Explicit]
        public void TestConcatenate_NCHW_9_1_1_1()
        {
            const int rows = 9;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 1, 1, 1);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 2);

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.05,
                num_epochs = 10,
                BatchSize = 3,
                lambdaL2Regularization = 0.07,
                ResourceIds = new List<int> { -1 }
            }
            .WithSGD(0.75, true);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestConcatenate_NCHW_9_1_1_1), false);

            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, sample.lambdaL2Regularization, true)
                .Convolution(1, 1, 1, ConvolutionLayer.PADDING_TYPE.VALID, sample.lambdaL2Regularization, true)
                .ConcatenateLayer(1, 2)
                .Flatten()
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            TestNetworkPropagation.FromNumpyArray("[[[[-0.007486820220947266]]]]").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[[-0.8230451345443726]]]]").CopyTo(model.Layers[2].Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.27234524488449097, 0.1896159052848816], [-0.014010012149810791, 0.5606575608253479]]").CopyTo(model.Layers[5].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);
            Assert.AreEqual(0.6930493116378784, loss_before, 1e-6);
            Assert.AreEqual(0.6860392093658447, loss_after, 1e-6);




        }


        [Test, Explicit]
        public void TestLeakyReluActivation_NCHW_10_1_4_4()
        {
            const int rows = 10;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 1, 4, 4);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 3);

            var sample = new NetworkSample
                {
                    LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                    ShuffleDatasetBeforeEachEpoch = false,
                    AutoSaveIntervalInMinutes = -1,
                    CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                    ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                    LogNetworkPropagation = true,
                    InitialLearningRate = 0.1,
                    num_epochs = 10,
                    BatchSize = 2,
                    lambdaL2Regularization = 0.17,
                    ResourceIds = new List<int> { -1 }
                }
                .WithSGD(0.66, true);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestLeakyReluActivation_NCHW_10_1_4_4), false);

            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Flatten()
                .Linear(3, true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU, Tensor.SingleFloat(0.1f))
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            TestNetworkPropagation.FromNumpyArray("[[-0.0018717050552368164, 0.13411089777946472, -0.20576128363609314, -0.18398475646972656, -0.09628859162330627, 0.06703934073448181, -0.004953294992446899, 0.19822236895561218, -0.02218601107597351, 0.0661531388759613, -0.07555326819419861, -0.04914134740829468, -0.2388371229171753, -0.16557052731513977, -0.10305577516555786, 0.009260892868041992], [0.09883379936218262, 0.15000569820404053, -0.16948527097702026, -0.10886570811271667, 0.09080427885055542, 0.20759698748588562, -0.05145004391670227, 0.18707793951034546, -0.04029583930969238, 0.026453524827957153, 0.22636905312538147, -0.2319175899028778, -0.1573844850063324, -0.06329131126403809, -0.09744998812675476, 0.21600019931793213], [-0.16204491257667542, -0.11508321762084961, -0.1746601164340973, -0.23414024710655212, -0.14593511819839478, 0.21489951014518738, 0.1115545928478241, 0.12116813659667969, 0.013147890567779541, -0.12817087769508362, 0.04229617118835449, -0.23342368006706238, -0.1806415617465973, -0.128882497549057, 0.1577344834804535, 0.146580308675766]]").CopyTo(model.Layers[2].Weights);
            TestNetworkPropagation.FromNumpyArray("[[0.5739630460739136, 0.229140043258667, 0.0779958963394165], [0.38710546493530273, -0.3399451971054077, 0.10758578777313232], [-0.4476228356361389, -0.4001534581184387, -0.29824966192245483]]").CopyTo(model.Layers[4].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);
            Assert.AreEqual(1.1183210611343384, loss_before, 1e-6);
            Assert.AreEqual(1.0901305675506592, loss_after, 1e-6);
        }

        [Test, Explicit]
        public void TestMultiply_NCHW_2_3_4_5_different_dimension()
        {
            const int rows = 2;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 3, 4, 5);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 2);

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.01,
                num_epochs = 10,
                BatchSize = 2,
                lambdaL2Regularization = 0.07,
                ResourceIds = new List<int> { -1 }
            }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(TestMultiply_NCHW_2_3_4_5_different_dimension), false);

            model
                .Input(X.Shape[1], X.Shape[2], X.Shape[3])
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true)
                .Convolution(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true)
                .GlobalMaxPooling()
                .MultiplyLayer(1, 3)
                .Flatten()
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            TestNetworkPropagation.FromNumpyArray("[[[[-0.004322528839111328]], [[0.3097158670425415]], [[-0.4751853346824646]]], [[[-0.4248945713043213]], [[-0.222368985414505]], [[0.1548207402229309]]]]").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[[-0.06275153160095215]], [[0.1871093511581421]]], [[[-0.2136968970298767]], [[-0.13899272680282593]]]]").CopyTo(model.Layers[2].Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.06517819315195084, 0.005857110023498535, 0.0625079870223999, 0.09487192332744598, -0.10719189792871475, -0.06885271519422531, 0.05742967128753662, 0.13129587471485138, -0.03253985941410065, 0.11831848323345184, -0.025485321879386902, 0.01673068106174469, 0.1431683450937271, -0.14667756855487823, -0.09953868389129639, -0.04002893716096878, -0.061632782220840454, 0.13661052286624908, -0.10248620063066483, -0.07278501987457275, -0.11046475172042847, -0.14808329939842224, -0.09229747205972672, 0.1359143704175949, 0.07055331766605377, 0.07663345336914062, 0.00831545889377594, -0.08106237649917603, 0.02675044536590576, -0.14763009548187256, -0.11424775421619415, -0.081512451171875, 0.099760040640831, 0.09270553290843964, -0.07012271881103516, -0.005705133080482483, 0.10112343728542328, 0.15718625485897064, 0.06275257468223572, 0.02136005461215973], [0.10601319372653961, -0.09309782087802887, 0.029463574290275574, -0.12258656322956085, -0.10958653688430786, -0.08167903125286102, 0.07154226303100586, 0.06358714401721954, -0.09365915507078171, 0.04776732623577118, 0.08680009841918945, -0.019956722855567932, 0.006037026643753052, 0.03663572669029236, 0.09809015691280365, 0.15182001888751984, -0.1218462884426117, -0.057943955063819885, 0.062140315771102905, 0.1310051530599594, 0.13759185373783112, 0.13951285183429718, 0.03146696090698242, -0.1374930888414383, 0.014545291662216187, -0.09891688823699951, -0.1473548859357834, 0.14048300683498383, 0.12022344768047333, -0.15772302448749542, 0.02959449589252472, -0.026635870337486267, -0.02601940929889679, -0.07237771898508072, 0.06080366671085358, -0.09365140646696091, 0.05796317756175995, 0.07995946705341339, 0.11318923532962799, 0.0591205358505249]]").CopyTo(model.Layers[6].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);
            Assert.AreEqual(0.6887038946151733, loss_before, 1e-6);
            Assert.AreEqual(0.6871695518493652, loss_after, 1e-6);
            //test_save_sharpnet(model, X);
        }



        [Test, Explicit]
        public void Test_SimpleRNN()
        {
            const int rows = 4;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 2, 1);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 2);
            
            int timeSteps = X.Shape[1];     //number of words in each sentence
            int inputSize = X.Shape[2];     //number of distinct words in the dictionary 
            const int hiddenSize = 2;

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.Mse,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = false,
                InitialLearningRate = 0.01,
                num_epochs = 10,
                BatchSize = 2,
                lambdaL2Regularization = 0.00,
                ResourceIds = new List<int> { 0 }
            }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(Test_SimpleRNN), false);

            model
                .Input(timeSteps, inputSize, -1)
                .SimpleRNN(hiddenSize, true, true, 2)
                .SimpleRNN(hiddenSize, false, false, 1)
                ;

            TestNetworkPropagation.FromNumpyArray("[[-0.0052939653396606445], [0.37932294607162476]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_ih[0]);
            TestNetworkPropagation.FromNumpyArray("[[-0.2136968970298767], [-0.13899272680282593]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_ih[1]);
            TestNetworkPropagation.FromNumpyArray("[[0.2568332552909851, 0.5871729254722595, -0.14552271366119385, 0.5291363596916199], [-0.11397385597229004, 0.07482188940048218, 0.6402683854103088, -0.6559619903564453]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_ih[2]);
            TestNetworkPropagation.FromNumpyArray("[[-0.41276684403419495, 0.6078276038169861, 0.31552404165267944, 0.34271520376205444], [0.03718787431716919, -0.3625219762325287, 0.11963164806365967, -0.6602218747138977]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_ih[3]);
            TestNetworkPropagation.FromNumpyArray("[[-0.5819807648658752, -0.5203874707221985], [-0.27234524488449097, 0.1896159052848816]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_hh[0]);
            TestNetworkPropagation.FromNumpyArray("[[-0.6755334138870239, -0.4683041572570801], [-0.2914857566356659, 0.026193737983703613]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_hh[1]);
            TestNetworkPropagation.FromNumpyArray("[[-0.4451505243778229, -0.17901486158370972], [-0.2756301760673523, 0.6109407544136047]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_hh[2]);
            TestNetworkPropagation.FromNumpyArray("[[-0.5109314918518066, -0.36453473567962646], [0.44614046812057495, 0.4145917296409607]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_hh[3]);

            TestNetworkPropagation.FromNumpyArray("[[0.28063809871673584, 0.09552508592605591, 0.4741054177284241, -0.41634613275527954], [0.13176512718200684, -0.5482237935066223, -0.4900858998298645, -0.3652797341346741]]").CopyTo(((RecurrentLayer)model.Layers[2])._weight_ih[0]);
            TestNetworkPropagation.FromNumpyArray("[[0.3199467062950134, 0.2843703627586365], [-0.4188564717769623, 0.21362197399139404]]").CopyTo(((RecurrentLayer)model.Layers[2])._weight_hh[0]);

            (float loss_before, float loss_after) = Train(model, X, Y);
            
            //pred before: [[0.02031463198363781, -0.005873197223991156], [0.030436206609010696, -0.001550998305901885], [0.04031758010387421, 0.003334005828946829], [0.04994407296180725, 0.008880063891410828]]
            Assert.AreEqual(0.48370790481567383, loss_before, 1e-6);
            //pred after:  [[0.6409029960632324, 0.6298636198043823], [0.6515954732894897, 0.6425012946128845], [0.6600877046585083, 0.654278039932251], [0.6665916442871094, 0.664940595626831]]
            Assert.AreEqual(0.2722862660884857, loss_after, 1e-6);
            //test_save_sharpnet(model, X);
        }


        private static void test_save_sharpnet(Network model, Tensor X)
        {
            model.EpochData.Clear(); model.Save("C:/Projects/SharpNet/Tests/NonReg", model.ModelName);
            var model_pytorch = Network.LoadTrainedNetworkModel("C:/Projects/SharpNet/Tests/NonReg", "model_pytorch");
            model_pytorch.Save("C:/Projects/SharpNet/Tests/NonReg", model_pytorch.ModelName + "_loaded");
            Log.Info($"PyTorch Y_pref_after: {model_pytorch.Predict(X, false).ToNumpy()}");
            var model_sharpnet = Network.LoadTrainedNetworkModel("C:/Projects/SharpNet/Tests/NonReg", model.ModelName);
            model_sharpnet.Save("C:/Projects/SharpNet/Tests/NonReg", model.ModelName + "_loaded");
            Log.Info($"SharpNet (reloaded) Y_pref_after: {model_sharpnet.Predict(X, false).ToNumpy()}");
        }



        [Test, Explicit]
        public void Test_EfficientNet_TOREMOVE()
        {
            var model_pytorch = Network.LoadTrainedNetworkModel("C:/Projects/SharpNet/Tests/NonReg", "model_pytorch");

            Log.Info(model_pytorch.ToString());
            var batch_size = 3;
            var X = TestNetworkPropagation.numpy_array_for_tests(batch_size, 3, 224, 224);

            //test_save_sharpnet(model_pytorch, X);
        }


        [TestCase(false, 1.0957681735356648, 1.0050253868103027)]
        [TestCase(true, 1.076534628868103, 0.9956841468811035)]
        public void Test_DotProductAttention(bool is_causal, double expected_loss_before, double expected_loss_after)
        {
            const int rows = 6;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 4, 5);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 3);
            const bool use_scale = false;
            //const bool is_causal = true;

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = false,
                InitialLearningRate = 0.1,
                num_epochs = 10,
                BatchSize = 2,
                lambdaL2Regularization = 0.07,
                ResourceIds = new List<int> { 0 }
            }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(Test_DotProductAttention), false);

            model
                .Input(X.Shape[1], X.Shape[2], -1)
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true, 0, "conv1D_Q")
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true, 0, "conv1D_K")
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true, 0, "conv1D_V")
                .ScaledDotProductAttention(use_scale, is_causal, 1, 2, 3)
                .Flatten()
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);


            TestNetworkPropagation.FromNumpyArray("[[[-0.003743410110473633], [0.26822179555892944], [-0.4115225672721863], [-0.3679695129394531]], [[-0.19257718324661255], [0.13407868146896362], [-0.009906589984893799], [0.39644473791122437]]]").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[-0.15110653638839722], [-0.09828269481658936], [-0.4776742458343506], [-0.33114105463027954]], [[-0.20611155033111572], [0.018521785736083984], [0.19766759872436523], [0.30001139640808105]]]").CopyTo(model.Layers[2].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[0.18160855770111084], [0.41519397497177124], [-0.10290008783340454], [0.3741558790206909]], [[-0.08059167861938477], [0.05290704965591431], [0.45273810625076294], [-0.4638351798057556]]]").CopyTo(model.Layers[3].Weights);
            TestNetworkPropagation.FromNumpyArray("[[-0.12326556444168091, 0.27322104573249817, -0.20497240126132965, -0.1455700397491455, -0.22092950344085693, -0.2961665987968445, -0.18459494411945343, 0.2718287408351898, 0.14110663533210754, 0.15326690673828125], [0.01663091778755188, -0.16212475299835205, 0.05350089073181152, -0.2952601909637451, -0.2284955084323883, -0.16302490234375, 0.199520081281662, 0.18541106581687927, -0.1402454376220703, -0.011410266160964966], [0.20224687457084656, 0.3143725097179413, 0.12550514936447144, 0.04272010922431946, 0.21202638745307922, -0.18619564175605774, 0.05892714858055115, -0.2451731264591217, -0.21917307376861572, -0.16335806250572205]]").CopyTo(model.Layers[6].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);

            Assert.AreEqual(expected_loss_before, loss_before, 1e-6);
            Assert.AreEqual(expected_loss_after, loss_after, 1e-6);
            //test_save_sharpnet(model, X);
        }


        [TestCase(false, 1.102099061012268, 1.0814517736434937)]
        [TestCase(true, 1.103563904762268, 1.0815070867538452)]
        public void Test_MultiHeadAttention_no_bias(bool is_causal, double expected_loss_before, double expected_loss_after)
        {
            const int rows = 10;
            const int num_heads = 3;
            const int key_dim = 2;
            const int value_dim = key_dim;
            const int embedding_dim = num_heads* key_dim;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 2, embedding_dim);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 3);
            const bool use_bias_Q_K_V = false;
            const bool use_bias_O = use_bias_Q_K_V;


            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.1,
                num_epochs = 10,
                BatchSize = 2,
                lambdaL2Regularization = 0.00,
                ResourceIds = new List<int> { 0 }
            }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(Test_MultiHeadAttention_no_bias), false);

            model
                .Input(X.Shape[1], X.Shape[2], -1)
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true, 0, "conv1D_Q")
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true, 0, "conv1D_K")
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true, 0, "conv1D_V")
                .MultiHeadAttention(num_heads, key_dim, value_dim, use_bias_Q_K_V, use_bias_O, is_causal, 1, 2, 3)
                .Flatten()
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);

            
            TestNetworkPropagation.FromNumpyArray("[[[-0.0052939653396606445], [0.37932294607162476]], [[-0.5819807648658752], [-0.5203874707221985]]]").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[-0.014010012149810791], [0.5606575608253479]], [[-0.06275153160095215], [0.1871093511581421]]]").CopyTo(model.Layers[2].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[-0.6755334138870239], [-0.4683041572570801]], [[-0.2914857566356659], [0.026193737983703613]]]").CopyTo(model.Layers[3].Weights);

            var mha = (MultiheadAttention)model.Layers[4];
            TestNetworkPropagation.FromNumpyArray("[[0.3352431654930115, -0.2944011688232422, 0.09317201375961304, -0.38765275478363037, -0.3465430736541748, -0.2582917809486389], [0.2262365221977234, 0.20108020305633545, -0.2961762547492981, 0.1510535478591919, 0.2744860053062439, -0.06310868263244629], [0.019090771675109863, 0.11585235595703125, 0.31018829345703125, 0.48009705543518066, -0.3853117823600769, -0.18323487043380737], [0.19650495052337646, 0.4142746925354004, 0.4351036548614502, 0.4411783814430237, 0.09950727224349976, -0.43479132652282715], [0.045996248722076416, -0.31280267238616943, -0.4659770727157593, 0.44424623250961304, 0.38017988204956055, -0.4987639784812927], [0.0935860276222229, -0.08423000574111938, -0.08228057622909546, -0.2288784384727478, 0.19227808713912964, -0.2961517572402954], [0.18329566717147827, 0.25285404920578003, 0.3579357862472534, 0.18695557117462158, -0.49486762285232544, -0.32434844970703125], [0.2496575117111206, 0.10465067625045776, -0.3900420069694519, -0.28790974617004395, 0.4703746438026428, 0.3369089365005493], [-0.21801257133483887, -0.1258423924446106, -0.4762990474700928, -0.008987069129943848, -0.3765294551849365, -0.38567835092544556], [-0.027549803256988525, 0.0750725269317627, -0.20476514101028442, 0.2966887950897217, -0.30426955223083496, 0.45368504524230957], [0.3426499366760254, -0.42164146900177, -0.12444216012954712, 0.022561311721801758, 0.07295054197311401, 0.11858713626861572], [0.19621413946151733, 0.029950082302093506, -0.24396437406539917, 0.23659449815750122, -0.4796244502067566, -0.2963533401489258], [-0.1251649260520935, -0.2435566782951355, -0.17491668462753296, -0.4098108410835266, -0.10635757446289062, 0.10687822103500366], [-0.32573288679122925, -0.025659680366516113, 0.3579254150390625, -0.05140012502670288, 0.01389610767364502, -0.043134450912475586], [0.10119068622589111, 0.3179197311401367, 0.4736230969429016, 0.3175279498100281, 0.474706768989563, -0.036160826683044434], [-0.44916075468063354, -0.23703861236572266, 0.3404526114463806, -0.003241240978240967, -0.24852317571640015, -0.383155882358551], [-0.467926025390625, -0.4220041036605835, -0.1014183759689331, 0.27420300245285034, 0.2703205347061157, -0.482215940952301], [0.31189101934432983, -0.391254723072052, -0.1057051420211792, -0.20273631811141968, -0.09630763530731201, -0.09817135334014893]]").CopyTo(mha.in_proj_weight);
            TestNetworkPropagation.FromNumpyArray("[[-0.276768296957016, -0.17777696251869202, 0.14828276634216309, 0.3390044569969177, -0.08401757478713989, 0.3054969906806946], [-0.06580284237861633, 0.04319843649864197, 0.3696591258049011, -0.37871986627578735, -0.25700777769088745, -0.10335427522659302], [-0.15913516283035278, 0.35272687673568726, -0.26461824774742126, -0.18793010711669922, -0.2852187752723694, -0.38234943151474], [-0.2383110523223877, 0.3509294390678406, 0.1821678876876831, 0.19786673784255981, 0.02147042751312256, -0.2093021720647812], [0.06906935572624207, -0.3811792731285095, -0.29498645663261414, -0.2104642391204834, 0.25757932662963867, 0.2393646240234375], [-0.18105609714984894, -0.014730572700500488, 0.26109957695007324, 0.40585315227508545, 0.16202646493911743, 0.05515143275260925]]").CopyTo(mha.out_proj_weight);

            TestNetworkPropagation.FromNumpyArray("[[-0.2590426206588745, -0.249253049492836, -0.04517173767089844, 0.0037331879138946533, -0.13113786280155182, 0.10874369740486145, -0.2598244845867157, -0.019481897354125977, 0.2538664937019348, -0.11774827539920807, 0.26067453622817993, 0.10454478859901428], [-0.26051801443099976, 0.18264397978782654, -0.03331151604652405, -0.1288665533065796, 0.2308400273323059, -0.23327699303627014, 0.03097626566886902, -0.060439541935920715, 0.20614656805992126, 0.08058208227157593, 0.1387099325656891, 0.10194820165634155], [-0.06941907107830048, -0.06070995330810547, -0.2378918081521988, 0.15641692280769348, 0.22920173406600952, 0.1975187063217163, -0.2036251723766327, 0.012874871492385864, -0.20349694788455963, -0.1589110940694809, -0.16821259260177612, 0.09865328669548035]]").CopyTo(model.Layers[6].Weights);
            
            (float loss_before, float loss_after) = Train(model, X, Y);

            Assert.AreEqual(expected_loss_before, loss_before, 1e-6);
            Assert.AreEqual(expected_loss_after, loss_after, 1e-6);
            //test_save_sharpnet(model, X);
        }


        [TestCase(false, 1.1075962781906128, 1.082484245300293)]
        [TestCase(true, 1.1090373992919922, 1.0816659927368164)]
        public void Test_MultiHeadAttention_with_bias(bool is_causal, double expected_loss_before, double expected_loss_after)
        {
            const int rows = 10;
            const int num_heads = 3;
            const int key_dim = 2;
            const int value_dim = key_dim;
            const int embedding_dim = num_heads * key_dim;
            var X = TestNetworkPropagation.numpy_array_for_tests(rows, 2, embedding_dim);
            var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 3);
            const bool use_bias_Q_K_V = true;
            const bool use_bias_O = use_bias_Q_K_V;

            var sample = new NetworkSample
            {
                LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
                ShuffleDatasetBeforeEachEpoch = false,
                AutoSaveIntervalInMinutes = -1,
                CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
                ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
                LogNetworkPropagation = true,
                InitialLearningRate = 0.1,
                num_epochs = 10,
                BatchSize = 2,
                lambdaL2Regularization = 0.00,
                ResourceIds = new List<int> { 0 }
            }
                .WithSGD(0.9, false);

            var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(Test_MultiHeadAttention_with_bias), false);

            model
                .Input(X.Shape[1], X.Shape[2], -1)
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true, 0, "conv1D_Q")
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true, 0, "conv1D_K")
                .Conv1D(2, 1, 1, ConvolutionLayer.PADDING_TYPE.SAME, sample.lambdaL2Regularization, true, 0, "conv1D_V")
                .MultiHeadAttention(num_heads, key_dim, value_dim, use_bias_Q_K_V, use_bias_O, is_causal, 1, 2, 3)
                .Flatten()
                .Linear(Y.Shape[1], true, sample.lambdaL2Regularization, true)
                .Activation(cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX);


            TestNetworkPropagation.FromNumpyArray("[[[-0.0052939653396606445], [0.37932294607162476]], [[-0.5819807648658752], [-0.5203874707221985]]]").CopyTo(model.Layers[1].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[-0.014010012149810791], [0.5606575608253479]], [[-0.06275153160095215], [0.1871093511581421]]]").CopyTo(model.Layers[2].Weights);
            TestNetworkPropagation.FromNumpyArray("[[[-0.6755334138870239], [-0.4683041572570801]], [[-0.2914857566356659], [0.026193737983703613]]]").CopyTo(model.Layers[3].Weights);

            var mha = (MultiheadAttention)model.Layers[4];
            TestNetworkPropagation.FromNumpyArray("[[0.2262365221977234, 0.20108020305633545, -0.2961762547492981, 0.1510535478591919, 0.2744860053062439, -0.06310868263244629], [0.019090771675109863, 0.11585235595703125, 0.31018829345703125, 0.48009705543518066, -0.3853117823600769, -0.18323487043380737], [0.19650495052337646, 0.4142746925354004, 0.4351036548614502, 0.4411783814430237, 0.09950727224349976, -0.43479132652282715], [0.045996248722076416, -0.31280267238616943, -0.4659770727157593, 0.44424623250961304, 0.38017988204956055, -0.4987639784812927], [0.0935860276222229, -0.08423000574111938, -0.08228057622909546, -0.2288784384727478, 0.19227808713912964, -0.2961517572402954], [0.18329566717147827, 0.25285404920578003, 0.3579357862472534, 0.18695557117462158, -0.49486762285232544, -0.32434844970703125], [0.2496575117111206, 0.10465067625045776, -0.3900420069694519, -0.28790974617004395, 0.4703746438026428, 0.3369089365005493], [-0.21801257133483887, -0.1258423924446106, -0.4762990474700928, -0.008987069129943848, -0.3765294551849365, -0.38567835092544556], [-0.027549803256988525, 0.0750725269317627, -0.20476514101028442, 0.2966887950897217, -0.30426955223083496, 0.45368504524230957], [0.3426499366760254, -0.42164146900177, -0.12444216012954712, 0.022561311721801758, 0.07295054197311401, 0.11858713626861572], [0.19621413946151733, 0.029950082302093506, -0.24396437406539917, 0.23659449815750122, -0.4796244502067566, -0.2963533401489258], [-0.1251649260520935, -0.2435566782951355, -0.17491668462753296, -0.4098108410835266, -0.10635757446289062, 0.10687822103500366], [-0.32573288679122925, -0.025659680366516113, 0.3579254150390625, -0.05140012502670288, 0.01389610767364502, -0.043134450912475586], [0.10119068622589111, 0.3179197311401367, 0.4736230969429016, 0.3175279498100281, 0.474706768989563, -0.036160826683044434], [-0.44916075468063354, -0.23703861236572266, 0.3404526114463806, -0.003241240978240967, -0.24852317571640015, -0.383155882358551], [-0.467926025390625, -0.4220041036605835, -0.1014183759689331, 0.27420300245285034, 0.2703205347061157, -0.482215940952301], [0.31189101934432983, -0.391254723072052, -0.1057051420211792, -0.20273631811141968, -0.09630763530731201, -0.09817135334014893], [-0.4486749768257141, -0.43171894550323486, -0.07823973894119263, 0.006466090679168701, -0.22713744640350342, 0.18834960460662842]]").CopyTo(mha.in_proj_weight);
            TestNetworkPropagation.FromNumpyArray("[[-0.276768296957016, -0.17777696251869202, 0.14828276634216309, 0.3390044569969177, -0.08401757478713989, 0.3054969906806946], [-0.06580284237861633, 0.04319843649864197, 0.3696591258049011, -0.37871986627578735, -0.25700777769088745, -0.10335427522659302], [-0.15913516283035278, 0.35272687673568726, -0.26461824774742126, -0.18793010711669922, -0.2852187752723694, -0.38234943151474], [-0.2383110523223877, 0.3509294390678406, 0.1821678876876831, 0.19786673784255981, 0.02147042751312256, -0.2093021720647812], [0.06906935572624207, -0.3811792731285095, -0.29498645663261414, -0.2104642391204834, 0.25757932662963867, 0.2393646240234375], [-0.18105609714984894, -0.014730572700500488, 0.26109957695007324, 0.40585315227508545, 0.16202646493911743, 0.05515143275260925]]").CopyTo(mha.out_proj_weight);

            TestNetworkPropagation.FromNumpyArray("[[-0.2598244845867157, -0.019481897354125977, 0.2538664937019348, -0.11774827539920807, 0.26067453622817993, 0.10454478859901428, -0.26051801443099976, 0.18264397978782654, -0.03331151604652405, -0.1288665533065796, 0.2308400273323059, -0.23327699303627014], [0.03097626566886902, -0.060439541935920715, 0.20614656805992126, 0.08058208227157593, 0.1387099325656891, 0.10194820165634155, -0.06941907107830048, -0.06070995330810547, -0.2378918081521988, 0.15641692280769348, 0.22920173406600952, 0.1975187063217163], [-0.2036251723766327, 0.012874871492385864, -0.20349694788455963, -0.1589110940694809, -0.16821259260177612, 0.09865328669548035, -0.17202532291412354, -0.006298094987869263, 0.012144029140472412, 0.18608665466308594, -0.21821531653404236, -0.19817900657653809]]").CopyTo(model.Layers[6].Weights);

            (float loss_before, float loss_after) = Train(model, X, Y);

            Assert.AreEqual(expected_loss_before, loss_before, 1e-6);
            Assert.AreEqual(expected_loss_after, loss_after, 1e-6);
            //test_save_sharpnet(model, X);
        }

        [Test, Explicit]
        public void Test_EfficientNetB0()
        {
            var networkBuilder = EfficientNetNetworkSample.CIFAR10();
            networkBuilder.SetResourceId(0);

            var model = networkBuilder.EfficientNetB0(
                networkBuilder.BuildNetworkWithoutLayers(GetDefaultWorkingDirectory(), "efficientnet_b0_small"),
                MobileBlocksDescription.Default(),
                true,
                null,
                new[] { 3, 224, 224 },
                1000);



            //var model = networkBuilder.EfficientNetB0(GetDefaultWorkingDirectory(), true, null, new[] { 3, 224, 224 });
            int batchSize = 1;
            Log.Info(model.Summary() + Environment.NewLine);
            Log.Info(model.ToPytorchModule(batchSize) + Environment.NewLine);

        }

        //[Test, Explicit]
        //public void Test_LSTM()
        //{
        //    const int rows = 4;
        //    var X = TestNetworkPropagation.numpy_array_for_tests(rows, 2, 1);
        //    var Y = TestNetworkPropagation.y_numpy_array_for_tests(rows, 1);

        //    int timeSteps = X.Shape[1];     //number of words in each sentence
        //    int inputSize = X.Shape[2];     //number of distinct words in the dictionary 
        //    const int hiddenSize = 1;
        //    const double dropout = 0.0;

        //    var sample = new NetworkSample
        //    {
        //        LossFunction = EvaluationMetricEnum.Mse,
        //        ShuffleDatasetBeforeEachEpoch = false,
        //        AutoSaveIntervalInMinutes = -1,
        //        CompatibilityMode = NetworkSample.CompatibilityModeEnum.PyTorch,
        //        ConvolutionAlgoPreference = ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM,
        //        LogNetworkPropagation = false,
        //        InitialLearningRate = 0.01,
        //        num_epochs = 1,
        //        BatchSize = 2,
        //        lambdaL2Regularization = 0.00,
        //        ResourceIds = new List<int> { 0 }
        //    }
        //        .WithSGD(0.9, false);

        //    var model = new Network(sample, null, GetDefaultWorkingDirectory(), nameof(Test_LSTM), false);

        //    model
        //        .Input(timeSteps, inputSize, -1)
        //        .LSTM(hiddenSize, true, true, 2, dropout, false)
        //        .LSTM(hiddenSize, false, false, 1, 0.0, false)
        //        ;

        //    TestNetworkPropagation.FromNumpyArray("[[-0.007486820220947266], [0.5364435911178589], [-0.8230451345443726], [-0.7359390258789062]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_ih[0]);
        //    TestNetworkPropagation.FromNumpyArray("[[0.39533519744873047], [0.6000227928161621], [-0.677941083908081], [-0.4354628324508667]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_ih[1]);
        //    TestNetworkPropagation.FromNumpyArray("[[-0.6481796503067017, -0.46033287048339844], [-0.6986404657363892, -0.9365609884262085], [-0.5837404727935791, 0.8595980405807495], [0.4462183713912964, 0.48467254638671875]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_ih[2]);
        //    TestNetworkPropagation.FromNumpyArray("[[0.39688217639923096, 0.1350928544998169], [0.670486330986023, -0.5888023376464844], [0.18634402751922607, -0.7753055095672607], [-0.6930861473083496, -0.5165835618972778]]").CopyTo(((RecurrentLayer)model.Layers[1])._weight_ih[3]);
        //    TestNetworkPropagation.FromNumpyArray("[[-0.3851543664932251], [0.26815736293792725], [-0.019813179969787598], [0.7928894758224487]]").CopyTo(((RecurrentLayer)model.Layers[1])._weights_hh[0]);
        //    TestNetworkPropagation.FromNumpyArray("[[0.3632171154022217], [0.8303879499435425], [-0.20580017566680908], [0.7483117580413818]]").CopyTo(((RecurrentLayer)model.Layers[1])._weights_hh[1]);
        //    TestNetworkPropagation.FromNumpyArray("[[0.052591562271118164], [-0.5126835107803345], [0.16918468475341797], [-0.9336947202682495]]").CopyTo(((RecurrentLayer)model.Layers[1])._weights_hh[2]);
        //    TestNetworkPropagation.FromNumpyArray("[[0.4524730443954468], [0.4021604061126709], [-0.5923525094985962], [0.3021070957183838]]").CopyTo(((RecurrentLayer)model.Layers[1])._weights_hh[3]);

        //    TestNetworkPropagation.FromNumpyArray("[[0.39300990104675293, 0.8285493850708008], [0.8702073097229004, 0.8823567628860474], [0.1990145444869995, -0.8695826530456543], [0.09199249744415283, -0.6256053447723389]]").CopyTo(((RecurrentLayer)model.Layers[2])._weight_ih[0]);
        //    TestNetworkPropagation.FromNumpyArray("[[-0.9319541454315186], [0.8884924650192261], [0.7603597640991211], [-0.9975279569625854]]").CopyTo(((RecurrentLayer)model.Layers[2])._weights_hh[0]);


        //    (float loss_before, float loss_after) = Train(model, X, Y);

        //    //Assert.AreEqual(0.48370790481567383, loss_before, 1e-6);
        //    //Assert.AreEqual(0.2722862660884857, loss_after, 1e-6);
        //}

    }
}

