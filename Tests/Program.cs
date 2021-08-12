using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Networks;

// ReSharper disable UnusedMember.Local

namespace SharpNetTests
{
    static class Program
    {

        private static readonly log4net.ILog Log = log4net.LogManager.GetLogger(typeof(Program));


        // ReSharper disable once UnusedMember.Global
        public static bool Accept(CancelDatabaseEntry entry, string mandatoryPrefix)
        {
            if (entry.RemovedDate.HasValue)
            {
                return false;
            }

            if (string.IsNullOrEmpty(entry.Cancel))
            {
                return false;
            }
            if (!string.IsNullOrEmpty(mandatoryPrefix) && !entry.Cancel.StartsWith(mandatoryPrefix))
            {
                return false;
            }
            return true;
        }

        //private static string ImageDatabaseManagementPath => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), @"AppData\Roaming\ImageDatabaseManagement");

        private static void Main()
        {
            SharpNet.Utils.ConfigureGlobalLog4netProperties(NetworkConfig.DefaultLogDirectory, "SharpNet");
            SharpNet.Utils.ConfigureThreadLog4netProperties(NetworkConfig.DefaultLogDirectory, "SharpNet");
            //235*200 : 34.28% errors (30 epochs)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_200_235_20200611_0716_30.txt");
            //235*200 :  11.46% errors (70 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_200_235_20200615_0848_70.txt");
            //235*200 :  1.58% errors (150 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_200_235_20200615_0848_150.txt");
            //235*200 :  0.67% errors (294 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_200_235_20200615_0848_294.txt");
            //235*200 :  0.65% errors (310 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_200_235_20200615_0848_310.txt");
            //470*400 :  1.11% errors (150 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_400_470_20200620_1427_310.txt");
            //470*400 :  0.48% errors (310 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_400_470_20200620_1427_310.txt");
            //470*400 :  0.18% errors (630 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Cancel_400_470_20200715_2244_630.txt");
            //new CancelDatabase().CreatePredictionFile(Path.Combine(ImageDatabaseManagementPath, "Prediction_def.csv"));return;


            //CFM60DataSet.EnsembleLearning(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions", new Dictionary<string, double>
            //{
            //    ["CFM60__20210605_1901.csv"] = 1,
            //    ["CFM60_DenseUnits_50_20210606_0825.csv"] = 1,
            //    ["CFM60_min_lr_10_20210606_1919.csv"] = 1
            //}); return;


            //CFM60DataSet.EnsembleLearning(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20200613", new Dictionary<string, double>
            //{
            //    ["CFM60_l2_0_00001_batch4096_V1_20210613_1158.csv"] = 1,
            //    ["CFM60_l2_0_00001_V1_20210613_1029.csv"] = 1,
            //    ["CFM60_l2_0_00005_batch4096_V1_20210613_1158.csv"] = 1,
            //    ["CFM60_l2_0_00005_V1_20210613_1029.csv"] = 1,
            //    ["CFM60_l2_0_0001_batch4096_V1_20210613_1158.csv"] = 1,
            //    ["CFM60_l2_0_0001_V2_20210613_1049.csv"] = 1,
            //    ["CFM60_l2_0_0005_batch4096_V1_20210613_1158.csv"] = 1,
            //    ["CFM60_l2_0_0005_V1_20210613_1029.csv"] = 1
            //}, 1.0, Math.Log(1.00)); return;

            //CFM60DataSet.EnsembleLearning(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20210712", new Dictionary<string, double>
            //{
            //    ["CFM60True_True_False_20210712_0738_0_3819.csv"] = 1,
            //    ["CFM60_20210708_0808_0_3702.csv"] = 1,
            //    ["CFM60_ADD_NOISE_0_03_Continuous_noise_0_1_20210711_1636_0_3818.csv"] = 1,
            //    ["CFM60_lr_0_001_divide_200_20210708_0808_0_3664.csv"] = 1,
            //    ["CFM60_lr_0_005_1cycle_divide500_20210709_1551_0_3797.csv"] = 1,
            //    ["CFM60_norm_deduce_mean_20210711_1823_0_3756.csv"] = 1,
            //    ["CFM60_Pid_EmbeddingDim_16_20210705_1245_0_3665.csv"] = 1,
            //}, 1.0, Math.Log(1.00)); return;


            //CFM60DataSet.EnsembleLearning(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20210716", null,  1.0, Math.Log(1.00));return;
            //CFM60DataSet.EnsembleLearning(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20210717b", null,  1.0, Math.Log(1.00));return;
            //CFM60DataSet.EnsembleLearning(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20210811", null,  1.0, Math.Log(1.06), Math.Log(1.06));return;  // -0.0020
            //CFM60DataSet.EnsembleLearning(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20210811", null,  1.0, Math.Log(1.03), Math.Log(1.09));return;  // -0.0028
            //CFM60DataSet.EnsembleLearning(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20210811", null,  1.0, Math.Log(1.00), Math.Log(1.12));return;  // -0.0031
            //CFM60DataSet.EnsembleLearning(@"C:\Users\fzibi\AppData\Local\SharpNet\CFM60\test_predictions\20210717", null,  1.0, Math.Log(1.06));return;


            //var builderIDM = new CancelDatabase(System.IO.Path.Combine(NetworkConfig.DefaultDataDirectory, "Cancel"));
            //var builder = new CancelDatabase(System.IO.Path.Combine(NetworkConfig.DefaultDataDirectory, "Cancel"));
            //builder.CreateIDM(Path.Combine(ImageDatabaseManagementPath, "Duplicates.csv"), e => !string.IsNullOrEmpty(e.CancelComment));
            //builder.AddAllFilesInPath(@"C:\SA\AnalyzedPictures");
            //using var network = Network.ValueOf(Path.Combine(NetworkConfig.DefaultLogDirectory, "Cancel", "efficientnet-b0_DA_SVHN_20200526_1736_70.txt"));
            //using var dataSet = builder.ExtractDataSet(e => e.HasExpectedWidthHeightRatio(xShape[3] / ((double)xShape[2]), 0.05), root);
            //network.Predict(dataSet, Path.Combine(ImageDatabaseManagementPath, "Prediction.csv"));

            //var p = CFM60NetworkBuilder.Default(); p.Use_y_LinearRegressionEstimate = true; p.Use_pid_y_avg = false; p.Use_Christmas_flag = true; p.Use_EndOfYear_flag = true; p.Use_EndOfTrimester_flag = true; p.Use_day = false; ; p.ExtraDescription = "_target_eoy_eoq_christmas"; using var network = Network.ValueOf(@"C:\Users\Franck\AppData\Local\SharpNet\CFM60\CFM60_target_eoy_eoq_christmas_20210118_2308_150.txt");
            //using var cfm60TrainingAndTestDataSet = new CFM60TrainingAndTestDataSet(p, s => Network.Log.Info(s));
            //var cfm60Test = (CFM60DataSet)cfm60TrainingAndTestDataSet.Test;
            //cfm60Test.CreatePredictionFile(network, 1024, @"C:\Users\Franck\AppData\Local\SharpNet\CFM60\CFM60_target_eoy_eoq_christmas_20210118_2308_150" + "_" + DateTime.Now.Ticks + ".csv");
            //return;

            //var p = new CFM60NetworkBuilder();
            //p.InputSize = 5; p.HiddenSize = 128; p.NumEpochs = 150; p.BatchSize = 1024; p.InitialLearningRate = 0.0005; p.DropoutRate = 0.2;
            //using var network = Network.ValueOf(@"C:\Users\Franck\AppData\Local\SharpNet\CFM60\CFM60-0-0_InputSize5_128_Drop_0_2_lr_0_0005_20210115_1842_30.txt");
            //using var cfm60TrainingAndTestDataSet = new CFM60TrainingAndTestDataSet(p, s => Network.Log.Info(s));
            //var cfm60 = (CFM60DataSet)cfm60TrainingAndTestDataSet.Training;
            //using var trainingValidation = cfm60.SplitIntoTrainingAndValidation(p.PercentageInTraining);
            //Network.Log.Info("training: "+Network.MetricsToString(network.ComputeMetricsForTestDataSet(1024, trainingValidation.Training), "train_"));
            //Network.Log.Info("validation: "+Network.MetricsToString(network.ComputeMetricsForTestDataSet(1024, trainingValidation.Test), "val_"));
            //return;

            //var p = new CFM60NetworkBuilder();
            //using var cfm60TrainingAndTestDataSet = new CFM60TrainingAndTestDataSet(p, s => Network.Log.Info(s));
            //Network.Log.Info(cfm60TrainingAndTestDataSet.ComputeStats(0.9));
            //Network.Log.Info(cfm60TrainingAndTestDataSet.ComputeStats(0.68));
            //return;


            //using var cfm60TrainingAndTestDataSet = new CFM60TrainingAndTestDataSet(new CFM60NetworkBuilder(), s => Network.Log.Info(s));
            //var IDToPredictions = CFM60TrainingAndTestDataSet.LoadPredictionFile(@"C:\Users\Franck\AppData\Local\SharpNet\CFM60\validation_predictions\CFM60_do_not_use_day_20210117_1923.csv");
            //var mse = ((CFM60DataSet) cfm60TrainingAndTestDataSet.Training).ComputeMeanSquareError(IDToPredictions, false);
            //Network.Log.Info("mse for CFM60_do_not_use_day = " + mse.ToString(CultureInfo.InvariantCulture));
            //return;


            //using var cfm60TrainingAndTestDataSet = new CFM60TrainingAndTestDataSet(new CFM60NetworkBuilder(), s => Network.Log.Info(s));
            //var training = ((CFM60DataSet) cfm60TrainingAndTestDataSet.Training);
            //var test = ((CFM60DataSet) cfm60TrainingAndTestDataSet.Test);
            //using var sub = training.SplitIntoTrainingAndValidation(0.68);
            //var subTraining = ((CFM60DataSet)sub.Training);
            //var subValidation = ((CFM60DataSet)sub.Test);
            //var pidToLinearRegressionBetweenDayAndY = training.ComputePidToLinearRegressionBetweenDayAndY();
            //test.ComputePredictions(  e => pidToLinearRegressionBetweenDayAndY[e.pid].Estimation(e.day), "linearRegression_0_68");
            //training.CreateSummaryFile("c:/temp/training.csv");
            //test.CreateSummaryFile("c:/temp/test.csv");
            //return;


            //new TestCFM60DataSet().TestDayToFractionOfYear();return;
            //new TestParallelRunCpuVersusGpu().TestSwitchSecondAndThirdDimension(new[] {2,3,5}, new[] { 2, 5, 3 });

            CFM60Tests();

            //new TestCpuTensor().TestMaxPooling3D();return;
            //new NonReg.ParallelRunWithTensorFlow().TestParallelRunWithTensorFlow_UnivariateTimeSeries(); return;
            //EfficientNetTests_Cancel(true);
            //WideResNetTests();
            //SVHNTests();
            //CIFAR100Tests();
            //ResNetTests();
            //DenseNetTests();
            //EfficientNetTests();
            //TestSpeed();return;
            //new NonReg.TestBenchmark().TestGPUBenchmark_Memory();new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
            //new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
            //new NonReg.TestBenchmark().BenchmarkDataAugmentation();
        }

        #region DenseNet Training
        private static void DenseNetTests()
        {
            var networkGeometries = new List<Action<DenseNetBuilder, int>>
            {
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.DenseNet_12_40);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.DenseNetBC_12_100);},
            /*(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.DenseNet_Fast_CIFAR10);},
            (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.DenseNet_12_10_CIFAR10);},
            (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.DenseNet_12_40);},
            (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.DenseNetBC_12_40);},
             */
        };

            var networkMetaParameters = new List<Func<DenseNetBuilder>>
            {
                //(p) =>{p.UseNesterov = false; p.NumEpochs = 50; p.ExtraDescription = "_50Epoch_no_nesterov";},
                //(p) =>{p.UseAdam = true; p.NumEpochs = 5; p.ExtraDescription = "_50Epoch_Adam";},
                //(p) =>{p.SaveNetworkStatsAfterEachEpoch = true; p.ExtraDescription = "_Adam_with_l2_inConv";},
                //(p) =>{p.SaveNetworkStatsAfterEachEpoch = false;p.SaveLossAfterEachMiniBatch = false;p.UseAdam = true;p.UseNesterov = false;p.BatchSize = -1;p.NumEpochs = 150; p.ExtraDescription = "_Adam";},
                //(p) =>{ p.ExtraDescription = "_OrigPaper";},

                () =>{var p = DenseNetBuilder.DenseNet_CIFAR10();p.NumEpochs = 240;p.BatchSize = -1;p.Config.WithSGD(0.9,true); p.Config.CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1;p.DA.CutoutPatchPercentage = 0.0;p.ExtraDescription = "_240Epochs_TensorFlowCompatibilityMode_CutoutPatchPercentage0_WithNesterov_EnhancedMemory";return p;},
                () =>{var p = DenseNetBuilder.DenseNet_CIFAR10();p.NumEpochs = 240;p.BatchSize = -1;p.Config.WithSGD(0.9,false); p.Config.CompatibilityMode = NetworkConfig.CompatibilityModeEnum.TensorFlow1;p.DA.CutoutPatchPercentage = 0.0;p.ExtraDescription = "_240Epochs_TensorFlowCompatibilityMode_CutoutPatchPercentage0_NoNesterov_EnhancedMemory";return p;},


                //(p) =>{p.NumEpochs = 300;p.BatchSize = -1;p.CutoutPatchPercentage = 0.25;p.ExtraDescription = "_200Epochs_L2InDense_CutoutPatchPercentage0_25;},
                //(p) =>{p.NumEpochs = 300;p.BatchSize = -1;p.CutoutPatchPercentage = 0.0;p.ExtraDescription = "_200Epochs_L2_InDense_CutoutPatchPercentage0";},
                //(p) =>{p.NumEpochs = 200;p.Config.WithSGD(0.9,false);p.ExtraDescription = "_200Epochs_NoNesterov";},

            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries);
        }
        #endregion


        #region EfficientNet Cancel DataSet Training

        private static void EfficientNetTests_Cancel(bool useMultiGpu)
        {
            const int targetHeight = 470; const int targetWidth = 400;var batchSize = 20;const double defaultInitialLearningRate = 0.01;
            //var targetHeight = 235;var targetWidth = 200;var batchSize = 80;var defaultInitialLearningRate = 0.02;
            //var targetHeight = 118;var targetWidth = 100;var batchSize = 300;var defaultInitialLearningRate = 0.05;
            //var targetHeight = 59;var targetWidth = 50;var batchSize = 1200;var defaultInitialLearningRate = ?;
            
            if (useMultiGpu) { batchSize *= GPUWrapper.GetDeviceCount(); }


            const int numEpochs = 150;

            var networkMetaParameters = new List<Func<EfficientNetBuilder>>
            {
                () =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.DataAugmentationType =ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_SVHN;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_Cancel_Augment_SVHN"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.CutoutPatchPercentage=0 ;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_Cancel_NoCutout"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.Config.WithSGD(0.9, true);p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_nesterov_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.DataAugmentationType =ImageDataGenerator.DataAugmentationEnum.DEFAULT;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_Cancel_Augment_DEFAULT"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.025;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_lr_0_25_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.015;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_lr_0_15_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.CutoutPatchPercentage=0.05 ;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_Cancel_Cutout_0_05_"+targetWidth+"_"+targetHeight;return p;},


                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.01;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_Cancel_lr_0_01_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.04;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_Cancel_lr_0_04_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.CutoutPatchPercentage=0 ;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_Cancel_NoCutout"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.CutoutPatchPercentage=0.05 ;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_Cancel_0_05_Cutout"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.DataAugmentationType =ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_IMAGENET;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_Cancel_AutoAugment_ImageNet"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.015;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_lr_0_15_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.025;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_lr_0_25_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.Config.WithSGD(0.9, true);p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_nesterov_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.BatchNormEpsilon=0.0001;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_BatchNormEpsilon_0_0001_Cancel_"+targetWidth+"_"+targetHeight;return p;},


                //() =>{var p = EfficientNetBuilder.EfficientNet_Cancel();p.InitialLearningRate = 0.01;p.BatchSize = batchSize;p.NumEpochs = 5;p.ExtraDescription = "_0_01";return p;},
                //() =>{var p = EfficientNetBuilder.EfficientNet_Cancel();p.BatchSize = batchSize;p.NumEpochs = 150;p.ExtraDescription = "";return p;},
                //() =>{var p = EfficientNetBuilder.EfficientNet_Cancel();p.InitialLearningRate = 0.01;p.BatchSize = batchSize;p.NumEpochs = 30;p.ExtraDescription = "_0_01";return p;},
                //() =>{var p = EfficientNetBuilder.EfficientNet_Cancel();p.InitialLearningRate = 0.30;p.BatchSize = batchSize;p.NumEpochs = 30;p.ExtraDescription = "_0_30";return p;},
                //() =>{var p = EfficientNetBuilder.EfficientNet_Cancel();p.InitialLearningRate = 0.10;p.BatchSize = batchSize;p.NumEpochs = 30;p.ExtraDescription = "_0_10";return p;},
            };


            //networkMetaParameters.Clear();for (int i = 0; i < 4; ++i){int j = i;networkMetaParameters.Add(() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.BatchSize = batchSize;p.NumEpochs = numEpochs;p.ExtraDescription = "_V"+j+(useMultiGpu?"_MultiThreaded":"");return p;});}

            // ReSharper disable once ConditionIsAlwaysTrueOrFalse
            var networkGeometries = new List<Action<EfficientNetBuilder, int>>
            {
                (p,gpuDeviceId) =>{p.SetResourceId(gpuDeviceId);Train_Cancel_EfficientNet(p, targetHeight, targetWidth);},
            };

            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries, useMultiGpu);
        }
        private static void Train_Cancel_EfficientNet(EfficientNetBuilder p, int targetHeight, int targetWidth)
        {
            var database = new CancelDatabase();
            //TODO Test with selection of only matching size input in the training set
            //using var dataset = database.ExtractDataSet(e=>e.HasExpectedWidthHeightRatio(targetWidth / ((double)targetHeight), 0.05) && CancelDatabase.IsValidNonEmptyCancel(e.Cancel), ResizeStrategyEnum.ResizeToHeightAndWidthSizeKeepingSameProportionWith5PercentTolerance);
            using var dataset = database.ExtractDataSet(e=>CancelDatabase.IsValidNonEmptyCancel(e.Cancel), ResizeStrategyEnum.BiggestCropInOriginalImageToKeepSameProportion);
            using var trainingAndValidation = dataset.SplitIntoTrainingAndValidation(0.9); //90% for training,  10% for validation

            var rootPrediction = CancelDatabase.Hierarchy.RootPrediction();
            using var network = p.EfficientNetB0(true, "", new[] { trainingAndValidation.Training.Channels, targetHeight, targetWidth }, rootPrediction.Length);
            network.SetSoftmaxWithHierarchy(rootPrediction);
            //network.LoadParametersFromH5File(@System.IO.Path.Combine(NetworkConfig.DefaultLogDirectory, "Cancel", "efficientnet-b0_Imagenet_400_470_20200620_1427_310.h5"), NetworkConfig.CompatibilityModeEnum.TensorFlow1);

            //using var network =Network.ValueOf(@System.IO.Path.Combine(NetworkConfig.DefaultLogDirectory, "Cancel", "efficientnet-b0_Cancel_400_470_20200713_1809_580.txt"));

            //network.FindBestLearningRate(cancelDataset, 1e-5, 10, p.BatchSize);return;
            var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
            Network.Log.Debug(database.Summary());
            network.Fit(trainingAndValidation.Training, learningRateComputer, p.NumEpochs, p.BatchSize, trainingAndValidation.Test);
        }
        #endregion

        #region EfficientNet Training
        private static void EfficientNetTests()
        {
            const bool useMultiGpu = true;
            var networkGeometries = new List<Action<EfficientNetBuilder, int>>
            {
                //(p,gpuDeviceId) =>{p.SetResourceId(gpuDeviceId);p.WeightForTransferLearning = "imagenet";p.Config.LastLayerNameToFreeze = "top_dropout";p.ExtraDescription += "_only_dense";Train_CIFAR10_EfficientNet(p);},
                //(p,gpuDeviceId) =>{p.SetResourceId(gpuDeviceId);p.WeightForTransferLearning = "imagenet";p.Config.LastLayerNameToFreeze = "block7a_project_bn";p.ExtraDescription += "_all_top";Train_CIFAR10_EfficientNet(p);},
                (p,gpuDeviceId) =>{p.SetResourceId(gpuDeviceId);p.ExtraDescription += "";Train_CIFAR10_EfficientNet(p);},
            };

            var networkMetaParameters = new List<Func<EfficientNetBuilder>>
            {
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.30;p.NumEpochs = 1;p.ExtraDescription = "_lr_0_30_batchAuto_test";return p;},
                
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.30;p.NumEpochs = 30;p.ExtraDescription = "_lr_0_30_batchAuto";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 64;p.InitialLearningRate = 0.01;p.NumEpochs = 30;p.ExtraDescription = "_lr_0_30_batch64_test";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 32;p.InitialLearningRate = 0.01;p.NumEpochs = 30;p.ExtraDescription = "_lr_0_30_batch32_test";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 16;p.InitialLearningRate = 0.30;p.NumEpochs = 2;p.ExtraDescription = "_lr_0_30_batch16_test";return p;},
                
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 16;p.InitialLearningRate = 0.30;p.NumEpochs = 30;p.ExtraDescription = "_lr_0_30_batchAuto";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 32;p.InitialLearningRate = 0.01;p.NumEpochs = 150;p.ExtraDescription = "_lr_0_01_batchAuto_zoom_150epochs";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.001;p.NumEpochs = 30;p.WeightForTransferLearning = "imagenet";p.Config.LastLayerNameToFreeze = "top_dropout";p.ExtraDescription = "_lr_0_001_batchAuto_zoom7_30epochs_only_probs";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.001;p.NumEpochs = 30;p.WeightForTransferLearning = "imagenet";p.Config.LastLayerNameToFreeze = "block7a_project_bn";p.ExtraDescription = "_lr_0_001_batchAuto_zoom7_30epochs_only_top";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.10;p.NumEpochs = 30;p.ExtraDescription = "_lr_0_10_batchAuto_zoom_30epochs";return p;},
            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries, useMultiGpu);
        }
        private static void Train_CIFAR10_EfficientNet(EfficientNetBuilder p)
        {
            const int zoomFactor = 7;
            using var cifar10Original = new CIFAR10DataSet();
            var zoomedInputShape = new []{CIFAR10DataSet.Shape_CHW[0], zoomFactor * CIFAR10DataSet.Shape_CHW[1],zoomFactor * CIFAR10DataSet.Shape_CHW[2]};
            using var cifar10Zoomed = new ZoomedTrainingAndTestDataSet(cifar10Original, CIFAR10DataSet.Shape_CHW, zoomFactor, zoomFactor);
            using var network = p.EfficientNetB0_CIFAR10(p.WeightForTransferLearning, zoomedInputShape);
            Log.Info(network.ToString());
            //network.FindBestLearningRate(cifar10.Training, 1e-5, 10, p.BatchSize);return;

            var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
            network.Fit(cifar10Zoomed.Training, learningRateComputer, p.NumEpochs, p.BatchSize, cifar10Zoomed.Test);
        }
        #endregion

        #region WideResNet Training
        private static void WideResNetTests()
        {
            const bool useMultiGpu = false;
            var networkGeometries = new List<Action<WideResNetBuilder, int>>
            {
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 16,4);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 16,8);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 40,4);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 28,8);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 16,10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 28,10);},
            };

            //var batchSize = useMultiGpu ? 256 : 128;
            var networkMetaParameters = new List<Func<WideResNetBuilder>>
            {
                () =>{var p = WideResNetBuilder.WRN_CIFAR10();p.NumEpochs = 310;p.BatchSize = 128;p.ExtraDescription = "_BatchSize128_310epochs";return p;},
                
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.BatchSize = 128;p.ExtraDescription = "_BatchSize128";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.BatchSize = batchSize;p.ExtraDescription = "_MultiGPU";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.BatchSize = 128;p.ExtraDescription = "_BatchSize128";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.BatchSize = 128;p.ExtraDescription = "_BatchSize128";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.BatchSize = 64;p.ExtraDescription = "_BatchSize64";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.BatchSize = -1;p.ExtraDescription = "_BatchSizeAuto";return p;},
            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries, useMultiGpu);
        }
        private static void Train_CIFAR10_WRN(WideResNetBuilder p, int WRN_depth, int WRN_k)
        {
            using var cifar10 = new CIFAR10DataSet();
            using var network = p.WRN(WRN_depth, WRN_k, CIFAR10DataSet.Shape_CHW, CIFAR10DataSet.CategoryCount);
            var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
            network.Fit(cifar10.Training, learningRateComputer, p.NumEpochs, p.BatchSize, cifar10.Test);
        }
        #endregion

        #region ResNet Training
        private static void ResNetTests()
        {
            var networkGeometries = new List<Action<ResNetBuilder, int>>
            {

                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet164V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet110V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet56V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet20V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet11V2_CIFAR10);},
                
                /*
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet20V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet32V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet44V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet56V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet110V1_CIFAR10);},
                */
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet164V1_CIFAR10);},
            };

            var networkMetaParameters = new List<Func<ResNetBuilder>>
            {
                () =>{var p = ResNetBuilder.ResNet_CIFAR10();p.Config.WithSGD(0.9,true);p.ExtraDescription = "";return p;},
            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries);
        }
        #endregion

        #region CIFAR-100 Training
        private static void CIFAR100Tests()
        {
            var networkGeometries = new List<Action<WideResNetBuilder, int>>
            {
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 16,4);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 16,8);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 40,4);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 16,10);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 28,8);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 28,10);},
            };
            var networkMetaParameters = new List<Func<WideResNetBuilder>>
            {
                () => {var p = WideResNetBuilder.WRN_CIFAR100();p.DA.AlphaMixup = 0.0;p.DA.AlphaCutMix = 1.0;p.DA.CutoutPatchPercentage = 0.0; p.ExtraDescription = "CutMix";return p;},
                () => {var p = WideResNetBuilder.WRN_CIFAR100();p.DA.AlphaMixup = 1.0;p.DA.AlphaCutMix = 0.0;p.DA.CutoutPatchPercentage = 0.0; p.ExtraDescription = "Mixup";return p;},
                () => {var p = WideResNetBuilder.WRN_CIFAR100();p.DA.AlphaMixup = 0.0;p.DA.AlphaCutMix = 0.0;p.DA.CutoutPatchPercentage = 20.0/32.0; p.ExtraDescription = "Cutout_0_625";return p;},
            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries);
        }

        private static void Train_CIFAR100_WRN(WideResNetBuilder p, int WRN_depth, int WRN_k)
        {
            using (var cifar100 = new CIFAR100DataSet())
            using (var network = p.WRN(WRN_depth, WRN_k, CIFAR100DataSet.Shape_CHW, CIFAR100DataSet.CategoryCount))
            {
                var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
                network.Fit(cifar100.Training, learningRateComputer, p.NumEpochs, p.BatchSize, cifar100.Test);
            }
        }
        #endregion

        #region SVHN Training
        private static void SVHNTests()
        {
            const bool useMultiGpu = false;
            var networkGeometries = new List<Action<WideResNetBuilder, int>>
            {
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 16,4);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 16,8);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 40,4);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 16,10);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 28,8);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 28,10);},
            };

            var networkMetaParameters = new List<Func<WideResNetBuilder>>
            {
                () =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs = 30;p.BatchSize=-1;p.ExtraDescription = "_30Epochs_MultiGPU";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs = 30;p.ExtraDescription = "_30Epochs";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs = 30;p.Config.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS;  p.ExtraDescription = "_30Epochs_USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS";return p;},
            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries, useMultiGpu);
        }

        private static void Train_SVHN_WRN(WideResNetBuilder p, bool loadExtraFileForTraining, int WRN_depth, int WRN_k)
        {
            using var svhn = new SVHNDataSet(loadExtraFileForTraining);
            using var network = p.WRN(WRN_depth, WRN_k, SVHNDataSet.Shape_CHW, SVHNDataSet.CategoryCount);
            //using var network = Network.ValueOf(@"C:\Users\Franck\AppData\Local\SharpNet\SVHN\WRN-16-10_30Epochs_MultiGPU_20200501_1147_30.txt");
            var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
            network.Fit(svhn.Training, learningRateComputer, p.NumEpochs, p.BatchSize, svhn.Test);
        }
        #endregion

        /// <summary>
        /// Train a network on CIFAR-10 data set 
        /// </summary>
        private static void Train_CIFAR10(NetworkBuilder p, Func<CIFAR10DataSet, Network> buildNetwork)
        {
            using (var cifar10 = new CIFAR10DataSet())
            using (var network = buildNetwork(cifar10))
            {
                var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
                network.Fit(cifar10.Training, learningRateComputer, p.NumEpochs, p.BatchSize, cifar10.Test);
            }
        }


        #region CFM60 Training
        private static void CFM60Tests()
        {
            const bool useMultiGpu = false;
            var networkGeometries = new List<Action<CFM60NetworkBuilder, int>>
            {
                (p,gpuDeviceId) =>{p.SetResourceId(gpuDeviceId);Train_CFM60(p);},
            };

            var networkMetaParameters = new List<Func<CFM60NetworkBuilder>>
            {
                () =>{var p = CFM60NetworkBuilder.Default(); p.ExtraDescription = "";return p;},
                () =>{var p = CFM60NetworkBuilder.Default();p.Encoder(2,20,0.2);p.Decoder(0,0,0); p.ExtraDescription = "_no_decoder";return p;},
                () =>{var p = CFM60NetworkBuilder.Default();p.Decoder_TimeSteps = 2; p.ExtraDescription = "_2predictions";return p;},
                () =>{var p = CFM60NetworkBuilder.Default();p.Decoder_TimeSteps = 50; p.ExtraDescription = "_50predictions";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder_TimeSteps = 80;p.Config.AlwaysUseFullTestDataSetForLossAndAccuracy=true; p.ExtraDescription = p.EncDesc();return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder_TimeSteps = 60;p.Config.AlwaysUseFullTestDataSetForLossAndAccuracy=true; p.ExtraDescription = p.EncDesc();return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder_TimeSteps = 40;p.Config.AlwaysUseFullTestDataSetForLossAndAccuracy=true; p.ExtraDescription = p.EncDesc();return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder_TimeSteps = 30;p.Config.AlwaysUseFullTestDataSetForLossAndAccuracy=true; p.ExtraDescription = p.EncDesc();return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.AlwaysUseFullTestDataSetForLossAndAccuracy=true; p.ExtraDescription = p.EncDesc();return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 6;p.Config.AlwaysUseFullTestDataSetForLossAndAccuracy=true; p.ExtraDescription = p.EncDesc()+"_"+p.Pid_EmbeddingDim+"embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 12;p.Config.AlwaysUseFullTestDataSetForLossAndAccuracy=true; p.ExtraDescription = p.EncDesc()+"_"+p.Pid_EmbeddingDim+"embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.Config.AlwaysUseFullTestDataSetForLossAndAccuracy=true; p.ExtraDescription = p.EncDesc()+"_"+p.Pid_EmbeddingDim+"embedding";return p;},

                //TODO:
                //try: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/

                #region already performed tests
                ////CFM60_02drop_encoder_2_20_decoder_2_1_02drop    423861  30  5.00E-06    1086.415833 0.313492924 0.361346334
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder(2, 20, 0.0);p.Decoder(2, 1, 0.2);p.DropoutRate=0.2;p.DropoutRate_After_Decoder=0.0;p.ExtraDescription = p.EncDesc();return p;},
                ////CFM60_0drop_encoder_1_20_decoder_1_1_02dropAfter    226229  30  5.00E-06    904.2233133 0.326655418 0.363409907
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder(1, 20, 0.0);p.Decoder(1, 1, 0.0);p.DropoutRate=0.0;p.DropoutRate_After_Decoder=0.2;p.ExtraDescription = p.EncDesc();return p;},
                ////CFM60_02drop_encoder_1_20_decoder_2_1_02dropAfter   325045  30  5.00E-06    914.494669  0.319439262 0.363662958
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder(1, 20, 0.0);p.Decoder(2, 1, 0.2);p.DropoutRate=0.2;p.DropoutRate_After_Decoder=0.2;p.ExtraDescription = p.EncDesc();return p;},
                ////CFM60_0drop_encoder_2_20_02drop_decoder_1_1 325045  30  5.00E-06    1052.748153 0.319399804 0.363881439
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder(2, 20, 0.2);p.Decoder(1, 1, 0.0);p.DropoutRate=0.0;p.DropoutRate_After_Decoder=0.0;p.ExtraDescription = p.EncDesc();return p;},
                ////CFM60_0drop_encoder_2_20_decoder_2_1_02drop 423861  30  5.00E-06    1025.255989 0.319993883 0.364001006
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder(2, 20, 0.0);p.Decoder(2, 1, 0.2);p.DropoutRate=0.0;p.DropoutRate_After_Decoder=0.0;p.ExtraDescription = p.EncDesc();return p;},
                ////CFM60_0drop_encoder_2_20_02drop 215437  30  5.00E-06    844.7426773 0.322956085 0.3643727
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder(2, 20, 0.2);p.Decoder(0, 0, 0);p.DropoutRate=0.0;p.DropoutRate_After_Decoder=0.0;p.ExtraDescription = p.EncDesc();return p;},
                ////CFM60_0drop_encoder_1_20_decoder_2_1    325045  30  5.00E-06    948.3809039 0.320686072 0.364709973
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder(1, 20, 0.0);p.Decoder(2, 1, 0);p.DropoutRate=0.0;p.DropoutRate_After_Decoder=0.0;p.ExtraDescription = p.EncDesc();return p;},
                ////CFM60_02drop_encoder_1_20_decoder_1_1_02dropAfter
                //() =>{var p = CFM60NetworkBuilder.Default();p.Encoder(1, 20, 0.0);p.Decoder(1, 1, 0.0);p.DropoutRate=0.2;p.DropoutRate_After_Decoder=0.2;p.ExtraDescription = p.EncDesc();return p;},
                #endregion
            };

            //double[] allDropRates = {0, 0.2};
            //int[] allNumLayers = {2, 1};
            //foreach (var dropRate in allDropRates)
            //foreach (var encoderNumLayers in allNumLayers)
            //foreach (var encoderTimeSteps in new[] { /*40,*/ 20 })
            //foreach (var encoderDropRate in allDropRates)
            //foreach (var useDecoder in new[] { false,true })
            ////foreach (var useDecoder in new[] { true })
            //foreach (var decoderNumLayers in useDecoder? allNumLayers : new []{0})
            //foreach (var decoderTimeSteps in useDecoder ? new[] { 1 } : new[] { 0 })
            //foreach (var decoderDropRate in useDecoder ? allDropRates : new[] { 0.0 })
            //foreach (var dropoutRate_After_Decoder in useDecoder ? allDropRates : new[] { 0.0 })
            //{
            //    if (   encoderNumLayers <= 1 && encoderDropRate > 1e-6
            //         ||decoderNumLayers <= 1 && decoderDropRate > 1e-6 )
            //    {
            //        continue;
            //    }
            //    networkMetaParameters.Add(
            //        () => { var p = CFM60NetworkBuilder.Default();
            //            p.DropoutRate = dropRate;
            //            p.Encoder(encoderNumLayers, encoderTimeSteps, encoderDropRate); 
            //            p.Decoder(decoderNumLayers, decoderTimeSteps, decoderDropRate);
            //            p.DropoutRate_After_Decoder = dropoutRate_After_Decoder;
            //            p.ExtraDescription = p.EncDesc(); return p; }
            //        );
            //}

            //networkMetaParameters = SharpNet.Utils.Repeat(networkMetaParameters, 2);

            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries, useMultiGpu);
        }



        private static void Train_CFM60(CFM60NetworkBuilder p)
        {
            using var network = p.CFM60();
            //using var network = Network.ValueOf(@"C:\Users\Franck\AppData\Local\SharpNet\CFM60\CFM60-0-0_InputSize4_64_DropoutRate0_2_20210115_1831_10.txt");
            using var cfm60TrainingAndTestDataSet = new CFM60TrainingAndTestDataSet(p, s=> Network.Log.Info(s));
            var cfm60 = (CFM60DataSet) cfm60TrainingAndTestDataSet.Training;
            //To compute feature importances, uncomment the following line
            //cfm60.ComputeFeatureImportances("c:/temp/cfm60_featureimportances.csv", false); return;
            using var trainingValidation = cfm60.SplitIntoTrainingAndValidation(p.PercentageInTraining);
            //var res = network.FindBestLearningRate(cfm60, 1e-7, 0.9, p.BatchSize);return;

            var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
            ((CFM60DataSet) trainingValidation.Training).OriginalTestDataSet = (CFM60DataSet)cfm60TrainingAndTestDataSet.Test;
            network.Fit(trainingValidation.Training, learningRateComputer, p.NumEpochs, p.BatchSize, trainingValidation.Test);
            ((CFM60DataSet)trainingValidation.Training).OriginalTestDataSet = null;

            //((CFM60DataSet)cfm60TrainingAndTestDataSet.Test).CreatePredictionFile(network);
            //((CFM60DataSet)trainingValidation.Test).CreatePredictionFile(network);
        }
        #endregion



        /// <summary>
        /// perform as much actions as possible among 'allActionsToPerform'
        /// </summary>
        /// <param name="gpuId">GPU deviceId to use 
        /// int.MaxValue means uses all available GPU</param>
        /// <param name="allActionsToPerform"></param>
        private static void PerformActionsInSingleGpu(int gpuId, List<Action<int>> allActionsToPerform)
        {
            for (; ; )
            {
                Action<int> nexActionToPerform;
                var gpuIdPrefix = "GpuId#" + (gpuId==int.MaxValue?"All":gpuId.ToString()) + " : ";
                lock (allActionsToPerform)
                {
                    Console.WriteLine(gpuIdPrefix + allActionsToPerform.Count + " remaining computation(s)");
                    if (allActionsToPerform.Count == 0)
                    {
                        return;
                    }
                    nexActionToPerform = allActionsToPerform[0];
                    allActionsToPerform.RemoveAt(0);
                }
                try
                {
                    Console.WriteLine(gpuIdPrefix+"starting new computation");
                    nexActionToPerform(gpuId);
                    Console.WriteLine(gpuIdPrefix + "ended new computation");
                }
                catch (Exception e)
                {
                    Console.WriteLine(gpuIdPrefix + e);
                    Console.WriteLine(gpuIdPrefix + "ignoring error");
                }
            }
        }
        private static void PerformAllActionsInAllGpu<T>(List<Func<T>> networkMetaParameters, List<Action<T, int>> networkGeometriesOrderedFromSmallestToBiggest, bool useMultiGPU = false) where T : NetworkBuilder, new()
        {
            var taskToBePerformed = new List<Action<int>>();

            //we'll start to compute the most time intensive (bigger) network, to end with the smallest one
            var networkGeometries = new List<Action<T, int>>(networkGeometriesOrderedFromSmallestToBiggest);
            networkGeometries.Reverse();

            foreach (var networkMetaParameter in networkMetaParameters)
            {
                foreach (var networkGeometry in networkGeometries)
                {
                    taskToBePerformed.Add(gpuDeviceId => networkGeometry(networkMetaParameter(), gpuDeviceId));
                }
            }

            Task[] gpuTasks;
            if (useMultiGPU)
            {
                //if multi GPU support is enabled, we'll use a single task that will use all GPU
                Console.WriteLine(taskToBePerformed.Count + " computation(s) will be done on All (" + GPUWrapper.GetDeviceCount() + ") GPU");
                gpuTasks = new Task[1];
                gpuTasks[0] = new Task(() => PerformActionsInSingleGpu(int.MaxValue, taskToBePerformed));
                gpuTasks[0].Start();
            }
            else
            {
                int nbGPUs = Math.Min(GPUWrapper.GetDeviceCount(), taskToBePerformed.Count);
                Console.WriteLine(taskToBePerformed.Count + " computation(s) will be done on " + nbGPUs + " GPU");
                gpuTasks = new Task[nbGPUs];
                for (int i = 0; i < gpuTasks.Length; ++i)
                {
                    var gpuId = i;
                    gpuTasks[i] = new Task(() => PerformActionsInSingleGpu(gpuId, taskToBePerformed));
                    gpuTasks[i].Start();
                }

            }
            Task.WaitAll(gpuTasks);
        }
    }
}
