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

                //TODO:
                //try: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
                
                () =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=40;  p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps_1cycle_drop_0_50";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.TimeSteps=40;  p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps_cyclic";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumLayersLastLSTM = 2;p.DropoutRateLastLSTM = p.DropoutRate;p.LSTMLayersReturningFullSequence = 0;p.TimeSteps=40;  p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps_singleLSTM_cyclic";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=40;  p.Config.lambdaL2Regularization*= 2;p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps_l2_mult_2";return p;},
                

                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.ExtraDescription = "_"+p.TimeSteps+"timesteps_1cycle_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.Config.WithAdamW(0.000005);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_1cycle_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.Config.WithAdamW(0.00001);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_1cycle_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.Config.WithAdamW(0.000025);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_1cycle_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.Config.WithAdamW(0.0001);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_1cycle_divide_20";return p;},


                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0); p.ExtraDescription = "_"+p.TimeSteps+"timesteps_cyclic";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0); p.Config.WithAdamW(0.000005);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0); p.Config.WithAdamW(0.00001);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0); p.Config.WithAdamW(0.000025);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0); p.Config.WithAdamW(0.0001);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic";return p;},

                
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0); p.ExtraDescription = "_"+p.TimeSteps+"timesteps_cyclic";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0); p.Config.WithAdamW(p.Config.AdamW_L2Regularization/2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0); p.Config.WithAdamW(p.Config.AdamW_L2Regularization*2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic";return p;},
                
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.ExtraDescription = "_"+p.TimeSteps+"timesteps_1cycle_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.Config.WithAdamW(p.Config.AdamW_L2Regularization/2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_1cycle_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.Config.WithAdamW(p.Config.AdamW_L2Regularization*2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_1cycle_divide_20";return p;},
                
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/20); p.ExtraDescription = "_"+p.TimeSteps+"timesteps_cyclic_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/20); p.Config.WithAdamW(p.Config.AdamW_L2Regularization/2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/20); p.Config.WithAdamW(p.Config.AdamW_L2Regularization*2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic_divide_20";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200); p.ExtraDescription = "_"+p.TimeSteps+"timesteps_cyclic_divide_200";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200); p.Config.WithAdamW(p.Config.AdamW_L2Regularization/2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic_divide_200";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200); p.Config.WithAdamW(p.Config.AdamW_L2Regularization*2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic_divide_200";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0); p.ExtraDescription = "_"+p.TimeSteps+"timesteps_cyclic";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0); p.Config.WithAdamW(p.Config.AdamW_L2Regularization/2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0); p.Config.WithAdamW(p.Config.AdamW_L2Regularization*2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.TimeSteps=60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.ExtraDescription = "_"+p.TimeSteps+"timesteps_1cycle_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.TimeSteps=60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.Config.WithAdamW(p.Config.AdamW_L2Regularization/2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_1cycle_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.TimeSteps=60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.Config.WithAdamW(p.Config.AdamW_L2Regularization*2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_1cycle_divide_20";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/20); p.ExtraDescription = "_"+p.TimeSteps+"timesteps_cyclic_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/20); p.Config.WithAdamW(p.Config.AdamW_L2Regularization/2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/20); p.Config.WithAdamW(p.Config.AdamW_L2Regularization*2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic_divide_20";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200); p.ExtraDescription = "_"+p.TimeSteps+"timesteps_cyclic_divide_200";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200); p.Config.WithAdamW(p.Config.AdamW_L2Regularization/2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic_divide_200";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default(); p.InitialLearningRate = 0.002; p.TimeSteps=60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200); p.Config.WithAdamW(p.Config.AdamW_L2Regularization*2);p.ExtraDescription = "_AdamWl2_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps_cyclic_divide_200";return p;},



                

            #region already performed tests
                //() =>{var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.lambdaL2Regularization/= 2;p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps_l2_divide_2";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.lambdaL2Regularization*= 2;p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps_l2_mult_2";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.lambdaL2Regularization=0;p.ExtraDescription = "_no_l2";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithAdamW(p.Config.lambdaL2Regularization);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithAdamW(p.Config.lambdaL2Regularization/2);p.ExtraDescription = "_l2_div2_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithAdamW(p.Config.lambdaL2Regularization*2);p.ExtraDescription = "_l2_mult2_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},


                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=40;  p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=40;  p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=40;  p.Config.lambdaL2Regularization/= 2;p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps_l2_divide_2";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=40;  p.Config.lambdaL2Regularization*= 2;p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps_l2_mult_2";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=40;  p.Config.lambdaL2Regularization=0;p.ExtraDescription = "_no_l2"+"_"+p.TimeSteps+"timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=40;  p.Config.WithAdamW(p.Config.lambdaL2Regularization);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=40;  p.Config.WithAdamW(p.Config.lambdaL2Regularization/2);p.ExtraDescription = "_l2_div2_A_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=40;  p.Config.WithAdamW(p.Config.lambdaL2Regularization*2);p.ExtraDescription = "_l2_mult2_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;  p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;  p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;  p.Config.lambdaL2Regularization/= 2;p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps_l2_divide_2";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;  p.Config.lambdaL2Regularization*= 2;p.ExtraDescription = ""+"_"+p.TimeSteps+"timesteps_l2_mult_2";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;  p.Config.lambdaL2Regularization=0;p.ExtraDescription = "_no_l2"+"_"+p.TimeSteps+"timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;  p.Config.WithAdamW(p.Config.lambdaL2Regularization);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;  p.Config.WithAdamW(p.Config.lambdaL2Regularization/2);p.ExtraDescription = "_l2_div2_A_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps=60;  p.Config.WithAdamW(p.Config.lambdaL2Regularization*2);p.ExtraDescription = "_l2_mult2_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_')+"_"+p.TimeSteps+"timesteps";return p;},


                //() =>{var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.NumLayersLastLSTM = 2;p.ExtraDescription = "_2LayersLastLSTM";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.NumLayersLastLSTM = 4;p.ExtraDescription = "_4LayersLastLSTM";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2 = true;p.ExtraDescription = "_UseBatchNorm2";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType = CFM60NetworkBuilder.InputNormalizationEnum.BATCH_NORM_LAYER;p.ExtraDescription = "_BATCH_NORM_LAYER";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType = CFM60NetworkBuilder.InputNormalizationEnum.DEDUCE_MEAN;p.ExtraDescription = "_DEDUCE_MEAN";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithAdamW(0.000005);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithAdamW(0.00001);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithAdamW(0.0001);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.10;p.ExtraDescription = "_0_10_dropout";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.30;p.ExtraDescription = "_0_30_dropout";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.0; p.ExtraDescription = "_no_dropout";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithAdamW(0.00005);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithAdamW( 0.0005);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithAdamW(  0.001);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithAdamW(  0.005);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithAdamW(   0.05);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default(); p.Config.WithAdamW(   0.1);p.ExtraDescription = "_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 60;p.ExtraDescription = "_60timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 60;p.NumLayersLastLSTM = 2;p.ExtraDescription = "_60timesteps_2LayersLastLSTM";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 60;p.ExtraDescription = "_60timesteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 60;p.Config.WithAdamW(0.00005);p.ExtraDescription = "_60timesteps_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 60;p.Config.WithAdamW( 0.0005);p.ExtraDescription = "_60timesteps_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 60;p.Config.WithAdamW(  0.001);p.ExtraDescription = "_60timesteps_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 60;p.Config.WithAdamW(  0.005);p.ExtraDescription = "_60timesteps_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 60;p.Config.WithAdamW(   0.05);p.ExtraDescription = "_60timesteps_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 60;p.Config.WithAdamW(   0.1);p.ExtraDescription = "_60timesteps_AdamW_"+p.Config.AdamW_L2Regularization.ToString(CultureInfo.InvariantCulture).Replace('.','_');return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.10;p.TimeSteps = 60;p.ExtraDescription = "_60timesteps_0_10_dropout";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.30;p.TimeSteps = 60;p.ExtraDescription = "_60timesteps_0_30_dropout";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.0;p.TimeSteps = 60;p.ExtraDescription = "_60timesteps_no_dropout";return p;},


                //() =>{var p = CFM60NetworkBuilder.Default();p.LSTMLayersReturningFullSequence = 0;p.NumLayersLastLSTM = 2;p.ExtraDescription = "_2layers_last";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.LSTMLayersReturningFullSequence = 0;p.DropoutRateLastLSTM=p.DropoutRate ;p.NumLayersLastLSTM = 2;p.ExtraDescription = "_2layers_last_dropout_everywhere";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.LSTMLayersReturningFullSequence = 0;p.DropoutRateLastLSTM=p.DropoutRate=0 ;p.NumLayersLastLSTM = 2;p.ExtraDescription = "_2layers_no_dropout_at_all";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.LSTMLayersReturningFullSequence = 0;p.DropoutRateLastLSTM=p.DropoutRate ;p.DropoutRate=0 ;p.NumLayersLastLSTM = 2;p.ExtraDescription = "_2layers_last_dropout_only_in_lstm";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.ExtraDescription = "_16embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.LSTMLayersReturningFullSequence = 0;p.NumLayersLastLSTM = 2;p.ExtraDescription = "_2layers_last_16embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.LSTMLayersReturningFullSequence = 0;p.DropoutRateLastLSTM=p.DropoutRate ;p.NumLayersLastLSTM = 2;p.ExtraDescription = "_2layers_last_dropout_everywhere_16embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.LSTMLayersReturningFullSequence = 0;p.DropoutRateLastLSTM=p.DropoutRate=0 ;p.NumLayersLastLSTM = 2;p.ExtraDescription = "_2layers_no_dropout_at_all_16embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.LSTMLayersReturningFullSequence = 0;p.DropoutRateLastLSTM=p.DropoutRate ;p.DropoutRate=0 ;p.NumLayersLastLSTM = 2;p.ExtraDescription = "_2layers_last_dropout_only_in_lstm_16embedding";return p;},
                


                //() =>{var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = 0; p.ExtraDescription = "_l2_0_0";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = 0.00001; p.ExtraDescription = "_l2_0_00001";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = 0.00005; p.ExtraDescription = "_l2_0_00005";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = 0.0001; p.ExtraDescription = "_l2_0_0001";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = 0.0005; p.ExtraDescription = "_l2_0_0005";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = 0.001; p.ExtraDescription = "_l2_0_001";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = 0.005; p.ExtraDescription = "_l2_0_005";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.ExtraDescription = "_Cyclic";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = 0.00005; p.ExtraDescription = "";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.ExtraDescription = "_Cyclic";return p;},


                //() =>{var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 10; p.ExtraDescription = "10TimeSteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 30; p.ExtraDescription = "30TimeSteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 40; p.ExtraDescription = "40TimeSteps";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithLinearLearningRateScheduler(20);p.NumEpochs = 10;p.TimeSteps = 45;p.InitialLearningRate = 0.002; p.ExtraDescription = "_45TimeSteps_lr_002_12embedding_linear";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithLinearLearningRateScheduler(20);p.NumEpochs = 10;p.TimeSteps = 55;p.InitialLearningRate = 0.002; p.ExtraDescription = "_55TimeSteps_lr_002_12embedding_linear";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithLinearLearningRateScheduler(20);p.NumEpochs = 10;p.TimeSteps = 65;p.InitialLearningRate = 0.002; p.ExtraDescription = "_65TimeSteps_lr_002_12embedding_linear";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithLinearLearningRateScheduler(20);p.NumEpochs = 10;p.TimeSteps = 75;p.InitialLearningRate = 0.002; p.ExtraDescription = "_75TimeSteps_lr_002_12embedding_linear";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithLinearLearningRateScheduler(20);p.NumEpochs = 10;p.TimeSteps = 85;p.InitialLearningRate = 0.002; p.ExtraDescription = "_85TimeSteps_lr_002_12embedding_linear";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithLinearLearningRateScheduler(20);p.NumEpochs = 10;p.TimeSteps = 90;p.InitialLearningRate = 0.002; p.ExtraDescription = "_90TimeSteps_lr_002_12embedding_linear";return p;},


                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 10;p.TimeSteps = 45;p.InitialLearningRate = 0.002; p.ExtraDescription = "_45TimeSteps_lr_002_12embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 10;p.TimeSteps = 55;p.InitialLearningRate = 0.002; p.ExtraDescription = "_55TimeSteps_lr_002_12embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 10;p.TimeSteps = 65;p.InitialLearningRate = 0.002; p.ExtraDescription = "_65TimeSteps_lr_002_12embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 10;p.TimeSteps = 75;p.InitialLearningRate = 0.002; p.ExtraDescription = "_75TimeSteps_lr_002_12embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 10;p.TimeSteps = 85;p.InitialLearningRate = 0.002; p.ExtraDescription = "_85TimeSteps_lr_002_12embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 10;p.TimeSteps = 90;p.InitialLearningRate = 0.002; p.ExtraDescription = "_90TimeSteps_lr_002_12embedding";return p;},

                

                //() =>{var p = CFM60NetworkBuilder.Default();p.BatchSize=1024;p.NumEpochs = 30;p.InitialLearningRate = 0.002; p.ExtraDescription = "_45TimeSteps_lr_002_16embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.BatchSize=1024;p.NumEpochs = 30;p.InitialLearningRate = 0.002; p.ExtraDescription = "_55TimeSteps_lr_002_16embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.BatchSize=1024;p.NumEpochs = 30;p.InitialLearningRate = 0.002; p.ExtraDescription = "_65TimeSteps_lr_002_16embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.BatchSize=1024;p.NumEpochs = 30;p.InitialLearningRate = 0.002; p.ExtraDescription = "_75TimeSteps_lr_002_16embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.BatchSize=1024;p.NumEpochs = 30;p.InitialLearningRate = 0.002; p.ExtraDescription = "_85TimeSteps_lr_002_16embedding";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.BatchSize=1024;p.NumEpochs = 30;p.InitialLearningRate = 0.002; p.ExtraDescription = "_90TimeSteps_lr_002_16embedding";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 10;p.TimeSteps = 60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.ExtraDescription = "_60TimeSteps_lr_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 10;p.TimeSteps = 100; p.ExtraDescription = "_60TimeSteps";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 10;p.TimeSteps = 60;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.ExtraDescription = "_60TimeSteps_lr_divide_20";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 10;p.TimeSteps = 100;p.InitialLearningRate = 0.002;p.Config.WithOneCycleLearningRateScheduler(20, 0.1); p.ExtraDescription = "_100TimeSteps_lr_0_002_divide_20";return p;},

                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 60;p.InitialLearningRate = 0.002;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/20); p.ExtraDescription = "_60TimeSteps_lr_0_002_cyclic";return p;},
                //() =>{var p = CFM60NetworkBuilder.Default();p.TimeSteps = 60;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2,p.InitialLearningRate/20); p.ExtraDescription = "_60TimeSteps_cyclic";return p;},

                


            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.20, true, true, 0.02);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.50, true, true, 0.03);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.20, true, true, 0.02);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.50, true, true, 0.03);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},


            //() =>{var p = CFM60NetworkBuilder.Default();p.LSTMLayersReturningFullSequence=2;p.ExtraDescription ="_2LSTMLayersReturningFullSequence";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.LSTMLayersReturningFullSequence=3;p.ExtraDescription ="_3LSTMLayersReturningFullSequence";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate=0.1;p.ExtraDescription ="_drop_0_10";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate=0.3;p.ExtraDescription ="_drop_0_30";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate=0.0;p.ExtraDescription ="_nodrop";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.WithSpecialEndV1=true;p.ExtraDescription ="_WithSpecialEndV1";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.05, true, true, 0.01);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.10, true, true, 0.01);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.20, true, true, 0.01);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.50, true, true, 0.01);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},



            //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.BATCH_NORM_LAYER;p.ExtraDescription ="_BATCH_NORM_LAYER";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.DEDUCE_MEAN;p.ExtraDescription ="_DEDUCE_MEAN";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.Z_SCORE;p.ExtraDescription ="_Z_SCORE";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.DEDUCE_MEAN_AND_BATCH_NORM_LAYER;p.ExtraDescription ="_DEDUCE_MEAN_AND_BATCH_NORM_LAYER";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2 = true;p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.BATCH_NORM_LAYER;p.ExtraDescription ="_BATCH_NORM_LAYER_UseBatchNorm2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2 = true;p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.DEDUCE_MEAN;p.ExtraDescription ="_DEDUCE_MEAN_UseBatchNorm2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2 = true;p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.Z_SCORE;p.ExtraDescription ="_Z_SCORE_UseBatchNorm2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2 = true;p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.DEDUCE_MEAN_AND_BATCH_NORM_LAYER;p.ExtraDescription ="_DEDUCE_MEAN_AND_BATCH_NORM_LAYER_UseBatchNorm2";return p;},


            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate=0.01;p.ExtraDescription ="_lr_0_01";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate=0.005;p.ExtraDescription ="_lr_0_005";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate=0.0005;p.ExtraDescription ="_lr_0_0005";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate=0.0001;p.ExtraDescription ="_lr_0_0001";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_ret_vol_start_and_end_only=true; p.ExtraDescription ="_Use_ret_vol_start_and_end_only";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_ret_vol=false; p.ExtraDescription ="_no_Use_ret_vol";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.HiddenSize=50;p.Use_ret_vol_start_and_end_only=true; p.ExtraDescription ="_Use_ret_vol_start_and_end_only_50HiddenSize";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.HiddenSize=50;p.Use_ret_vol=false; p.ExtraDescription ="_no_Use_ret_vol_50HiddenSize";return p;},


            //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithLinearLearningRateScheduler(200); p.ExtraDescription ="_linear_divide_200";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithLinearLearningRateScheduler(2000); p.ExtraDescription ="_linear_divide_2000";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithLinearLearningRateScheduler(100); p.ExtraDescription ="_linear_divide_100";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01;p.Config.WithLinearLearningRateScheduler(200); p.ExtraDescription ="_lr_0_01_linear_divide_200";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.HiddenSize=64;p.NumEpochs = 70; p.ExtraDescription ="_64HiddenSize_70epochs";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.BatchSize = 4096;p.HiddenSize=64; p.ExtraDescription ="_64HiddenSize_4096batchSize";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.HiddenSize=50; p.ExtraDescription ="_50HiddenSize";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DenseUnits=100; p.ExtraDescription ="_100DenseUnits";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DenseUnits=60; p.ExtraDescription ="_60DenseUnits";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DenseUnits=40; p.ExtraDescription ="_40DenseUnits";return p;},


            //() =>{var p = CFM60NetworkBuilder.Default();p.BatchSize = 4096;p.DenseUnits=25; p.ExtraDescription ="_25DenseUnits_4096batchSize";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate / 200);p.ExtraDescription = "_mean_lr_divider_200";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2;return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate / 200);p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_mean_lr_divider_200";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01;p.ExtraDescription = "_lr_0_01";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate=0.1; p.ExtraDescription ="_drop_0_10";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim=16; p.ExtraDescription ="_16embeddding";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2=true; p.ExtraDescription ="_false_true";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.BATCH_NORM_LAYER;p.UseBatchNorm2=false; p.ExtraDescription ="_BATCH_NORM_LAYER";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.DEDUCE_MEAN;p.UseBatchNorm2=false; p.ExtraDescription ="_DEDUCE_MEAN";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.DEDUCE_MEAN_AND_BATCH_NORM_LAYER;p.UseBatchNorm2=false; p.ExtraDescription ="_DEDUCE_MEAN_AND_BATCH_NORM_LAYER";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.HiddenSize=128; p.ExtraDescription ="_128HiddenSize";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.HiddenSize=64; p.ExtraDescription ="_64HiddenSize";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DenseUnits=100; p.ExtraDescription ="_100DenseUnits";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2=true;p.Pid_EmbeddingDim=16;p.UseBatchNorm2=true; p.ExtraDescription ="_False_True_0_10_dropout_16embeddding";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2=true;p.Pid_EmbeddingDim=16;p.ExtraDescription ="_0_10_dropout_16embeddding";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm1=true;p.UseBatchNorm2=false; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2;return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2=false;p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2;return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.3;p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_0_30_dropout";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2;return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.001;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_lr_0_001";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_16embeddding";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 32;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_32embeddding";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.1;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_0_10_dropout";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_1cycle";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.DropoutRate = 0.1;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_0_10_dropout_16embeddding";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.Pid_EmbeddingDim = 16;p.DropoutRate = 0.1;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_0_10_dropout_16embeddding_1cycle";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.Pid_EmbeddingDim = 16;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_16embeddding_1cycle";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ActivationFunctionAfterFirstDense =cudnnActivationMode_t.CUDNN_ACTIVATION_ELU ; p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_elu";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ActivationFunctionAfterFirstDense =cudnnActivationMode_t.CUDNN_ACTIVATION_RELU ; p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_relu";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ActivationFunctionAfterFirstDense =cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX ; p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_softmax";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ActivationFunctionAfterFirstDense =cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH ; p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_swish";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ActivationFunctionAfterFirstDense =cudnnActivationMode_t.CUDNN_ACTIVATION_TANH ; p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_tanh";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2;return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.001;p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_lr_0_001";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_16embeddding";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 32;p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_32embeddding";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.1;p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_0_10_dropout";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_1cycle";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.DropoutRate = 0.1;p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_0_10_dropout_16embeddding";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.Pid_EmbeddingDim = 16;p.DropoutRate = 0.1;p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_0_10_dropout_16embeddding_1cycle";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.Pid_EmbeddingDim = 16;p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_16embeddding_1cycle";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ActivationFunctionAfterFirstDense =cudnnActivationMode_t.CUDNN_ACTIVATION_ELU ; p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_elu";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ActivationFunctionAfterFirstDense =cudnnActivationMode_t.CUDNN_ACTIVATION_RELU ; p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_relu";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ActivationFunctionAfterFirstDense =cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX ; p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_softmax";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ActivationFunctionAfterFirstDense =cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH ; p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_swish";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ActivationFunctionAfterFirstDense =cudnnActivationMode_t.CUDNN_ACTIVATION_TANH ; p.UseBatchNorm1=true;p.UseBatchNorm2=true; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_tanh";return p;},


            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm1=true;p.UseBatchNorm2=false;  p.DropoutRate = 0; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_nodrop";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2=true;  p.DropoutRate = 0; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_nodrop";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm2=false;  p.DropoutRate = 0; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_nodrop";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.UseBatchNorm1=true;p.UseBatchNorm2=true;  p.DropoutRate = 0; p.ExtraDescription = p.UseBatchNorm1+"_"+p.UseBatchNorm2+"_nodrop";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.ExtraDescription =  p.DA.TimeSeriesDescription();return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate=0.5;p.ExtraDescription = "_drop_0_50";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate=0.3;p.ExtraDescription = "_drop_0_30";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate=0.1;p.ExtraDescription = "_drop_0_10";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.NO_NORMALIZATION;p.ExtraDescription = "_no_norm";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.Z_SCORE_NORMALIZATION;p.ExtraDescription = "_norm_z_score_V1";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.Z_SCORE_NORMALIZATION;p.ExtraDescription = "_norm_z_score_V2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputNormalizationType=CFM60NetworkBuilder.InputNormalizationEnum.DEDUCE_MEAN_NORMALIZATION;p.ExtraDescription = "_norm_deduce_mean";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.05, true, true, 0.20);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.05, true, true, 0.05);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.03, true, true, 0.20);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.03, true, true, 0.05);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.05, true, false, 0.20);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.05, true, false, 0.05);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.03, true, false, 0.20);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.03, true, false, 0.05);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.01, true, false, 0.10);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.01, true, false, 0.50);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.01, true, false, 1.00);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.03, true, false, 0.10);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.03, true, false, 0.50);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.03, true, false, 1.00);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.05, true, false, 0.10);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.05, true, false, 0.50);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE, 0.05, true, false, 1.00);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_ZERO, 0.01);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_ZERO, 0.03);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_ZERO, 0.05);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_ZERO, 0.07);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_ZERO, 0.10);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_ZERO, 0.01);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_MEAN, 0.03);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_MEAN, 0.05);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_MEAN, 0.07);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DA.WithTimeSeriesDataAugmentation(DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_MEAN, 0.10);p.ExtraDescription = p.DA.TimeSeriesDescription();return p;},


            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01;;p.Use_mean_abs_ret=true ;p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.ExtraDescription = "_lr_0_01_1cycle_divide200_mean_abs_ret";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;;p.Use_mean_abs_ret=true ;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200);p.ExtraDescription = "_lr_0_005_divide_200_mean_abs_ret";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;;p.Use_mean_abs_ret=true ;p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.ExtraDescription = "_lr_0_005_1cycle_divide200_mean_abs_ret";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01; p.NumEpochs=30;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200);p.ExtraDescription = "_lr_0_01_divide_200_NumEpochs_30";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01;p.NumEpochs=30;p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.ExtraDescription = "_lr_0_01_1cycle_divide200_NumEpochs_30";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;;p.NumEpochs=30;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200);p.ExtraDescription = "_lr_0_005_divide_200_NumEpochs_30";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;;p.NumEpochs=30;p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.ExtraDescription = "_lr_0_005_1cycle_NumEpochs_30";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200);p.ExtraDescription = "_lr_0_01_divide_200";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/100);p.ExtraDescription = "_lr_0_01_divide_100";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/500);p.ExtraDescription = "_lr_0_01_divide_500";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0);p.ExtraDescription = "_lr_0_01_0_min";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01; p.Config.WithOneCycleLearningRateScheduler(100,0.1);p.ExtraDescription = "_lr_0_01_1cycle_divide100";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01; p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.ExtraDescription = "_lr_0_01_1cycle_divide200";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01; p.Config.WithOneCycleLearningRateScheduler(500,0.1);p.ExtraDescription = "_lr_0_01_1cycle_divide500";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.02; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200);p.ExtraDescription = "_lr_0_02_divide_200";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.02; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/100);p.ExtraDescription = "_lr_0_02_divide_100";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.02; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/500);p.ExtraDescription = "_lr_0_02_divide_500";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.02; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0);p.ExtraDescription = "_lr_0_02_0_min";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.02; p.Config.WithOneCycleLearningRateScheduler(100,0.1);p.ExtraDescription = "_lr_0_02_1cycle_divide100";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.02; p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.ExtraDescription = "_lr_0_02_1cycle_divide200";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.02; p.Config.WithOneCycleLearningRateScheduler(500,0.1);p.ExtraDescription = "_lr_0_02_1cycle_divide500";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200);p.ExtraDescription = "_lr_0_005_divide_200";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/100);p.ExtraDescription = "_lr_0_005_divide_100";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/500);p.ExtraDescription = "_lr_0_005_divide_500";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, 0);p.ExtraDescription = "_lr_0_005_0_min";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005; p.Config.WithOneCycleLearningRateScheduler(100,0.1);p.ExtraDescription = "_lr_0_005_1cycle_divide100";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005; p.Config.WithOneCycleLearningRateScheduler(200,0.1);p.ExtraDescription = "_lr_0_005_1cycle_divide200";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005; p.Config.WithOneCycleLearningRateScheduler(500,0.1);p.ExtraDescription = "_lr_0_005_1cycle_divide500";return p;},

            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_mean_abs_ret=true;p.InitialLearningRate = 0.01; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200);p.ExtraDescription = "_lr_0_01_divide_200_mean_abs_ret";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_abs_ret_Volatility=true;p.InitialLearningRate = 0.01; p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2, p.InitialLearningRate/200);p.ExtraDescription = "_lr_0_01_divide_200__abs_ret_Volatility";return p;},


            //() => {var p = CFM60NetworkBuilder.Default();p.Use_ret_vol_start_and_end_only=true;p.ExtraDescription = "_ret_vol_start_and_end_only";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_mean_abs_ret=true;p.ExtraDescription = "_mean_abs_ret";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_abs_ret_Volatility=true;p.ExtraDescription = "_abs_ret_Volatility";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_mean_abs_ret=true;p.Use_abs_ret_Volatility=true;p.ExtraDescription = "_mean_abs_ret_abs_ret_Volatility";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_day_Divider=250;p.ExtraDescription = "_day_Divider_250";return p;},

            //() => {var p = CFM60NetworkBuilder.Default();p.LambdaL2RegularizationForEmbedding=0;p.ExtraDescription = "_0_LambdaL2RegularizationForEmbedding";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = p.LambdaL2RegularizationForEmbedding = 0.00001;p.ExtraDescription = "_l2_0_00001";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = p.LambdaL2RegularizationForEmbedding = 0.0001;p.ExtraDescription  = "_l2_0_0001";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_abs_ret = false;p.ExtraDescription = "_no_abs_ret";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_fraction_of_year = true;p.ExtraDescription = "_with_fraction_of_year";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_LS = true;p.ExtraDescription = "_with_LS";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_NLV = false;p.ExtraDescription = "_no_NLV";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.TimeSteps = 40;p.ExtraDescription = "_40timeSteps";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.TimeSteps = 30;p.ExtraDescription = "_30timeSteps";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.TimeSteps = 10;p.ExtraDescription = "_10timeSteps";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.ExtraDescription = "_lr_0_002";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.0005;p.ExtraDescription = "_lr_0_0005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.0001;p.ExtraDescription = "_lr_0_0001";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 4;p.ExtraDescription = "_Pid_EmbeddingDim_4";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 12;p.ExtraDescription = "_Pid_EmbeddingDim_12";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.ExtraDescription = "_Pid_EmbeddingDim_16";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NormalizeNLV = true;p.ExtraDescription = "_NormalizeNLV";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NormalizeNLV_V2 = true;p.ExtraDescription = "_NormalizeNLV_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NormalizeLS = true;p.Use_LS = true;p.ExtraDescription = "_with_LS_NormalizeLS";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NormalizeLS_V2 = true;p.Use_LS = true;p.ExtraDescription = "_with_LS_NormalizeLS_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ClipValueForGradients = 1000;p.DivideGradientsByTimeSteps = true;p.ExtraDescription = "_clip1000_divide_bytimesteps_V1";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = p.LambdaL2RegularizationForEmbedding = 0.00000;p.ExtraDescription = "_no_l2_batch4096_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 0;p.LambdaL2Regularization = p.LambdaL2RegularizationForEmbedding = 0.0001;p.ExtraDescription = "_l2_0_0001_no_embedding_batch4096_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 1;p.ExtraDescription = "_Pid_EmbeddingDim_1_V4";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_GRU_instead_of_LSTM = true;p.ExtraDescription = "_GRU_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_abs_ret = false;p.ExtraDescription = "_no_Use_abs_ret";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 0;p.ExtraDescription = "_Pid_EmbeddingDim_0_V4";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_fraction_of_year = true;p.ExtraDescription = "_with_fraction_of_year";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_abs_ret = false;p.Use_ret_vol = false;p.Use_ret_vol_Volatility = true;p.ExtraDescription = "_only_ret_vol_Volatility_no_abs_ret_no_tet_vol";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_GRU_instead_of_LSTM = true;p.ExtraDescription = "_GRU_V4";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.DenseUnits = 75;p.ExtraDescription = "_DenseUnits_75";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_day_Divider = 250f;p.ExtraDescription = "_divider_250";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_fraction_of_year = true;p.ExtraDescription = "_with_fraction_of_year";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_day_Divider = 250f;p.Use_fraction_of_year = true;p.ExtraDescription = "_divider_250_with_fraction_of_year";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.002;p.ExtraDescription = "_lr_0_002";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;p.ExtraDescription = "_lr_0_005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.0005;p.ExtraDescription = "_lr_0_0005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 4;p.ExtraDescription = "_Pid_EmbeddingDim_4";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 16;p.ExtraDescription = "_Pid_EmbeddingDim_16";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_ret_vol_Volatility = true;p.ExtraDescription = "_Use_ret_vol_Volatility";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_Bidirectional_RNN = false;p.ExtraDescription = "_mono_directional_RNN";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_GRU_instead_of_LSTM = true;p.ExtraDescription = "_GRU_instead_of_LSTM";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2, p.InitialLearningRate/10);p.ExtraDescription = "_min_lr_10";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2, p.InitialLearningRate/100);p.ExtraDescription = "_min_lr_100";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.05;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2, 0.01);p.ExtraDescription = "_lr_0_05_0_01";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.05;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2, 0.001);p.ExtraDescription = "_lr_0_05_0_001";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2, 0.001);p.ExtraDescription = "_lr_0_01_0_001";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2, 0.0001);p.ExtraDescription = "_lr_0_01_0_0001";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 30;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(p.NumEpochs,2);p.ExtraDescription = "_1Cycle_30epochs_V1";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(p.NumEpochs,2);p.ExtraDescription = "_1Cycle";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(p.NumEpochs,2);p.ExtraDescription = "_1Cycle_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "_V3";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.DivideGradientsByTimeSteps = true;p.ExtraDescription = "_bytimesteps";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ClipValueForGradients = 100;p.ExtraDescription = "_clip100";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ClipValueForGradients = 1000;p.ExtraDescription = "_clip1000";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ClipValueForGradients = 100;p.DivideGradientsByTimeSteps = true;p.ExtraDescription = "_clip100_divide_bytimesteps";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ClipValueForGradients = 1000;p.DivideGradientsByTimeSteps = true;p.ExtraDescription = "_clip1000_divide_bytimesteps";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = p.LambdaL2RegularizationForEmbedding = 0.0001;p.ExtraDescription = "_l2_0_0001";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = p.LambdaL2RegularizationForEmbedding = 0.0005;p.ExtraDescription = "_l2_0_0005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.LambdaL2Regularization = p.LambdaL2RegularizationForEmbedding = 0.00005;p.ExtraDescription = "_l2_0_00005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ClipValueForGradients = 1000;p.LambdaL2Regularization = p.LambdaL2RegularizationForEmbedding = 0.0001;p.ExtraDescription = "_l2_0_0001_clip1000";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ClipValueForGradients = 1000;p.LambdaL2Regularization = p.LambdaL2RegularizationForEmbedding = 0.0005;p.ExtraDescription = "_l2_0_0005_clip1000";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ClipValueForGradients = 1000;p.LambdaL2Regularization = p.LambdaL2RegularizationForEmbedding = 0.00005;p.ExtraDescription = "_l2_0_00005_clip1000";return p;},                //() => {var p = CFM60NetworkBuilder.Default();p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY);p.ExtraDescription = "_Custom_1_Identity";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_Tanh";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID);p.ExtraDescription = "_Custom_Sigmoid";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY);p.Use_fraction_of_year = true;p.ExtraDescription = "_Custom_Identity_Use_fraction_of_year";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY);p.InitialLearningRate = 0.005;p.ExtraDescription = "_Custom_Identity_Use_fraction_of_year_lr_0_005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY);p.Use_day_Divider = 250f;p.ExtraDescription = "_Custom_Identity_Use_fraction_of_year_day_divider_250";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY);p.Pid_EmbeddingDim = 8;p.ExtraDescription = "_Custom_Identity_Pid_EmbeddingDim_8";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Config.WithCyclicCosineAnnealingLearningRateScheduler(70,2);p.ExtraDescription = "_1Cycle";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Config.WithOneCycleLearningRateScheduler(100,0.1);p.ExtraDescription = "_1Cycle_100_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Config.WithOneCycleLearningRateScheduler(1000,0.1);p.ExtraDescription = "_1Cycle_1000_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Config.WithOneCycleLearningRateScheduler(10000,0.1);p.ExtraDescription = "_1Cycle_10000_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Config.WithOneCycleLearningRateScheduler(10,0.1);p.ExtraDescription = "_1Cycle_10_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(70,2);p.ExtraDescription = "_1Cycle_lr_0_005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;p.Config.WithOneCycleLearningRateScheduler(100,0.1);p.ExtraDescription = "_1Cycle_100_V2_lr_0_005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;p.Config.WithOneCycleLearningRateScheduler(1000,0.1);p.ExtraDescription = "_1Cycle_1000_V2_lr_0_005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;p.Config.WithOneCycleLearningRateScheduler(10000,0.1);p.ExtraDescription = "_1Cycle_10000_V2_lr_0_005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;p.Config.WithOneCycleLearningRateScheduler(10,0.1);p.ExtraDescription = "_1Cycle_10_V2_lr_0_005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.0005;p.Config.WithOneCycleLearningRateScheduler(100,0.1);p.ExtraDescription = "_1Cycle_100_V2_lr_0_0005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.0005;p.Config.WithOneCycleLearningRateScheduler(1000,0.1);p.ExtraDescription = "_1Cycle_1000_V2_lr_0_0005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.0005;p.Config.WithOneCycleLearningRateScheduler(10000,0.1);p.ExtraDescription = "_1Cycle_10000_V2_lr_0_0005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.0005;p.Config.WithOneCycleLearningRateScheduler(10,0.1);p.ExtraDescription = "_1Cycle_10_V2_lr_0_0005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.TimeSteps = 40;p.ExtraDescription = "_TimeSteps_40";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.TimeSteps = 30;p.ExtraDescription = "_TimeSteps_30";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.TimeSteps = 10;p.ExtraDescription = "_TimeSteps_10";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.01;p.ExtraDescription = "_lr_0_01";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;p.ExtraDescription = "_lr_0_005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim = 6;p.ExtraDescription = "_Pid_EmbeddingDim_6";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "_V2";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "_V3";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "_V4";return p;},
            //() => { var p = CFM60NetworkBuilder.Default(); p.WithSpecialEndV1 = true; p.ExtraDescription = "_WithSpecialEndV1"; return p; },
            //() => { var p = CFM60NetworkBuilder.Default(); p.HiddenSize = 64; p.ExtraDescription = "_HiddenSize_64"; return p; },
            //() => { var p = CFM60NetworkBuilder.Default(); p.HiddenSize = 32; p.ExtraDescription = "_HiddenSize_32"; return p; },
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_pid_y_vol = false;p.ExtraDescription = "_no_Use_pid_y_vol";return p;},
            //() => {var p = CFM60NetworkBuilder.Default(); p.Use_y_LinearRegressionEstimate = false;p.Use_pid_y_vol = false;p.ExtraDescription = "_no_Use_y_LinearRegressionEstimate_Use_pid_y_vol";return p;},
            ////() => {var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.5;p.ExtraDescription = "_default2_drop_0_50";return p;},
            ////() => {var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.0;p.ExtraDescription = "_default2_no_drop";return p;},
            ////() => {var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.4;p.ExtraDescription = "_default2_drop_0_40";return p;},
            ////() => {var p = CFM60NetworkBuilder.Default();p.Use_Bidirectional_RNN=false;p.ExtraDescription = "_default2_no_Bidirectional_RNN";return p;},
            ////() => {var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim=8;p.ExtraDescription = "_default2_Pid_EmbeddingDim_8";return p;},
            ////() => {var p = CFM60NetworkBuilder.Default();p.Pid_EmbeddingDim=16;p.ExtraDescription = "_default2_Pid_EmbeddingDim_16";return p;},
            ////() => {var p = CFM60NetworkBuilder.Default();p.Use_abs_ret=false;p.ExtraDescription = "_default2_no_abs_ret";return p;},
            ////() => {var p = CFM60NetworkBuilder.Default();p.Use_LS=false;p.ExtraDescription = "_default2_no_Use_LS";return p;},
            ////() => {var p = CFM60NetworkBuilder.Default();p.Use_NLV=false;p.ExtraDescription = "_default2_no_Use_NLV";return p;},
            ////() => {var p = CFM60NetworkBuilder.Default();p.InitialLearningRate = 0.005;p.ExtraDescription = "_default2_lr_0_005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.TimeSteps = 10;p.ExtraDescription = "_TimeSteps_10";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.TimeSteps = 30;p.ExtraDescription = "_TimeSteps_30";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.ExtraDescription = "_default";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_pid_y_vol = false;p.ExtraDescription = "_no_Use_pid_y_vol";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_y_LinearRegressionEstimate = false;p.Use_pid_y_vol = false;p.ExtraDescription = "_no_Use_y_LinearRegressionEstimate_Use_pid_y_vol";return p;},
            //public bool Use_y_LinearRegressionEstimate { get; set; } = true; //validated on 19-jan-2021: -0.0501 (with other changes)
            //public bool Use_pid_y_vol { get; set; } = true; //validated on 17-jan-2021: -0.0053
            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_GRU_instead_of_LSTM = true;p.NumEpochs = 30;p.ExtraDescription = "_GRU";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_Bidirectional_RNN = false;p.NumEpochs = 30;p.ExtraDescription = "_MonoDirectional";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_Bidirectional_RNN = false;p.Use_GRU_instead_of_LSTM = true;p.NumEpochs = 30;p.ExtraDescription = "_GRU_MonoDirectional";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY);p.ExtraDescription = "_Custom_1_Identity";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(5f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_5_TANH";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_1_TANH";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(3f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_3_TANH";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(5f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_5_TANH_no_y_LinearRegressionEstimate";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_1_TANH_no_y_LinearRegressionEstimate";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(3f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_3_TANH_no_y_LinearRegressionEstimate";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_pid_y_vol=false;p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(5f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_5_TANH_no_y_LinearRegressionEstimate_no_y_vol";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_pid_y_vol=false;p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_1_TANH_no_y_LinearRegressionEstimate_no_y_vol";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_pid_y_vol=false;p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(3f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_3_TANH_no_y_LinearRegressionEstimate_no_y_vol";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_day = true;p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(5f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_5_TANH_no_y_LinearRegressionEstimate_with_day";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_day = true;p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_1_TANH_no_y_LinearRegressionEstimate_with_day";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_day = true;p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(3f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_3_TANH_no_y_LinearRegressionEstimate_with_day";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_pid_y_vol=false;p.Use_day = true;p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(5f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_5_TANH_no_y_LinearRegressionEstimate_no_y_vol_with_day";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_pid_y_vol=false;p.Use_day = true;p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(1f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_1_TANH_no_y_LinearRegressionEstimate_no_y_vol_with_day";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.Use_pid_y_vol=false;p.Use_day = true;p.Use_y_LinearRegressionEstimate = false;p.NumEpochs = 30;p.WithCustomLinearFunctionLayer(3f, cudnnActivationMode_t.CUDNN_ACTIVATION_TANH);p.ExtraDescription = "_Custom_3_TANH_no_y_LinearRegressionEstimate_no_y_vol_with_day";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 1;p.NumEpochs = 30;p.BatchSize=4096;p.InitialLearningRate = 0.00025;p.ExtraDescription = "_InputSize1";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 2;p.NumEpochs = 30;p.BatchSize=4096;p.InitialLearningRate = 0.00025;p.ExtraDescription = "_InputSize2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 3;p.NumEpochs = 30;p.BatchSize=4096;p.InitialLearningRate = 0.00025;p.ExtraDescription = "_InputSize3";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.NumEpochs = 30;p.BatchSize=4096;p.InitialLearningRate = 0.00025;p.ExtraDescription = "_InputSize4";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.NumEpochs = 30;p.BatchSize=1024;p.InitialLearningRate = 0.01;p.ExtraDescription = "_InputSize5_1";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.ExtraDescription = "_InputSize4_64";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.0;p.ExtraDescription = "_InputSize5_64_NoDropoutRate";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_64_DropoutRate_0_2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.5;p.ExtraDescription = "_InputSize5_64_DropoutRate_0_5";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.LSTMLayersReturningFullSequence =2;p.ExtraDescription = "_InputSize5_64_NoDropoutRate_LSTMLayersReturningFullSequence_2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.LSTMLayersReturningFullSequence =2;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_64_DropoutRate_0_2_LSTMLayersReturningFullSequence_2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.LSTMLayersReturningFullSequence =2;p.DropoutRate = 0.5;p.ExtraDescription = "_InputSize5_64_DropoutRate_0_5_LSTMLayersReturningFullSequence_2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.0;p.ExtraDescription = "_InputSize4_64_NoDropoutRate";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize4_64_DropoutRate0_2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.ExtraDescription = "_InputSize4_128_NoDrop";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.ExtraDescription = "_InputSize5_128_NoDrop";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.UseBatchNorm = true;p.ExtraDescription = "_InputSize4_128_BatchNorm";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.UseBatchNorm = true;p.ExtraDescription = "_InputSize5_128_BatchNorm";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize4_128_Drop_0_2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_128_Drop_0_2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.PercentageInTraining = 0.68;p.ExtraDescription = "_InputSize5_128_Drop_0_2_0_68_InTraining";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.PercentageInTraining = 0.50;p.ExtraDescription = "_InputSize5_128_Drop_0_2_0_50_InTraining";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.001;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_128_Drop_0_2_lr_0_001";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.0005;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_128_Drop_0_2_lr_0_0005";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.0001;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_128_Drop_0_2_lr_0_0001";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.00005;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_128_Drop_0_2_lr_0_00005";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.00001;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_128_Drop_0_2_lr_0_00001";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.ActivationFunctionAfterDense = cudnnActivationMode_t.CUDNN_ACTIVATION_RELU;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_128_Drop_0_2_RELU";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.ActivationFunctionAfterDense = cudnnActivationMode_t.CUDNN_ACTIVATION_RELU;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_128_Drop_0_2_TANH";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.DenseUnits=200;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_128_Drop_0_2_200Units";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.DenseUnits=300;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.ExtraDescription = "_InputSize5_128_Drop_0_2_300Units";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.LSTMLayersReturningFullSequence =2;p.ExtraDescription = "_InputSize4_128_Drop_0_2_2LSTMLayersReturningFullSequence";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.LSTMLayersReturningFullSequence =2;p.ExtraDescription = "_InputSize5_128_Drop_0_2_2LSTMLayersReturningFullSequence";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.LSTMLayersReturningFullSequence =3;p.ExtraDescription = "_InputSize4_128_Drop_0_2_3LSTMLayersReturningFullSequence";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.DropoutRate = 0.2;p.LSTMLayersReturningFullSequence =3;p.ExtraDescription = "_InputSize5_128_Drop_0_2_3LSTMLayersReturningFullSequence";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.UseBatchNorm = true;p.LSTMLayersReturningFullSequence =2;p.ExtraDescription = "_InputSize4_128_BatchNorm_2LSTMLayersReturningFullSequence";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.HiddenSize = 128;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.005;p.UseBatchNorm = true;p.LSTMLayersReturningFullSequence =2;p.ExtraDescription = "_InputSize5_128_BatchNorm_2LSTMLayersReturningFullSequence";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 4;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.001;p.ExtraDescription = "_InputSize4_64_001";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.NumEpochs = 150;p.BatchSize=1024;p.InitialLearningRate = 0.001;p.ExtraDescription = "_InputSize5_64_001";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.NumEpochs = 30;p.BatchSize=1024;p.InitialLearningRate = 0.001;p.ExtraDescription = "_InputSize5_3";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InputSize = 5;p.NumEpochs = 30;p.BatchSize=1024;p.InitialLearningRate = 0.0001;p.ExtraDescription = "_InputSize5_4";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.InitialLearningRate = 0.0005;p.ExtraDescription = "_lr_0_0005";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.InitialLearningRate = 0.0005;p.DenseUnits = 300;p.ExtraDescription = "_lr_0_0005_300DenseUnits";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.InitialLearningRate = 0.0005;p.DenseUnits = 200;p.ExtraDescription = "_lr_0_0005_200DenseUnits";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.ExtraDescription = "_lr_0_0005";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.PercentageInTraining = 0.68;p.ExtraDescription = "_lr_0_0005_0_68";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.NormalizeLS=true;p.ExtraDescription = "_lr_0_0005_normalize_ls";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.NormalizeNLV=true;p.ExtraDescription = "_lr_0_0005_normalize_nlv";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.NormalizeLS=true;p.NormalizeNLV=true;p.ExtraDescription = "_lr_0_0005_normalize_ls_and_nlv";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.LinearLayer_a=1f;p.LinearLayer_b=-1.97f;p.ExtraDescription = "_lr_0_0005_linear_1_97";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.NormalizeLS=true;p.NormalizeNLV=true;p.LinearLayer_a=1f;p.LinearLayer_b=-1.97f;p.ExtraDescription = "_lr_0_0005_linear_1_97_normalize_ls_and_nlv";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.PercentageInTraining = 0.68;p.NormalizeLS=true;p.NormalizeNLV=true;p.LinearLayer_a=1f;p.LinearLayer_b=-1.97f;p.ExtraDescription = "_lr_0_0005_linear_1_97_normalize_ls_and_nlv_0_98";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.Use_LS=false;p.ExtraDescription = "_lr_0_0005_no_ls";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.Use_NLV=false;p.ExtraDescription = "_lr_0_0005_no_nlv";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.Use_LS=false;p.Use_NLV=false;p.ExtraDescription = "_lr_0_0005_no_ls_no_nlv";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0005;p.Use_abs_ret=false;p.ExtraDescription = "_lr_0_0005_no_abs_ret";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.PercentageInTraining = 0.95;p.ExtraDescription = "_0_95_in_training";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_day=false;p.ExtraDescription = "_no_day";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.ModuloForDay=250;p.ExtraDescription = "_ModuloForDay_250";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.WithConv1D(1, ConvolutionLayer.PADDING_TYPE.SAME, false);p.ExtraDescription = "_Conv1D";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.WithConv1D(1, ConvolutionLayer.PADDING_TYPE.SAME);p.ExtraDescription = "_Conv1D_BatchNorm";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.WithConv1D(1, ConvolutionLayer.PADDING_TYPE.SAME, false);p.ExtraDescription = "_Conv1D_Relu";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.WithConv1D(1, ConvolutionLayer.PADDING_TYPE.SAME, true);p.ExtraDescription = "_Conv1D_BatchNorm_Relu";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.InitialLearningRate=0.001;p.ExtraDescription = "_lr_0_001";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.InitialLearningRate=0.0001;p.ExtraDescription = "_lr_0_0001";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.LinearLayer_a=1/0.909245f;p.LinearLayer_b=+1.958691f/0.909245f;p.ExtraDescription = "_linear_1_95";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.SplitTrainingAndValidationBasedOnDays=false;p.ExtraDescription = "_SplitTrainingAndValidation_Random";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.NormalizeLS_V2=true;p.ExtraDescription = "_NormalizeLS_V2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.NormalizeNLV_V2=true;p.ExtraDescription = "_NormalizeNLV_V2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.NormalizeNLV_V2=true;p.NormalizeLS_V2=true;p.ExtraDescription = "_NormalizeLS_V2_NormalizeNLV_V2";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.LinearLayer_a=1/0.909245f;p.LinearLayer_b=+1.958691f/0.909245f;p.NormalizeNLV_V2=true;p.NormalizeLS_V2=true;p.ExtraDescription = "_NormalizeLS_V2_NormalizeNLV_V2_linear_1_95";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.DropoutRate = 0.1;p.ExtraDescription = "_drop0_0_10";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.DropoutRate = 0.15;p.ExtraDescription = "_drop0_0_15";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.DropoutRate = 0.25;p.ExtraDescription = "_drop0_0_25";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.DropoutRate = 0.3;p.ExtraDescription = "_drop0_0_30";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.NormalizeLS=true;p.ExtraDescription = "_NormalizeLS";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.NumEpochs = 150;p.NormalizeNLV=true;p.ExtraDescription = "_NormalizeNLV";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_pid_y_vol = true;p.ExtraDescription = "_Use_pid_y_vol";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.1;p.ExtraDescription = "_drop0_0_10";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.LinearLayer_a=0.909245f;p.LinearLayer_b=+1.958691f;p.ExtraDescription = "_linear_1_95";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate=0.002;p.ExtraDescription = "_lr_0_002";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.InitialLearningRate=0.0005;p.ExtraDescription = "_lr_0_0005";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_pid_y_vol = true;p.SplitTrainingAndValidationBasedOnDays=false;p.ExtraDescription = "_SplitTrainingAndValidation_Random_Use_pid_y_vol";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.1;p.SplitTrainingAndValidationBasedOnDays=false;p.ExtraDescription = "_SplitTrainingAndValidation_Random_drop_0_10";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.WithConv1D(1, ConvolutionLayer.PADDING_TYPE.SAME, true);p.ExtraDescription = "_Conv1D_BatchNorm_Relu";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "_default";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.15;p.ExtraDescription = "_drop0_0_15";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.DropoutRate = 0.0;p.ExtraDescription = "_nodrop";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_day=false;p.ExtraDescription = "_do_not_use_day";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.Use_day=false;p.PercentageInTraining = 0.68;p.ExtraDescription = "_do_not_use_day_split_0_68";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.SplitTrainingAndValidationBasedOnDays=false;p.ExtraDescription = "_SplitTrainingAndValidation_Random";return p;},
            //() =>{var p = CFM60NetworkBuilder.Default();p.ExtraDescription = "_default";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.ExtraDescription = "_default";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_pid_y_vol=false;p.ExtraDescription = "_no_pid_y_vol";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_Christmas_flag=false;p.ExtraDescription = "_no_Christmas_flag";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_EndOfYear_flag=false;p.ExtraDescription = "_no_EndOfYear_flag";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_EndOfTrimester_flag=false;p.ExtraDescription = "_no_EndOfTrimester_flag";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.SplitTrainingAndValidationBasedOnDays=false;p.ExtraDescription = "_SplitTrainingAndValidationBasedOnDays_random";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.PercentageInTraining = 0.68;p.ExtraDescription = "_split_0_68";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.DropoutRate = 0.1;p.ExtraDescription = "_drop_0_10";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.DropoutRate = 0.3;p.ExtraDescription = "_drop_0_30";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.DropoutRate = 0.0;p.ExtraDescription = "_nodrop";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_day=false;p.ExtraDescription = "_no_day";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_NLV=false;p.ExtraDescription = "_no_NLV";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_LS=false;p.ExtraDescription = "_no_LS";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.DenseUnits=50;p.ExtraDescription = "_DenseUnits_050";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.DenseUnits=100;p.ExtraDescription = "_DenseUnits_100";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.DenseUnits=300;p.ExtraDescription = "_DenseUnits_300";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.ActivationFunctionAfterDense=cudnnActivationMode_t.CUDNN_ACTIVATION_TANH;p.ExtraDescription = "_Dense_then_TANH";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_day=true;p.ExtraDescription = "_Use_day";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.Use_LS=false;p.Use_NLV=false;p.ExtraDescription = "_no_LS_no_NLV";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.LSTMLayersReturningFullSequence=2;p.ExtraDescription = "_2LSTMLayersReturningFullSequence";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.0001; p.LSTMLayersReturningFullSequence=2;p.ExtraDescription = "_2LSTMLayersReturningFullSequence_lr_0_0001";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.00005; p.LSTMLayersReturningFullSequence=2;p.ExtraDescription = "_2LSTMLayersReturningFullSequence_lr_0_00005";return p;},
            //() => {var p = CFM60NetworkBuilder.Default();p.NumEpochs = 70;p.InitialLearningRate = 0.00001; p.LSTMLayersReturningFullSequence=2;p.ExtraDescription = "_2LSTMLayersReturningFullSequence_lr_0_00001";return p;},
            #endregion
            };

            networkMetaParameters = SharpNet.Utils.Repeat(networkMetaParameters, 2);

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
