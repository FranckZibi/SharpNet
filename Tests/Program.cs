using System.Collections.Generic;
using System.Threading.Tasks;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Models;
using SharpNet.Networks;
using System;

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
            SharpNet.Utils.ConfigureGlobalLog4netProperties(NetworkSample.DefaultWorkingDirectory, "SharpNet");
            SharpNet.Utils.ConfigureThreadLog4netProperties(NetworkSample.DefaultWorkingDirectory, "SharpNet");
            //235*200 : 34.28% errors (30 epochs)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_200_235_20200611_0716_30");
            //235*200 :  11.46% errors (70 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_200_235_20200615_0848_70");
            //235*200 :  1.58% errors (150 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_200_235_20200615_0848_150");
            //235*200 :  0.67% errors (294 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_200_235_20200615_0848_294");
            //235*200 :  0.65% errors (310 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_200_235_20200615_0848_310");
            //470*400 :  1.11% errors (150 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_400_470_20200620_1427_310");
            //470*400 :  0.48% errors (310 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Imagenet_400_470_20200620_1427_310");
            //470*400 :  0.18% errors (630 epochs, lr=0.02)
            //new CancelDatabase().UpdateSuggestedCancelForAllDatabase("efficientnet-b0_Cancel_400_470_20200715_2244_630");
            //new CancelDatabase().CreatePredictionFile(Path.Combine(ImageDatabaseManagementPath, "Prediction_def.csv"));return;
            //QRT72NetworkSample.Run(new QRT72Hyperparameters());
            //var builderIDM = new CancelDatabase(System.IO.Path.Combine(NetworkSample.DefaultDataDirectory, "Cancel"));
            //var builder = new CancelDatabase(System.IO.Path.Combine(NetworkSample.DefaultDataDirectory, "Cancel"));
            //builder.CreateIDM(Path.Combine(ImageDatabaseManagementPath, "Duplicates.csv"), e => !string.IsNullOrEmpty(e.CancelComment));
            //builder.AddAllFilesInPath(@"C:\SA\AnalyzedPictures");
            //using var network = Network.ValueOf(Path.Combine(NetworkSample.DefaultLogDirectory, "Cancel", "efficientnet-b0_DA_SVHN_20200526_1736_70.txt"));
            //using var dataSet = builder.ExtractDataSet(e => e.HasExpectedWidthHeightRatio(xShape[3] / ((double)xShape[2]), 0.05), root);
            //network.Predict(dataSet, Path.Combine(ImageDatabaseManagementPath, "Prediction.csv"));
            //new TestCpuTensor().TestMaxPooling3D();return;
            //new NonReg.ParallelRunWithTensorFlow().Test_Speed_MultiHeadAttention(); return;
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

            //SharpNet.Datasets.EffiSciences95.EffiSciences95Utils.InferenceUnlabeledEffiSciences95("C:/Projects/Challenges/EffiSciences95/", "F1040C26F7_FULL", true); return;
            //SharpNet.Datasets.EffiSciences95.EffiSciences95Utils.Launch_HPO(30, -1); return;
            
            //SharpNet.Networks.Transformers.TextTransformersUtils.Run();  return;
            //SharpNet.Networks.Transformers.MyNameIsGrootUtils.Run();  return;

            //SharpNet.Datasets.PlumeLabs88.PlumeLabs88Utils.Run();
            //SharpNet.Datasets.CFM84.CFM84Utils.Run();return;
            //SharpNet.Datasets.Biosonar85.Biosonar85Utils.Run();return;
            //SharpNet.Datasets.SNCF89.SNCF89Utils.Run(); return;
            //SharpNet.Datasets.BNP104.BNP104Utils.Run(); return;
        }


        #region DenseNet Training
        private static void DenseNetTests()
        {
            var networkGeometries = new List<Action<NetworkSample, int>>
            {
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(()=> DenseNetNetworkSample.DenseNet_12_40(x));},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(()=> DenseNetNetworkSample.DenseNetBC_12_100(x));},
            /*(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.DenseNet_Fast_CIFAR10);},
            (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.DenseNet_12_10_CIFAR10);},
            (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.DenseNet_12_40);},
            (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.DenseNetBC_12_40);},
             */
        };

            var networkMetaParameters = new List<Func<NetworkSample>>
            {
                //(p) =>{p.UseNesterov = false; p.Config.NumEpochs = 50; p.Config.ExtraDescription = "_50Epoch_no_nesterov";},
                //(p) =>{p.UseAdam = true; p.Config.NumEpochs = 5; p.Config.ExtraDescription = "_50Epoch_Adam";},
                //(p) =>{p.SaveNetworkStatsAfterEachEpoch = true; p.Config.ExtraDescription = "_Adam_with_l2_inConv";},
                //(p) =>{p.SaveNetworkStatsAfterEachEpoch = false;p.SaveLossAfterEachMiniBatch = false;p.UseAdam = true;p.UseNesterov = false;p.Config.BatchSize = -1;p.Config.NumEpochs = 150; p.Config.ExtraDescription = "_Adam";},
                //(p) =>{ p.Config.ExtraDescription = "_OrigPaper";},

                () =>{var p = DenseNetNetworkSample.CIFAR10();p.NumEpochs = 240;p.BatchSize = -1;p.WithSGD(0.9,true); p.CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow;p.CutoutPatchPercentage = 0.0;return p;},
                () =>{var p = DenseNetNetworkSample.CIFAR10();p.NumEpochs = 240;p.BatchSize = -1;p.WithSGD(0.9,false); p.CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow;p.CutoutPatchPercentage = 0.0;return p;},


                //(p) =>{p.Config.NumEpochs = 300;p.Config.BatchSize = -1;p.CutoutPatchPercentage = 0.25;p.Config.ExtraDescription = "_200Epochs_L2InDense_CutoutPatchPercentage0_25;},
                //(p) =>{p.Config.NumEpochs = 300;p.Config.BatchSize = -1;p.CutoutPatchPercentage = 0.0;p.Config.ExtraDescription = "_200Epochs_L2_InDense_CutoutPatchPercentage0";},
                //(p) =>{p.Config.NumEpochs = 200;p.Config.WithSGD(0.9,false);p.Config.ExtraDescription = "_200Epochs_NoNesterov";},

            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries);
        }
        #endregion


        #region EfficientNet Cancel DataSet Training

        private static void EfficientNetTests_Cancel(bool useMultiGpu)
        {
            const int channels = 3;
            const int targetHeight = 470; const int targetWidth = 400;var batchSize = 20;const double defaultInitialLearningRate = 0.01;
            //var targetHeight = 235;var targetWidth = 200;var batchSize = 80;var defaultInitialLearningRate = 0.02;
            //var targetHeight = 118;var targetWidth = 100;var batchSize = 300;var defaultInitialLearningRate = 0.05;
            //var targetHeight = 59;var targetWidth = 50;var batchSize = 1200;var defaultInitialLearningRate = ?;
            
            if (useMultiGpu) { batchSize *= GPUWrapper.GetDeviceCount(); }


            const int numEpochs = 150;

            var networkMetaParameters = new List<Func<EfficientNetNetworkSample>>
            {
                () =>{var p = EfficientNetNetworkSample.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.BatchSize = batchSize;p.NumEpochs = numEpochs;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.DataAugmentationType =ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_SVHN;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_Cancel_Augment_SVHN"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.CutoutPatchPercentage=0 ;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_Cancel_NoCutout"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.Config.WithSGD(0.9, true);p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_nesterov_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.DataAugmentationType =ImageDataGenerator.DataAugmentationEnum.DEFAULT;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_Cancel_Augment_DEFAULT"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.025;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_lr_0_25_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.015;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_lr_0_15_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.CutoutPatchPercentage=0.05 ;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_Cancel_Cutout_0_05_"+targetWidth+"_"+targetHeight;return p;},


                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.01;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_Cancel_lr_0_01_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.04;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_Cancel_lr_0_04_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.CutoutPatchPercentage=0 ;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_Cancel_NoCutout"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.CutoutPatchPercentage=0.05 ;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_Cancel_0_05_Cutout"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.DA.DataAugmentationType =ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_IMAGENET;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_Cancel_AutoAugment_ImageNet"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.015;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_lr_0_15_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = 0.025;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_lr_0_25_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.Config.WithSGD(0.9, true);p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_nesterov_Cancel_"+targetWidth+"_"+targetHeight;return p;},
                //() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.BatchNormEpsilon=0.0001;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_BatchNormEpsilon_0_0001_Cancel_"+targetWidth+"_"+targetHeight;return p;},


                //() =>{var p = EfficientNetBuilder.EfficientNet_Cancel();p.InitialLearningRate = 0.01;p.Config.BatchSize = batchSize;p.Config.NumEpochs = 5;p.Config.ExtraDescription = "_0_01";return p;},
                //() =>{var p = EfficientNetBuilder.EfficientNet_Cancel();p.Config.BatchSize = batchSize;p.Config.NumEpochs = 150;p.Config.ExtraDescription = "";return p;},
                //() =>{var p = EfficientNetBuilder.EfficientNet_Cancel();p.InitialLearningRate = 0.01;p.Config.BatchSize = batchSize;p.Config.NumEpochs = 30;p.Config.ExtraDescription = "_0_01";return p;},
                //() =>{var p = EfficientNetBuilder.EfficientNet_Cancel();p.InitialLearningRate = 0.30;p.Config.BatchSize = batchSize;p.Config.NumEpochs = 30;p.Config.ExtraDescription = "_0_30";return p;},
                //() =>{var p = EfficientNetBuilder.EfficientNet_Cancel();p.InitialLearningRate = 0.10;p.Config.BatchSize = batchSize;p.Config.NumEpochs = 30;p.Config.ExtraDescription = "_0_10";return p;},
            };


            //networkMetaParameters.Clear();for (int i = 0; i < 4; ++i){int j = i;networkMetaParameters.Add(() =>{var p = EfficientNetBuilder.Cancel();p.InitialLearningRate = defaultInitialLearningRate;p.Config.BatchSize = batchSize;p.Config.NumEpochs = numEpochs;p.Config.ExtraDescription = "_V"+j+(useMultiGpu?"_MultiThreaded":"");return p;});}

            // ReSharper disable once ConditionIsAlwaysTrueOrFalse
            var networkGeometries = new List<Action<EfficientNetNetworkSample, int>>
            {
                (p,gpuDeviceId) =>{p.SetResourceId(gpuDeviceId);Train_Cancel_EfficientNet(p, channels, targetHeight, targetWidth);},
            };

            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries, useMultiGpu);
        }
        private static void Train_Cancel_EfficientNet(EfficientNetNetworkSample p, int channels, int targetHeight, int targetWidth)
        {
            var database = new CancelDatabase();
            //TODO Test with selection of only matching size input in the training set
            //using var dataset = database.ExtractDataSet(e=>e.HasExpectedWidthHeightRatio(targetWidth / ((double)targetHeight), 0.05) && CancelDatabase.IsValidNonEmptyCancel(e.Cancel), ResizeStrategyEnum.ResizeToHeightAndWidthSizeKeepingSameProportionWith5PercentTolerance);
            using var dataset = database.ExtractDataSet(e=>CancelDatabase.IsValidNonEmptyCancel(e.Cancel), ResizeStrategyEnum.BiggestCropInOriginalImageToKeepSameProportion);
            using var trainingAndValidation = dataset.SplitIntoTrainingAndValidation(0.9, false, false); //90% for training,  10% for validation

            var rootPrediction = CancelDatabase.Hierarchy.RootPrediction();
            using var network = p.EfficientNetB0(DenseNetNetworkSample.CancelWorkingDirectory, true, "", new[] { channels, targetHeight, targetWidth }, rootPrediction.Length);
            network.SetSoftmaxWithHierarchy(rootPrediction);
            //network.LoadParametersFromH5File(@System.IO.Path.Combine(NetworkSample.DefaultLogDirectory, "Cancel", "efficientnet-b0_Imagenet_400_470_20200620_1427_310.h5"), NetworkSample.CompatibilityModeEnum.TensorFlow1);

            //using var network =Network.ValueOf(@System.IO.Path.Combine(NetworkSample.DefaultLogDirectory, "Cancel", "efficientnet-b0_Cancel_400_470_20200713_1809_580.txt"));

            //network.FindBestLearningRate(cancelDataset, 1e-5, 10, p.Config.BatchSize);return;
            Model.Log.Debug(database.Summary());
            network.Fit(trainingAndValidation.Training, trainingAndValidation.Test);
        }
        #endregion


        #region EfficientNet Training
        private static void EfficientNetTests()
        {
            const bool useMultiGpu = true;
            var networkGeometries = new List<Action<EfficientNetNetworkSample, int>>
            {
                //(p,gpuDeviceId) =>{p.SetResourceId(gpuDeviceId);p.WeightForTransferLearning = "imagenet";p.Config.LastLayerNameToFreeze = "top_dropout";p.Config.ExtraDescription += "_only_dense";Train_CIFAR10_EfficientNet(p);},
                //(p,gpuDeviceId) =>{p.SetResourceId(gpuDeviceId);p.WeightForTransferLearning = "imagenet";p.Config.LastLayerNameToFreeze = "block7a_project_bn";p.Config.ExtraDescription += "_all_top";Train_CIFAR10_EfficientNet(p);},
                (p,gpuDeviceId) =>{p.SetResourceId(gpuDeviceId);Train_CIFAR10_EfficientNet(p);},
            };

            var networkMetaParameters = new List<Func<EfficientNetNetworkSample>>
            {
                () =>{var p = EfficientNetNetworkSample.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.30;p.NumEpochs = 1;return p;},
                
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.Config.BatchSize = -1;p.InitialLearningRate = 0.30;p.Config.NumEpochs = 30;p.Config.ExtraDescription = "_lr_0_30_batchAuto";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.Config.BatchSize = 64;p.InitialLearningRate = 0.01;p.Config.NumEpochs = 30;p.Config.ExtraDescription = "_lr_0_30_batch64_test";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.Config.BatchSize = 32;p.InitialLearningRate = 0.01;p.Config.NumEpochs = 30;p.Config.ExtraDescription = "_lr_0_30_batch32_test";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.Config.BatchSize = 16;p.InitialLearningRate = 0.30;p.Config.NumEpochs = 2;p.Config.ExtraDescription = "_lr_0_30_batch16_test";return p;},
                
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.Config.BatchSize = 16;p.InitialLearningRate = 0.30;p.Config.NumEpochs = 30;p.Config.ExtraDescription = "_lr_0_30_batchAuto";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.Config.BatchSize = 32;p.InitialLearningRate = 0.01;p.Config.NumEpochs = 150;p.Config.ExtraDescription = "_lr_0_01_batchAuto_zoom_150epochs";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.Config.BatchSize = -1;p.InitialLearningRate = 0.001;p.Config.NumEpochs = 30;p.WeightForTransferLearning = "imagenet";p.Config.LastLayerNameToFreeze = "top_dropout";p.Config.ExtraDescription = "_lr_0_001_batchAuto_zoom7_30epochs_only_probs";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.Config.BatchSize = -1;p.InitialLearningRate = 0.001;p.Config.NumEpochs = 30;p.WeightForTransferLearning = "imagenet";p.Config.LastLayerNameToFreeze = "block7a_project_bn";p.Config.ExtraDescription = "_lr_0_001_batchAuto_zoom7_30epochs_only_top";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.Config.BatchSize = -1;p.InitialLearningRate = 0.10;p.Config.NumEpochs = 30;p.Config.ExtraDescription = "_lr_0_10_batchAuto_zoom_30epochs";return p;},
            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries, useMultiGpu);
        }
        private static void Train_CIFAR10_EfficientNet(EfficientNetNetworkSample p)
        {
            using var cifar10 = new CIFAR10DataSet();
            using var network = p.EfficientNetB0_CIFAR10(p.WeightForTransferLearning, CIFAR10DataSet.Shape_CHW);
            Log.Info(network.ToString());
            network.Fit(cifar10.Training, cifar10.Test);
        }
        #endregion

        #region WideResNet Training
        private static void WideResNetTests()
        {
            const bool useMultiGpu = false;
            var networkGeometries = new List<Action<WideResNetNetworkSample, int>>
            {
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 16,4);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 16,8);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 40,4);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 28,8);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 16,10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10_WRN(x, 28,10);},
            };

            //var batchSize = useMultiGpu ? 256 : 128;
            var networkMetaParameters = new List<Func<WideResNetNetworkSample>>
            {
                () =>{var p = WideResNetNetworkSample.CIFAR10();p.NumEpochs = 310;p.BatchSize = 128;return p;},
                
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.Config.BatchSize = 128;p.Config.ExtraDescription = "_BatchSize128";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.Config.BatchSize = batchSize;p.Config.ExtraDescription = "_MultiGPU";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.Config.BatchSize = 128;p.Config.ExtraDescription = "_BatchSize128";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.Config.BatchSize = 128;p.Config.ExtraDescription = "_BatchSize128";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.Config.BatchSize = 64;p.Config.ExtraDescription = "_BatchSize64";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.Config.BatchSize = -1;p.Config.ExtraDescription = "_BatchSizeAuto";return p;},
            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries, useMultiGpu);
        }
        private static void Train_CIFAR10_WRN(WideResNetNetworkSample p, int WRN_depth, int WRN_k)
        {
            using var cifar10 = new CIFAR10DataSet();
            using var network = p.WRN(DenseNetNetworkSample.Cifar10WorkingDirectory, WRN_depth, WRN_k, CIFAR10DataSet.Shape_CHW, CIFAR10DataSet.NumClass);
            network.Fit(cifar10.Training, cifar10.Test);
        }
        #endregion

        #region ResNet Training
        private static void ResNetTests()
        {
            var networkGeometries = new List<Action<ResNetNetworkSample, int>>
            {

                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x.ResNet164V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x.ResNet110V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x.ResNet56V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x.ResNet20V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x.ResNet11V2_CIFAR10);},
                
                /*
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet20V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet32V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet44V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet56V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet110V1_CIFAR10);},
                */
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR10(x, x.ResNet164V1_CIFAR10);},
            };

            var networkMetaParameters = new List<Func<ResNetNetworkSample>>
            {
                () =>{var p = ResNetNetworkSample.CIFAR10();p.WithSGD(0.9,true);return p;},
            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries);
        }
        #endregion

        #region CIFAR-100 Training
        private static void CIFAR100Tests()
        {
            var networkGeometries = new List<Action<WideResNetNetworkSample, int>>
            {
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 16,4);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 16,8);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 40,4);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 16,10);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 28,8);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_CIFAR100_WRN(x, 28,10);},
            };
            var networkMetaParameters = new List<Func<WideResNetNetworkSample>>
            {
                () => {var p = WideResNetNetworkSample.CIFAR100();p.AlphaMixup = 0.0;p.AlphaCutMix = 1.0;p.CutoutPatchPercentage = 0.0; return p;},
                () => {var p = WideResNetNetworkSample.CIFAR100();p.AlphaMixup = 1.0;p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.0; return p;},
                () => {var p = WideResNetNetworkSample.CIFAR100();p.AlphaMixup = 0.0;p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 20.0/32.0; return p;},
            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries);
        }

        private static void Train_CIFAR100_WRN(WideResNetNetworkSample p, int WRN_depth, int WRN_k)
        {
            using (var cifar100 = new CIFAR100DataSet())
            using (var network = p.WRN(DenseNetNetworkSample.Cifar100WorkingDirectory, WRN_depth, WRN_k, CIFAR100DataSet.Shape_CHW, CIFAR100DataSet.NumClass))
            {
                network.Fit(cifar100.Training, cifar100.Test);
            }
        }
        #endregion

        #region SVHN Training
        private static void SVHNTests()
        {
            const bool useMultiGpu = false;
            var networkGeometries = new List<Action<WideResNetNetworkSample, int>>
            {
                (x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 16,4);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 16,8);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 40,4);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 16,10);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 28,8);},
                //(x,gpuDeviceId) =>{x.SetResourceId(gpuDeviceId);Train_SVHN_WRN(x, true, 28,10);},
            };

            var networkMetaParameters = new List<Func<WideResNetNetworkSample>>
            {
                () =>{var p = WideResNetNetworkSample.WRN_SVHN();p.NumEpochs = 30;p.BatchSize=-1;return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.Config.NumEpochs = 30;p.Config.ExtraDescription = "_30Epochs";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.Config.NumEpochs = 30;p.Config.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS;  p.Config.ExtraDescription = "_30Epochs_USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS";return p;},
            };
            PerformAllActionsInAllGpu(networkMetaParameters, networkGeometries, useMultiGpu);
        }

        private static void Train_SVHN_WRN(WideResNetNetworkSample p, bool loadExtraFileForTraining, int WRN_depth, int WRN_k)
        {
            using var svhn = new SvhnDataset(loadExtraFileForTraining);
            using var network = p.WRN(DenseNetNetworkSample.SVHNWorkingDirectory, WRN_depth, WRN_k, SvhnDataset.Shape_CHW, SvhnDataset.NumClass);
            //using var network = Network.ValueOf(@"C:\Users\Franck\AppData\Local\SharpNet\SVHN\WRN-16-10_30Epochs_MultiGPU_20200501_1147_30.txt");
            network.Fit(svhn.Training, svhn.Test);
        }
        #endregion

        /// <summary>
        /// Train a network on CIFAR-10 data set 
        /// </summary>
        private static void Train_CIFAR10(Func<Network> buildNetwork)
        {
            using var cifar10 = new CIFAR10DataSet();
            using var network = buildNetwork();
            network.Fit(cifar10.Training, cifar10.Test);
        }

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

        private static void PerformAllActionsInAllGpu<T>(List<Func<T>> networkMetaParameters, List<Action<T, int>> networkGeometriesOrderedFromSmallestToBiggest, bool useMultiGPU = false) where T : NetworkSample
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
