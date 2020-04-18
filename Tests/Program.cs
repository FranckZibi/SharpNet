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
        private static void Main()
        {
            //new NonReg.ParallelRunWithTensorFlow().TestParallelRunWithTensorFlow_Efficientnet(); return;
            //new NonReg.ParallelRunWithTensorFlow().TestParallelRunWithTensorFlow_Convolution(); return;
            //new SharpNetTests.NonReg.TestEnsembleLearning().TestSVHN();return;
            //WideResNetTests();
            //SVHNTests();
            //CIFAR100Tests();
            //ResNetTests();
            //DenseNetTests();
            EfficientNetTests();
            //TestSpeed();return;
            //new TestGradient().TestGradientForDenseLayer(true, true);
            //new NonReg.TestBenchmark().TestGPUBenchmark_Memory();new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
            //new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
        }

        #region DenseNet Training
        private static void DenseNetTests()
        {
            var todo = new List<Action<DenseNetBuilder, int>>
            {
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_12_40);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNetBC_12_100);},
            /*(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_Fast_CIFAR10);},
            (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_12_10_CIFAR10);},
            (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_12_40);},
            (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNetBC_12_40);},
            //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\SharpNet\DenseNet_12_40_CIFAR10_200Epochs_NoNesterov_20190512_0743_200.txt"));},
            //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\SharpNet\DenseNet_12_40_CIFAR10_200Epochs_20190511_1946_154.txt"));},
             */
        };

            var metaParametersModifiers = new List<Func<DenseNetBuilder>>
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

                //(p) =>{p.Config.WithSGD(0.9,false);p.BatchSize = -1;p.NumEpochs = 300; p.ExtraDescription = "_SGD";},
                //(p) =>{p.UseAdam = false;p.UseNesterov = true;p.BatchSize = -1;p.NumEpochs = 300; p.ExtraDescription = "_SGDNesterov";},
                //(p) =>{p.UseAdam = true;p.UseNesterov = false;p.BatchSize = -1;p.NumEpochs = 200;p.InitialLearningRate = 0.001;p.ExtraDescription = "_Adam_0_001";},
                //(p) =>{p.UseAdam = true;p.BatchSize = 50;p.SaveNetworkStatsAfterEachEpoch = true; p.NumEpochs = 1; p.ExtraDescription = "_Adam_with_l2_inConv";},
                //(p) =>{p.UseAdam = true; p.lambdaL2Regularization = 0.0;p.SaveNetworkStatsAfterEachEpoch = true; p.NumEpochs = 1; p.ExtraDescription = "_Adam_no_l2_inConv";},

                //(p) =>{p.UseAdam = true;p.SaveNetworkStatsAfterEachEpoch = true; p.lambdaL2Regularization = 0.0;p.NumEpochs = 2; p.ExtraDescription = "_Adam_no_lambdaL2Regularization";},
                //(p) =>{p.lambdaL2Regularization = 0.0;p.UseNesterov = false;p.NumEpochs = 50; p.ExtraDescription = "_50Epoch_no_nesterov_no_lambdaL2Regularization";},
                #region already performed tests
            #endregion
            };
            PerformAllActionsInAllGpu(metaParametersModifiers, todo);
        }
        #endregion

        #region EfficientNet Training

        private static void EfficientNetTests()
        {
            var todo = new List<Action<EfficientNetBuilder, int>>
            {
                //(p,gpuDeviceId) =>{p.GpuDeviceId=gpuDeviceId;p.WeightForTransferLearning = "imagenet";p.Config.LastLayerNameToFreeze = "top_dropout";p.ExtraDescription += "_only_probs";Train_CIFAR10_EfficientNet(p);},
                //(p,gpuDeviceId) =>{p.GpuDeviceId=gpuDeviceId;p.WeightForTransferLearning = "imagenet";p.Config.LastLayerNameToFreeze = "block7a_project_bn";p.ExtraDescription += "_only_top";Train_CIFAR10_EfficientNet(p);},
                (p,gpuDeviceId) =>{p.GpuDeviceId=gpuDeviceId;p.ExtraDescription += "_no_freezing";Train_CIFAR10_EfficientNet(p);},
            };

            var modifiers = new List<Func<EfficientNetBuilder>>
            {
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.01;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_01_batch512_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.03;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_03_batch512_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.05;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_05_batch512_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.07;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_07_batch512_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.1;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_1_batch512_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.2;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_2_batch512_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.4;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_4_batch512_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.5;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_5_batch512_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.6;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_6_batch512_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.8;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_8_batch512_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 1.0;p.NumEpochs = 310;p.ExtraDescription = "_lr_1_0_batch512_310epochs";return p;},

                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.01;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_01_batchAuto_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.03;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_03_batchAuto_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.05;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_05_batchAuto_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.07;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_07_batchAuto_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.1;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_1_batchAuto_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.2;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_2_batchAuto_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.4;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_4_batchAuto_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.5;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_5_batchAuto_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.6;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_6_batchAuto_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.8;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_8_batchAuto_310epochs";return p;},
                () =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 1.0;p.NumEpochs = 310;p.ExtraDescription = "_lr_1_0_batchAuto_310epochs";return p;},

                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 1.0;p.NumEpochs = 150;p.ExtraDescription = "_lr_1_0_batch512_150epochs";return p;},
                ////() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 1.0;p.NumEpochs = 310;p.ExtraDescription = "_lr_1_0_batch512_310epochs";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.1;p.NumEpochs = 150;p.ExtraDescription = "_lr_0_1_batch512_150epochs";return p;},
                ////() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.1;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_1_batch512_310epochs";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.01;p.NumEpochs = 150;p.ExtraDescription = "_lr_0_01_batch512_150epochs";return p;},
                ////() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = 512;p.InitialLearningRate = 0.01;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_01_batch512_310epochs";return p;},

                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 1.0;p.NumEpochs = 150;p.ExtraDescription = "_lr_1_0_batchAuto_150epochs";return p;},
                ////() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 1.0;p.NumEpochs = 310;p.ExtraDescription = "_lr_1_0_batchAuto_310epochs";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.1;p.NumEpochs = 150;p.ExtraDescription = "_lr_0_1_batchAuto_150epochs";return p;},
                ////() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.1;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_1_batchAuto_310epochs";return p;},
                //() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.01;p.NumEpochs = 150;p.ExtraDescription = "_lr_0_01_batchAuto_150epochs";return p;},
                ////() =>{var p = EfficientNetBuilder.CIFAR10();p.BatchSize = -1;p.InitialLearningRate = 0.01;p.NumEpochs = 310;p.ExtraDescription = "_lr_0_01_batchAuto_310epochs";return p;},

            };
            PerformAllActionsInAllGpu(modifiers, todo);
        }
        private static void Train_CIFAR10_EfficientNet(EfficientNetBuilder p)
        {
            //const int zoomFactor = 3;
            //using (var cifar10Original = new CIFAR10DataSet())
            //using (var cifar10 = new ZoomedTrainingAndTestDataSet(cifar10Original, zoomFactor, zoomFactor))
            using (var cifar10 = new CIFAR10DataSet())

            using (var network = p.EfficientNetB0_CIFAR10(p.WeightForTransferLearning, cifar10.Training.InputShape_CHW))
            {
                network.Info(network.ToString());

                //network.Info("training only last layer");
                //network.Layers.ForEach(l => l.Trainable = false);
                //network.LastFrozenLayer().Trainable = true;

                // batchSize = 32 => best Learning Rate = 0.005
                // batchSize = 128 => best Learning Rate = 0.05

                //network.FindBestLearningRate(cifar10.Training, 1e-7, 10, p.BatchSize);return;

                var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
                network.Fit(cifar10.Training, learningRateComputer, p.NumEpochs, p.BatchSize, cifar10.Test);
            }

        }
        #endregion

        #region WideResNet Training
        private static void WideResNetTests()
        {
            var todo = new List<Action<WideResNetBuilder, int>>
            {
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 16,4);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 16,10);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 40,4);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 16,8);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 28,8);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 28,10);},

                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_DogsVsCat_WRN_TransferLearning(x);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_DogsVsCat_WRN(x, 10, 4);},

                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_RecursionCellularImageClassification_WRN(x, 10, 4);},
            };

            var modifiers = new List<Func<WideResNetBuilder>>
            {
                () =>{var p = WideResNetBuilder.WRN_CIFAR10();p.BatchSize = 128;p.Config.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST; p.ExtraDescription = "_FASTEST_DETERMINIST_BatchSize128";return p;},
                () =>{var p = WideResNetBuilder.WRN_CIFAR10();p.BatchSize = 64;p.Config.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST; p.ExtraDescription = "_FASTEST_DETERMINIST_BatchSize64";return p;},
                () =>{var p = WideResNetBuilder.WRN_CIFAR10();p.BatchSize = -1;p.Config.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST; p.ExtraDescription = "_FASTEST_DETERMINIST_BatchSizeAuto";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.WithRandAugment(3,5); p.ExtraDescription = "_RandAugment_3_5_";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.WithRandAugment(3,3); p.ExtraDescription = "_RandAugment_3_3_";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.WithRandAugment(2,4); p.ExtraDescription = "_RandAugment_2_4_";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.WithRandAugment(4,4); p.ExtraDescription = "_RandAugment_4_4_";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.WithRandAugment(3,4); p.ExtraDescription = "_RandAugment_3_4_";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10; p.ExtraDescription = "_AutoAugment";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.WRN_PoolingBeforeDenseLayer = NetworkBuilder.POOLING_BEFORE_DENSE_LAYER.AveragePooling_2;p.ExtraDescription = "_AvgPoolingSize_2";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.WRN_PoolingBeforeDenseLayer = NetworkBuilder.POOLING_BEFORE_DENSE_LAYER.AveragePooling_2;p.ExtraDescription = "_AvgPoolingSize_2";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.WRN_PoolingBeforeDenseLayer = NetworkBuilder.POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling;p.ExtraDescription = "_GlobalAveragePooling";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.WRN_PoolingBeforeDenseLayer = NetworkBuilder.POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling_And_GlobalMaxPooling; p.ExtraDescription = "_GAP_and_MAX";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.WRN_PoolingBeforeDenseLayer = NetworkBuilder.POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling; p.ExtraDescription = "_GlobalMaxPooling";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT; p.ExtraDescription = "";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10; p.ExtraDescription = "_AUTO_AUGMENT_CIFAR10";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_CUTOUT_CUTMIX_MIXUP; p.ExtraDescription = "_AUTO_AUGMENT_CIFAR10_CUTOUT_CUTMIX_MIXUP";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_CUTMIX; p.ExtraDescription = "_AUTO_AUGMENT_CIFAR10_AND_MANDATORY_CUTMIX";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_MIXUP; p.ExtraDescription = "_AUTO_AUGMENT_CIFAR10_AND_MANDATORY_MIXUP";return p;},


                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.EqualizeOperationProbability = 0.5; p.ExtraDescription = "_Equalize_0_50";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.AutoContrastOperationProbability = 0.5; p.ExtraDescription = "_AutoContrast_0_50";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.InvertOperationProbability = 0.5; p.ExtraDescription = "_Invert_0_50";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.BrightnessOperationProbability = 0.5;p.DA.BrightnessOperationEnhancementFactor = 1.0; p.ExtraDescription = "_Brightness_0_50_1_00";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.ColorOperationProbability = 0.5;p.DA.ColorOperationEnhancementFactor = 1.0; p.ExtraDescription = "_Color_0_50_1_00";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.ContrastOperationProbability = 0.5;p.DA.ContrastOperationEnhancementFactor = 1.0; p.ExtraDescription = "_Contrast_0_50_1_10";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.EqualizeOperationProbability = 0.5;p.DA.AutoContrastOperationProbability = 0.5; p.ExtraDescription = "_Equalize_0_50_AutoContrast_0_50";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.BrightnessOperationProbability = 0.5;p.DA.BrightnessOperationEnhancementFactor = 1.0;p.DA.ColorOperationProbability = 0.5;p.DA.ColorOperationEnhancementFactor = 1.0; p.ExtraDescription = "_Brightness_0_50_1_00_Color_0_50_1_00";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.EqualizeOperationProbability = 0.5;p.DA.AutoContrastOperationProbability = 0.5;p.DA.BrightnessOperationProbability = 0.5;p.DA.BrightnessOperationEnhancementFactor = 1.0;p.DA.ColorOperationProbability = 0.5;p.DA.ColorOperationEnhancementFactor = 1.0; p.ExtraDescription = "_Equalize_0_50_AutoContrast_0_50_Brightness_0_50_1_00_Color_0_50_1_00";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.BrightnessOperationProbability = 0.5;p.DA.BrightnessOperationEnhancementFactor = 1.0;p.DA.ColorOperationProbability = 0.5;p.DA.ColorOperationEnhancementFactor = 1.0; p.ExtraDescription = "_Brightness_0_50_1_00_Color_0_50_1_00";return p;},

                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.AlphaMixup = 0.0;p.DA.AlphaCutMix = 0.0;p.DA.CutoutPatchPercentage = 0.5; p.ExtraDescription = "_CutoutOnly";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DA.AlphaMixup = 0.0;p.DA.AlphaCutMix = 1.0;p.DA.CutoutPatchPercentage = 0.5; p.ExtraDescription = "_CutMix_Cutout";return p;},

                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_CUTMIX; p.ExtraDescription = "_AUTO_AUGMENT_CIFAR10_AND_MANDATORY_CUTMIX";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10; p.ExtraDescription = "_AUTO_AUGMENT_CIFAR10";return p;},

                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_CUTOUT_CUTMIX_MIXUP; p.ExtraDescription = "_AUTO_AUGMENT_CIFAR10_CUTOUT_CUTMIX_MIXUP";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_CUTMIX; p.ExtraDescription = "_AUTO_AUGMENT_CIFAR10_AND_MANDATORY_CUTMIX";return p;},
                //() =>{var p = WideResNetBuilder.WRN_CIFAR10();p.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_CIFAR10_AND_MANDATORY_MIXUP; p.ExtraDescription = "_AUTO_AUGMENT_CIFAR10_AND_MANDATORY_MIXUP";return p;},

                //() =>{p.ZoomRange = 0.1;p.ExtraDescription = "_ZoomRange_0_1";},
                //() =>{p.AvgPoolingSize = 8;p.ExtraDescription = "_AvgPoolingSize_8";},
                //() => { p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.0;p.AlphaMixup = 1.0;p.ExtraDescription = "_Mixup_only";},
                //() => { p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.5;p.AlphaMixup = 0.0;p.ExtraDescription = "_Cutout_only";},
                ////10-aug-2019: -10 bps
                //() =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(1, 2);p.NumEpochs = 127;p.ExtraDescription = "_CyclicCosineAnnealing_1_2_127Epochs";},


                //() =>{p.RecursionCellularImageClassification();p.ExtraDescription = "_RecursionCellular";},
                //() => { p.CutMix = false;p.CutoutPatchPercentage = 0.5;},
                //() => { p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.0;p.AlphaMixup = 1.0;},
                //() =>{p.NumEpochs=200;p.ExtraDescription = "_200epochs";},
                //() =>{p.Config.WithCifar10WideResNetLearningRateScheduler(true, true, false);p.ExtraDescription = "_WithCifar10WideResNetLearningRateScheduler";},
                //with CutMix: tested on 12-aug-2019: +15 bps
                //() =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.CutMix = true;p.CutoutPatchPercentage = 0.0;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_CutMix_150epochs";},
                //new formula of cutout : tested on 12-aug-2019: -43 bps
                //() =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_CutoutV3_150epochs";},
                #region already performed tests
                //CutMix V2 : validated on 18-aug-2019 : -4bps (in CutMix V2 we ensure that we keep at least 50% of the original element)
                //(p) =>{p.ExtraDescription = "_CutMixV2";},
                //tested on 18-aug-219: 'CutMix + Cutout 10%' : -43bps vs 'CutMix only'
                //(p) =>{p.CutoutPatchPercentage = 0.1;p.ExtraDescription = "_CutMixV2_Cutout_0_10";},
                //tested on 16-aug-219: 'Mixup only' : -63bps vs 'CutMix only'
                //(p) => { p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.0;p.AlphaMixup = 1.0;p.ExtraDescription = "_only_MixUp";},
                //tested on 16-aug-219: 'Mixup & CutOut' : -75bps vs 'CutMix only'
                //(p) => { p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.5;p.AlphaMixup = 1.0;p.ExtraDescription = "_MixUp_and_cutout";},
                //tested on 15-aug-219: 'CutMix & CutOut' : -55bps vs 'CutMix only'
                //(p) => { p.CutMix = true;p.CutoutPatchPercentage = 0.5;p.ExtraDescription = "_CutMix_Cutout";},
                //tested on 15-aug-219: 'Cutout only' : -45bps vs 'CutMix only'
                //(p) => { p.CutMix = false;p.CutoutPatchPercentage = 0.5;p.ExtraDescription = "_noCutMix_Cutout";},
                //CutMix : tested on 14-aug-2019 : +15 bps
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.CutMix = true;p.CutoutPatchPercentage = 0.0;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_CutMixV2_150epochs";},
                //CutMix + Cutout : tested on 14-aug-2019 : -15 bps
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.CutMix = true;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_CutMixV2_Cutout_150epochs";},
                //10-aug-2019: -10 bps
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(1, 2);p.NumEpochs = 150;p.ExtraDescription = "_CyclicCosineAnnealing_1_2_150epochs";},
                //10-aug-2019: -5 bps
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.ZoomRange = 0.2;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_ZoomRange_0_2_150epochs";},
                //10-aug-2019: +4 bps
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.ZoomRange = 0.1;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_ZoomRange_0_1_150epochs";},
                //10-aug-2019: -42 bps
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.RotationRangeInDegrees = 10;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_RotationRangeInDegrees_10_150epochs";},
                //9-aug-2019: -80 bps
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.RotationRangeInDegrees = 25;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_RotationRangeInDegrees_25_150epochs";},
                //9-aug-2019: -33 bps
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(1, 2);p.NumEpochs = 127;p.ExtraDescription = "_CyclicCosineAnnealing_1_2_127epochs";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_150epochs";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.CutoutPatchPercentage = 0.0;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_no_cutout_150epochs";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.CutoutPatchPercentage = 0.25;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_cutout_0_25_150epochs";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.WidthShiftRange = 0.2;p.HeightShiftRange = 0.2;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_WidthShiftRange_0_2_150epochs";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.BatchSize = 512;p.NumEpochs = 150;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_BatchSize512_150epochs";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.ExtraDescription = "_CyclicCosineAnnealing_10_2";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.CutoutPatchPercentage = 0.0;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_no_cutout";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.CutoutPatchPercentage = 0.25;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_cutout_0_25";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(200,1);p.ExtraDescription = "_CyclicCosineAnnealing_200_1";},
                //(p) =>{p.DropOutAfterDenseLayer = 0.1;p.NumEpochs = 150;p.ExtraDescription = "_010DropOutAfterDenseLayer_150epochs";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2);p.NumEpochs = 150;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_150epochs";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(150,1);p.NumEpochs = 150;p.ExtraDescription = "_CyclicCosineAnnealing_150_1_150epochs";},
                //(p) =>{p.DropOutAfterDenseLayer = 0.3;p.NumEpochs = 150;p.ExtraDescription = "_030DropOutAfterDenseLayer_150epochs";},
                //(p) =>{p.CutoutPatchlength=0;p.ExtraDescription = "_0CutoutPatchlength";},
                //(p) =>{p.AvgPoolingSize=2;p.ExtraDescription = "_2AvgPoolingSize";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(50,1);p.ExtraDescription = "_CyclicCosineAnnealing_50_1";},
                //(p) =>{p.ExtraDescription = "";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2);p.NumEpochs = 150;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_150epochs";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(150,1);p.NumEpochs = 150;p.ExtraDescription = "_CyclicCosineAnnealing_150_1_150epochs";},
                //(p) =>{p.DropOut = 0.3;p.ExtraDescription = "_030_dropout";},
                //(p) =>{p.CutoutPatchLength=0;p.ExtraDescription = "_0CutoutPatchlength";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(1,2);p.ExtraDescription = "_CyclicCosineAnnealing_1_2";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2);p.ExtraDescription = "_CyclicCosineAnnealing_10_2";},
                //(p) =>{p.ExtraDescription = "";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.RotationRangeInDegrees = 15;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_RotationRangeInDegrees_15_150epochs";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.ZoomRange = 0.1;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_ZoomRange_0_1_150epochs";},
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.WidthShiftRange = 0.0;p.HeightShiftRange = 0.0;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_WidthShiftRange_0_0_150epochs";},
                #endregion
            };
            PerformAllActionsInAllGpu(modifiers, todo);
        }
        private static void Train_CIFAR10_WRN(WideResNetBuilder p, int WRN_depth, int WRN_k)
        {
            using (var cifar10 = new CIFAR10DataSet())
            using (var network = p.WRN(WRN_depth, WRN_k, cifar10.InputShape_CHW, cifar10.CategoryCount))
            {
                var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
                network.Fit(cifar10.Training, learningRateComputer, p.NumEpochs, p.BatchSize, cifar10.Test);
            }
        }
        #endregion

        #region ResNet Training
        private static void ResNetTests()
        {
            var todo = new List<Action<ResNetBuilder, int>>
            {

                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet164V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet110V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet56V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet20V2_CIFAR10);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet11V2_CIFAR10);},
                
                /*
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet20V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet32V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet44V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet56V1_CIFAR10);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet110V1_CIFAR10);},
                */
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet164V1_CIFAR10);},
            };

            var modifiers = new List<Func<ResNetBuilder>>
            {
                () =>{var p = ResNetBuilder.ResNet_CIFAR10();p.Config.WithSGD(0.9,true);p.ExtraDescription = "";return p;},
            };
            PerformAllActionsInAllGpu(modifiers, todo);
        }
        #endregion



        #region CIFAR-100 Training
        private static void CIFAR100Tests()
        {
            var todo = new List<Action<WideResNetBuilder, int>>
            {
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR100_WRN(x, 16,4);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 40,4);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 16,8);},

                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 16,10);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 28,8);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 28,10);},
            };

            var modifiers = new List<Func<WideResNetBuilder>>
            {
                () => {var p = WideResNetBuilder.WRN_CIFAR100();p.DA.AlphaMixup = 0.0;p.DA.AlphaCutMix = 1.0;p.DA.CutoutPatchPercentage = 0.0; p.ExtraDescription = "CutMix";return p;},
                () => {var p = WideResNetBuilder.WRN_CIFAR100();p.DA.AlphaMixup = 1.0;p.DA.AlphaCutMix = 0.0;p.DA.CutoutPatchPercentage = 0.0; p.ExtraDescription = "Mixup";return p;},
                () => {var p = WideResNetBuilder.WRN_CIFAR100();p.DA.AlphaMixup = 0.0;p.DA.AlphaCutMix = 0.0;p.DA.CutoutPatchPercentage = 20.0/32.0; p.ExtraDescription = "Cutout_0_625";return p;},
            };
            PerformAllActionsInAllGpu(modifiers, todo);
        }

        private static void Train_CIFAR100_WRN(WideResNetBuilder p, int WRN_depth, int WRN_k)
        {
            using (var cifar100 = new CIFAR100DataSet())
            using (var network = p.WRN(WRN_depth, WRN_k, cifar100.InputShape_CHW, cifar100.CategoryCount))
            {
                var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
                network.Fit(cifar100.Training, learningRateComputer, p.NumEpochs, p.BatchSize, cifar100.Test);
            }
        }
        #endregion

        #region SVHN Training
        private static void SVHNTests()
        {
            var todo = new List<Action<WideResNetBuilder, int>>
            {
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_SVHN_WRN(x, true, 16,4);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_SVHN_WRN(x, true, 40,4);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_SVHN_WRN(x, true, 16,8);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_SVHN_WRN(x, true, 16,10);},

                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_SVHN_WRN(x, true, 16,4);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_SVHN_WRN(x, true, 40,4);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_SVHN_WRN(x, true, 16,8);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_SVHN_WRN(x, true, 16,10);},

                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_SVHN_WRN(x, true, 28,8);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_SVHN_WRN(x, true, 28,10);},
            };

            var modifiers = new List<Func<WideResNetBuilder>>
            {
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs = 30;p.Config.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.FASTEST_DETERMINIST;  p.ExtraDescription = "_30Epochs_FASTEST_DETERMINIST";return p;},
                () =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs = 30;p.Config.ConvolutionAlgoPreference = GPUWrapper.ConvolutionAlgoPreference.USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS;  p.ExtraDescription = "_30Epochs_USE_CUDNN_GET_CONVOLUTION_ALGORITHM_METHODS";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs = 30;  p.ExtraDescription = "_30Epochs";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs = 30;p.DA.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_SVHN;  p.ExtraDescription = "_30Epochs_AutoAugment";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs = 30;p.DA.WithRandAugment(3,9);  p.ExtraDescription = "_30Epochs_RandAugment_3_9";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs = 30;p.WRN_PoolingBeforeDenseLayer = NetworkBuilder.POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling;  p.ExtraDescription = "_GlobalMaxPooling_30Epochs";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs = 30;p.WRN_PoolingBeforeDenseLayer = NetworkBuilder.POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling_And_GlobalMaxPooling;  p.ExtraDescription = "_GAP_AND_GlobalMaxPooling_30Epochs";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs = 30;p.WRN_PoolingBeforeDenseLayer = NetworkBuilder.POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling;  p.ExtraDescription = "_GAP_30Epochs";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs=150;p.ExtraDescription = "_150Epochs_smallTrain";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs=150;p.DA.DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.AUTO_AUGMENT_SVHN;p.ExtraDescription = "_150Epochs_AutoAugment_smallTrain";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.NumEpochs=150;p.ExtraDescription = "_30Epochs";return p;},
                //() =>{var p = WideResNetBuilder.WRN_SVHN();p.BatchSize = -1;p.NumEpochs=30;p.ExtraDescription = "_30Epochs_AutoBatchSize";return p;},
            };
            PerformAllActionsInAllGpu(modifiers, todo);
        }

        private static void Train_SVHN_WRN(WideResNetBuilder p, bool loadExtraFileForTraining, int WRN_depth, int WRN_k)
        {
            using (var svhn = new SVHNDataSet(loadExtraFileForTraining))
            using (var network = p.WRN(WRN_depth, WRN_k, svhn.InputShape_CHW, svhn.CategoryCount))
            {
                var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
                network.Fit(svhn.Training, learningRateComputer, p.NumEpochs, p.BatchSize, svhn.Test);
            }
        }
        #endregion



        /// <summary>
        /// Train a network on CIFAR-10 data set 
        /// </summary>
        private static void Train_CIFAR10(NetworkBuilder p, Func<AbstractTrainingAndTestDataSet, Network> buildNetwork)
        {
            using (var cifar10 = new CIFAR10DataSet())
            using (var network = buildNetwork(cifar10))
            {
                var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
                network.Fit(cifar10.Training, learningRateComputer, p.NumEpochs, p.BatchSize, cifar10.Test);
            }
        }

        /// <summary>
        /// perform as much actions as possible among 'allActionsToPerform'
        /// </summary>
        /// <param name="gpuId"></param>
        /// <param name="allActionsToPerform"></param>
        private static void PerformActionsInSingleGpu(int gpuId, List<Action<int>> allActionsToPerform)
        {
            for (; ; )
            {
                Action<int> nexActionToPerform;
                lock (allActionsToPerform)
                {
                    Console.WriteLine("GpuId#" + gpuId + " : " + allActionsToPerform.Count + " remaining computation(s)");
                    if (allActionsToPerform.Count == 0)
                    {
                        return;
                    }
                    nexActionToPerform = allActionsToPerform[0];
                    allActionsToPerform.RemoveAt(0);
                }
                try
                {
                    Console.WriteLine("GpuId#" + gpuId + " : starting new computation");
                    nexActionToPerform(gpuId);
                    Console.WriteLine("GpuId#" + gpuId + " : ended new computation");
                }
                catch (Exception e)
                {
                    Console.WriteLine("GpuId#" + gpuId + " : " + e);
                    Console.WriteLine("GpuId#" + gpuId + " : ignoring error");
                }
            }
        }
        private static void PerformAllActionsInAllGpu<T>(List<Func<T>> networkDeformers, List<Action<T, int>> networks) where T : NetworkBuilder, new()
        {
            var taskToBePerformed = new List<Action<int>>();
            foreach (var networkDeformer in networkDeformers)
            {
                foreach (var network in networks)
                {
                    taskToBePerformed.Add(gpuDeviceId => network(networkDeformer(), gpuDeviceId));
                }
            }
            int nbGPUs = Math.Min(GPUWrapper.GetDeviceCount(), taskToBePerformed.Count);
            Console.WriteLine(taskToBePerformed.Count + " computation(s) will be done on " + nbGPUs + " GPU(s)");
            var gpuTasks = new Task[nbGPUs];
            for (int i = 0; i < nbGPUs; ++i)
            {
                var gpuId = i;
                gpuTasks[i] = new Task(() => PerformActionsInSingleGpu(gpuId, taskToBePerformed));
                gpuTasks[i].Start();
            }
            Task.WaitAll(gpuTasks);
        }
    }
}
