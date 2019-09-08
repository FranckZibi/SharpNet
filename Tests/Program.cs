using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
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
            /*
            Console.WriteLine("Epoch1");
            Console.WriteLine(Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\Network_19064_1.txt").ContentStats());
            Console.WriteLine("Epoch7");
            Console.WriteLine(Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\Network_19064_7.txt").ContentStats());
            return;
            */
            //var x = new DenseNetBuilder {NumEpochs = 5,BatchSize = -1, GpuDeviceId=-1};Train_CIFAR10(x, x.DenseNet_12_40_CIFAR10());return;


            //var csvFilename = @"C:\temp\aptos2019-blindness-detection\test_images\test.csv";
            //var dataDirectory = @"C:\temp\aptos2019-blindness-detection\test_images";
            //var testDirectory = Aptos2019BlindnessDetection.ValueOf(csvFilename, dataDirectory, -1, -1, null);
            //testDirectory.CropBorder(true).MakeSquarePictures(true, false, null, true).Resize(32, 32, true); return;





            //var dogsVsCatDirectory = @"C:\Users\Franck\AppData\Local\SharpNet\Data\dogs-vs-cats\train";
            //var original = DogsVsCats.ValueOf(dogsVsCatDirectory, null);
            //var filtered = original.Filter(x => 1.2*Math.Min(x.GetWidth(),x.GetHeight()) > Math.Max(x.GetWidth(), x.GetHeight()));
            //var resized = filtered.Resize(32, 32);
            //return;


            //new TestEnsembleLearning().Test(); return;
            //WideResNetTests();
            CIFAR100Tests(); //?D
            //ResNetTests();
            //DenseNetTests();

            //TestSpeed();return;
            //new TestGradient().TestGradientForDenseLayer(true, true);
            //new NonReg.TestMNIST().Test();
            ////new NonReg.TestNetworkPropagation().TestParallelRunWithTensorFlow();
            //new NonReg.TestBenchmark().TestGPUBenchmark_Memory();new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
            //new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
        }

        #region DenseNet Training
        private static void DenseNetTests()
        {
            var todo = new List<Action<DenseNetBuilder, int>>
            {
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_12_40_CIFAR10);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNetBC_12_100_CIFAR10);},
            /*(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_Fast_CIFAR10);},
            (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_12_10_CIFAR10);},
            (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_12_40_CIFAR10);},
            (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNetBC_12_40_CIFAR10);},
            //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\SharpNet\DenseNet_12_40_CIFAR10_200Epochs_NoNesterov_20190512_0743_200.txt"));},
            //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\SharpNet\DenseNet_12_40_CIFAR10_200Epochs_20190511_1946_154.txt"));},
             */
        };

            var metaParametersModifiers = new List<Action<DenseNetBuilder>>
            {
                //(p) =>{p.UseNesterov = false; p.NumEpochs = 50; p.ExtraDescription = "_50Epoch_no_nesterov";},
                //(p) =>{p.UseAdam = true; p.NumEpochs = 5; p.ExtraDescription = "_50Epoch_Adam";},
                //(p) =>{p.SaveNetworkStatsAfterEachEpoch = true; p.ExtraDescription = "_Adam_with_l2_inConv";},
                //(p) =>{p.SaveNetworkStatsAfterEachEpoch = false;p.SaveLossAfterEachMiniBatch = false;p.UseAdam = true;p.UseNesterov = false;p.BatchSize = -1;p.ForceTensorflowCompatibilityMode = false;p.NumEpochs = 150; p.ExtraDescription = "_Adam";},
                //(p) =>{ p.ExtraDescription = "_OrigPaper";},

                (p) =>{p.NumEpochs = 240;p.BatchSize = -1;p.Config.WithSGD(0.9,true); p.Config.ForceTensorflowCompatibilityMode = true;p.CutoutPatchPercentage = 0.0;p.ExtraDescription = "_240Epochs_ForceTensorflowCompatibilityMode_CutoutPatchPercentage0_WithNesterov_EnhancedMemory";},
                (p) =>{p.NumEpochs = 240;p.BatchSize = -1;p.Config.WithSGD(0.9,false); p.Config.ForceTensorflowCompatibilityMode = true;p.CutoutPatchPercentage = 0.0;p.ExtraDescription = "_240Epochs_ForceTensorflowCompatibilityMode_CutoutPatchPercentage0_NoNesterov_EnhancedMemory";},


                //(p) =>{p.NumEpochs = 300;p.BatchSize = -1;p.CutoutPatchPercentage = 0.25;p.ExtraDescription = "_200Epochs_L2InDense_CutoutPatchPercentage0_25;},
                //(p) =>{p.NumEpochs = 300;p.BatchSize = -1;p.CutoutPatchPercentage = 0.0;p.ExtraDescription = "_200Epochs_L2_InDense_CutoutPatchPercentage0";},
                //(p) =>{p.NumEpochs = 200;p.Config.WithSGD(0.9,false);p.ExtraDescription = "_200Epochs_NoNesterov";},

                //(p) =>{p.Config.WithSGD(0.9,false);p.BatchSize = -1;p.Config.ForceTensorflowCompatibilityMode = false;p.NumEpochs = 300; p.ExtraDescription = "_SGD";},
                //(p) =>{p.UseAdam = false;p.UseNesterov = true;p.BatchSize = -1;p.ForceTensorflowCompatibilityMode = false;p.NumEpochs = 300; p.ExtraDescription = "_SGDNesterov";},
                //(p) =>{p.UseAdam = true;p.UseNesterov = false;p.BatchSize = -1;p.ForceTensorflowCompatibilityMode = false;p.NumEpochs = 200;p.InitialLearningRate = 0.001;p.ExtraDescription = "_Adam_0_001";},
                //(p) =>{p.UseAdam = true;p.BatchSize = 50;p.SaveNetworkStatsAfterEachEpoch = true; p.NumEpochs = 1; p.ExtraDescription = "_Adam_with_l2_inConv";},
                //(p) =>{p.UseAdam = true; p.lambdaL2Regularization = 0.0;p.SaveNetworkStatsAfterEachEpoch = true; p.NumEpochs = 1; p.ExtraDescription = "_Adam_no_l2_inConv";},

                //(p) =>{p.UseAdam = true;p.SaveNetworkStatsAfterEachEpoch = true; p.lambdaL2Regularization = 0.0;p.NumEpochs = 2; p.ExtraDescription = "_Adam_no_lambdaL2Regularization";},
                //(p) =>{p.lambdaL2Regularization = 0.0;p.UseNesterov = false;p.NumEpochs = 50; p.ExtraDescription = "_50Epoch_no_nesterov_no_lambdaL2Regularization";},
                #region already performed tests
            #endregion
            };
            PerformTestSet(metaParametersModifiers, todo);
        }
        #endregion

        const int WidthAptos2019 = 128;

        #region WideResNet Training
        private static void WideResNetTests()
        {
            var todo = new List<Action<WideResNetBuilder, int>>
            {

                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_DogsVsCat_WRN_TransferLearning(x);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_DogsVsCat_WRN(x, 10, 4);},

                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_RecursionCellularImageClassification_WRN(x, 10, 4);},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_Aptos2019Blindness_WRN(x, 16,8,WidthAptos2019);},

                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 16,4);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 40,4);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 16,8);},

                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 16,10);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 28,8);},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10_WRN(x, 28,10);},
            };



            var modifiers = new List<Action<WideResNetBuilder>>
                            {
                ////ref 8-aug-2019: 0 bps
                (p) =>{},
                //(p) =>{p.ZoomRange = 0.1;p.ExtraDescription = "_ZoomRange_0_1";},
                //(p) =>{p.AvgPoolingSize = 8;p.ExtraDescription = "_AvgPoolingSize_8";},
                //(p) => { p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.0;p.AlphaMixup = 1.0;p.ExtraDescription = "_Mixup_only";},
                //(p) => { p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.5;p.AlphaMixup = 0.0;p.ExtraDescription = "_Cutout_only";},
                ////10-aug-2019: -10 bps
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(1, 2);p.NumEpochs = 127;p.ExtraDescription = "_CyclicCosineAnnealing_1_2_127Epochs";},
                

                //(p) =>{p.RecursionCellularImageClassification();p.ExtraDescription = "_RecursionCellular";},
                //(p) => { p.CutMix = false;p.CutoutPatchPercentage = 0.5;},
                //(p) => { p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.0;p.AlphaMixup = 1.0;},
                //(p) =>{p.NumEpochs=200;p.ExtraDescription = "_200epochs";},
                //(p) =>{p.Config.WithCifar10WideResNetLearningRateScheduler(true, true, false);p.ExtraDescription = "_WithCifar10WideResNetLearningRateScheduler";},
                //with CutMix: tested on 12-aug-2019: +15 bps
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.CutMix = true;p.CutoutPatchPercentage = 0.0;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_CutMix_150epochs";},
                //new formula of cutout : tested on 12-aug-2019: -43 bps
                //(p) =>{p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10, 2);p.NumEpochs = 150;p.ExtraDescription = "_CyclicCosineAnnealing_10_2_CutoutV3_150epochs";},


                (p) =>{p.BatchSize = 32;p.NumEpochs = 150;p.WidthShiftRange = p.HeightShiftRange = 0;p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.5;p.AlphaMixup = 1.0;p.InitialLearningRate = 0.01;p.CutoutPatchPercentage = 0.0;p.RotationRangeInDegrees=90;p.ExtraDescription = "_Aptos2019_noCutout_noCutMix_Mixup_Rotation90_"+WidthAptos2019;},
                //(p) =>{p.BatchSize = 32;p.NumEpochs = 150;p.WidthShiftRange = p.HeightShiftRange = 0;p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.5;p.AlphaMixup = 1.0;p.InitialLearningRate = 0.01;p.CutoutPatchPercentage = 0.0;p.RotationRangeInDegrees=0;p.ExtraDescription = "_Aptos2019_noCutout_noCutMix_Mixup_noRotation"+WidthAptos2019;},
                //(p) =>{p.BatchSize = 32;p.NumEpochs = 70;p.WidthShiftRange = p.HeightShiftRange = 0;p.AlphaCutMix= 0.0;p.CutoutPatchPercentage = 0.0;p.InitialLearningRate = 0.01;p.ExtraDescription = "_Aptos2019_noCutout_noCutMix_noRotation"+WidthAptos2019;},
                //(p) =>{p.BatchSize = 32;p.NumEpochs = 70;p.CutMix = true;p.WidthShiftRange = p.HeightShiftRange = 0;p.CutMix = false;p.CutoutPatchPercentage = 0.0;p.InitialLearningRate = 0.01;p.ExtraDescription = "_Aptos2019_noCutout_CutMix_noRotation"+WidthAptos2019;},
                //(p) =>{p.BatchSize = 32;p.NumEpochs = 70;p.CutoutPatchPercentage = 0.1;p.WidthShiftRange = p.HeightShiftRange = 0;p.CutMix = false;p.CutoutPatchPercentage = 0.0;p.InitialLearningRate = 0.01;p.ExtraDescription = "_Aptos2019_Cutout_0_10_noCutMix_noRotation"+WidthAptos2019;},
                //(p) =>{p.BatchSize = 32;p.NumEpochs = 70;p.CutoutPatchPercentage = 0.5;p.WidthShiftRange = p.HeightShiftRange = 0;p.CutMix = false;p.CutoutPatchPercentage = 0.0;p.InitialLearningRate = 0.01;p.ExtraDescription = "_Aptos2019_Cutout_0_50_noCutMix_noRotation"+WidthAptos2019;},
                //(p) =>{p.BatchSize = 32;p.NumEpochs = 70;p.WidthShiftRange = p.HeightShiftRange = 0;p.CutMix = false;p.CutoutPatchPercentage = 0.0;p.InitialLearningRate = 0.01;p.RotationRangeInDegrees=10;p.ExtraDescription = "_Aptos2019_noCutout_noCutMix_Rotation10"+WidthAptos2019;},
                //(p) =>{p.BatchSize = 32;p.NumEpochs = 70;p.WidthShiftRange = p.HeightShiftRange = 0;p.CutMix = false;p.CutoutPatchPercentage = 0.0;p.InitialLearningRate = 0.01;p.RotationRangeInDegrees=90;p.ExtraDescription = "_Aptos2019_noCutout_noCutMix_Rotation90"+WidthAptos2019;},
                //(p) =>{p.BatchSize = 32;p.NumEpochs = 70;p.WidthShiftRange = p.HeightShiftRange = 0;p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.0;p.InitialLearningRate = 0.01;p.RotationRangeInDegrees=180;p.ExtraDescription = "_Aptos2019_noCutout_noCutMix_Rotation180"+WidthAptos2019;},
                //(p) =>{p.BatchSize = 32;p.NumEpochs = 70;p.WidthShiftRange = p.HeightShiftRange = 0;p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.0;p.InitialLearningRate = 0.01;p.CutoutPatchPercentage = 0.1;p.RotationRangeInDegrees=90;p.ExtraDescription = "_Aptos2019_Cutout_0_10_noCutMix_Rotation90"+WidthAptos2019;},
                //(p) =>{p.BatchSize = 32;p.NumEpochs = 70;p.WidthShiftRange = p.HeightShiftRange = 0;p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.5;p.InitialLearningRate = 0.01;p.CutoutPatchPercentage = 0.1;p.RotationRangeInDegrees=90;p.ExtraDescription = "_Aptos2019_Cutout_0_50_noCutMix_Rotation90"+WidthAptos2019;},
                //(p) =>{p.BatchSize = 32;p.NumEpochs = 150;p.WidthShiftRange = p.HeightShiftRange = 0;p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.5;p.AlphaMixup = 1.0;p.InitialLearningRate = 0.01;p.CutoutPatchPercentage = 0.1;p.RotationRangeInDegrees=90;p.ExtraDescription = "_Aptos2019_Cutout_0_10_noCutMix_Mixup_Rotation90_"+WidthAptos2019;},

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
            PerformTestSet(modifiers, todo);
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

            var modifiers = new List<Action<ResNetBuilder>>
            {
                (p) =>{p.Config.WithSGD(0.9,true);p.ExtraDescription = "";},

                
                /*
                (p) =>{p.Config.WithSGD(0.9,true);p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2);p.ExtraDescription = "_CyclicCosineAnnealing_10_2";},
                (p) =>{p.Config.WithSGD(0.9,true);p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2);p.ExtraDescription = "_CyclicCosineAnnealing_10_2";},
                (p) =>{p.Config.WithSGD(0.9,true);p.NumEpochs=300;p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2);p.ExtraDescription = "_CyclicCosineAnnealing_10_2_300Epochs";},
                (p) =>{p.Config.WithSGD(0.9,true);p.Config.WithCyclicCosineAnnealingLearningRateScheduler(200,1);p.ExtraDescription = "_CyclicCosineAnnealing_200_1";},
                (p) =>{p.Config.WithSGD(0.9,false);p.Config.WithCyclicCosineAnnealingLearningRateScheduler(10,2);p.ExtraDescription = "_CyclicCosineAnnealing_10_2_NoNesterov";},
                */
                /*
                (p) =>{p.Config.WithSGD(0.9,true);p.ExtraDescription = "";},
                (p) =>{p.Config.WithSGD(0.9, true);p.BatchSize = -1;p.ExtraDescription = "_AutoMiniBatchSize";},
                (p) =>{p.Config.WithSGD(0.9, true);p.NumEpochs=300;p.ExtraDescription = "_300Epochs";}, */
                #region already performed tests
                /*
                (p) =>{p.Config.WithSGD(0.9,true);p.Config.ForceTensorflowCompatibilityMode = true;p.ExtraDescription = "_SGD_WithNesterov_ForceTensorflowCompatibilityMode_EnhancedMemory";},
                (p) =>{p.Config.WithSGD(0.9,false);p.ExtraDescription = "_SGD_NoNesterov_EnhancedMemory";},
                (p) =>{p.Config.WithSGD(0.9,false);p.Config.ForceTensorflowCompatibilityMode = true;p.ExtraDescription = "_SGD_NoNesterov_ForceTensorflowCompatibilityMode_EnhancedMemory";},
                //https://sgugger.github.io/the-1cycle-policy.html
                //(param) => {}, //used to check new speed
                //(param) => {param.UseAdam=true;param.ExtraDescription = "_UseAdam";},
                //(param) => {param.WidthShiftRange=param.HeightShiftRange=0.2;param.ExtraDescription = "_ShiftRange_0_20";},
                //(param) => {param.DivideBy10OnPlateau=true;param.ExtraDescription = "_DivideBy10OnPlateau";},
                (param) =>{param.OneCycleLearningRate =true;param.NumEpochs = 70;param.BatchSize = -1;param.InitialLearningRate = 0.8;param.OneCycleDividerForMinLearningRate = 10;param.ExtraDescription = "_OneCycle_080_008_70Epochs";},
                (param) =>{param.OneCycleLearningRate =true;param.NumEpochs = 100;param.BatchSize = -1;param.InitialLearningRate = 0.8;param.OneCycleDividerForMinLearningRate = 10;param.ExtraDescription = "_OneCycle_080_008_100Epochs";},
                (param) =>{param.BatchSize=-1;param.ExtraDescription = "_Auto_BatchSize";},
                (param) =>{param.FillMode=ImageDataGenerator.FillModeEnum.Reflect;param.ExtraDescription = "_FillMode_Reflect";},
                (param) =>{param.HorizontalFlip=false;param.ExtraDescription = "_no_HorizontalFlip";},
                (param) =>{param.lambdaL2Regularization=0.0005;param.ExtraDescription = "_lambdaL2Regularization_0_0005";},
                (param) =>{param.WidthShiftRange=0.0;param.HeightShiftRange=0.0;param.HorizontalFlip=false;param.ExtraDescription = "_no_DataAugmentation_onlyShuffling";},
                //https://sgugger.github.io/the-1cycle-policy.html
                (param) =>{param.OneCycleLearningRate =true;param.NumEpochs = 50;param.BatchSize = -1;param.InitialLearningRate = 3.0;param.OneCycleDividerForMinLearningRate = 20;param.ExtraDescription = "_OneCycle_300_015_50Epochs";},
                //https://sgugger.github.io/the-1cycle-policy.html
                (param) =>{param.OneCycleLearningRate =true;param.NumEpochs = 70;param.BatchSize = -1;param.InitialLearningRate = 3.0;param.OneCycleDividerForMinLearningRate = 20;param.ExtraDescription = "_OneCycle_300_015_70Epochs";},
                */
                #endregion
            };
            PerformTestSet(modifiers, todo);
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

            var modifiers = new List<Action<WideResNetBuilder>>
            {
                (p) => { p.WRN_CIFAR100();p.AlphaMixup = 0.0;p.AlphaCutMix = 1.0;p.CutoutPatchPercentage = 0.0; p.ExtraDescription = "CutMix";},
                (p) => { p.WRN_CIFAR100();p.AlphaMixup = 1.0;p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 0.0; p.ExtraDescription = "Mixup";},
                (p) => { p.WRN_CIFAR100();p.AlphaMixup = 0.0;p.AlphaCutMix = 0.0;p.CutoutPatchPercentage = 20.0/32.0; p.ExtraDescription = "Cutout_0_625";},
            };
            PerformTestSet(modifiers, todo);
        }

        private static void Train_CIFAR100_WRN(WideResNetBuilder p, int WRN_depth, int WRN_k)
        {
            var network = p.WRN(WRN_depth, WRN_k, CIFAR100DataLoader.InputShape_CHW, CIFAR100DataLoader.Categories);
            using (var loader = new CIFAR100DataLoader())
            {
                var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
                network.Fit(loader.Training, learningRateComputer, p.NumEpochs, p.BatchSize, loader.Test);
            }
            network.Dispose();
        }
        #endregion


        /// <summary>
        /// Train a network on CIFAR-10 data set 
        /// </summary>
        private static void Train_CIFAR10(NetworkBuilder p, Func<Network> buildNetwork)
        {
            using (var loader = new CIFAR10DataLoader())
            {
                var network = buildNetwork();
                var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
                network.Fit(loader.Training, learningRateComputer, p.NumEpochs, p.BatchSize, loader.Test);
                network.Dispose();
            }
        }

        private static void Train_CIFAR10_WRN(WideResNetBuilder p, int WRN_depth, int WRN_k)
        {
            var network = p.WRN(WRN_depth, WRN_k, CIFAR10DataLoader.InputShape_CHW, CIFAR10DataLoader.Categories);
            using (var loader = new CIFAR10DataLoader())
            {
                var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
                network.Fit(loader.Training, learningRateComputer, p.NumEpochs, p.BatchSize, loader.Test);
            }
            network.Dispose();
        }

        private static void Train_DogsVsCat_WRN_TransferLearning(WideResNetBuilder p)
        {
            int heightAndWidth = 32;
            var dataDirectory = @"C:\Users\Franck\AppData\Local\SharpNet\Data\dogs-vs-cats\train_filter_resize_32_32";
            var trainingDirectory = DogsVsCats.ValueOf(dataDirectory, heightAndWidth, heightAndWidth, null).Shuffle(new Random(0)).Take(2000);
            var trainingAndValidationSet = trainingDirectory.SplitIntoTrainingAndValidation(0.5);

            var network = Network.ValueOf(@"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-4_20190816_1810_150.txt", p.GpuDeviceId);
            //network.Config.WithCifar10WideResNetLearningRateScheduler(true, true, false);
            network.ChangeNumberOfCategoriesForTransferLearning(trainingDirectory.Categories);

            network.Info("training only last layer");
            network.Layers.ForEach(l=>l.Trainable = false);
            network.LastFrozenLayer().Trainable = true;
            network.EpochDatas.Clear();
            p.NumEpochs = 150;
            var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
            network.Fit(trainingAndValidationSet.Training, learningRateComputer, p.NumEpochs, p.BatchSize, trainingAndValidationSet.Test);
            network.Dispose();
            trainingAndValidationSet.Dispose();
        }

        private static void Train_DogsVsCat_WRN(WideResNetBuilder p, int WRN_depth, int WRN_k)
        {
            int heightAndWidth = 32;
            var dataDirectory = @"C:\Users\Franck\AppData\Local\SharpNet\Data\dogs-vs-cats\train_filter_resize_32_32";
            var trainingDirectory = DogsVsCats.ValueOf(dataDirectory, heightAndWidth, heightAndWidth, null).Shuffle(new Random(0)).Take(2000);
            var trainingAndValidationSet = trainingDirectory.SplitIntoTrainingAndValidation(0.5);
            var network = p.WRN(WRN_depth, WRN_k, trainingDirectory.InputShape_CHW, trainingDirectory.Categories);
            var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
            network.Fit(trainingAndValidationSet.Training, learningRateComputer, p.NumEpochs, p.BatchSize, trainingAndValidationSet.Test);
            network.Dispose();
            trainingAndValidationSet.Dispose();
        }
        private static void Train_RecursionCellularImageClassification_WRN(WideResNetBuilder p, int WRN_depth, int WRN_k)
        {
            //int heightAndWidth = 512;
            //var dataDirectory = @"C:\Users\fzibi\AppData\Local\SharpNet\Data\recursion-cellular-image-classification\train";
            //var csvFilename = @"C:\Users\fzibi\AppData\Local\SharpNet\Data\recursion-cellular-image-classification\train.csv";
            //var trainingDirectory = new RecursionCellularImageClassificationDataLoader(originalCsvFilename, dataDirectory, heightAndWidth, heightAndWidth, 0.9, networkLogger);
            int heightAndWidth = 256;
            var dataDirectory = @"C:\Users\fzibi\AppData\Local\SharpNet\Data\recursion-cellular-image-classification\train_resize_" + heightAndWidth + "_" + heightAndWidth;
            var csvFilename = @"C:\Users\fzibi\AppData\Local\SharpNet\Data\recursion-cellular-image-classification\train.csv";
            var trainingDirectory = RecursionCellularImageClassification.ValueOf(csvFilename, dataDirectory, heightAndWidth, heightAndWidth, p.NetworkLogger("WRN-" + WRN_depth + "-" + WRN_k));
            var trainingAndValidationSet = trainingDirectory.SplitIntoTrainingAndValidation(0.9);
            //var network = p.WRN(WRN_depth, WRN_k, trainingDirectory.InputShape_CHW, trainingDirectory.Categories);
            var network = p.WRN_ImageNet(WRN_depth, WRN_k, trainingDirectory.InputShape_CHW, trainingDirectory.Categories);
            var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
            network.Fit(trainingAndValidationSet.Training, learningRateComputer, p.NumEpochs, p.BatchSize, trainingAndValidationSet.Test);
            network.Dispose();
            trainingAndValidationSet.Dispose();
        }

        private static void Train_Aptos2019Blindness_WRN(WideResNetBuilder p, int WRN_depth, int WRN_k, int heightAndWidth)
        {
            var csvFilename = @"C:\temp\aptos2019-blindness-detection\train_images\train.csv";
            var dataDirectory = @"C:\temp\aptos2019-blindness-detection\train_images_cropped_square_resize_" + heightAndWidth + "_" + heightAndWidth;
            var trainingDirectory = Aptos2019BlindnessDetection.ValueOf(csvFilename, dataDirectory, heightAndWidth, heightAndWidth, p.NetworkLogger("WRN-" + WRN_depth + "-" + WRN_k));
            var trainingAndValidationSet = trainingDirectory.SplitIntoTrainingAndValidation(0.9);
            var network = p.WRN(WRN_depth, WRN_k, trainingDirectory.InputShape_CHW, trainingDirectory.Categories);
            var learningRateComputer = network.Config.GetLearningRateComputer(p.InitialLearningRate, p.NumEpochs);
            network.Fit(trainingAndValidationSet.Training, learningRateComputer, p.NumEpochs, p.BatchSize, trainingAndValidationSet.Test);
            network.Dispose();
            trainingAndValidationSet.Dispose();
        }
       
        private static void ConsumersLaunchingTests(int gpuDeviceId, BlockingCollection<Action<int>> produced)
        {
            Console.WriteLine("Computations on GPU " + gpuDeviceId+" have started (ThreadId"+Thread.CurrentThread.ManagedThreadId+")");
            foreach (var action in produced.GetConsumingEnumerable())
            {
                action(gpuDeviceId);
            }
            Console.WriteLine("Last computation on GPU " + gpuDeviceId+ " is in progress");
        }
        private static void PerformTestSet<T>(List<Action<T>> networkDeformers, List<Action<T, int>> networks) where T: NetworkBuilder, new()
        {
            int nbGPUs = GPUWrapper.GetDeviceCount();
            var totalTests = networkDeformers.Count * networks.Count;
            nbGPUs = Math.Min(nbGPUs, totalTests);
            Console.WriteLine("Computation will be done on "+nbGPUs+" GPU(s)");
            var taskToBePerformed = new BlockingCollection<Action<int>>(1);
            var consumers = Enumerable.Range(0, nbGPUs).Select(gpuDeviceId => Task.Run(() => ConsumersLaunchingTests(gpuDeviceId, taskToBePerformed))).ToArray();
            var nbPerformedTests = 0;
            for (int networkDeformerIndex = 0; networkDeformerIndex < networkDeformers.Count; ++networkDeformerIndex)
            {
                var networkMetaParametersDeformer = networkDeformers[networkDeformerIndex];
                for (int networkIndex = 0; networkIndex < networks.Count; ++networkIndex)
                {
                    var network = networks[networkIndex];
                    int testIdx = networkDeformerIndex* networks.Count + networkIndex+1;
                    var networkMetaParameters = new T();
                    networkMetaParametersDeformer(networkMetaParameters);
                    Console.WriteLine("Adding test " + (networkDeformerIndex + 1) + "." + (networkIndex + 1) + " (#" + testIdx + "/" + totalTests + ") in queue  ('" + networkMetaParameters.ExtraDescription + "')");
                    taskToBePerformed.Add(gpuDeviceId => network(networkMetaParameters, gpuDeviceId));
                    ++nbPerformedTests;
                    Console.WriteLine(new string('-', 80));
                    Console.WriteLine("Progress: " + ((100.0 * nbPerformedTests) / totalTests));
                    Console.WriteLine(new string('-', 80));
                }
            }
            taskToBePerformed.CompleteAdding();
            Task.WaitAll(consumers);
        }
    }
}
