using System;
using System.Collections.Generic;
using SharpNet;
using SharpNet.Pictures;

namespace SharpNetTests
{
    static class Program
    {
        private static void Main()
        {
            var todo = new List<Action>
            {
                //()=> new NonReg.TestResNetCIFAR10().TestResNet11V2_CIFAR10(),
                //()=> new NonReg.TestResNetCIFAR10().TestResNet20V1_CIFAR10(),
                ()=> new NonReg.TestResNetCIFAR10().TestResNet20V2_CIFAR10(),
                ()=> new NonReg.TestResNetCIFAR10().TestResNet32V1_CIFAR10(),
                () => new NonReg.TestResNetCIFAR10().TestResNet44V1_CIFAR10(),
                () => new NonReg.TestResNetCIFAR10().TestResNet56V1_CIFAR10(),
                //() => new NonReg.TestResNetCIFAR10().TestResNet56V2_CIFAR10(),
                //() => new NonReg.TestResNetCIFAR10().TestResNet110V1_CIFAR10(),
                //() => new NonReg.TestResNetCIFAR10().TestResNet110V2_CIFAR10(),
                //() => new NonReg.TestResNetCIFAR10().TestResNet164V1_CIFAR10(),
                //() => new NonReg.TestResNetCIFAR10().TestResNet164V2_CIFAR10()
            };

            var modifiers = new List<Action>
            {
                //https://sgugger.github.io/the-1cycle-policy.html
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 70;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 0.8;ResNetUtils.OneCycleDividerForMinLearningRate = 10;ResNetUtils.ExtraDescription = "_OneCycle_080_008_70Epochs";},
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 100;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 0.8;ResNetUtils.OneCycleDividerForMinLearningRate = 10;ResNetUtils.ExtraDescription = "_OneCycle_080_008_100Epochs";},
                () =>{ResNetUtils.BatchSize=-1;ResNetUtils.ExtraDescription = "_Auto_BatchSize";},
                () => {},
                () =>{ResNetUtils.FillMode=ImageDataGenerator.FillModeEnum.Reflect;ResNetUtils.ExtraDescription = "_FillMode_Reflect";},
                () =>{ResNetUtils.HorizontalFlip=false;ResNetUtils.ExtraDescription = "_no_HorizontalFlip";},
                () =>{ResNetUtils.lambdaL2Regularization=0.0005;ResNetUtils.ExtraDescription = "_lambdaL2Regularization_0_0005";},

                /*
                () =>{ResNetUtils.WidthShiftRange=0.0;ResNetUtils.HeightShiftRange=0.0;ResNetUtils.ExtraDescription = "_no_ShiftRange";},
                () =>{ResNetUtils.WidthShiftRange=0.0;ResNetUtils.HeightShiftRange=0.0;ResNetUtils.HorizontalFlip=false;ResNetUtils.ExtraDescription = "_no_DataAugmentation_onlyShuffling";},
                */

            //https://sgugger.github.io/the-1cycle-policy.html
            //() =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 50;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 3.0;ResNetUtils.OneCycleDividerForMinLearningRate = 20;ResNetUtils.ExtraDescription = "_OneCycle_300_015_50Epochs";},
            //https://sgugger.github.io/the-1cycle-policy.html
            //() =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 70;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 3.0;ResNetUtils.OneCycleDividerForMinLearningRate = 20;ResNetUtils.ExtraDescription = "_OneCycle_300_015_70Epochs";},

        };

            var totalTest = modifiers.Count * todo.Count;
            var nbPerformedtests = 0;
            for (int modifierIndex = 0; modifierIndex < modifiers.Count; ++modifierIndex)
            {
                Console.WriteLine("Starting all tests of family " + (modifierIndex + 1) + ".*");
                for (int testIndex = 0; testIndex < todo.Count; ++testIndex)
                {

                    ResNetUtils.lambdaL2Regularization = 1e-4;
                    ResNetUtils.LinearLearningRate = false;
                    ResNetUtils.ExtraDescription = "";
                    ResNetUtils.NumEpochs = 160;
                    ResNetUtils.BatchSize = 128;
                    ResNetUtils.OneCycleLearningRate = false;
                    ResNetUtils.InitialLearningRate = 0.1;
                    ResNetUtils.OneCycleDividerForMinLearningRate = 10;
                    ResNetUtils.OneCyclePercentInAnnealing = 0.2;
                    ResNetUtils.CutoutPatchlength = 16;
                    ResNetUtils.WidthShiftRange = 0.1;
                    ResNetUtils.HeightShiftRange = 0.1;
                    ResNetUtils.HorizontalFlip = true;
                    ResNetUtils.VerticalFlip = false;
                    ResNetUtils.FillMode = ImageDataGenerator.FillModeEnum.Nearest;

                    modifiers[modifierIndex]();
                    Console.WriteLine("Starting test " + (modifierIndex + 1) + "." + (testIndex + 1) + " ('" + ResNetUtils.ExtraDescription + "'), last one will be " + modifiers.Count + "." + todo.Count);
                    {
                        Console.WriteLine("Starting test: '" + ResNetUtils.ExtraDescription+"' (this test includes "+ todo.Count+" networks)");
                    }
                    todo[testIndex]();
                    ++nbPerformedtests;
                    if (modifierIndex == modifiers.Count-1)
                    {
                        Console.WriteLine("End of test: '" + ResNetUtils.ExtraDescription + "'");
                    }
                    Console.WriteLine(new string('-',80));
                    Console.WriteLine("Progress: " + ((100.0 * nbPerformedtests) / totalTest));
                    Console.WriteLine(new string('-', 80));
                }
            }

            /*
            new NonReg.TestResNetCIFAR10().TestResNet56V1_CIFAR10();
            new NonReg.TestResNetCIFAR10().TestResNet11V2_CIFAR10();
            new NonReg.TestResNetCIFAR10().TestResNet20V2_CIFAR10();
            new NonReg.TestResNetCIFAR10().TestResNet29V2_CIFAR10();
            new NonReg.TestResNetCIFAR10().TestResNet56V2_CIFAR10();
            */
            /*
            new NonReg.TestResNetCIFAR10().TestResNet20V1_CIFAR10();
            new NonReg.TestResNetCIFAR10().TestResNet20V2_CIFAR10();
            new NonReg.TestResNetCIFAR10().TestResNet32V1_CIFAR10();
            new NonReg.TestResNetCIFAR10().TestResNet44V1_CIFAR10();
            new NonReg.TestResNetCIFAR10().TestResNet56V1_CIFAR10();
            new NonReg.TestResNetCIFAR10().TestResNet56V2_CIFAR10();
            */
            /*
            new NonReg.TestResNetCIFAR10().TestResNet110V1_CIFAR10(),
            new NonReg.TestResNetCIFAR10().TestResNet110V2_CIFAR10(),
            new NonReg.TestResNetCIFAR10().TestResNet164V1_CIFAR10(),
            new NonReg.TestResNetCIFAR10().TestResNet164V2_CIFAR10(),
            */
            //new NonReg.TestResNetCIFAR10().TestResNet1202V1_CIFAR10();
            //new NonReg.TestResNetCIFAR10().TestResNet1001V2_CIFAR10();
            //new NonReg.TestResNetCIFAR10().TestAllResNetV1_CIFAR10();
            //new NonReg.TestResNetCIFAR10().TestResNet20V1_CIFAR10();
            //new TestGradient().TestGradientForDenseLayer(true, true);
            //new NonReg.TestMNIST().Test();
            //new NonReg.TestNetworkPropagation().TestParallelRunWithTensorFlow();
            //new NonReg.TestBenchmark().TestGPUBenchmark_Memory();new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
            //new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
        }
    }
}
