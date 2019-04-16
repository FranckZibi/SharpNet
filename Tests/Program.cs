using System;
using System.Collections.Generic;
using SharpNet;

namespace SharpNetTests
{
    static class Program
    {
        private static void Main()
        {
            var todo = new List<Action>
            {
                () => new NonReg.TestResNetCIFAR10().TestResNet56V1_CIFAR10(),
                ()=> new NonReg.TestResNetCIFAR10().TestResNet11V2_CIFAR10(),
                ()=> new NonReg.TestResNetCIFAR10().TestResNet20V1_CIFAR10(),
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
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 50;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 3.0;ResNetUtils.OneCycleDividerForMinLearningRate = 20;ResNetUtils.ExtraDescription = "_OneCycle_300_015_50Epochs";},
                //https://sgugger.github.io/the-1cycle-policy.html
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 70;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 3.0;ResNetUtils.OneCycleDividerForMinLearningRate = 20;ResNetUtils.ExtraDescription = "_OneCycle_300_015_70Epochs";},
                //https://sgugger.github.io/the-1cycle-policy.html
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 100;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 0.8;ResNetUtils.OneCycleDividerForMinLearningRate = 10;ResNetUtils.ExtraDescription = "_OneCycle_080_008_100Epochs";},
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 100;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 0.1;ResNetUtils.OneCycleDividerForMinLearningRate = 10;ResNetUtils.ExtraDescription = "_OneCycle_010_001_100Epochs";},

                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 50;ResNetUtils.lambdaL2Regularization = 5*1e-4;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 3.0;ResNetUtils.OneCycleDividerForMinLearningRate = 20;ResNetUtils.ExtraDescription = "_OneCycle_300_015_L2_0_0005_50Epochs";},
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 70;ResNetUtils.lambdaL2Regularization = 5*1e-4;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 3.0;ResNetUtils.OneCycleDividerForMinLearningRate = 20;ResNetUtils.ExtraDescription = "_OneCycle_300_015_L2_0_0005_70Epochs";},
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 100;ResNetUtils.lambdaL2Regularization = 5*1e-4;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 0.8;ResNetUtils.OneCycleDividerForMinLearningRate = 10;ResNetUtils.ExtraDescription = "_OneCycle_080_008_L2_0_0005_100Epochs";},
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 100;ResNetUtils.lambdaL2Regularization = 5*1e-4;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 0.1;ResNetUtils.OneCycleDividerForMinLearningRate = 10;ResNetUtils.ExtraDescription = "_OneCycle_010_001_L2_0_0005_100Epochs";},

                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 50;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 3.0;ResNetUtils.OneCycleDividerForMinLearningRate = 20;ResNetUtils.OneCyclePercentInAnnealing = 0.4;ResNetUtils.ExtraDescription = "_OneCycle_300_015_40Annealing_50Epochs";},
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 70;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 3.0;ResNetUtils.OneCycleDividerForMinLearningRate = 20;ResNetUtils.OneCyclePercentInAnnealing = 0.4;ResNetUtils.ExtraDescription = "_OneCycle_300_015_40Annealing_70Epochs";},
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 100;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 0.8;ResNetUtils.OneCycleDividerForMinLearningRate = 10;ResNetUtils.OneCyclePercentInAnnealing = 0.4;ResNetUtils.ExtraDescription = "_OneCycle_080_008_40Annealing_100Epochs";},
                () =>{ResNetUtils.OneCycleLearningRate =true;ResNetUtils.NumEpochs = 100;ResNetUtils.BatchSize = -1;ResNetUtils.InitialLearningRate = 0.1;ResNetUtils.OneCycleDividerForMinLearningRate = 10;ResNetUtils.OneCyclePercentInAnnealing = 0.4;ResNetUtils.ExtraDescription = "_OneCycle_010_001_40Annealing_100Epochs";},

                //() => {},
                //() =>{ResNetUtils.lambdaL2Regularization = 5*1e-4;ResNetUtils.ExtraDescription = "_L2_0_0005";},
        };
            

            var totalTest = modifiers.Count * todo.Count;
            var nbPerformedtests = 0;
            for (int testIndex = 0; testIndex < todo.Count; ++testIndex)
            {
                for (int modifierIndex = 0; modifierIndex < modifiers.Count; ++modifierIndex)
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

                    modifiers[modifierIndex]();
                    todo[testIndex]();
                    ++nbPerformedtests;
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
