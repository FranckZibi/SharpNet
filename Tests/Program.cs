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
                ()=> new NonReg.TestResNetCIFAR10().TestResNet11V2_CIFAR10(),
                ()=> new NonReg.TestResNetCIFAR10().TestResNet20V1_CIFAR10(),
                ()=> new NonReg.TestResNetCIFAR10().TestResNet20V2_CIFAR10(),
                ()=> new NonReg.TestResNetCIFAR10().TestResNet32V1_CIFAR10(),
                () => new NonReg.TestResNetCIFAR10().TestResNet44V1_CIFAR10(),
                () => new NonReg.TestResNetCIFAR10().TestResNet56V1_CIFAR10(),
                () => new NonReg.TestResNetCIFAR10().TestResNet56V2_CIFAR10(),
                () => new NonReg.TestResNetCIFAR10().TestResNet110V1_CIFAR10(),
                () => new NonReg.TestResNetCIFAR10().TestResNet110V2_CIFAR10(),
                () => new NonReg.TestResNetCIFAR10().TestResNet164V1_CIFAR10(),
                () => new NonReg.TestResNetCIFAR10().TestResNet164V2_CIFAR10()
            };

            var modifiers = new List<Action>
            {
                () => {ResNetUtils.ExtraDescription = "";}, 
                //() =>{ResNetUtils.lambdaL2Regularization = 3*1e-3;ResNetUtils.ForceTensorflowCompatibilityMode = false;ResNetUtils.ExtraDescription = "_L2_0_003";},
                //() =>{ResNetUtils.lambdaL2Regularization = 1e-4;ResNetUtils.ForceTensorflowCompatibilityMode = true;ResNetUtils.ExtraDescription = "_ForceTensorflowCompatibilityMode";},
            };


            var totalTest = modifiers.Count * todo.Count;
            var nbPerformedtests = 0;
            for (int testIndex = 0; testIndex < todo.Count; ++testIndex)
            {
                for (int modifierIndex = 0; modifierIndex < modifiers.Count; ++modifierIndex)
                {
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
