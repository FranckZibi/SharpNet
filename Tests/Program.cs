using System;
using System.Collections.Generic;
using SharpNet;

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
            
            DenseNetTests();
            //ResNetTests();

            //TestSpeed();return;
            //new TestGradient().TestGradientForDenseLayer(true, true);
            //new NonReg.TestMNIST().Test();
            //new NonReg.TestNetworkPropagation().TestParallelRunWithTensorFlow();
            //new NonReg.TestBenchmark().TestGPUBenchmark_Memory();new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
            //new NonReg.TestBenchmark().TestGPUBenchmark_Speed();
        }

        /*
        private static void TestConcatSpeed()
        {
            var gpuWrapper = GPUWrapper.Default;
            var rand = new Random(0);
            var x1 = TestCpu
            or.RandomFloatTensor(new[] { 128, 32, 16, 16 }, rand, -1.5, +1.5, "a");
            var x2 = TestCpuTensor.RandomFloatTensor(new[] { x1.Shape[0], 320, x1.Shape[2], x1.Shape[3] }, rand, -1.5, +1.5, "a");
            var concat = TestCpuTensor.RandomFloatTensor(new[] { x1.Shape[0], x1.Shape[1] + x2.Shape[1], x1.Shape[2], x1.Shape[3] }, rand, -1.5, +1.5, "a");

            var x1Gpu = x1.ToGPU<float>(gpuWrapper);
            var x2Gpu = x2.ToGPU<float>(gpuWrapper);
            var concatGpu = concat.ToGPU<float>(gpuWrapper);
            int nbTests = 15000;

            concat.Concatenate(x1, x2); //WarmUp
            concat.Concatenate(x1, x2); //WarmUp
            var spGPU = Stopwatch.StartNew();
            for (int i = 0; i < nbTests; ++i)
            {
                concatGpu.Concatenate(x1Gpu, x2Gpu);
                concatGpu.Split(x1Gpu, x2Gpu);
            }
            spGPU.Stop();
            Console.WriteLine("Elapsed GPU (ms): " + spGPU.Elapsed.TotalMilliseconds / nbTests);
        }
        */

        private static void DenseNetTests()
        {
            var todo = new List<Action<DenseNetConfig>>
            {
                //x => x.TrainDenseNet_12_10_CIFAR10(),
                x => x.TrainDenseNet_12_40_CIFAR10(),
                x => x.TrainDenseNetBC_12_40_CIFAR10(),
                //x=> x.TrainDenseNetBC_12_100_CIFAR10()
            };

            var modifiers = new List<Action<DenseNetConfig>>
            {
                //(p) =>{p.UseNesterov = false; p.NumEpochs = 50; p.ExtraDescription = "_50Epoch_no_nesterov";},
                //(p) =>{p.UseAdam = true; p.NumEpochs = 5; p.ExtraDescription = "_50Epoch_Adam";},
                //(p) =>{p.SaveNetworkStatsAfterEachEpoch = true; p.ExtraDescription = "_Adam_with_l2_inConv";},
                //(p) =>{p.SaveNetworkStatsAfterEachEpoch = false;p.SaveLossAfterEachMiniBatch = false;p.UseAdam = true;p.UseNesterov = false;p.BatchSize = -1;p.ForceTensorflowCompatibilityMode = false;p.NumEpochs = 150; p.ExtraDescription = "_Adam";},
                (p) =>{ p.ExtraDescription = "_OrigPaper";},
                (p) =>{p.NumEpochs = 150;p.ExtraDescription = "_150Epochs";},
                (p) =>{p.NumEpochs = 150;p.Config.WithSGD(0.9,false);p.ExtraDescription = "_150Epochs_NoNesterov";},
                //(p) =>{p.Config.WithSGD(0.9,false);p.BatchSize = -1;p.Config.ForceTensorflowCompatibilityMode = false;p.NumEpochs = 300; p.ExtraDescription = "_SGD";},
                //(p) =>{p.UseAdam = false;p.UseNesterov = true;p.BatchSize = -1;p.ForceTensorflowCompatibilityMode = false;p.NumEpochs = 300; p.ExtraDescription = "_SGDNesterov";},
                //(p) =>{p.UseAdam = true;p.UseNesterov = false;p.BatchSize = -1;p.ForceTensorflowCompatibilityMode = false;p.NumEpochs = 200;p.InitialLearningRate = 0.001;p.ExtraDescription = "_Adam_0_001";},
                //(p) =>{p.SaveNetworkStatsAfterEachEpoch = true;p.UseDoublePrecision = true;p.UseAdam = true;p.BatchSize = -1;p.ExtraDescription = "_ForceTensorflowCompatibilityMode_UseDoublePrecision";},
                //(p) =>{p.UseAdam = true;p.BatchSize = 50;p.SaveNetworkStatsAfterEachEpoch = true; p.NumEpochs = 1; p.ExtraDescription = "_Adam_with_l2_inConv";},
                //(p) =>{p.UseAdam = true; p.lambdaL2Regularization = 0.0;p.SaveNetworkStatsAfterEachEpoch = true; p.NumEpochs = 1; p.ExtraDescription = "_Adam_no_l2_inConv";},

                //(p) =>{p.UseAdam = true;p.SaveNetworkStatsAfterEachEpoch = true; p.lambdaL2Regularization = 0.0;p.NumEpochs = 2; p.ExtraDescription = "_Adam_no_lambdaL2Regularization";},
                //(p) =>{p.lambdaL2Regularization = 0.0;p.UseNesterov = false;p.NumEpochs = 50; p.ExtraDescription = "_50Epoch_no_nesterov_no_lambdaL2Regularization";},
                #region already performed tests
            #endregion
            };
            PerformTestSet(modifiers, todo);
        }

        public static void ResNetTests()
        {
            var todo = new List<Action<ResNetConfig>>
            {
                x=> x.TrainResNet20V1_CIFAR10(),

                /*
                //TrainResNet.TrainResNet11V2_CIFAR10,
                TrainResNet.TrainResNet20V1_CIFAR10,
                //TrainResNet.TrainResNet20V2_CIFAR10,
                TrainResNet.TrainResNet32V1_CIFAR10,
                TrainResNet.TrainResNet44V1_CIFAR10,
                TrainResNet.TrainResNet56V1_CIFAR10,
                //TrainResNet.TrainResNet56V2_CIFAR10,
                //TrainResNet.TrainResNet110V1_CIFAR10,
                //TrainResNet.TrainResNet110V2_CIFAR10,
                //TrainResNet.TrainResNet164V1_CIFAR10,
                //TrainResNet.TrainResNet164V2_CIFAR10 */
            };

            var modifiers = new List<Action<ResNetConfig>>
            {
                (p) =>{p.Config.WithSGD(0.9,false);p.BatchSize = -1;p.NumEpochs = 21;p.ExtraDescription = "_SGD";},
                #region already performed tests
                /*
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
        private static void PerformTestSet<T>(List<Action<T>> modifiers, List<Action<T>> todo) where T: IMetaParameters, new()
        {
            var totalTest = modifiers.Count * todo.Count;
            var nbPerformedtests = 0;
            for (int modifierIndex = 0; modifierIndex < modifiers.Count; ++modifierIndex)
            {
                Console.WriteLine("Starting all tests of family " + (modifierIndex + 1) + ".*");
                for (int testIndex = 0; testIndex < todo.Count; ++testIndex)
                {
                    var param = new T();
                    modifiers[modifierIndex](param);
                    Console.WriteLine("Starting test " + (modifierIndex + 1) + "." + (testIndex + 1) + " ('" + param.ExtraDescription + "'), last one will be " + modifiers.Count + "." + todo.Count);
                    {
                        Console.WriteLine("Starting test: '" + param.ExtraDescription + "' (this test includes " + todo.Count + " networks)");
                    }
                    todo[testIndex](param);
                    ++nbPerformedtests;
                    if (modifierIndex == modifiers.Count - 1)
                    {
                        Console.WriteLine("End of test: '" + param.ExtraDescription + "'");
                    }

                    Console.WriteLine(new string('-', 80));
                    Console.WriteLine("Progress: " + ((100.0 * nbPerformedtests) / totalTest));
                    Console.WriteLine(new string('-', 80));
                }
            }
        }
    }
}
