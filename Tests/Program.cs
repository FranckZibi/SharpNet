using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using SharpNet;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.Optimizers;

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


            ResNetTests();
            DenseNetTests();

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

        #region Training
        /// <summary>
        /// Train a network on CIFAR10 data set 
        /// </summary>
        ///
        private static void Train_CIFAR10(NetworkBuilder p, Network network, ILearningRateScheduler lrScheduler = null, bool autoBatchSize = false)
        {
            CIFAR10.LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            network.Fit(xTrain, yTrain, lrScheduler ?? p.Cifar10LearningRateScheduler(), p.Cifar10ReduceLROnPlateau(), p.NumEpochs, autoBatchSize ? -1 : p.BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        #endregion

        private static void DenseNetTests()
        {
            var todo = new List<Action<DenseNetBuilder,int>>
            {
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_12_40_CIFAR10());},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNetBC_12_100_CIFAR10());},

                /*(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_Fast_CIFAR10());},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_12_10_CIFAR10());},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNet_12_40_CIFAR10());},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.DenseNetBC_12_40_CIFAR10());},
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

                (p) =>{p.NumEpochs = 240;p.BatchSize = -1;p.Config.WithSGD(0.9,true); p.Config.ForceTensorflowCompatibilityMode = true;p.CutoutPatchlength = 0;p.ExtraDescription = "_240Epochs_ForceTensorflowCompatibilityMode_CutoutPatchlength0_WithNesterov_EnhancedMemory";},
                (p) =>{p.NumEpochs = 240;p.BatchSize = -1;p.Config.WithSGD(0.9,false); p.Config.ForceTensorflowCompatibilityMode = true;p.CutoutPatchlength = 0;p.ExtraDescription = "_240Epochs_ForceTensorflowCompatibilityMode_CutoutPatchlength0_NoNesterov_EnhancedMemory";},


                //(p) =>{p.NumEpochs = 300;p.BatchSize = -1;p.CutoutPatchlength = 16;p.ExtraDescription = "_200Epochs_L2InDense_CutoutPatchlength16";},
                //(p) =>{p.NumEpochs = 300;p.BatchSize = -1;p.CutoutPatchlength = 0;p.ExtraDescription = "_200Epochs_L2_InDense_CutoutPatchlength0";},
                //(p) =>{p.NumEpochs = 200;p.Config.WithSGD(0.9,false);p.ExtraDescription = "_200Epochs_NoNesterov";},

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
            PerformTestSet(metaParametersModifiers, todo);
        }

        public static void ResNetTests()
        {
            var todo = new List<Action<ResNetBuilder, int>>
            {
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet11V2_CIFAR10());},
                //(x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet20V2_CIFAR10());},

                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet20V1_CIFAR10());},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet32V1_CIFAR10());},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet44V1_CIFAR10());},
                (x,gpuDeviceId) =>{x.GpuDeviceId=gpuDeviceId;Train_CIFAR10(x, x.ResNet56V1_CIFAR10());},
            };

            var modifiers = new List<Action<ResNetBuilder>>
            {
                (p) =>{p.Config.WithSGD(0.9,true);p.ExtraDescription = "_SGD_WithNesterov_EnhancedMemory";},
                (p) =>{p.Config.WithSGD(0.9,true);p.Config.ForceTensorflowCompatibilityMode = true;p.ExtraDescription = "_SGD_WithNesterov_ForceTensorflowCompatibilityMode_EnhancedMemory";},
                (p) =>{p.Config.WithSGD(0.9,false);p.ExtraDescription = "_SGD_NoNesterov_EnhancedMemory";},
                (p) =>{p.Config.WithSGD(0.9,false);p.Config.ForceTensorflowCompatibilityMode = true;p.ExtraDescription = "_SGD_NoNesterov_ForceTensorflowCompatibilityMode_EnhancedMemory";},
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


        private static void ConsumersLauchingTests(int gpuDeviceId, BlockingCollection<Action<int>> produced)
        {
            Console.WriteLine("Computations on GPU " + gpuDeviceId+" have started (ThreadId"+Thread.CurrentThread.ManagedThreadId+")");
            foreach (var action in produced.GetConsumingEnumerable())
            {
                action(gpuDeviceId);
            }
            Console.WriteLine("Computations on GPU " + gpuDeviceId + " has ended");
        }
        private static void PerformTestSet<T>(List<Action<T>> metaParameterModifiers, List<Action<T, int>> allNetworks) where T: NetworkBuilder, new()
        {
            int nbGPUs = GPUWrapper.GetDeviceCount();
            Console.WriteLine("Computation will be done on "+nbGPUs+" GPU(s)");
            var taskToBePerformed = new BlockingCollection<Action<int>>(1);
            var consumers = Enumerable.Range(0, nbGPUs).Select(gpuDeviceId => Task.Run(() => ConsumersLauchingTests(gpuDeviceId, taskToBePerformed))).ToArray();
            var totalTest = metaParameterModifiers.Count * allNetworks.Count;
            var nbPerformedtests = 0;
            for (int metaParameterModifiersIndex = 0; metaParameterModifiersIndex < metaParameterModifiers.Count; ++metaParameterModifiersIndex)
            {
                for (int networkIndex = 0; networkIndex < allNetworks.Count; ++networkIndex)
                {
                    int testIdx = metaParameterModifiersIndex* allNetworks.Count + networkIndex+1;
                    int totalTests = metaParameterModifiers.Count*allNetworks.Count;
                    var metaParameters = new T();
                    metaParameterModifiers[metaParameterModifiersIndex](metaParameters);
                    var action = allNetworks[networkIndex];
                    Console.WriteLine("Adding test  " + (metaParameterModifiersIndex + 1) + "." + (networkIndex + 1) + "(#" + testIdx + "/" + totalTests + ") in queue  ('" + metaParameters.ExtraDescription + "')");
                    taskToBePerformed.Add(gpuDeviceId => action(metaParameters, gpuDeviceId));
                    ++nbPerformedtests;
                    Console.WriteLine(new string('-', 80));
                    Console.WriteLine("Progress: " + ((100.0 * nbPerformedtests) / totalTest));
                    Console.WriteLine(new string('-', 80));
                }
            }
            taskToBePerformed.CompleteAdding();
            Task.WaitAll(consumers);
        }
    }
}
