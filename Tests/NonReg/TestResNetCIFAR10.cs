using System.Diagnostics;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;

namespace SharpNetTests.NonReg
{
    /// <summary>
    /// Train a ResNet V1 network on Cifar10 dataset has described in https://arxiv.org/pdf/1512.03385.pdf
    /// </summary>
    [TestFixture]
    public class TestResNetCIFAR10
    {
        private const int NumEpochs = 200;
        private const int BatchSize = 128;

        [Test, Explicit]
        public void TestResNet20V1_CIFAR10()
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet20V1_CIFAR10(true, false, Logger(nameof(ResNetUtils.ResNet20V1_CIFAR10)));
            network.Fit(xTrain, yTrain, ResNetUtils.Cifar10LearningRateScheduler(), NumEpochs, BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        [Test, Explicit]
        public void TestResNet32V1_CIFAR10()
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet32V1_CIFAR10(true, false, Logger(nameof(ResNetUtils.ResNet32V1_CIFAR10)));
            network.Fit(xTrain, yTrain, ResNetUtils.Cifar10LearningRateScheduler(), NumEpochs, BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        [Test, Explicit]
        public void TestResNet44V1_CIFAR10()
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet44V1_CIFAR10(true, false, Logger(nameof(ResNetUtils.ResNet44V1_CIFAR10)));
            network.Fit(xTrain, yTrain, ResNetUtils.Cifar10LearningRateScheduler(), NumEpochs, BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        [Test, Explicit]
        public void TestResNet56V1_CIFAR10()
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet56V1_CIFAR10(true, false, Logger(nameof(ResNetUtils.ResNet56V1_CIFAR10)));
            network.Fit(xTrain, yTrain, ResNetUtils.Cifar10LearningRateScheduler(), NumEpochs, BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        [Test, Explicit]
        public void TestResNet110V1_CIFAR10()
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet110V1_CIFAR10(true, false, Logger(nameof(ResNetUtils.ResNet110V1_CIFAR10)));
            network.Fit(xTrain, yTrain, ResNetUtils.Cifar10LearningRateScheduler(), NumEpochs, BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        [Test, Explicit]
        public void TestResNet1202V1_CIFAR10()
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet1202V1_CIFAR10(true, false, Logger(nameof(ResNetUtils.ResNet1202V1_CIFAR10)));
            network.Fit(xTrain, yTrain, ResNetUtils.Cifar10LearningRateScheduler(), NumEpochs, BatchSize, xTest, yTest);
            network.ClearMemory();
        }

        private Logger Logger(string networkName)
        {
            var logFileName = Utils.ConcatenatePathWithFileName(@"c:\temp\ML\",
                networkName + "_" + Process.GetCurrentProcess().Id + "_" +
                System.Threading.Thread.CurrentThread.ManagedThreadId + ".log");
            return new Logger(logFileName, true);
        }
        private static void LoadCifar10(out CpuTensor<float> xTrain, out CpuTensor<float> yTrain, out CpuTensor<float> xTest, out CpuTensor<float> yTest)
        {
            ResNetUtils.Load(out CpuTensor<byte> xTrainingSet, out var yTrainingSet, out var xTestSet, out var yTestSet);
            ResNetUtils.ToWorkingSet(xTrainingSet, yTrainingSet, out xTrain, out yTrain);
            ResNetUtils.ToWorkingSet(xTestSet, yTestSet, out xTest, out yTest);
            //We remove the mean 
            var mean = xTrain.Mean();
            xTrain.Add(-mean);
            xTest.Add(-mean);
        }
    }
}
