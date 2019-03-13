using System.Diagnostics;
using NUnit.Framework;
using SharpNet;
using SharpNet.CPU;


/*
SharpNet on 12-march-2019
LearningRate = LearningRateScheduler.ConstantByInterval(1, initialLearningRate, 80, initialLearningRate / 10, 120, initialLearningRate / 100);
BatchSize = 128
SGD
L1 = 1-e4
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)  
# ---------------------------------------------------------------------------
# ResNet20  | 3 (2)| 91.07     | 91.25     | -----     | -----     | 10 (---) 
# ResNet32  | 5(NA)| 92.12     | 92.49     | -----     | NA        | 16 (---) 
# ResNet44  | 7(NA)| 91.06     | 92.83     | -----     | NA        | 21 (---) 
# ResNet56  | 9 (6)| 91.57     | 93.03     | -----     | NA        | 27 (---) 
# ResNet110 |18(12)| 90.65     | 93.39+-.16| -----     | 93.63     | 52 (---)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------

TensorFlow results: (see https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
BatchSize = 32
Adam
# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 5

*/
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
        public void TestResNet164V1_CIFAR10()
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet164V1_CIFAR10(true, false, Logger(nameof(ResNetUtils.ResNet164V1_CIFAR10)));
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
