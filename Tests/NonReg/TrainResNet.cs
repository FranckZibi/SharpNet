﻿using System.Diagnostics;
using SharpNet;
using SharpNet.CPU;


/*
SharpNet on 12-march-2019
LearningRate = LearningRateScheduler.ConstantByInterval(1, initialLearningRate, 80, initialLearningRate / 10, 120, initialLearningRate / 100);
BatchSize = 128
EpochCount = 160
SGD with momentum = 0.9 & L2 = 1-e4
Cutout 16 / FillMode = Reflect / DivideBy10OnPlateau
# ----------------------------------------------------------------------------
#           |      | 160-epoch | Orig Paper| 160-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)  
# ---------------------------------------------------------------------------
# ResNet11  | - (1)| NA        | NA        | 89.26     | -----     | NA   (8.8) 
# ResNet20  | 3 (2)| 91.50     | 91.25     | 90.97     | -----     | 8.8  (14.9) 
# ResNet32  | 5(NA)| 93.52     | 92.49     | NA        | NA        | 13.8 ( NA) 
# ResNet44  | 7(NA)| 93.99     | 92.83     | NA        | NA        | 18.8 ( NA) 
# ResNet56  | 9 (6)| 93.82     | 93.03     | 75.75     | -----     | 23.8 (41.9) 
# ResNet110 |18(12)| 94.57     | 93.39+-.16| 44.17     | 93.63     | 47.1 (85.5)
# ResNet164 |27(18)| 93.01     | 94.07     | -----     | 94.54     | 78.9 (141)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| --- (---)
# ---------------------------------------------------------------------------
*/
namespace SharpNetTests.NonReg
{
    /// <summary>
    /// Train a ResNet Network (V1 & V2) on Cifar10 dataset has described in https://arxiv.org/pdf/1512.03385.pdf & https://arxiv.org/pdf/1603.05027.pdf
    /// </summary>
    public static class TrainResNet
    {
        public static void TrainResNet20V1_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet20V1_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet20V1_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, param.BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        public static void TrainResNet32V1_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet32V1_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet32V1_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, param.BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        public static void TrainResNet44V1_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet44V1_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet44V1_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, param.BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        public static void TrainResNet56V1_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet56V1_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet56V1_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, param.BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        public static void TrainResNet110V1_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet110V1_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet110V1_CIFAR10)));
            network.Fit(xTrain, yTrain, param.ResNet110LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, -1 /*BatchSize*/, xTest, yTest);
            network.ClearMemory();
        }
        public static void TrainResNet164V1_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet164V1_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet164V1_CIFAR10)));
            network.Fit(xTrain, yTrain, param.ResNet110LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, -1 /*BatchSize*/, xTest, yTest);
            network.ClearMemory();
        }
        public static void TrainResNet1202V1_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet1202V1_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet1202V1_CIFAR10)));
            network.Fit(xTrain, yTrain, param.ResNet110LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, -1 /*BatchSize*/, xTest, yTest);
            network.ClearMemory();
        }
       
        public static void TrainResNet11V2_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet11V2_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet11V2_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, param.BatchSize, xTest, yTest);
            network.ClearMemory();
        }

        public static void TrainResNet20V2_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet20V2_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet20V2_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, param.BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        public static void TrainResNet29V2_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet29V2_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet29V2_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, param.BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        public static void TrainResNet56V2_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet56V2_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet56V2_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, param.BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        public static void TrainResNet110V2_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet110V2_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet110V2_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, -1 /*BatchSize*/, xTest, yTest);
            network.ClearMemory();
        }
        public static void TrainResNet164V2_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet164V2_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet164V2_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, -1 /*BatchSize*/, xTest, yTest);
            network.ClearMemory();
        }
        public static void TrainResNet1001V2_CIFAR10(ResNetMetaParameters param)
        {
            LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = ResNetUtils.ResNet1001V2_CIFAR10(param, Utils.Logger(nameof(ResNetUtils.ResNet1001V2_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, -1 /*BatchSize*/, xTest, yTest);
            network.ClearMemory();
        }

        public static void LoadCifar10(out CpuTensor<float> xTrain, out CpuTensor<float> yTrain, out CpuTensor<float> xTest, out CpuTensor<float> yTest)
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
