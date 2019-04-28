using SharpNet;

namespace SharpNetTests.NonReg
{
    /// <summary>
    /// Train a DenseNet Network on Cifar10 dataset 
    /// </summary>
    public static class TrainDenseNet
    {
        public static void TrainDenseNet40_CIFAR10(DenseNetMetaParameters param)
        {
            TrainResNet.LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = DenseNetUtils.DenseNet40_CIFAR10(param, Utils.Logger(nameof(DenseNetUtils.DenseNet40_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, param.BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        public static void DenseNet100_BC_k_12_CIFAR10(DenseNetMetaParameters param)
        {
            TrainResNet.LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            var network = DenseNetUtils.DenseNet100_BC_k_12_CIFAR10(param, Utils.Logger(nameof(DenseNetUtils.DenseNet100_BC_k_12_CIFAR10)));
            network.Fit(xTrain, yTrain, param.Cifar10LearningRateScheduler(), param.Cifar10ReduceLROnPlateau(), param.NumEpochs, param.BatchSize, xTest, yTest);
            network.ClearMemory();
        }
    }
}
