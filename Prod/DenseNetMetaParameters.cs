using System;
using SharpNet.Optimizers;
using SharpNet.Pictures;

namespace SharpNet
{
    public class DenseNetMetaParameters : IMetaParameters
    {
        public bool UseGPU { get; set; } = true;
        public bool UseDoublePrecision { get; set; } = false;
        public bool DivideBy10OnPlateau { get; set; } = true;
        public bool UseAdam { get; set; } = false;
        public bool UseNesterov { get; set; } = true;
        public NetworkConfig Config()
        {
            var networkConfig = new NetworkConfig(UseGPU)
            {
                UseDoublePrecision = UseDoublePrecision,
                LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                Logger = Logger.ConsoleLogger
            };
            networkConfig = UseAdam ? networkConfig.WithAdam() : networkConfig.WithSGD(0.9, 0, UseNesterov);
            return networkConfig;
        }
        public bool OneCycleLearningRate { get; set; } = false;
        public double lambdaL2Regularization { get; set; } = 1e-4;
        public int NumEpochs { get; set; } = 300;
        public int BatchSize { get; set; } = 64;
        public double InitialLearningRate { get; set; } = 0.1;
        //!D public int CutoutPatchlength { get; set; } = 16;
        public int CutoutPatchlength { get; set; } = 0;
        public double WidthShiftRange { get; set; } = 0.1;
        public double HeightShiftRange { get; set; } = 0.1;
        public bool HorizontalFlip { get; set; } = true;
        public bool VerticalFlip { get; set; } = false;
        public ImageDataGenerator.FillModeEnum FillMode { get; set; } = ImageDataGenerator.FillModeEnum.Reflect;
        public string ExtraDescription { get; set; } = "";
        public ILearningRateScheduler Cifar10LearningRateScheduler()
        {
            return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, 150, InitialLearningRate / 10, 225, InitialLearningRate / 100);

        }
        public ImageDataGenerator DataGenerator()
        {
            return new ImageDataGenerator(WidthShiftRange, HeightShiftRange, HorizontalFlip, VerticalFlip, FillMode, 0.0, CutoutPatchlength);
        }
        public ReduceLROnPlateau Cifar10ReduceLROnPlateau()
        {
            return null;
            if (OneCycleLearningRate)
            {
                return null;
            }
            var factorForReduceLrOnPlateau = DivideBy10OnPlateau ? 0.1 : Math.Sqrt(0.1);
            return new ReduceLROnPlateau(factorForReduceLrOnPlateau, 5, 5);
        }
    }
}