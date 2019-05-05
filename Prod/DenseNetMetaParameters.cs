using SharpNet.Optimizers;
using SharpNet.Pictures;

namespace SharpNet
{
    public class DenseNetMetaParameters : IMetaParameters
    {
        public bool UseGPU { get; set; } = true;
        public bool UseDoublePrecision { get; set; }
        public bool UseAdam { get; set; }
        public bool SaveLossAfterEachMiniBatch { get; set; }
        public bool ForceTensorflowCompatibilityMode { get; set; }
        public bool UseNesterov { get; set; } = true;
        public bool SaveStatsWhenSavingNetwork { get; set; }

        public NetworkConfig Config()
        {
            var config = new NetworkConfig(UseGPU);
            config.UseDoublePrecision = UseDoublePrecision;
            config.SaveStatsWhenSavingNetwork = SaveStatsWhenSavingNetwork;
            config.LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy;
            config.Logger = Logger.ConsoleLogger;
            config = UseAdam ? config.WithAdam() : config.WithSGD(0.9, 0, UseNesterov);
            config.ForceTensorflowCompatibilityMode = ForceTensorflowCompatibilityMode;
            config.SaveLossAfterEachMiniBatch = SaveLossAfterEachMiniBatch;
            return config;
        }
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
        }
    }
}