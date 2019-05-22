using SharpNet.Datasets;
using SharpNet.Optimizers;
using SharpNet.Pictures;

/*
SharpNet on 8-may-2019
LearningRate = Orig Paper
BatchSize = 64
EpochCount = 300
SGD with momentum = 0.9 & Nesterov & L2 = 1-e4 
Cutout 16 / FillMode = Reflect
Orig Paper = https://arxiv.org/pdf/1608.06993.pdf
# ------------------------------------------------------------
# Model     | k |Depth|Epochs| SharpNet  | Orig Paper| sec/epoch
#           |   |     |      | %Accuracy | %Accuracy | GTX1080
# ---------------------------|--------------------------------
# DenseNet  | 12|   40|  300 | 91.48     | 94.76     |  94.0        (9-may-2019, 1 078 018 params)
# DenseNet  | 12|  100|      | -----     | 95.90     | -----
# DenseNet  | 24|  100|      | -----     | 96.26     | -----
# DenseNetBC| 12|   40|  300 | 89.18     | NA        |  36.5        (9-may-2019, 181 210 params)
# DenseNetBC| 12|  100|  300 | 93.76     | 95.49     |   156        (9-may-2019, 793 150 params)
# DenseNetBC| 24|  250|      | -----     | 96.38     | -----
# DenseNetBC| 40|  190|      | -----     | 97.54     | -----
*/

namespace SharpNet
{
    public class DenseNetBuilder : NetworkBuilder
    {
        public DenseNetBuilder()
        {
            Config = new NetworkConfig
            { 
                UseDoublePrecision = false,
                LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                lambdaL2Regularization = 1e-4
            };
            Config.WithSGD(0.9, true);
        }

        public Network DenseNet_Fast_CIFAR10()
        {
            var network = Build(nameof(DenseNet_Fast_CIFAR10));
            network.AddDenseNet(
                new[] { 1, CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width },
                CIFAR10.Categories,
                false,
                new[] { 2, 2},
                false,
                12,
                0.5,
                null);
            return network;
        }


        //!D TO REMOVE
        public Network DenseNet_12_10_CIFAR10()
        {
            var network = Build(nameof(DenseNet_12_10_CIFAR10));
            network.AddDenseNet(
                new[] { 1, CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width },
                CIFAR10.Categories,
                false,
                new[] { 2, 2, 2 },
                false,
                12,
                1.0,
                null);
            return network;
        }
        public Network DenseNet_12_40_CIFAR10()
        {
            //return Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\Network_15576_14.txt");
            var network = Build(nameof(DenseNet_12_40_CIFAR10));
            network.AddDenseNet(
                new[] { 1, CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width },
                CIFAR10.Categories,
                false,
                new[] { 12, 12, 12 },
                false,
                12,
                1.0,
                null);
            return network;
        }
        public Network DenseNetBC_12_40_CIFAR10()
        {
            var network = Build(nameof(DenseNetBC_12_40_CIFAR10));
            network.AddDenseNet(
                new[] { 1, CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width },
                CIFAR10.Categories,
                false,
                new[] { 12 / 2, 12 / 2, 12 / 2 },
                true,
                12,
                0.5,
                null);
            return network;
        }
        public Network DenseNetBC_12_100_CIFAR10()
        {
            var network = Build(nameof(DenseNetBC_12_100_CIFAR10));
            network.AddDenseNet(
                new[] { 1, CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width },
                CIFAR10.Categories,
                false,
                new[] { 32 / 2, 32 / 2, 32 / 2 },
                true,
                12,
                0.5,
                null);
            return network;
        }

        public bool DivideBy10OnPlateau { get; set; } = true;
        public bool DisableLogging { get; set; }
        //for one cycle policy: by how much we have to divide the max learning rate to reach the min learning rate
        public int OneCycleDividerForMinLearningRate { get; set; } = 10;
        public double OneCyclePercentInAnnealing { get; set; } = 0.2;
        public bool OneCycleLearningRate { get; set; } = false;
        public override int NumEpochs { get; set; } = 300;
        public override int BatchSize { get; set; } = 64;
        public double InitialLearningRate { get; set; } = 0.1;

        #region Data Augmentation
        public int CutoutPatchlength { get; set; } = 16;
        public double WidthShiftRange { get; set; } = 0.1;
        public double HeightShiftRange { get; set; } = 0.1;
        public bool HorizontalFlip { get; set; } = true;
        public bool VerticalFlip { get; set; } = false;
        public ImageDataGenerator.FillModeEnum FillMode { get; set; } = ImageDataGenerator.FillModeEnum.Reflect;

        protected override ImageDataGenerator DataGenerator()
        {
            return new ImageDataGenerator(WidthShiftRange, HeightShiftRange, HorizontalFlip, VerticalFlip, FillMode, 0.0, CutoutPatchlength);
        }
        #endregion


        public override ReduceLROnPlateau Cifar10ReduceLROnPlateau()
        {
            return null;
            /*
            if (OneCycleLearningRate)
            {
                return null;
            }
            var factorForReduceLrOnPlateau = DivideBy10OnPlateau ? 0.1 : System.Math.Sqrt(0.1);
            return new ReduceLROnPlateau(factorForReduceLrOnPlateau, 5, 5);*/
        }

        public override ILearningRateScheduler Cifar10LearningRateScheduler()
        {
            //return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, NumEpochs/2, InitialLearningRate / 10, (3* NumEpochs)/4, InitialLearningRate / 100);
            return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, 150, InitialLearningRate / 10, 225, InitialLearningRate / 100);
        }

    }
}