using SharpNet.Datasets;
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
            NumEpochs = 300;
            BatchSize = 64;
            InitialLearningRate = 0.1;
            CutoutPatchlength = 16;
            WidthShiftRange = 0.1;
            HeightShiftRange = 0.1;
            HorizontalFlip = true;
            VerticalFlip = false;
            FillMode = ImageDataGenerator.FillModeEnum.Reflect;
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



    }
}