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
# -----------------------------------------------------
# Model     |  k|Depth| SharpNet  | Orig Paper| sec/epoch
#           |   |     | %Accuracy | %Accuracy | GTX1080
# -----------------------------------------------------
# DenseNet  | 12|   40| -----     | 94.76     | -----
# DenseNet  | 12|  100| -----     | 95.90     | -----
# DenseNet  | 24|  100| -----     | 96.26     | -----
# DenseNetBC| 12|   40| -----     | NA        | -----
# DenseNetBC| 12|  100| 93.76     | 95.49     |   156       (9-may-2019)
# DenseNetBC| 24[  250| -----     | 96.38     | -----
# DenseNetBC| 40[  190| -----     | 97.54     | -----
# -----------------------------------------------------

*/

namespace SharpNet
{
    public class DenseNetConfig : IMetaParameters
    {
        public DenseNetConfig()
        {
            Config = new NetworkConfig(true)
            { 
                UseDoublePrecision = false,
                LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                lambdaL2Regularization = 1e-4
            };
            Config.WithSGD(0.9, true);
        }


        #region Training
        /// <summary>
        /// Train a DenseNet Network on CIFAR10 data set 
        /// </summary>
        ///
        private void TrainDenseNet_CIFAR10(Network network)
        {
            CIFAR10.LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            network.Fit(xTrain, yTrain, Cifar10LearningRateScheduler(), Cifar10ReduceLROnPlateau(), NumEpochs, BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        /// 
        public void TrainDenseNet_12_10_CIFAR10() {TrainDenseNet_CIFAR10(DenseNet_12_10_CIFAR10());}
        public void TrainDenseNet_12_40_CIFAR10() {TrainDenseNet_CIFAR10(DenseNet_12_40_CIFAR10());}
        public void TrainDenseNetBC_12_40_CIFAR10() {TrainDenseNet_CIFAR10(DenseNetBC_12_40_CIFAR10());}
        public void TrainDenseNetBC_12_100_CIFAR10() { TrainDenseNet_CIFAR10(DenseNetBC_12_100_CIFAR10()); }
        #endregion

        //!D TO REMOVE
        public Network DenseNet_12_10_CIFAR10()
        {
            return GetNetwork(nameof(DenseNet_12_10_CIFAR10)).DenseNet(
                new[] { 1, CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width },
                CIFAR10.Categories,
                false,
                new[] { 2, 2, 2 },
                false,
                12,
                1.0,
                null);
        }
        public Network DenseNet_12_40_CIFAR10()
        {
            //return Network.ValueOf(@"C:\Users\fzibi\AppData\Local\Temp\Network_15576_14.txt");
            return GetNetwork(nameof(DenseNet_12_40_CIFAR10)).DenseNet(
                new[] { 1, CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width },
                CIFAR10.Categories,
                false,
                new[] { 12, 12, 12 },
                false,
                12,
                1.0,
                null);
        }
        public Network DenseNetBC_12_40_CIFAR10()
        {
            return GetNetwork(nameof(DenseNetBC_12_40_CIFAR10)).DenseNet(
                new[] { 1, CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width },
                CIFAR10.Categories,
                false,
                new[] { 12 / 2, 12 / 2, 12 / 2 },
                true,
                12,
                0.5,
                null);
        }
        public Network DenseNetBC_12_100_CIFAR10()
        {
            return GetNetwork(nameof(DenseNetBC_12_100_CIFAR10)).DenseNet(
                new[] { 1, CIFAR10.Channels, CIFAR10.Height, CIFAR10.Width },
                CIFAR10.Categories,
                false,
                new[] { 32 / 2, 32 / 2, 32 / 2 },
                true,
                12,
                0.5,
                null);
        }

        public Network GetNetwork(string networkName)
        {
            Config.Logger = DisableLogging?Logger.NullLogger:Utils.Logger(networkName + ExtraDescription);
            var network = new Network(Config, DataGenerator());
            network.Description = networkName + ExtraDescription;
            return network;
        }

        public NetworkConfig Config { get; }
        public bool DivideBy10OnPlateau { get; set; } = true;
        public bool DisableLogging { get; set; }
        //for one cycle policy: by how much we have to divide the max learning rate to reach the min learning rate
        public int OneCycleDividerForMinLearningRate { get; set; } = 10;
        public double OneCyclePercentInAnnealing { get; set; } = 0.2;
        public bool OneCycleLearningRate { get; set; } = false;
        public int NumEpochs { get; set; } = 300;
        public int BatchSize { get; set; } = 64;
        public double InitialLearningRate { get; set; } = 0.1;
        public string ExtraDescription { get; set; } = "";

        #region Data Augmentation
        public int CutoutPatchlength { get; set; } = 16;
        public double WidthShiftRange { get; set; } = 0.1;
        public double HeightShiftRange { get; set; } = 0.1;
        public bool HorizontalFlip { get; set; } = true;
        public bool VerticalFlip { get; set; } = false;
        public ImageDataGenerator.FillModeEnum FillMode { get; set; } = ImageDataGenerator.FillModeEnum.Reflect;
        public ImageDataGenerator DataGenerator()
        {
            return new ImageDataGenerator(WidthShiftRange, HeightShiftRange, HorizontalFlip, VerticalFlip, FillMode, 0.0, CutoutPatchlength);
        }
        #endregion


        private ReduceLROnPlateau Cifar10ReduceLROnPlateau()
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

        private ILearningRateScheduler Cifar10LearningRateScheduler()
        {
            return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, NumEpochs/2, InitialLearningRate / 10, (3* NumEpochs)/4, InitialLearningRate / 100);
        }

    }
}