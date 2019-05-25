using SharpNet.Pictures;


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
# ResNet20  | 3 (2)| 91.96     | 91.25     | 90.97     | -----     | 8.8  (14.9) 
# ResNet32  | 5(NA)| 93.28     | 92.49     | NA        | NA        | 13.8 ( NA) 
# ResNet44  | 7(NA)| 93.41     | 92.83     | NA        | NA        | 18.8 ( NA) 
# ResNet56  | 9 (6)| 93.92     | 93.03     | 89.29     | -----     | 23.8 (41.9) 
# ResNet110 |18(12)| 94.07     | 93.39+-.16| 44.17     | 93.63     | 47.1 (85.5)
# ResNet164 |27(18)| 93.51     | 94.07     | -----     | 94.54     | 78.9 (141)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| --- (---)
# ---------------------------------------------------------------------------
*/

namespace SharpNet
{
    public abstract class NetworkBuilder
    {
        public string ExtraDescription { get; set; }
        public NetworkConfig Config { get; set; }
        public int GpuDeviceId { private get; set; }
        public int NumEpochs { get; set; }
        public int BatchSize { get; set; }
        public double InitialLearningRate { get; set; }
        public bool DisableLogging { get; set; }
        #region Data Augmentation
        public double WidthShiftRange { get; set; }
        public double HeightShiftRange { get; set; }
        public bool HorizontalFlip { get; set; }
        public bool VerticalFlip { get; set; }
        public ImageDataGenerator.FillModeEnum FillMode { get; set; }
        public int CutoutPatchlength { get; set; }
        #endregion

        public Network Build(string networkName)
        {
            Config.Logger = Utils.Logger(networkName + ExtraDescription);
            var network = new Network(Config, DataGenerator(), GpuDeviceId);
            network.Description = networkName + ExtraDescription;
            return network;
        }

        private ImageDataGenerator DataGenerator()
        {
            return new ImageDataGenerator(WidthShiftRange, HeightShiftRange, HorizontalFlip, VerticalFlip, FillMode, 0.0, CutoutPatchlength);
        }
    }

    public class ResNetBuilder : NetworkBuilder
    {
        public ResNetBuilder()
        {
            Config = new NetworkConfig
            {
                UseDoublePrecision = false,
                LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                lambdaL2Regularization = 1e-4
            };
            Config.WithSGD(0.9, false); // SGD : validated on 19-apr-2019: +70 bps
            NumEpochs = 160; //64k iterations
            BatchSize = 128;
            InitialLearningRate = 0.1;
            WidthShiftRange = 0.1; //validated on 18-apr-2019: +300 bps (for both using WidthShiftRange & HeightShiftRange)
            HeightShiftRange = 0.1;
            HorizontalFlip  = true; // 'true' : validated on 18-apr-2019: +70 bps
            VerticalFlip = false;
            FillMode = ImageDataGenerator.FillModeEnum.Reflect; //validated on 18-apr-2019: +50 bps
            CutoutPatchlength = 16; // '16' : validated on 17-apr-2019: +70 bps
        }

        #region ResNet V1
        public Network ResNet18_V1(int[] xShape, int nbCategories)
        {
            return Build(nameof(ResNet18_V1)).ResNetV1(new[] { 2, 2, 2, 2 }, false, xShape, nbCategories);
        }
        public Network ResNet34_V1(int[] xShape, int nbCategories)
        {
            return Build(nameof(ResNet34_V1)).ResNetV1(new[] { 3, 4, 6, 3 }, false, xShape, nbCategories);
        }
        public Network ResNet50_V1(int[] xShape, int nbCategories)
        {
            return Build(nameof(ResNet50_V1)).ResNetV1(new[] { 3, 4, 6, 3 }, true, xShape, nbCategories);
        }
        public Network ResNet101_V1(int[] xShape, int nbCategories)
        {
            return Build(nameof(ResNet101_V1)).ResNetV1(new[] { 3, 4, 23, 3 }, true, xShape, nbCategories);
        }
        public Network ResNet152_V1(int[] xShape, int nbCategories)
        {
            return Build(nameof(ResNet152_V1)).ResNetV1(new[] { 3, 8, 36, 3 }, true, xShape, nbCategories);
        }
        #endregion

        #region ResNetV1 for CIFAR10
        private Network BuildResNetV1_CIFAR10(int numResBlocks)
        {
            var networkName = "ResNet" + (6 * numResBlocks + 2) + "V1_CIFAR10";
            return Build(networkName).GetResNetV1_CIFAR10(numResBlocks);
        }
        public Network ResNet20V1_CIFAR10() {return BuildResNetV1_CIFAR10(3);}
        public Network ResNet32V1_CIFAR10() {return BuildResNetV1_CIFAR10(5);}
        public Network ResNet44V1_CIFAR10() {return BuildResNetV1_CIFAR10(7);}
        public Network ResNet56V1_CIFAR10() {return BuildResNetV1_CIFAR10(9);}
        public Network ResNet110V1_CIFAR10() {return BuildResNetV1_CIFAR10(18);}
        public Network ResNet164V1_CIFAR10() {return BuildResNetV1_CIFAR10(27);}
        public Network ResNet1202V1_CIFAR10() {return BuildResNetV1_CIFAR10(200);}
        #endregion

        #region ResNetV2 for CIFAR10
        private Network BuildResNetV2_CIFAR10(int numResBlocks)
        {
            var networkName = "ResNet"+(9* numResBlocks +2)+ "V2_CIFAR10";
            return Build(networkName).GetResNetV2_CIFAR10(1);
        }
        public Network ResNet11V2_CIFAR10() {return BuildResNetV2_CIFAR10(1);}
        public Network ResNet20V2_CIFAR10() {return BuildResNetV2_CIFAR10(2);}
        public Network ResNet29V2_CIFAR10() {return BuildResNetV2_CIFAR10(3);}
        public Network ResNet56V2_CIFAR10() {return BuildResNetV2_CIFAR10(6);}
        public Network ResNet110V2_CIFAR10() {return BuildResNetV2_CIFAR10(12);}
        public Network ResNet164V2_CIFAR10() {return BuildResNetV2_CIFAR10(18);}
        public Network ResNet1001V2_CIFAR10() {return BuildResNetV2_CIFAR10(111);}
        #endregion
    }
}
