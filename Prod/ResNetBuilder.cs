using System;
using SharpNet.Optimizers;
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
# ResNet20  | 3 (2)| 91.95     | 91.25     | 90.97     | -----     | 8.8  (14.9) 
# ResNet32  | 5(NA)| 92.86     | 92.49     | NA        | NA        | 13.8 ( NA) 
# ResNet44  | 7(NA)| 93.78     | 92.83     | NA        | NA        | 18.8 ( NA) 
# ResNet56  | 9 (6)| 93.67     | 93.03     | 75.75     | -----     | 23.8 (41.9) 
# ResNet110 |18(12)| 94.57     | 93.39+-.16| 44.17     | 93.63     | 47.1 (85.5)
# ResNet164 |27(18)| 93.01     | 94.07     | -----     | 94.54     | 78.9 (141)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| --- (---)
# ---------------------------------------------------------------------------
*/

namespace SharpNet
{
    public abstract class NetworkBuilder
    {
        public string ExtraDescription { get; set; }
        public int GpuDeviceId { private get; set; } = 0;
        public NetworkConfig Config { get; set; }


        public abstract int NumEpochs { get; set; }
        public abstract int BatchSize { get; set; }


        protected abstract ImageDataGenerator DataGenerator();
        public abstract ReduceLROnPlateau Cifar10ReduceLROnPlateau();
        public abstract ILearningRateScheduler Cifar10LearningRateScheduler();


        public Network Build(string networkName)
        {
            Config.Logger = Utils.Logger(networkName + ExtraDescription);
            var network = new Network(Config, DataGenerator(), GpuDeviceId);
            network.Description = networkName + ExtraDescription;
            return network;
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


        //for one cycle policy: by how much we have to divide the max learning rate to reach the min learning rate
        public int OneCycleDividerForMinLearningRate { get; set; } = 10;
        public double OneCyclePercentInAnnealing { get; set; } = 0.2;
        public bool OneCycleLearningRate { get; set; } = false;
        public bool LinearLearningRate { get; set; } = false;
        public override int NumEpochs { get; set; } = 160; //64k iterations
        public override int BatchSize { get; set; } = 128;
        public double InitialLearningRate { get; set; } = 0.1;
        public bool DivideBy10OnPlateau { get; set; } = true; // 'true' : validated on 19-apr-2019: +20 bps
        #region Data Augmentation
        public double WidthShiftRange { get; set; } = 0.1; //validated on 18-apr-2019: +300 bps (for both using WidthShiftRange & HeightShiftRange)
        public double HeightShiftRange { get; set; } = 0.1;
        public bool HorizontalFlip { get; set; } = true; // 'true' : validated on 18-apr-2019: +70 bps
        public bool VerticalFlip { get; set; } = false;
        public ImageDataGenerator.FillModeEnum FillMode { get; set; } = ImageDataGenerator.FillModeEnum.Reflect; //validated on 18-apr-2019: +50 bps
        public int CutoutPatchlength { get; set; } = 16; // '16' : validated on 17-apr-2019: +70 bps

        protected override ImageDataGenerator DataGenerator()
        {
            return new ImageDataGenerator(WidthShiftRange, HeightShiftRange, HorizontalFlip, VerticalFlip, FillMode, 0.0, CutoutPatchlength);
        }
        #endregion

      

        public override ILearningRateScheduler Cifar10LearningRateScheduler()
        {
            if (OneCycleLearningRate)
            {
                return new OneCycleLearningRateScheduler(InitialLearningRate, OneCycleDividerForMinLearningRate, OneCyclePercentInAnnealing, NumEpochs);
            }
            if (LinearLearningRate)
            {
                return LearningRateScheduler.InterpolateByInterval(1, InitialLearningRate, 80, InitialLearningRate / 10, 120, InitialLearningRate / 100);
            }
            return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate, 80, InitialLearningRate / 10, 120,InitialLearningRate / 100);
        }

        public override ReduceLROnPlateau Cifar10ReduceLROnPlateau()
        {
            if (OneCycleLearningRate)
            {
                return null;
            }
            var factorForReduceLrOnPlateau = DivideBy10OnPlateau ? 0.1 : Math.Sqrt(0.1);
            return new ReduceLROnPlateau(factorForReduceLrOnPlateau, 5, 5);
        }
        private ILearningRateScheduler ResNet110LearningRateScheduler()
        {
            if (OneCycleLearningRate)
            {
                return new OneCycleLearningRateScheduler(InitialLearningRate, OneCycleDividerForMinLearningRate, OneCyclePercentInAnnealing, NumEpochs);
            }
            return LearningRateScheduler.ConstantByInterval(1, InitialLearningRate / 10, 2, InitialLearningRate, 80, InitialLearningRate / 10, 120, InitialLearningRate / 100);
        }
    }
}
