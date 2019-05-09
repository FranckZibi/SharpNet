using System;
using SharpNet.Datasets;
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
# ResNet20  | 3 (2)| 91.50     | 91.25     | 90.97     | -----     | 8.8  (14.9) 
# ResNet32  | 5(NA)| 93.52     | 92.49     | NA        | NA        | 13.8 ( NA) 
# ResNet44  | 7(NA)| 93.99     | 92.83     | NA        | NA        | 18.8 ( NA) 
# ResNet56  | 9 (6)| 93.82     | 93.03     | 75.75     | -----     | 23.8 (41.9) 
# ResNet110 |18(12)| 94.57     | 93.39+-.16| 44.17     | 93.63     | 47.1 (85.5)
# ResNet164 |27(18)| 93.01     | 94.07     | -----     | 94.54     | 78.9 (141)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| --- (---)
# ---------------------------------------------------------------------------
*/

namespace SharpNet
{
    public interface IMetaParameters
    {
        string ExtraDescription { get; set; }
    }

    public class ResNetConfig : IMetaParameters
    {
        public ResNetConfig()
        {
            Config = new NetworkConfig(true)
            {
                UseDoublePrecision = false,
                LossFunction = NetworkConfig.LossFunctionEnum.CategoricalCrossentropy,
                lambdaL2Regularization = 1e-4
            };
            Config.WithSGD(0.9, false); // SGD : validated on 19-apr-2019: +70 bps
        }

        public Network ResNet18_V1(int[] xShape, int nbCategories)
        {
            return GetNetwork(nameof(ResNet18_V1)).ResNetV1(new[] { 2, 2, 2, 2 }, false, xShape, nbCategories);
        }
        public Network ResNet34_V1(int[] xShape, int nbCategories)
        {
            return GetNetwork(nameof(ResNet34_V1)).ResNetV1(new[] { 3, 4, 6, 3 }, false, xShape, nbCategories);
        }
        public Network ResNet50_V1(int[] xShape, int nbCategories)
        {
            return GetNetwork(nameof(ResNet50_V1)).ResNetV1(new[] { 3, 4, 6, 3 }, true, xShape, nbCategories);
        }
        public Network ResNet101_V1(int[] xShape, int nbCategories)
        {
            return GetNetwork(nameof(ResNet101_V1)).ResNetV1(new[] { 3, 4, 23, 3 }, true, xShape, nbCategories);
        }
        public Network ResNet152_V1(int[] xShape, int nbCategories)
        {
            return GetNetwork(nameof(ResNet152_V1)).ResNetV1(new[] { 3, 8, 36, 3 }, true, xShape, nbCategories);
        }

        #region ResNetV1 for CIFAR10
        private Network ResNetV1_CIFAR10(int numResBlocks)
        {
            var networkName = "ResNet" + (6 * numResBlocks + 2) + "V1_CIFAR10";
            return GetNetwork(networkName).GetResNetV1_CIFAR10(numResBlocks);
        }
        public Network ResNet20V1_CIFAR10() {return ResNetV1_CIFAR10(3);}
        public Network ResNet32V1_CIFAR10() {return ResNetV1_CIFAR10(5);}
        public Network ResNet44V1_CIFAR10() {return ResNetV1_CIFAR10(7);}
        public Network ResNet56V1_CIFAR10() {return ResNetV1_CIFAR10(9);}
        public Network ResNet110V1_CIFAR10() {return ResNetV1_CIFAR10(18);}
        public Network ResNet164V1_CIFAR10() {return ResNetV1_CIFAR10(27);}
        public Network ResNet1202V1_CIFAR10() {return ResNetV1_CIFAR10(200);}
        #endregion

        #region ResNetV2 for CIFAR10

        private Network ResNetV2_CIFAR10(int numResBlocks)
        {
            var networkName = "ResNet"+(9* numResBlocks +2)+ "V2_CIFAR10";
            return GetNetwork(networkName).GetResNetV2_CIFAR10(1);
        }
        public Network ResNet11V2_CIFAR10() {return ResNetV2_CIFAR10(1);}
        public Network ResNet20V2_CIFAR10() {return ResNetV2_CIFAR10(2);}
        public Network ResNet29V2_CIFAR10() {return ResNetV2_CIFAR10(3);}
        public Network ResNet56V2_CIFAR10() {return ResNetV2_CIFAR10(6);}
        public Network ResNet110V2_CIFAR10() {return ResNetV2_CIFAR10(12);}
        public Network ResNet164V2_CIFAR10() {return ResNetV2_CIFAR10(18);}
        public Network ResNet1001V2_CIFAR10() {return ResNetV2_CIFAR10(111);}
        #endregion

        #region CIFAR10 Training 
        /// <summary>
        /// Train a ResNet Network (V1 & V2) on Cifar10 dataset has described in https://arxiv.org/pdf/1512.03385.pdf & https://arxiv.org/pdf/1603.05027.pdf
        /// </summary>
        ///
        /// 
        public void TrainResNet_CIFAR10(Network network, ILearningRateScheduler lrScheduler = null, bool autoBatchSize = false)
        {
            CIFAR10.LoadCifar10(out var xTrain, out var yTrain, out var xTest, out var yTest);
            lrScheduler = lrScheduler ?? Cifar10LearningRateScheduler();
            network.Fit(xTrain, yTrain, lrScheduler, Cifar10ReduceLROnPlateau(), NumEpochs, autoBatchSize?-1:BatchSize, xTest, yTest);
            network.ClearMemory();
        }
        public void TrainResNet20V1_CIFAR10() {TrainResNet_CIFAR10(ResNet20V1_CIFAR10());}
        public void TrainResNet32V1_CIFAR10() {TrainResNet_CIFAR10(ResNet32V1_CIFAR10());}
        public void TrainResNet44V1_CIFAR10() {TrainResNet_CIFAR10(ResNet44V1_CIFAR10());}
        public void TrainResNet56V1_CIFAR10() {TrainResNet_CIFAR10(ResNet56V1_CIFAR10());}
        public void TrainResNet110V1_CIFAR10() {TrainResNet_CIFAR10(ResNet110V1_CIFAR10());}
        public void TrainResNet164V1_CIFAR10() {TrainResNet_CIFAR10(ResNet164V1_CIFAR10());}
        public void TrainResNet1202V1_CIFAR10() {TrainResNet_CIFAR10(ResNet1202V1_CIFAR10(), ResNet110LearningRateScheduler(), true);}

        public void TrainResNet11V2_CIFAR10() {TrainResNet_CIFAR10(ResNet11V2_CIFAR10());}
        public void TrainResNet20V2_CIFAR10() {TrainResNet_CIFAR10(ResNet20V2_CIFAR10());}
        public void TrainResNet29V2_CIFAR10() {TrainResNet_CIFAR10(ResNet29V2_CIFAR10());}
        public void TrainResNet56V2_CIFAR10() { TrainResNet_CIFAR10(ResNet56V2_CIFAR10()); }
        public void TrainResNet110V2_CIFAR10() { TrainResNet_CIFAR10(ResNet110V2_CIFAR10(), Cifar10LearningRateScheduler(), true); }
        public void TrainResNet164V2_CIFAR10() { TrainResNet_CIFAR10(ResNet164V2_CIFAR10(), Cifar10LearningRateScheduler(), true); }
        public void TrainResNet1001V2_CIFAR10() { TrainResNet_CIFAR10(ResNet1001V2_CIFAR10(), Cifar10LearningRateScheduler(), true); }
        #endregion

        //for one cycle policy: by how much we have to divide the max learning rate to reach the min learning rate
        public int OneCycleDividerForMinLearningRate { get; set; } = 10;
        public double OneCyclePercentInAnnealing { get; set; } = 0.2;
        public bool OneCycleLearningRate { get; set; } = false;
        public bool LinearLearningRate { get; set; } = false;
        public int NumEpochs { get; set; } = 160; //64k iterations
        public int BatchSize { get; set; } = 128;
        public double InitialLearningRate { get; set; } = 0.1;
        public NetworkConfig Config { get; }
        public bool DivideBy10OnPlateau { get; set; } = true; // 'true' : validated on 19-apr-2019: +20 bps
        public string ExtraDescription { get; set; } = "";
        #region Data Augmentation
        public double WidthShiftRange { get; set; } = 0.1; //validated on 18-apr-2019: +300 bps (for both using WidthShiftRange & HeightShiftRange)
        public double HeightShiftRange { get; set; } = 0.1;
        public bool HorizontalFlip { get; set; } = true; // 'true' : validated on 18-apr-2019: +70 bps
        public bool VerticalFlip { get; set; } = false;
        public ImageDataGenerator.FillModeEnum FillMode { get; set; } = ImageDataGenerator.FillModeEnum.Reflect; //validated on 18-apr-2019: +50 bps
        public int CutoutPatchlength { get; set; } = 16; // '16' : validated on 17-apr-2019: +70 bps
        public ImageDataGenerator ResNetImageDataGenerator()
        {
            return new ImageDataGenerator(WidthShiftRange, HeightShiftRange, HorizontalFlip, VerticalFlip, FillMode, 0.0, CutoutPatchlength);
        }
        #endregion

        private Network GetNetwork(string networkName)
        {
            Config.Logger = Utils.Logger(networkName + ExtraDescription);
            var network = new Network(Config, ResNetImageDataGenerator());
            network.Description = networkName + ExtraDescription;
            return network;
        }
        private ILearningRateScheduler Cifar10LearningRateScheduler()
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
        private ReduceLROnPlateau Cifar10ReduceLROnPlateau()
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
