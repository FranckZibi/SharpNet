using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.CPU;
using SharpNet.Networks;
/*
BatchSize = 128
EpochCount = 150
SGD with momentum = 0.9 & L2 = 0.5* 1-e4
CutMix / no Cutout / no Mixup / FillMode = Reflect / Disable DivideBy10OnPlateau
AvgPoolingStride = 2
# ------------------------------------------------------------------------------------------------
#           |             |    200-epoch  |   150-epoch   |   200-epoch   |   Orig Paper  | sec/epoch
# Model     |   #Params   |      WRN      |   SGDR 10-2   |   SGDR 10-2   |      WRN      | GTX1080
#           |             |   %Accuracy   |   %Accuracy   |   %Accuracy   |   %Accuracy   | 
#           |             |   (dropout)   |(Ens. Learning)|(Ens. Learning)|   (dropout)   | 
# -----------------------------------------------------------------------------------------------
# WRN-16-4  |   3,159,636 | ----- (-----) | 77.91 (-----) | ----- (-----) | 75.97 (76.09) |  27.1
# WRN-40-4  |   9,367,124 | ----- (-----) | 79.88 (-----) | ----- (-----) | 77.11         |  79.5
# WRN-16-8  |  11,782,740 | ----- (-----) | 79.69 (-----) | ----- (-----) | 77.93         |  80.9
# WRN-16-10 |  ---------- | ----- (-----) | ----- (-----) | ----- (-----) | 78.41         | -----
# WRN-28-8  |  ---------- | ----- (-----) | ----- (-----) | ----- (-----) | 78.78         | -----
# WRN-28-10 |  37,514,836 | ----- (-----) | ----- (-----) | ----- (-----) | 80.75 (81.15) |   292
# -----------------------------------------------------------------------------------------------
*/


namespace SharpNet.Datasets
{
    public class CIFAR100DataSet : AbstractTrainingAndTestDataSet
    {
        private static readonly string[] CategoryIndexToDescription = new[] {
            "beaver", "dolphin", "otter", "seal", "whale",
            "aquarium fish", "flatfish", "ray", "shark", "trout",
            "orchids", "poppies", "roses", "sunflowers", "tulips",
            "bottles", "bowls", "cans", "cups", "plates",
            "apples", "mushrooms", "oranges", "pears", "sweet peppers",
            "clock", "computer keyboard", "lamp", "telephone", "television",
            "bed", "chair", "couch", "table", "wardrobe",
            "bee", "beetle", "butterfly", "caterpillar", "cockroach",
            "bear", "leopard", "lion", "tiger", "wolf",
            "bridge", "castle", "house", "road", "skyscraper",
            "cloud", "forest", "mountain", "plain", "sea",
            "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
            "fox", "porcupine", "possum", "raccoon", "skunk",
            "crab", "lobster", "snail", "spider", "worm",
            "baby", "boy", "girl", "man", "woman",
            "crocodile", "dinosaur", "lizard", "snake", "turtle",
            "hamster", "mouse", "rabbit", "shrew", "squirrel",
            "maple", "oak", "palm", "pine", "willow",
            "bicycle", "bus", "motorcycle", "pickup truck", "train",
            "lawn-mower", "rocket", "streetcar", "tank", "tractor" };

        public static int CategoryCount => CategoryIndexToDescription.Length;

        //private readonly string[] SuperclassIdToDescription = new[]
        //{
        //    "aquatic mammals",
        //    "fish",
        //    "flowers",
        //    "food containers",
        //    "fruit and vegetables",
        //    "household electrical devices",
        //    "household furniture",
        //    "insects",
        //    "large carnivores",
        //    "large man-made outdoor things",
        //    "large natural outdoor scenes",
        //    "large omnivores and herbivores",
        //    "medium-sized mammals",
        //    "non-insect invertebrates",
        //    "people",
        //    "reptiles",
        //    "small mammals",
        //    "trees",
        //    "vehicles 1",
        //    "vehicles 2"
        //};

        public override IDataSet Training { get; }
        public override IDataSet Test { get; }

        public static readonly int[] Shape_CHW = { 3, 32, 32 };

        public const string NAME = "CIFAR-100";
        public CIFAR100DataSet() : base(NAME)
        {
            var path = Path.Combine(NetworkConfig.DefaultDataDirectory, Name);

            //We load the training set
            var xTrainingSet = new CpuTensor<byte>(new[] { 50000, Shape_CHW[0], Shape_CHW[1], Shape_CHW[2] });
            var yTrainingSet = new CpuTensor<byte>(new[] { 50000, 1, 1, 1 });
            Load(Path.Combine(path, "train.bin"), xTrainingSet, yTrainingSet);
            //We normalize the input with 0 mean / 1 volatility
            var meanAndVolatilityOfEachChannelInTrainingSet = new List<Tuple<float, float>>{Tuple.Create(129.304165605469f, 68.1702428992064f),Tuple.Create(124.069962695312f, 65.3918080438575f),Tuple.Create(112.434050058594f, 70.4183701880494f)};
            var xTrain = AbstractDataSet.ToXWorkingSet(xTrainingSet, meanAndVolatilityOfEachChannelInTrainingSet);
            var yTrain = AbstractDataSet.ToYWorkingSet(yTrainingSet, CategoryCount, CategoryByteToCategoryIndex);

            AbstractDataSet.AreCompatible_X_Y(xTrain, yTrain);

            //We load the test set
            var xTestSet = new CpuTensor<byte>(new[] { 10000, Shape_CHW[0], Shape_CHW[1], Shape_CHW[2]});
            var yTestSet = new CpuTensor<byte>(new[] { 10000, 1, 1, 1 });
            Load(Path.Combine(path, "test.bin"), xTestSet, yTestSet);
            //We normalize the test set with 0 mean / 1 volatility (coming from the training set)
            var xTest = AbstractDataSet.ToXWorkingSet(xTestSet, meanAndVolatilityOfEachChannelInTrainingSet);
            var yTest = AbstractDataSet.ToYWorkingSet(yTestSet, CategoryCount, CategoryByteToCategoryIndex);

            AbstractDataSet.AreCompatible_X_Y(xTest, yTest);
            //Uncomment the following line to take only the first elements
            //xTrain = (CpuTensor<float>)xTrain.Slice(0, 1000);yTrain = (CpuTensor<float>)yTrain.Slice(0, xTrain.Shape[0]); xTest = (CpuTensor<float>)xTest.Slice(0, 1000); ; yTest = (CpuTensor<float>)yTest.Slice(0, xTest.Shape[0]);

            Training = new InMemoryDataSet(xTrain, yTrain, Name, Objective_enum.Classification,meanAndVolatilityOfEachChannelInTrainingSet, CategoryIndexToDescription);
            Test = new InMemoryDataSet(xTest, yTest, Name, Objective_enum.Classification, meanAndVolatilityOfEachChannelInTrainingSet, CategoryIndexToDescription);
        }

        private static void Load(string path, CpuTensor<byte> x, CpuTensor<byte> y)
        {
            Debug.Assert(x.Shape[1] == Shape_CHW[0]); //channels
            Debug.Assert(x.Shape[2] == Shape_CHW[1]); //height
            Debug.Assert(x.Shape[3] == Shape_CHW[2]); //width
            var b = File.ReadAllBytes(path);
            for (int count = 0; count < x.Shape[0]; ++count)
            {
                int bIndex = count * (2 + x.MultDim0);
                int xIndex = (count) * x.MultDim0;
                int yIndex = (count) * y.MultDim0;
                y[yIndex] = b[bIndex + 1]; //fine label . The coarse label is at b[bIndex]
                for (int j = 0; j < x.MultDim0; ++j)
                {
                    x[xIndex + j] = b[bIndex + 2 + j];
                }
            }
        }
    }
}
