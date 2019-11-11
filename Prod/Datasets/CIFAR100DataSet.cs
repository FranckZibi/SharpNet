using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
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
    public class CIFAR100DataSet : ITrainingAndTestDataSet
    {
        private readonly string[] CategoryIdToDescription = new[] { "beaver", "dolphin", "otter", "seal", "whale",
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

        private readonly string[] SuperclassIdToDescription = new[]
        {
            "aquatic mammals",
            "fish",
            "flowers",
            "food containers",
            "fruit and vegetables",
            "household electrical devices",
            "household furniture",
            "insects",
            "large carnivores",
            "large man-made outdoor things",
            "large natural outdoor scenes",
            "large omnivores and herbivores",
            "medium-sized mammals",
            "non-insect invertebrates",
            "people",
            "reptiles",
            "small mammals",
            "trees",
            "vehicles 1",
            "vehicles 2"
        };

        public string Name => "CIFAR-100";
        public const int Channels = 3;
        public const int Height = 32;
        public const int Width = Height;
        public const int Categories = 100;

        public static readonly int[] InputShape_CHW = { Channels, Height, Width };


        public IDataSet Training { get; }
        public IDataSet Test { get; }

        public CIFAR100DataSet()
        {
            var path = Path.Combine(NetworkConfig.DefaultDataDirectory, Name);

            //We load the training set
            var xTrainingSet = new CpuTensor<byte>(new[] { 50000, Channels, Height, Width }, "xTrainingSet");
            var yTrainingSet = new CpuTensor<byte>(new[] { 50000, 1, 1, 1 }, "yTrainingSet");
            Load(Path.Combine(path, "train.bin"), xTrainingSet, yTrainingSet);
            //We normalize the input with 0 mean / 1 volatility
            var meanAndVolatilityOfEachChannelInTrainingSet = new List<Tuple<float, float>>{Tuple.Create(129.304165605469f, 68.1702428992064f),Tuple.Create(124.069962695312f, 65.3918080438575f),Tuple.Create(112.434050058594f, 70.4183701880494f)};
            ToWorkingSet(xTrainingSet, yTrainingSet, out var xTrain, out var yTrain, meanAndVolatilityOfEachChannelInTrainingSet);
            AbstractDataSet.AreCompatible_X_Y(xTrain, yTrain);
            int[] trainElementIdToCategoryId = yTrainingSet.Content.Select(x => (int)x).ToArray();
            Debug.Assert(trainElementIdToCategoryId.Length == xTrainingSet.Shape[0]);

            //We load the test set
            var xTestSet = new CpuTensor<byte>(new[] { 10000, Channels, Height, Width }, "xTestSet");
            var yTestSet = new CpuTensor<byte>(new[] { 10000, 1, 1, 1 }, "yTestSet");
            Load(Path.Combine(path, "test.bin"), xTestSet, yTestSet);
            //We normalize the test set with 0 mean / 1 volatility (coming from the training set)
            ToWorkingSet(xTestSet, yTestSet, out var xTest, out var yTest, meanAndVolatilityOfEachChannelInTrainingSet);
            AbstractDataSet.AreCompatible_X_Y(xTest, yTest);
            int[] testElementIdToCategoryId = yTestSet.Content.Select(x => (int)x).ToArray();
            Debug.Assert(testElementIdToCategoryId.Length == xTestSet.Shape[0]);

            //Uncomment the following line to take only the first elements
            //xTrain = (CpuTensor<float>)xTrain.ExtractSubTensor(0, 1000);yTrain = (CpuTensor<float>)yTrain.ExtractSubTensor(0, xTrain.Shape[0]); xTest = (CpuTensor<float>)xTest.ExtractSubTensor(0, 1000); ; yTest = (CpuTensor<float>)yTest.ExtractSubTensor(0, xTest.Shape[0]);

            Training = new InMemoryDataSet(xTrain, yTrain, trainElementIdToCategoryId, CategoryIdToDescription, Name, meanAndVolatilityOfEachChannelInTrainingSet);
            Test = new InMemoryDataSet(xTest, yTest, testElementIdToCategoryId, CategoryIdToDescription, Name, meanAndVolatilityOfEachChannelInTrainingSet);
        }

        public void Dispose()
        {
            Training?.Dispose();
            Test?.Dispose();
        }

        private static void ToWorkingSet(CpuTensor<byte> x, CpuTensor<byte> y, out CpuTensor<float> xWorkingSet, out CpuTensor<float> yWorkingSet, List<Tuple<float, float>> meanAndVolatilityOfEachChannel)
        {
            xWorkingSet = x.Select((n, c, val) => (float)((val - meanAndVolatilityOfEachChannel[c].Item1) / Math.Max(meanAndVolatilityOfEachChannel[c].Item2, 1e-9)));
            //xWorkingSet = x.Select((n, c, val) => (float)val/255f);
            yWorkingSet = y.ToCategorical(1f, out _);
        }
        private static void Load(string path, CpuTensor<byte> x, CpuTensor<byte> y)
        {
            var b = File.ReadAllBytes(path);
            for (int count = 0; count < x.Shape[0]; ++count)
            {
                int bIndex = count * (2 + Height * Width * Channels);
                int xIndex = (count) * x.MultDim0;
                int yIndex = (count) * y.MultDim0;
                y[yIndex] = b[bIndex+1]; //fine label . The coarse label is at b[bIndex]
                for (int j = 0; j < Height * Width * Channels; ++j)
                {
                    x[xIndex + j] = b[bIndex + 2 + j];
                }
            }
        }
    }
}
