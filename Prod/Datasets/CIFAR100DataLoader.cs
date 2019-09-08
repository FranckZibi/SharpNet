using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Networks;

namespace SharpNet.Datasets
{
    public class CIFAR100DataLoader : IDataSet
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

        private readonly string[] SuperclasIdToDescription = new[]
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


        public IDataSetLoader Training { get; }
        public IDataSetLoader Test { get; }

        public CIFAR100DataLoader()
        {
            var path = Path.Combine(NetworkConfig.DefaultDataDirectory, Name);

            //We load the training set
            var xTrainingSet = new CpuTensor<byte>(new[] { 50000, Channels, Height, Width }, "xTrainingSet");
            var yTrainingSet = new CpuTensor<byte>(new[] { 50000, 1, 1, 1 }, "yTrainingSet");
            Load(Path.Combine(path, "train.bin"), xTrainingSet, yTrainingSet);
            //We normalize the input with 0 mean / 1 volatility
            var meanAndVolatilityOfEachChannelInTrainingSet = new List<Tuple<double, double>>{Tuple.Create(129.304165605469, 68.1702428992064),Tuple.Create(124.069962695312, 65.3918080438575),Tuple.Create(112.434050058594, 70.4183701880494)};
            ToWorkingSet(xTrainingSet, yTrainingSet, out var xTrain, out var yTrain, meanAndVolatilityOfEachChannelInTrainingSet);
            AbstractDataSetLoader.AreCompatible_X_Y(xTrain, yTrain);
            int[] trainElementIdToCategoryId = yTrainingSet.Content.Select(x => (int)x).ToArray();
            Debug.Assert(trainElementIdToCategoryId.Length == xTrainingSet.Shape[0]);

            //We load the test set
            var xTestSet = new CpuTensor<byte>(new[] { 10000, Channels, Height, Width }, "xTestSet");
            var yTestSet = new CpuTensor<byte>(new[] { 10000, 1, 1, 1 }, "yTestSet");
            Load(Path.Combine(path, "test.bin"), xTestSet, yTestSet);
            //We normalize the test set with 0 mean / 1 volatility (coming from the training set)
            ToWorkingSet(xTestSet, yTestSet, out var xTest, out var yTest, meanAndVolatilityOfEachChannelInTrainingSet);
            AbstractDataSetLoader.AreCompatible_X_Y(xTest, yTest);
            int[] testElementIdToCategoryId = yTestSet.Content.Select(x => (int)x).ToArray();
            Debug.Assert(testElementIdToCategoryId.Length == xTestSet.Shape[0]);

            //Uncomment the following line to take only the first elements
            //xTrain = (CpuTensor<float>)xTrain.ExtractSubTensor(0, 1000);yTrain = (CpuTensor<float>)yTrain.ExtractSubTensor(0, xTrain.Shape[0]); xTest = (CpuTensor<float>)xTest.ExtractSubTensor(0, 1000); ; yTest = (CpuTensor<float>)yTest.ExtractSubTensor(0, xTest.Shape[0]);

            Training = new InMemoryDataSetLoader(xTrain, yTrain, trainElementIdToCategoryId, CategoryIdToDescription, Name);
            Test = new InMemoryDataSetLoader(xTest, yTest, testElementIdToCategoryId, CategoryIdToDescription, Name);
        }

        public void Dispose()
        {
            Training?.Dispose();
            Test?.Dispose();
        }

        private static void ToWorkingSet(CpuTensor<byte> x, CpuTensor<byte> y, out CpuTensor<float> xWorkingSet, out CpuTensor<float> yWorkingSet, List<Tuple<double, double>> meanAndVolatilityOfEachChannel)
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
