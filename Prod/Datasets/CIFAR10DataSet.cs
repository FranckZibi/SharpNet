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
# WRN-16-4  |   2,752,506 | 94.61 (-----) | 95.67 (--.--) | 94.99 (95.69) | 94.98 (94.76) |  27.4
# WRN-40-4  |   8,959,994 | 95.43 (-----) | 96.29 (96.18) | 95.67 (96.30) | 95.43 (-----) |  77.3
# WRN-16-8  |  10,968,570 | 95.20 (-----) | 95.99 (--.--) | 95.69 (96.03) | 95.73         |  83.0
# WRN-16-10 |  17,125,626 | ----- (-----) | ----- (-----) | ----- (-----) | NA            | 136.0
# WRN-28-8  |  23,369,210 | ----- (-----) | ----- (-----) | ----- (-----) | NA            | 173.0
# WRN-28-10 |  36,497,146 | 95.28 (-----) | 96.40 (96.79) | ----- (-----) | 96.00 (96.11) | 296.5
# -----------------------------------------------------------------------------------------------
*/

namespace SharpNet.Datasets
{
    public class CIFAR10DataSet : ITrainingAndTestDataSet
    {
        private readonly string[] CategoryIdToDescription = new[] { "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" };

        public string Name => "CIFAR-10";
        public const int Channels = 3;
        public const int Height = 32;
        public const int Width = Height;
        public const int Categories = 10;

        public static readonly int[] InputShape_CHW = { Channels, Height, Width };


        public IDataSet Training { get; }
        public IDataSet Test { get; }

        public CIFAR10DataSet()
        {
            var path = Path.Combine(NetworkConfig.DefaultDataDirectory, Name);

            //We load the training set
            var xTrainingSet = new CpuTensor<byte>(new[] { 50000, Channels, Height, Width }, "xTrainingSet");
            var yTrainingSet = new CpuTensor<byte>(new[] { 50000, 1, 1, 1 }, "yTrainingSet");
            for (int i = 0; i < 5; ++i)
            {
                LoadAt(Path.Combine(path, "data_batch_" + (i + 1) + ".bin"), xTrainingSet, yTrainingSet, 10000 * i);
            }
            //We normalize the input with 0 mean / 1 volatility
            //var meanAndVolatilityOfEachChannelInTrainingSet = xTrainingSet.ComputeMeanAndVolatilityOfEachChannel(x=>(double)x);
            var meanAndVolatilityOfEachChannelInTrainingSet = new List<Tuple<float, float>> { Tuple.Create(125.306918046875f, 62.9932192781369f), Tuple.Create(122.950394140625f, 62.0887076400142f), Tuple.Create(113.865383183594f, 66.7048996406309f) };
            ToWorkingSet(xTrainingSet, yTrainingSet, out var xTrain, out var yTrain, meanAndVolatilityOfEachChannelInTrainingSet);
            AbstractDataSet.AreCompatible_X_Y(xTrain, yTrain);
            int[] trainElementIdToCategoryId = yTrainingSet.Content.Select(x => (int)x).ToArray();
            Debug.Assert(trainElementIdToCategoryId.Length == xTrainingSet.Shape[0]);

            //We load the test set
            var xTestSet = new CpuTensor<byte>(new[] { 10000, Channels, Height, Width }, "xTestSet");
            var yTestSet = new CpuTensor<byte>(new[] { 10000, 1, 1, 1 }, "yTestSet");
            LoadAt(Path.Combine(path, "test_batch.bin"), xTestSet, yTestSet, 0);
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
        private static void LoadAt(string path, CpuTensor<byte> x, CpuTensor<byte> y, int indexFirst)
        {
            var b = File.ReadAllBytes(path);
            for (int count = 0; count < 10000; ++count)
            {
                int bIndex = count * (1 + Height * Width * Channels);
                int xIndex = (count + indexFirst) * x.MultDim0;
                int yIndex = (count + indexFirst) * y.MultDim0;
                y[yIndex] = b[bIndex];
                for (int j = 0; j < Height * Width * Channels; ++j)
                {
                    x[xIndex + j] = b[bIndex + 1 + j];
                }
            }
        }
    }
}
