using System;
using System.Collections.Generic;
using System.IO;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public static class CIFAR10
    {
        public const int Channels = 3;
        public const int Height = 32;
        public const int Width = Height;
        public const int Categories = 10;

        public static void LoadCifar10(out CpuTensor<float> xTrain, out CpuTensor<float> yTrain, out CpuTensor<float> xTest, out CpuTensor<float> yTest)
        {
            Load(out CpuTensor<byte> xTrainingSet, out var yTrainingSet, out var xTestSet, out var yTestSet);
            //We normalize the input with 0 mean / 1 volatility
            //var meanAndVolatilityOfEachChannel = xTrainingSet.ComputeMeanAndVolatilityOfEachChannel(x=>(double)x);
            var meanAndVolatilityOfEachChannel = new List<Tuple<double, double>>{Tuple.Create(125.306918046875, 62.9932192781369), Tuple.Create(122.950394140625, 62.0887076400142),Tuple.Create(113.865383183594, 66.7048996406309)};
            ToWorkingSet(xTrainingSet, yTrainingSet, out xTrain, out yTrain, meanAndVolatilityOfEachChannel);
            ToWorkingSet(xTestSet, yTestSet, out xTest, out yTest, meanAndVolatilityOfEachChannel);
            
            //Uncomment the following line to take only the first 1000 elements;
            //xTest = null;yTest = null;xTrain = (CpuTensor<float>)xTrain.ExtractSubTensor(0, 1000);yTrain = (CpuTensor<float>)yTrain.ExtractSubTensor(0, xTrain.Shape[0]);
        }
        private static void Load(out CpuTensor<byte> xTrainingSet, out CpuTensor<byte> yTrainingSet, out CpuTensor<byte> xTestSet, out CpuTensor<byte> yTestSet)
        {
            var path = @"C:\Projects\SharpNet\Tests\Data\cifar-10-batches-bin\";
            xTrainingSet = new CpuTensor<byte>(new[] {50000, Channels, Height, Width }, "xTrainingSet");
            yTrainingSet = new CpuTensor<byte>(new[] {50000, 1, 1, 1}, "yTrainingSet");
            xTestSet = new CpuTensor<byte>(new[] {10000, Channels, Height, Width }, "xTestSet");
            yTestSet = new CpuTensor<byte>(new[] {10000, 1, 1, 1}, "yTestSet");
            for (int i = 0; i < 5; ++i)
            {
                LoadAt(Path.Combine(path, "data_batch_" + (i + 1) + ".bin"), xTrainingSet, yTrainingSet, 10000 * i);
            }

            LoadAt(Path.Combine(path, "test_batch.bin"), xTestSet, yTestSet, 0);
        }

        private static void ToWorkingSet(CpuTensor<byte> x, CpuTensor<byte> y, out CpuTensor<float> xWorkingSet, out CpuTensor<float> yWorkingSet, List<Tuple<double, double>> meanAndVolatilityOfEachChannel)
        {
            xWorkingSet = x.Select((n, c, val) =>(float) ((val - meanAndVolatilityOfEachChannel[c].Item1) /Math.Max(meanAndVolatilityOfEachChannel[c].Item2, 1e-9)));
            yWorkingSet = y.ToCategorical(1.0f, out _);
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
