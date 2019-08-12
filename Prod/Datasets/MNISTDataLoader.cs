using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Networks;
using SharpNet.Pictures;

namespace SharpNet.Datasets
{
    public class MNISTDataLoader<T> : IDataSet<T> where T: struct
    {
        private readonly string[] CategoryIdToDescription = new[] { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

        public IDataSetLoader<T> Training { get; }
        public IDataSetLoader<T> Test { get; }

        public MNISTDataLoader(ImageDataGenerator imageDataGenerator)
        {
            var trainingSet = PictureTools.ReadInputPictures(FileNameToPath("train-images.idx3-ubyte"), FileNameToPath("train-labels.idx1-ubyte"));
            var trainWorkingSet = ToWorkingSet(trainingSet);
            var xTrain = trainWorkingSet.Item1;
            var yTrain = trainWorkingSet.Item2;

            var trainElementIdToCategoryId = trainingSet.Select(x=>x.Value).ToArray();
            Training = (IDataSetLoader<T>)new InMemoryDataSetLoader<double>(xTrain, yTrain, trainElementIdToCategoryId, CategoryIdToDescription, imageDataGenerator);

            var testSet = PictureTools.ReadInputPictures(FileNameToPath("t10k-images.idx3-ubyte"), FileNameToPath("t10k-labels.idx1-ubyte"));
            var testWorkingSet = ToWorkingSet(testSet);
            var xTest = testWorkingSet.Item1;
            var yTest = testWorkingSet.Item2;
            var testElementIdToCategoryId = testSet.Select(x => x.Value).ToArray();
            Test = (IDataSetLoader <T>)new InMemoryDataSetLoader<double>(xTest, yTest, testElementIdToCategoryId, CategoryIdToDescription, imageDataGenerator);
        }


        public void Dispose()
        {
            Training?.Dispose();
            Test?.Dispose();
        }

        private static Tuple<CpuTensor<double>, CpuTensor<double>> ToWorkingSet(List<KeyValuePair<CpuTensor<byte>, int>> t)
        {
            int setSize = t.Count;

            var X = new CpuTensor<double>(new[] { setSize, 1, t[0].Key.Height, t[0].Key.Width }, "X");
            var Y = new CpuTensor<double>(new[] { setSize, 10 }, "Y");
            for (int m = 0; m < setSize; ++m)
            {
                var matrix = t[m].Key;
                for (int row = 0; row < matrix.Height; ++row)
                {
                    for (int col = 0; col < matrix.Width; ++col)
                    {
                        X.Set(m, 0, row, col, matrix.Get(row, col) / 255.0);
                    }
                }
                Y.Set(m, t[m].Value, 1);
            }
            return Tuple.Create(X, Y);
        }
        private static string FileNameToPath(string fileName)
        {
            return Path.Combine(NetworkConfig.DefaultDataDirectory, "MNIST", fileName);
        }
    }
}
