using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.CPU;
using SharpNet.Networks;
using SharpNet.Pictures;

namespace SharpNet.Datasets
{
    public class MnistDataset : AbstractTrainingAndTestDataset
    {
        private static readonly string[] CategoryIndexToDescription = new[] { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };
        public static int CategoryCount => CategoryIndexToDescription.Length;

        public override DataSet Training { get; }
        public override DataSet Test { get; }


        public static readonly int[] Shape_CHW = {3, 32, 32};

        public MnistDataset() : base ("MNIST")
        {
            var trainingSet = PictureTools.ReadInputPictures(FileNameToPath("train-images.idx3-ubyte"), FileNameToPath("train-labels.idx1-ubyte"));
            var trainWorkingSet = ToWorkingSet(trainingSet);
            var xTrain = trainWorkingSet.Item1;
            var yTrain = trainWorkingSet.Item2;

            Training = new InMemoryDataSet(xTrain, yTrain, Name, Objective_enum.Classification, null);

            var testSet = PictureTools.ReadInputPictures(FileNameToPath("t10k-images.idx3-ubyte"), FileNameToPath("t10k-labels.idx1-ubyte"));
            var testWorkingSet = ToWorkingSet(testSet);
            var xTest = testWorkingSet.Item1;
            var yTest = testWorkingSet.Item2;
            Test = new InMemoryDataSet(xTest, yTest, Name, Objective_enum.Classification, null);
        }


        private static Tuple<CpuTensor<float>, CpuTensor<float>> ToWorkingSet(List<KeyValuePair<CpuTensor<byte>, int>> t)
        {
            Debug.Assert(t[0].Key.Dimension == 2);
            int setSize = t.Count;

            var height = t[0].Key.Shape[0];
            var width = t[0].Key.Shape[1];
            var X = new CpuTensor<float>(new[] { setSize, 1, height, width });
            var Y = new CpuTensor<float>(new[] { setSize, 10 });
            for (int m = 0; m < setSize; ++m)
            {
                var matrix = t[m].Key;
                for (int row = 0; row < height; ++row)
                {
                    for (int col = 0; col < width; ++col)
                    {
                        X.Set(m, 0, row, col, matrix.Get(row, col) / 255f);
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
