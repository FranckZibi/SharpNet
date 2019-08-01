using System;
using System.Collections.Generic;
using System.IO;
using SharpNet.CPU;
using SharpNet.Networks;
using SharpNet.Pictures;

namespace SharpNet.Datasets
{
    public static class MNIST
    {
        public static void Load(out CpuTensor<double> X_train, out CpuTensor<double> Y_train, out CpuTensor<double> X_test, out CpuTensor<double> Y_test)
        {
            var trainTuple = ToWorkingSet(TrainingSet);
            X_train = trainTuple.Item1;
            Y_train = trainTuple.Item2;
            var testTuple = ToWorkingSet(TestSet);
            X_test = testTuple.Item1;
            Y_test = testTuple.Item2;
        }


        private static Tuple<CpuTensor<double>, CpuTensor<double>> ToWorkingSet(List<KeyValuePair<CpuTensor<byte>, int>> t)
        {
            int setSize = t.Count;

            //setSize = Math.Min(5000,setSize);

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

        private static List<KeyValuePair<CpuTensor<byte>, int>> TrainingSet => PictureTools.ReadInputPictures(FileNameToPath("train-images.idx3-ubyte"), FileNameToPath("train-labels.idx1-ubyte"));
        private static List<KeyValuePair<CpuTensor<byte>, int>> TestSet => PictureTools.ReadInputPictures(FileNameToPath("t10k-images.idx3-ubyte"), FileNameToPath("t10k-labels.idx1-ubyte"));
    }
}
