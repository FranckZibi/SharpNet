using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Datasets
{
    public class ImdbTrainingAndTestDataset : AbstractTrainingAndTestDataset
    {
        // ReSharper disable once CollectionNeverQueried.Local
        private readonly IDictionary<string, int> _wordIndex = new Dictionary<string, int>();

        public override DataSet Training { get; }
        public override DataSet Test { get; }


        public ImdbTrainingAndTestDataset() : base("IMDB")
        {
            var (xTrain, yTrain) = ReadImdDataSet(FileNameToPath("x_train.txt"), FileNameToPath("y_train.txt"));
            Training = new InMemoryDataSet(xTrain, yTrain, Name);

            var (xTest, yTest) = ReadImdDataSet(FileNameToPath("x_test.txt"), FileNameToPath("y_test.txt"));
            Test = new InMemoryDataSet(xTest, yTest, Name);

            foreach (var token in File.ReadAllText(FileNameToPath("dico.txt")).Split(','))
            {
                var idAndWord = token.Trim().Split(':').Select(x => x.Trim()).ToArray();
                _wordIndex[idAndWord[1]] = int.Parse(idAndWord[0]);
            }
        }

        private static (CpuTensor<float> x, CpuTensor<float> y) ReadImdDataSet(string xFilePath, string yFilePath)
        {
            var x = (CpuTensor<float>)TensorExtensions.FromNumpyArray(File.ReadAllText(xFilePath)) ;
            var y = (CpuTensor<float>)TensorExtensions.FromNumpyArray(File.ReadAllText(yFilePath));
            y.ReshapeInPlace(new[] { y.Count, 1 });

            //var yTmpAsSpan = ((CpuTensor<float>)TensorExtensions.FromNumpyArray(File.ReadAllText(yFilePath))).AsReadonlyFloatCpuContent;
            //var y = new CpuTensor<float>(new []{ yTmpAsSpan.Length, 2});
            //var yAsSpan = y.AsFloatCpuSpan;
            //for (int i = 0; i < yTmpAsSpan.Length; ++i)
            //{
            //    yAsSpan[2 * i + (int)(0.1+yTmpAsSpan[i])] = 1;
            //}

            return (x, y);
        }

        private static string FileNameToPath(string fileName)
        {
            return Path.Combine(NetworkSample.DefaultDataDirectory, "IMDB", fileName);
        }
    }
}