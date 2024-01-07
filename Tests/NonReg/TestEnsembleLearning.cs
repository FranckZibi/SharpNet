using System;
using System.IO;
using NUnit.Framework;
using SharpNet.Datasets;
using SharpNet.Networks;

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestEnsembleLearning
    {
        [Test, Explicit]
        public void TestCIFAR10()
        {
            Console.WriteLine("loading "+ CIFAR10DataSet.NAME);
            var cifar10 = new CIFAR10DataSet();

            //94.94 <= 94.82 (160 epochs) + 93.57 (70 epochs)
            //var files_ResNet56V2 = new[]
            //{
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "ResNet56V2_CIFAR10_CyclicCosineAnnealing_10_2_20190528_1732_160.txt"),
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "ResNet56V2_CIFAR10_CyclicCosineAnnealing_10_2_20190528_1732_70.txt")
            //};

            //95.83 <= 95.21 (160 epochs) + (70 epoch)
            //var files_ResNet110V2 = new[]
            //{
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "ResNet110V2_CIFAR10_CyclicCosineAnnealing_10_2_20190528_1358_160.txt"),
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "ResNet110V2_CIFAR10_CyclicCosineAnnealing_10_2_20190528_1358_70.txt")
            //};

            //96.05 <= 95.67 (150 epochs) + 94.91 (70 epochs)
            //var files = new[]
            //{
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-16-4_GAP_MAX_20200224_0815_150.txt"),
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-16-4_GAP_MAX_20200224_0815_70.txt")
            //};



            //96.30 <= 95.84 (150 epochs) + 95.41 (70 epochs)
            //var files = new[]
            //{
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-40-4_GAP_MAX_20200224_0815_150.txt"),
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-40-4_GAP_MAX_20200224_0815_70.txt")
            //};

            //96.55 <= 96.30 (150 epochs) + 95.94 (70 epochs)
            var files = new[]
            {
                Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-16-8_GAP_MAX_20200224_0922_150.txt"),
                Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-16-8_GAP_MAX_20200224_0922_70.txt")
            };

            //96.76 <= 96.65 (150 epochs) + 95.88 (70 epochs)
            //var files = new[]
            //{
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-16-10_GAP_MAX_20200224_1134_150.txt"),
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-16-10_GAP_MAX_20200224_1134_70.txt"),
            //};

            //96.84 <= 96.58 (150 epochs) + 96.00 (70 epochs)
            //var files = new[]
            //{
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-28-8_GAP_MAX_20200224_1245_150.txt"),
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-28-8_GAP_MAX_20200224_1245_70.txt"),
            //};

            //96.79 <= 96.40 (150 epochs) + 96.05 (70 epochs)
            //var files = new[]
            //{
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-28-10_20190824_2117_150.txt"),
            //    Path.Combine(NetworkSample.DefaultDataDirectory, "WRN-28-10_20190824_0947_70.txt"),
            //};

            var ensembleLearning = new EnsembleLearning(files);
            ensembleLearning.Predict(cifar10.Test, 64);
        }
    }
}
