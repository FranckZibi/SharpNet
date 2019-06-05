using System;
using NUnit.Framework;
using SharpNet.Datasets;
using SharpNet.Networks;
namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestEnsembleLearning
    {
        [Test, Explicit]
        public void Test()
        {
            Console.WriteLine("loading CIFAR10");
            CIFAR10.LoadCifar10(out var _, out var _, out var xTestCpu, out var yExpectedCpu);

            //95.65 <= 95.32 (200 epochs) + 94.02 (70 epochs)
            //var files_WRN_40_4 = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-40-4_CyclicCosineAnnealing_10_2_020learningRate_20190527_0827_200.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-40-4_CyclicCosineAnnealing_10_2_020learningRate_20190527_0827_70.txt",
            //};

            //94.94 <= 94.82 (160 epochS) + 93.57 (70 epochs)
            //var files_ResNet56V2 = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\ResNet56V2_CIFAR10_CyclicCosineAnnealing_10_2_20190528_1732_160.txt"
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\ResNet56V2_CIFAR10_CyclicCosineAnnealing_10_2_20190528_1732_70.txt",
            //};

            //95.83 <= 95.21 (160 epochs) + (70 epoch)
            //var files_ResNet110V2 = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\ResNet110V2_CIFAR10_CyclicCosineAnnealing_10_2_20190528_1358_160.txt"
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\ResNet110V2_CIFAR10_CyclicCosineAnnealing_10_2_20190528_1358_70.txt",
            //};

            //95.99 <= 95.46 (150 epochs) + 95.04 (70 epochs)
            //var files_WRN_28_10 = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-28-10_CyclicCosineAnnealing_10_2_150epochs_20190529_0709_150.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-28-10_CyclicCosineAnnealing_10_2_150epochs_20190529_0709_70.txt",
            //};


            //95.22 <= 93.92 (200 epochs) + 93.56 (150 epochs) + 94.56 (100 epochs)
            //var files_WRN_16_4_CyclicCosineAnnealing_50_1 = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-4_CyclicCosineAnnealing_50_1_20190604_1801_200.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-4_CyclicCosineAnnealing_50_1_20190604_1801_150.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-4_CyclicCosineAnnealing_50_1_20190604_1801_100.txt"
            //};

            //96.00 <= 95.81 (200 epochs) + 94.99 (138 epochs)
            //var files_WRN_40_4_2AvgPoolingSize = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-40-4_2AvgPoolingSize_20190604_1339_200.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-40-4_2AvgPoolingSize_20190604_1339_138.txt",
            //};

            //95.94 <= 95.51 (200 epochs) + 95.50 (168 epochs)
            //var files_WRN_16_8_2AvgPoolingSize = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-8_2AvgPoolingSize_20190604_1400_200.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-8_2AvgPoolingSize_20190604_1400_168.txt",
            //};

            //95.5 <= 94.79 (150 epochs) + 94.72 (70 epochs)
            //var files_WRN_16_4_CyclicCosineAnnealing_10_2_150epochs = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-4_CyclicCosineAnnealing_10_2_150epochs_20190605_0813_150.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-4_CyclicCosineAnnealing_10_2_150epochs_20190605_0813_70.txt",
            //};

            //96.15 <= 95.5 (150 epochs) + 95.09 (70 epochs)
            //var files_WRN_40_4_CyclicCosineAnnealing_10_2_150epochs = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-40-4_CyclicCosineAnnealing_10_2_150epochs_20190605_0813_150.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-40-4_CyclicCosineAnnealing_10_2_150epochs_20190605_0813_70.txt",
            //};

            //96.01 <= 95.39 (150 epochs) + 95.19 (70 epochs)
            //var files_WRN_16_8_CyclicCosineAnnealing_10_2_150epochs = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-8_CyclicCosineAnnealing_10_2_150epochs_20190605_0921_150.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-8_CyclicCosineAnnealing_10_2_150epochs_20190605_0921_70.txt",
            //};

            //96.28 <= 95.28 (200 epochs) + 94.59 (171 epoch) + 92.07 (152 epoch)
            //96.14 <= 95.28 (200 epochs) + 94.59 (171 epoch)
            var files_WRN_28_10_v2 = new[]
            {
                @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-28-10_20190529_0709_200.txt",
                @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-28-10_20190529_0709_171.txt",
                @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-28-10_20190529_0709_152.txt",
            };
            var ensembleLearning = new EnsembleLearning(files_WRN_28_10_v2);
            ensembleLearning.Predict(xTestCpu, yExpectedCpu);
        }

    }
}
