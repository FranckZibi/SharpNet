using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using NUnit.Framework;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.Networks;

namespace SharpNetTests.NonReg
{
    [TestFixture]
    public class TestEnsembleLearning
    {

        [Test, Explicit]
        public void TestAptos2019BlindnessDetection()
        {
            int heightAndWidth = 128;
            var csvFilename = @"C:\temp\aptos2019-blindness-detection\train_images\train.csv";
            var dataDirectory = @"C:\temp\aptos2019-blindness-detection\train_images_cropped_square_resize_" + heightAndWidth + "_" + heightAndWidth;
            var directoryLoader = Aptos2019BlindnessDetection.ValueOf(csvFilename, dataDirectory, heightAndWidth, heightAndWidth, null).SplitIntoTrainingAndValidation(0.9).Test;

            //var csvFilename = @"C:\temp\aptos2019-blindness-detection\test_images\test.csv";
            //var dataDirectory = @"C:\temp\aptos2019-blindness-detection\test_images_cropped_square_resize_" + heightAndWidth + "_" + heightAndWidth;
            //var directoryLoader = Aptos2019BlindnessDetection.ValueOf(csvFilename, dataDirectory, heightAndWidth, heightAndWidth, null);

            var networks = new[]
            {
                @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-8_Aptos2019_Cutout_0_10_noCutMix_Mixup_Rotation90_20190828_1201_150.txt"
                //@"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-8_Aptos2019_noCutout_noCutMix_Mixup_Rotation90_128_20190816_1520_150.txt",
                //@"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-8_Aptos2019_noCutout_noCutMix_Mixup_Rotation90_128_20190816_1520_70.txt",
            };
            var ensembleLearning = new EnsembleLearning(networks);
            var predictionsAndAccuracy = ensembleLearning.Predict(directoryLoader);

            var elementIdToPredictedCategory = predictionsAndAccuracy.Item1.ComputePrediction();
            var categoryIdToElementIdDescriptions = new Dictionary<int, List<string>>();
            for (int elementId = 0; elementId < directoryLoader.Count; ++elementId)
            {
                var categoryId = elementIdToPredictedCategory[elementId];
                if (!categoryIdToElementIdDescriptions.ContainsKey(categoryId))
                {
                    categoryIdToElementIdDescriptions[categoryId] = new List<string>();
                }
                categoryIdToElementIdDescriptions[categoryId].Add(directoryLoader.ElementIdToDescription(elementId));
            }

            var pythonFile = "c:/temp/toto.py";
            File.Delete(pythonFile);
            foreach (var e in categoryIdToElementIdDescriptions)
            {
                File.AppendAllText(pythonFile, "_"+e.Key+"_set = { "+string.Join(",", e.Value.Select(x=>"'"+x+"'"))+ "}"+Environment.NewLine);
            }
            directoryLoader.CreatePredictionFile(predictionsAndAccuracy.Item1, "c:/temp/toto.txt");
        }

        [Test, Explicit]
        public void Test()
        {
            Console.WriteLine("loading CIFAR10");
            //CIFAR10.LoadCifar10(out var _, out var _, out var xTestCpu, out var yExpectedCpu);

            IDataSet loader = new CIFAR10DataLoader();

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

            //95.69 <= 94.99 (200 epochs) + 94.5 (70 epochs)
            //var files_WRN_16_4_CyclicCosineAnnealing_10_2_200epochs = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-4_CyclicCosineAnnealing_10_2_20190606_0803_200.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-4_CyclicCosineAnnealing_10_2_20190606_0803_70.txt",
            //};
            //96.30 <= 95.67 (200 epochs) + 95.11 (70 epochs)
            //var files_WRN_40_4_CyclicCosineAnnealing_10_2_200epochs = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-40-4_CyclicCosineAnnealing_10_2_20190606_0803_200.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-40-4_CyclicCosineAnnealing_10_2_20190606_0803_70.txt",
            //};
            //96.03 <= 95.69 (200 epochs) + 95.27 (70 epochs)
            //var files_WRN_16_8_CyclicCosineAnnealing_10_2_200epochs = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-8_CyclicCosineAnnealing_10_2_20190606_0934_200.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-8_CyclicCosineAnnealing_10_2_20190606_0934_70.txt",
            //};

            //96.18 <= 96.15 (150 epochs) + 95.00 (70 epochs)
            //var files_WRN_40_4_CyclicCosineAnnealing_10_2_150epochs = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-40-4_CyclicCosineAnnealing_10_2_150epochs_20190806_2315_150.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-40-4_CyclicCosineAnnealing_10_2_150epochs_20190806_2315_70.txt",
            //};


            //96.79 <= 96.40 (150 epochs) + 96.05 (70 epochs)
            var files_WRN_28_10_150epochs = new[]
            {
                @"C:\Users\Franck\AppData\Local\SharpNet\WRN-28-10_20190824_2117_150.txt",
                @"C:\Users\Franck\AppData\Local\SharpNet\WRN-28-10_20190824_0947_70.txt",
            };

            

            //96.01 <= 95.39 (150 epochs) + 95.19 (70 epochs)
            //var files_WRN_16_8_CyclicCosineAnnealing_10_2_150epochs = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-8_CyclicCosineAnnealing_10_2_150epochs_20190605_0921_150.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-16-8_CyclicCosineAnnealing_10_2_150epochs_20190605_0921_70.txt",
            //};

            //96.28 <= 95.28 (200 epochs) + 94.59 (171 epoch) + 92.07 (152 epoch)
            //96.14 <= 95.28 (200 epochs) + 94.59 (171 epoch)
            //var files_WRN_28_10_v2 = new[]
            //{
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-28-10_20190529_0709_200.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-28-10_20190529_0709_171.txt",
            //    @"C:\Users\fzibi\AppData\Local\SharpNet\WRN-28-10_20190529_0709_152.txt",
            //};
            var ensembleLearning = new EnsembleLearning(files_WRN_28_10_150epochs);
            ensembleLearning.Predict(loader.Test);
        }

    }
}
