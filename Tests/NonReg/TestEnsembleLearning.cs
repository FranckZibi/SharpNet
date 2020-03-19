using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
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
            var categoryIndexToElementIdDescriptions = new Dictionary<int, List<string>>();
            for (int elementId = 0; elementId < directoryLoader.Count; ++elementId)
            {
                var categoryIndex = elementIdToPredictedCategory[elementId];
                if (!categoryIndexToElementIdDescriptions.ContainsKey(categoryIndex))
                {
                    categoryIndexToElementIdDescriptions[categoryIndex] = new List<string>();
                }
                categoryIndexToElementIdDescriptions[categoryIndex].Add(elementId.ToString());
            }

            var pythonFile = "c:/temp/toto.py";
            File.Delete(pythonFile);
            foreach (var e in categoryIndexToElementIdDescriptions)
            {
                File.AppendAllText(pythonFile, "_" + e.Key + "_set = { " + string.Join(",", e.Value.Select(x => "'" + x + "'")) + "}" + Environment.NewLine);
            }
            directoryLoader.CreatePredictionFile(predictionsAndAccuracy.Item1, "c:/temp/toto.txt");
        }

        [Test, Explicit]
        public void TestCIFAR10()
        {
            Console.WriteLine("loading CIFAR-10");
            var cifar10 = new CIFAR10DataSet();

            //94.94 <= 94.82 (160 epochs) + 93.57 (70 epochs)
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

            //96.05 <= 95.67 (150 epochs) + 94.91 (70 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-4_GAP_MAX_20200224_0815_150.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-4_GAP_MAX_20200224_0815_70.txt"
            //};

            //96.30 <= 95.84 (150 epochs) + 95.41 (70 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-40-4_GAP_MAX_20200224_0815_150.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-40-4_GAP_MAX_20200224_0815_70.txt"
            //};

            //96.55 <= 96.30 (150 epochs) + 95.94 (70 epochs)
            var files = new[]
            {
                @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-8_GAP_MAX_20200224_0922_150.txt",
                @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-8_GAP_MAX_20200224_0922_70.txt"
            };

            //96.76 <= 96.65 (150 epochs) + 95.88 (70 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-10_GAP_MAX_20200224_1134_150.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-10_GAP_MAX_20200224_1134_70.txt",
            //};

            //96.84 <= 96.58 (150 epochs) + 96.00 (70 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-28-8_GAP_MAX_20200224_1245_150.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-28-8_GAP_MAX_20200224_1245_70.txt",
            //};

            //96.79 <= 96.40 (150 epochs) + 96.05 (70 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-28-10_20190824_2117_150.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-28-10_20190824_0947_70.txt",
            //};

            var ensembleLearning = new EnsembleLearning(files);
            ensembleLearning.Predict(cifar10.Test);
        }

        [Test, Explicit]
        public void TestSVHN()
        {
            Console.WriteLine("loading SVHN");
            var svhn = new SVHNDataSet(true);

            //WRN 16-4 30 epochs
            //98.33<= 98.24 (30 epochs) + 98.18 (10 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-4_30Epochs_20200120_0932_30.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-4_30Epochs_20200120_0932_10.txt",
            //};

            //WRN 40-4 30 epochs
            //98.38 <= 98.36 (30 epochs) + 98.09 (10 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-40-4_30Epochs_20200120_1210_30.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-40-4_30Epochs_20200120_1210_10.txt",
            //}

            //WRN 16-8 30 epochs
            //98.28 <= 98.13 (30 epochs) + 98.16 (10 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-8_30Epochs_20200120_1945_30.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-8_30Epochs_20200120_1945_10.txt",
            //};

            //WRN 16-10 30 epochs
            //98.40 <= 98.22 (30 epochs) + 98.28 (10 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-10_30Epochs_20200120_1954_30.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-10_30Epochs_20200120_1954_10.txt",
            //};

            //WRN 16-10 30 epochs
            //98.40 <= 98.33 (30 epochs) + 98.26 (10 epochs)
            var files = new[]
            {
                @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-10_GAP_AND_GlobalMaxPooling_30Epochs_20200306_2040_30.txt",
                @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-10_GAP_AND_GlobalMaxPooling_30Epochs_20200306_2040_10.txt",
            };

            //WRN 16-4 30 epochs AutoAugment
            //97.93<= 97.87 (30 epochs) + 97.57 (10 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-4_30Epochs_AutoAugment_20200119_2128_30.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-4_30Epochs_AutoAugment_20200119_2128_10.txt",
            //};

            //WRN 16-10 30 epochs AutoAugment
            //98.12<= 98.06 (30 epochs) + 97.87 (10 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-10_30Epochs_AutoAugment_20200120_0509_30.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-10_30Epochs_AutoAugment_20200120_0509_10.txt",
            //};


            //WRN 40-4 150 epochs small training dataSet
            //97.4<= 97.0 (150 epochs) + 97.00 (70 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-40-4_150Epochs_smallTrain_20200121_2232_150.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-40-4_150Epochs_smallTrain_20200121_2232_70.txt",
            //};

            //WRN 16-8 150 epochs small training dataSet
            //97.34<= 96.91 (150 epochs) + 96.88 (70 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-8_150Epochs_smallTrain_20200122_0021_150.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-8_150Epochs_smallTrain_20200122_0021_70.txt",
            //};

            //WRN 16-8 150 epochs AutoAugment small training dataSet
            //97.30<= 96.88 (150 epochs) + 96.83 (70 epochs)
            //var files = new[]
            //{
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-8_150Epochs_AutoAugment_smallTrain_20200122_0624_150.txt",
            //    @"C:\Users\Franck\AppData\Local\SharpNet\WRN-16-8_150Epochs_AutoAugment_smallTrain_20200122_0624_70.txt",
            //};

            var ensembleLearning = new EnsembleLearning(files);
            ensembleLearning.Predict(svhn.Test);
        }

    }
}
