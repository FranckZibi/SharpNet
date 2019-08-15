﻿namespace SharpNet.Datasets
{
    public class Aptos2019BlindnessDetectionDataLoader : IDataSet<float>
    {
        private static readonly string[] CategoryIdToDescription = new[] { "No DR", "Mild", "Moderate", "Severe", "Proliferative DR" };
        public const int Channels = 3;
        public const int Categories = 5;
        public IDataSetLoader<float> Training { get; }
        public IDataSetLoader<float> Test { get; }


        public int Height {get; }
        public int Width { get; }


        public static DirectoryDataSetLoader<float> ResizeTrainingDirectory(string csvFilename, string trainingSetDirectory, int tagetHeightAndWidth, double percentageInTrainingSet, Logger logger)
        {
            logger = logger ?? new Logger(csvFilename + ".log", true);
            var trainingSet = new DirectoryDataSetLoader<float>(csvFilename, trainingSetDirectory, logger, Channels, -1, -1, CategoryIdToDescription);

            var resizedTrainingSet = trainingSet
                .CropBorder()
                .MakeSquarePictures(true)
                .Resize(32, 32);
            return resizedTrainingSet;
        }
        public static IDataSet<float> ValueOf(int widthAndHeight, double percentageInTrainingSet, Logger logger)
        {
            var csvFilename = @"C:\temp\aptos2019-blindness-detection\train_images\train.csv";
            var trainingSetDirectory = @"C:\temp\aptos2019-blindness-detection\train_images_cropped_square_resize_" + widthAndHeight + "_" + widthAndHeight;
            return new Aptos2019BlindnessDetectionDataLoader(csvFilename, trainingSetDirectory, widthAndHeight,
                widthAndHeight, percentageInTrainingSet, logger);
        }

        public Aptos2019BlindnessDetectionDataLoader(string csvFilename, string trainingSetDirectory, int height, int width, double percentageInTrainingSet, Logger logger)
        {
            Height = height;
            Width = width;
            var fullTestSet = new DirectoryDataSetLoader<float>(csvFilename, trainingSetDirectory, logger, Channels, height, width, CategoryIdToDescription);
            if (percentageInTrainingSet >= 0.999)
            {
                Training = fullTestSet;
                Test = null;
            }
            else
            {
                Training = fullTestSet.Sub(0, (int)(percentageInTrainingSet * fullTestSet.Count));
                Test = fullTestSet.Sub(Training.Count, fullTestSet.Count - 1);
            }
        }
        public void Dispose()
        {
            Training?.Dispose();
            Test?.Dispose();
        }

       
    }
}
