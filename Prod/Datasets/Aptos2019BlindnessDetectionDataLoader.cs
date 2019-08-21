namespace SharpNet.Datasets
{
    public class Aptos2019BlindnessDetectionDataLoader : IDataSet
    {
        private static readonly string[] CategoryIdToDescription = new[] { "No DR", "Mild", "Moderate", "Severe", "Proliferative DR" };
        public const int Channels = 3;
        public const int Categories = 5;
        public IDataSetLoader Training { get; }
        public IDataSetLoader Test { get; }


        //public static DirectoryDataSetLoader ResizeTrainingDirectory(string csvFilename, string trainingSetDirectory, int targetHeightAndWidth, Logger logger)
        //{
        //    logger = logger ?? Logger.ConsoleLogger;
        //    var trainingSet = new DirectoryDataSetLoader(csvFilename, trainingSetDirectory, logger, Channels, -1, -1, CategoryIdToDescription);

        //    var resizedTrainingSet = trainingSet
        //        .CropBorder()
        //        .MakeSquarePictures(true)
        //        .Resize(targetHeightAndWidth, targetHeightAndWidth);
        //    return resizedTrainingSet;
        //}
        public Aptos2019BlindnessDetectionDataLoader(string csvFilename, string trainingSetDirectory, int height, int width, double percentageInTrainingSet, Logger logger)
        {
            var fullTestSet = new DirectoryDataSetLoader(csvFilename, trainingSetDirectory, logger, Channels, height, width, CategoryIdToDescription, true, DirectoryDataSetLoader.DefaultCompute_CategoryId_Description_FullName);
            if (percentageInTrainingSet >= 0.999)
            {
                Training = fullTestSet;
                Test = null;
            }
            else
            {
                int lastElementIdIncludedInTrainingSet = (int) (percentageInTrainingSet * fullTestSet.Count);
                Training = new SubDataSetLoader(fullTestSet, id => id<= lastElementIdIncludedInTrainingSet);
                Test = new SubDataSetLoader(fullTestSet, id => id > lastElementIdIncludedInTrainingSet);
            }
        }
        public void Dispose()
        {
            Training?.Dispose();
            Test?.Dispose();
        }
    }
}
