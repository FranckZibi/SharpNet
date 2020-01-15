namespace SharpNet.Datasets
{
    public static class Aptos2019BlindnessDetection
    {
        public static DirectoryDataSet ValueOf(string csvFilename, string trainingSetDirectory, int height, int width, Logger logger)
        {
            return new DirectoryDataSet(
                csvFilename, 
                trainingSetDirectory, 
                logger,
                "Aptos2019BlindnessDetection", //Name
                3, //Channels
                height, 
                width, 
                new[] { "No DR", "Mild", "Moderate", "Severe", "Proliferative DR" },
                true,
                DirectoryDataSet.DefaultCompute_CategoryIndex_Description_FullName);
        }
    }
}
