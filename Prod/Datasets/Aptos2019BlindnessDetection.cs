namespace SharpNet.Datasets
{
    public static class Aptos2019BlindnessDetection
    {
        public static DirectoryDataSetLoader ValueOf(string csvFilename, string trainingSetDirectory, int height, int width, Logger logger)
        {
            return new DirectoryDataSetLoader(
                csvFilename, 
                trainingSetDirectory, 
                logger,
                "Aptos2019BlindnessDetection", //Name
                3, //Channels
                height, 
                width, 
                new[] { "No DR", "Mild", "Moderate", "Severe", "Proliferative DR" },
                true,
                DirectoryDataSetLoader.DefaultCompute_CategoryId_Description_FullName);
        }
    }
}
