using System;
using System.Collections.Generic;
using System.IO;
using SharpNet.Networks;

/*
BatchSize = 128
EpochCount = 30
SGD with momentum = 0.9 & L2 = 0.5* 1-e4
CutMix / no Cutout / no Mixup / FillMode = Reflect / Disable DivideBy10OnPlateau
AvgPoolingStride = 2
# --------------------------------------------------------------------------------
#           |             |    30-epoch   |   150-epoch   |   Orig Paper  | sec/epoch
# Model     |   #Params   |   SGDR 10-2   |   SGDR 10-2   |      WRN      | GTX1080
#           |             |   %Accuracy   |   %Accuracy   |   %Accuracy   | 
#           |             |(Ens. Learning)|(Ens. Learning)|   (dropout)   | 
# -------------------------------------------------------------------------------
# WRN-16-4  |   2,790,906 | 98.24 (98.33) | ----- (-----) | NA            |   314
# WRN-40-4  |   8,998,394 | 98.36 (98.38) | ----- (-----) | NA            |   927
# WRN-16-8  |  11,045,370 | 98.13 (98.28) | ----- (-----) | NA            |  1072
# WRN-16-10 |  17,221,626 | 98.22 (98.40) | ----- (-----) | NA            |  1715
# -------------------------------------------------------------------------------
*/


namespace SharpNet.Datasets
{
    //http://ufldl.stanford.edu/housenumbers/
    /*
    TRAIN	 73257
    TEST	 26032
    EXTRA	531131
    ==============
            630420
    */

    public class SVHNDataSet : AbstractTrainingAndTestDataSet
    {
        private static readonly string[] CategoryIndexToDescription = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
        public static int CategoryCount => CategoryIndexToDescription.Length;

        public override IDataSet Training { get; }
        public override IDataSet Test { get; }

        public override int CategoryByteToCategoryIndex(byte categoryByte)
        {
            return (categoryByte == 10) ? 0 : categoryByte;
        }
        public override byte CategoryIndexToCategoryByte(int categoryIndex)
        {
            return (categoryIndex == 0) ? (byte)10 : (byte)categoryIndex;
        }


        public static readonly int[] Shape_CHW = { 3, 32, 32 };

        /// <summary>
        /// 
        /// </summary>
        /// <param name="loadExtraFileForTraining">
        /// true if we should load both the train file (73.257 entries) and the extra file (531.131 entries) for training
        /// false if we should only load the train file (73.257 entries) for training</param>
        public SVHNDataSet(bool loadExtraFileForTraining = true) : base("SVHN")
        {
            var meanAndVolatilityOfEachChannelInTrainingSet = new List<Tuple<float, float>> { Tuple.Create(109.8823f, 50.11187f), Tuple.Create(109.7114f, 50.57312f), Tuple.Create(113.8187f, 50.85124f) };
            
            var directory = Path.Combine(NetworkConfig.DefaultDataDirectory, Name);
            var trainFiles = SplittedFileDataSet.AllBinFilesInDirectory(directory, loadExtraFileForTraining?new []{ "data_batch", "extra_batch"} : new [] { "data_batch"});
            Training = new SplittedFileDataSet(trainFiles, Name, CategoryIndexToDescription, Shape_CHW, meanAndVolatilityOfEachChannelInTrainingSet, CategoryByteToCategoryIndex);
            //to recompute the mean and volatility of each channel, uncomment the following line
            //meanAndVolatilityOfEachChannelInTrainingSet = ((SplittedFileDataSet)Training).ComputeMeanAndVolatilityForEachChannel();

            var testFiles = SplittedFileDataSet.AllBinFilesInDirectory(directory, "test_batch");
            Test = new SplittedFileDataSet(testFiles, Name, CategoryIndexToDescription, Shape_CHW, meanAndVolatilityOfEachChannelInTrainingSet, CategoryByteToCategoryIndex);
        }

        //public static void CreateBinFile()
        //{
        //    var matFile = Path.Combine(NetworkConfig.DefaultDataDirectory, "SVHN", "extra_32x32.mat");
        //    //var matFile = Path.Combine(NetworkConfig.DefaultDataDirectory, "SVHN", "train_32x32.mat");
        //    //var matFile = Path.Combine(NetworkConfig.DefaultDataDirectory, "SVHN", "test_32x32.mat");
        //    var reader = new MatReader(matFile);
        //    var X = reader.Read<byte[,,,]>("X");
        //    var y = reader.Read<byte[,]>("y");
        //    int Channels = 3;
        //    int Height = 32;
        //    int Width = 32;
        //    int totalCount = y.GetLength(0);
        //    int bytesByElement = Channels * Height * Width + 1; //+1 to store the category associated with each element
        //    int maxBytesByFile = 100 * 1024 * 1024; //100 MBytes
        //    int maxElementByFile = maxBytesByFile / bytesByElement;
        //    var bytes = new byte[Math.Min(totalCount, maxElementByFile) * bytesByElement];
        //    int idx = 0;
        //    int nextFileIndex = 1;
        //    for (int count = 0; count < totalCount; ++count)
        //    {
        //        bytes[idx++] = y[count, 0];
        //        for (int channel = 0; channel < Channels; ++channel)
        //        {
        //            for (int row = 0; row < Height; ++row)
        //            {
        //                for (int col = 0; col < Width; ++col)
        //                {
        //                    bytes[idx++] = X[row, col, channel, count];
        //                }
        //            }
        //        }
        //        if (idx >= bytes.Length)
        //        {
        //            File.WriteAllBytes(Path.Combine(NetworkConfig.DefaultDataDirectory, "SVHN", "extra_batch_" + (nextFileIndex++) + ".bin"), bytes);
        //            int remainingElement = totalCount - count - 1;
        //            bytes = new byte[Math.Min(remainingElement, maxElementByFile) * bytesByElement];
        //            idx = 0;
        //        }
        //    }
        //}

    }
}
