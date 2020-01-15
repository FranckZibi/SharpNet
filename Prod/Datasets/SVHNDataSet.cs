using System;
using System.Collections.Generic;
using System.IO;
using SharpNet.Networks;

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
        //private readonly string[] CategoryIndexToDescription = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

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

        public SVHNDataSet() : base("SVHN", 3,32,32,10)
        {
            var meanAndVolatilityOfEachChannelInTrainingSet = new List<Tuple<float, float>> { Tuple.Create(109.8823f, 50.11187f), Tuple.Create(109.7114f, 50.57312f), Tuple.Create(113.8187f, 50.85124f) };
            
            var directory = Path.Combine(NetworkConfig.DefaultDataDirectory, Name);

            var trainFiles = SplittedFileDataSet.AllBinFilesInDirectory(directory, "data_batch", "extra_batch");
            Training = new SplittedFileDataSet(trainFiles, Name, Categories, InputShape_CHW, meanAndVolatilityOfEachChannelInTrainingSet, CategoryByteToCategoryIndex);
            //to recompute the mean and volatility of each channel, uncomment the following line
            //meanAndVolatilityOfEachChannelInTrainingSet = ((SplittedFileDataSet)Training).ComputeMeanAndVolatilityForEachChannel();

            var testFiles = SplittedFileDataSet.AllBinFilesInDirectory(directory, "test_batch");
            Test = new SplittedFileDataSet(testFiles, Name, Categories, InputShape_CHW, meanAndVolatilityOfEachChannelInTrainingSet, CategoryByteToCategoryIndex);
        }

        //public static void CreateBinFile()
        //{
        //    var matFile = @"C:\Users\Franck\AppData\Local\SharpNet\Data\SVHN\extra_32x32.mat";
        //    //var matFile = @"C:\Users\Franck\AppData\Local\SharpNet\Data\SVHN\train_32x32.mat";
        //    //var matFile = @"C:\Users\Franck\AppData\Local\SharpNet\Data\SVHN\test_32x32.mat";
        //    var reader = new MatReader(matFile);

        //    var X = reader.Read<byte[,,,]>("X");
        //    var y = reader.Read<byte[,]>("y");

        //    int Channels = 3;
        //    int Height = 32;
        //    int Width = 32;
            
        //    int totalCount = y.GetLength(0);
        //    int bytesByElement = Channels * Height * Width  + 1; //+1 to store the category associated with each element
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
        //            File.WriteAllBytes(@"C:\Users\Franck\AppData\Local\SharpNet\Data\SVHN\extra_batch_" + (nextFileIndex++) + ".bin", bytes);
        //            int remainingElement = totalCount - count - 1;
        //            bytes = new byte[Math.Min(remainingElement, maxElementByFile) * bytesByElement];
        //            idx = 0;
        //        }
        //    }
        //}

    }
}
