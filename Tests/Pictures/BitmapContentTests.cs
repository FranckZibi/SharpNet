using System.IO;
using NUnit.Framework;
using SharpNet;
using SharpNet.Datasets;

namespace SharpNetTests.Pictures
{
    [TestFixture]
    public class BitmapContentTests
    {
        [Test, Explicit]
        public void Test0()
        {
            var csvFile = @"C:\temp\aptos2019-blindness-detection\train_images0\train.csv";
            var logger = new Logger(csvFile+".log", true);
            var infos = new DirectoryDataSetLoader<float>(csvFile, Path.GetDirectoryName(csvFile), logger, 3,-1,-1,null);
            var resize_256_256 = infos
                .CropBorder()
                .MakeSquarePictures(true)
                .Resize(256, 256);
        }
    }
}
