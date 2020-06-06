using System.IO;
using NUnit.Framework;
using SharpNet.Datasets;
using SharpNet.Layers;
using SharpNet.Networks;

namespace SharpNetTests.Datasets
{
    [TestFixture]
    public class TestCategoryHierarchy
    {
        [Test]
        public void TestGetExpected()
        {
            var stars = StarExample();
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { -300, 0, 21, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 0, 0, 0, 0, 0, 0, 0, 0, 0 },  
                stars.ExpectedPrediction(new string[] { }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"full" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"1digit" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 91, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"1digit", "5" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new[] { "1digit", "*" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 1, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"2digits" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 1, 21, -40, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"2digits", "*", "6" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 1, 21, 30, 1, 0, 0, 100, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"2digits", "1", "6" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 1, 21, 30, 1, 0, 0, -110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"2digits", "1", "*" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 1, 21, -40, 0, 0, 0, -110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"2digits", "*", "*" }), 1e-6));
        }


        [Test, Explicit]
        public void TestPredictions()
        {
            var root = CategoryHierarchy.ComputeRootNode();
            var rootPrediction = root.RootPrediction();
            //trained on 235x200
            //var network = Network.ValueOf(Path.Combine(NetworkConfig.DefaultLogDirectory, "CancelDataset","efficientnet-b0_0_05_200_235_20200602_2054_70.txt"), "", new[] { 0 });
            //trained on 235x200
            var network = Network.ValueOf(Path.Combine(NetworkConfig.DefaultLogDirectory, "CancelDataset", "efficientnet-b0_Imagenet_20200603_1803_150.txt"), "", new[] { 0 });

            ((InputLayer)network.Layers[0]).SetInputHeightAndWidth(235, 200);
            using var dataSet = DirectoryDataSet.FromDirectory(@"C:\Download\ToWork", rootPrediction.Length, root);
            var p = network.Predict(dataSet, @"C:\Users\fzibi\AppData\Roaming\ImageDatabaseManagement\Prediction.csv");
            for (int row = 0; row < p.Shape[0]; ++row)
            {
                var rowPrediction = p.RowSlice(row, 1);
                var mostProba = root.ExtractPrediction(rowPrediction.AsReadonlyFloatCpuContent);
                Network.Log.Error(mostProba);
            }

        }

        public static CategoryHierarchy StarExample()
        {
            var root = CategoryHierarchy.NewRoot("star");
            root.AddAllNumbersWithSameNumberOfDigits("2digits", 39);
            root.Add("full");
            root.AddAllNumbersWithSameNumberOfDigits("1digit", 9);
            return root;
        }
    }
}