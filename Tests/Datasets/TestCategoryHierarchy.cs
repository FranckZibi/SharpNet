﻿using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NUnit.Framework;
using SharpNet.Datasets;
using SharpNet.Layers;
using SharpNet.Networks;
using SharpNet.Pictures;

namespace SharpNetTests.Datasets
{
    [TestFixture]
    public class TestCategoryHierarchy
    {

        [Test]
        public void TestCategoryNameToPrediction()
        {
            var stars = StarExample();
            foreach (var expectedCancelName in new[]{ "etoile_pleine", "etoile*", "etoile7", "etoile**", "etoile*7", "etoile3*", "etoile37"})
            {
                var path = ToPathForStars(expectedCancelName);
                var prediction = stars.ExpectedPrediction(path);
                var observedCancelName = stars.ExtractPrediction(prediction);
                Assert.AreEqual(expectedCancelName, observedCancelName);
            }
        }


        [TestCase(new[] { "full" }, "etoile_pleine")]
        [TestCase(new[] { "1digit"}, "etoile*")]
        [TestCase(new[] { "1digit", "7" }, "etoile7")]
        [TestCase(new[] { "2digits", "3", "*" }, "etoile3*")]
        [TestCase(new[] { "2digits", "*", "7" }, "etoile*7")]
        [TestCase(new[] { "2digits", "3", "7" }, "etoile37")]
        [TestCase(new[] { "2digits"}, "etoile**")]
        public void TestToPathForStars(string[] expectedPath, string cancelName)
        {
            Assert.AreEqual(expectedPath, ToPathForStars(cancelName));
        }

        private static string[] ToPathForStars(string cancelName)
        {
            if (cancelName.StartsWith("etoile_pleine")) return new[] {"full" };
            if (cancelName.Equals("etoile*")) return new[] {"1digit" };
            if (cancelName.Equals("etoile**")) return new[] {"2digits" };
            if (cancelName.Length == 7)
            {
                return new[] {"1digit", cancelName[6].ToString()}; 
            }
            if (cancelName.Length == 8)
            {
                return new[] {"2digits", cancelName[6].ToString(), cancelName[7].ToString() };
            }
            return null;
        }

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
            var root = CancelDatabase.Hierarchy;
            //trained on 235x200
            //var network = Network.ValueOf(Path.Combine(NetworkConfig.DefaultLogDirectory, "Cancel","efficientnet-b0_0_05_200_235_20200602_2054_70.txt"), "", new[] { 0 });
            //trained on 235x200
            var network = Network.ValueOf(Path.Combine(NetworkConfig.DefaultLogDirectory, "Cancel", "efficientnet-b0_Imagenet_20200603_1803_150.txt"), "", new[] { 0 });

            ((InputLayer)network.Layers[0]).SetInputHeightAndWidth(235, 200);
            using var dataSet = FromDirectory(@"C:\Download\ToWork", root);
            var p = network.Predict(dataSet);
            for (int row = 0; row < p.Shape[0]; ++row)
            {
                var rowPrediction = p.RowSlice(row, 1);
                var mostProba = root.ExtractPrediction(rowPrediction.AsReadonlyFloatCpuContent);
                Network.Log.Error(mostProba);
            }

        }

        private static DirectoryDataSet FromDirectory(string path, CategoryHierarchy hierarchy)
        {
            Debug.Assert(hierarchy != null);
            var allFiles = Directory.GetFiles(path, "*.*", SearchOption.AllDirectories).Where(PictureTools.IsPicture).ToList();
            var elementIdToDescription = allFiles.ToList();
            var elementIdToPaths = new List<List<string>>();
            var elementIdToCategoryIndex = new List<int>();
            foreach (var f in allFiles)
            {
                elementIdToPaths.Add(new List<string> { f });
                elementIdToCategoryIndex.Add(-1);
            }

            int nbCategories = hierarchy.RootPrediction().Length;
            var categoryDescriptions = Enumerable.Range(0, nbCategories).Select(i => i.ToString()).ToArray();

            return new DirectoryDataSet(
                elementIdToPaths,
                elementIdToDescription,
                elementIdToCategoryIndex,
                null,
                path,
                3,
                categoryDescriptions,
                CancelDatabase.CancelMeanAndVolatilityForEachChannel,
                ResizeStrategyEnum.ResizeToTargetSize,
                hierarchy);

        }

        public static CategoryHierarchy StarExample()
        {
            var root = CategoryHierarchy.NewRoot("star", "etoile");
            root.AddAllNumbersWithSameNumberOfDigits("2digits", "", 39);
            root.Add("full", "_pleine");
            root.AddAllNumbersWithSameNumberOfDigits("1digit", "", 9);
            return root;
        }
    }
}