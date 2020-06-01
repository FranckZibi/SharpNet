using NUnit.Framework;
using SharpNet.Datasets;

namespace SharpNetTests.Datasets
{
    [TestFixture]
    public class TestCategoryHierarchy
    {
        [Test]
        public void TestGetExpected()
        {
            var stars = StarExample();
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { -300, 0, 20, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0 },  
                stars.ExpectedPrediction(new string[] { }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"full" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"1digit" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 90, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"1digit", "5" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 0, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new[] { "1digit", "*" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 1, -160, 30, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"2digits" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 1, 20, -40, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"2digits", "*", "6" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 1, 20, 30, 1, 0, 0, 100, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"2digits", "1", "6" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 1, 20, 30, 1, 0, 0, -110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"2digits", "1", "*" }), 1e-6));
            Assert.IsTrue(SharpNet.Utils.SameContent(new float[] { 30, 1, 20, -40, 0, 0, 0, -110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -100, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                stars.ExpectedPrediction(new [] {"2digits", "*", "*" }), 1e-6));
        }

        public static CategoryHierarchy StarExample()
        {
            var root = new CategoryHierarchy("star");
            root.AddAllNumbersWithSameNumberOfDigits("2digits", 39);
            root.Add(new CategoryHierarchy("full"));
            root.AddAllNumbersWithSameNumberOfDigits("1digit", 9);
            return root;
        }
    }
}