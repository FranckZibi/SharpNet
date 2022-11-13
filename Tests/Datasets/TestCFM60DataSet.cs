using System.Linq;
using NUnit.Framework;
using SharpNet.Datasets.CFM60;

namespace SharpNetTests.Datasets
{
    [TestFixture]
    public class TestCFM60DataSet
    {
        [Test]
        public void TestDayToFractionOfYear()
        {
            foreach (var endOfYear in CFM60DatasetSample.EndOfYear)
            {
                Assert.AreEqual(1f, CFM60DatasetSample.DayToFractionOfYear(endOfYear), 1e-6);
            }
            Assert.AreEqual(1f, CFM60DatasetSample.DayToFractionOfYear(CFM60DatasetSample.EndOfYear.Max() + 250), 1e-6);
            Assert.AreEqual(232 / 250f, CFM60DatasetSample.DayToFractionOfYear(1), 1e-6);
            Assert.AreEqual(0.5f, CFM60DatasetSample.DayToFractionOfYear(CFM60DatasetSample.EndOfYear.Min() + 125), 1e-6);
        }
    }
}