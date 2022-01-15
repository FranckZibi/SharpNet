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
            foreach (var endOfYear in CFM60DataSet.EndOfYear)
            {
                Assert.AreEqual(1f, CFM60DataSet.DayToFractionOfYear(endOfYear), 1e-6);
            }
            Assert.AreEqual(1f, CFM60DataSet.DayToFractionOfYear(CFM60DataSet.EndOfYear.Max()+250), 1e-6);
            Assert.AreEqual(232/250f, CFM60DataSet.DayToFractionOfYear(1), 1e-6);
            Assert.AreEqual(0.5f, CFM60DataSet.DayToFractionOfYear(CFM60DataSet.EndOfYear.Min() + 125), 1e-6);
        }
    }
}