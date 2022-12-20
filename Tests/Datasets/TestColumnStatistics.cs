using System;
using NUnit.Framework;
using SharpNet.Datasets;

namespace SharpNetTests.Datasets;

[TestFixture]
public class TestColumnStatistics
{
    [TestCase("0", 0.0)]
    [TestCase("-12.3456", -12.3456)]
    [TestCase("", Double.NaN)]
    [TestCase(" abd ", Double.NaN)]
    [TestCase(" abd 1", 1.0)]
    [TestCase("[[12", 12.0)]
    [TestCase("$1,410.00", 1410.0)]
    [TestCase("USD -1 410.001", -1410.001)]
    [TestCase("USD -1 410.00 ] 55", -1410.0)]
    public void TestExtractDouble(string featureStringValue, double expectedDoubleValue)
    {
        var observedExtractedValue = ColumnStatistics.ExtractDouble(featureStringValue);
        if (Double.IsNaN(observedExtractedValue))
        {
            Assert.IsTrue(Double.IsNaN(expectedDoubleValue));
        }
        if (Double.IsNaN(expectedDoubleValue))
        {
            Assert.IsTrue(Double.IsNaN(observedExtractedValue));
        }
        Assert.AreEqual(expectedDoubleValue, observedExtractedValue, 1e-6);
    }

    [TestCase(true)]
    [TestCase(false)]
    public void TestProperties(bool standardizeDoubleValues)
    {
        var testDatasetSample = new TesDatasetEncoder.TestDatasetSample(new [] { "cat2", "id" }, new[] { "id" }, new[] { "y" });
        var encoder = new DatasetEncoder(testDatasetSample, standardizeDoubleValues, true);
        var rows = TesDatasetEncoder.SimpleTestDataFrame();
        encoder.Fit(rows);

        var idStats = encoder["id"];
        Assert.AreEqual(3, idStats.Count);
        Assert.AreEqual(0, idStats.CountEmptyElements);
        Assert.IsTrue(idStats.IsCategorical);
        Assert.IsTrue(idStats.IsId);
        Assert.IsFalse(idStats.IsTargetLabel);

        var num1Stats = encoder["num1"];
        Assert.AreEqual(3, num1Stats.Count);
        Assert.AreEqual(2, num1Stats.CountEmptyElements);
        Assert.IsFalse(num1Stats.IsCategorical);
        Assert.IsFalse(num1Stats.IsId);
        Assert.IsFalse(num1Stats.IsTargetLabel);

        var num2Stats = encoder["num2"];
        Assert.AreEqual(3, num2Stats.Count);
        Assert.AreEqual(0, num2Stats.CountEmptyElements);
        Assert.IsFalse(num2Stats.IsCategorical);
        Assert.IsFalse(num2Stats.IsId);
        Assert.IsFalse(num2Stats.IsTargetLabel);

        var cat2Stats = encoder["cat2"];
        Assert.AreEqual(3, cat2Stats.Count);
        Assert.AreEqual(1, cat2Stats.CountEmptyElements);
        Assert.IsTrue(cat2Stats.IsCategorical);
        Assert.IsFalse(cat2Stats.IsId);
        Assert.IsFalse(cat2Stats.IsTargetLabel);

        var yStats = encoder["y"];
        Assert.AreEqual(3, yStats.Count);
        Assert.AreEqual(0, yStats.CountEmptyElements);
        Assert.IsFalse(yStats.IsCategorical);
        Assert.IsFalse(yStats.IsId);
        Assert.IsTrue(yStats.IsTargetLabel);
    }
}
