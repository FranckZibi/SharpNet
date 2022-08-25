using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.Datasets;

namespace SharpNetTests.Datasets;

[TestFixture]
public class TestFeatureStats
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
        var observedExtractedValue = FeatureStats.ExtractDouble(featureStringValue);
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

    [Test]
    public void TestProperties()
    {
        var encoder = new DatasetEncoder(new List<string> { "cat2", "id"}, new List<string> { "id" }, new List<string> { "y" });
        var rows = TesDatasetEncoder.SimpleTestDataset();
        encoder.NumericalEncoding(rows, "");

        var idStats = encoder["id"];
        Assert.AreEqual(3, idStats.Count);
        Assert.AreEqual(0, idStats.CountEmptyFeatures);
        Assert.IsTrue(idStats.IsCategoricalFeature);
        Assert.IsTrue(idStats.IsId);
        Assert.IsFalse(idStats.IsTargetFeature);

        var num1Stats = encoder["num1"];
        Assert.AreEqual(3, num1Stats.Count);
        Assert.AreEqual(2, num1Stats.CountEmptyFeatures);
        Assert.IsFalse(num1Stats.IsCategoricalFeature);
        Assert.IsFalse(num1Stats.IsId);
        Assert.IsFalse(num1Stats.IsTargetFeature);

        var num2Stats = encoder["num2"];
        Assert.AreEqual(3, num2Stats.Count);
        Assert.AreEqual(0, num2Stats.CountEmptyFeatures);
        Assert.IsFalse(num2Stats.IsCategoricalFeature);
        Assert.IsFalse(num2Stats.IsId);
        Assert.IsFalse(num2Stats.IsTargetFeature);

        var cat2Stats = encoder["cat2"];
        Assert.AreEqual(3, cat2Stats.Count);
        Assert.AreEqual(1, cat2Stats.CountEmptyFeatures);
        Assert.IsTrue(cat2Stats.IsCategoricalFeature);
        Assert.IsFalse(cat2Stats.IsId);
        Assert.IsFalse(cat2Stats.IsTargetFeature);

        var yStats = encoder["y"];
        Assert.AreEqual(3, yStats.Count);
        Assert.AreEqual(0, yStats.CountEmptyFeatures);
        Assert.IsFalse(yStats.IsCategoricalFeature);
        Assert.IsFalse(yStats.IsId);
        Assert.IsTrue(yStats.IsTargetFeature);
    }
}
