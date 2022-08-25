using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.Datasets;

namespace SharpNetTests.Datasets;

[TestFixture]
public class TesDatasetEncoder
{

    public static List<string[]> SimpleTestDataset()
    {
        List<string[]> rows = new()
        {
            new[]{ "id", "num1", "num2", "cat2", "y" }, //header
            new[]{ "12", " missing", "$25", "cat", "1" },
            new[]{"13", " 1.5GBP ", "€ 1,025", "dog", "0"},
            new[]{"14", "bad", "€ 5 555 euros", "     ", "5"}
        };
        return rows;

    }

    [Test]
    public void TestEncodingDecoding()
    {
        var encoder = new DatasetEncoder(new List<string> { "cat2" }, new List<string> { "id" }, new List<string> { "y" });

        var rows = SimpleTestDataset();
        var df = encoder.NumericalEncoding(rows, "");
        var observedResult = encoder.NumericalDecoding(df, ',', "NA" );
        var expectedResult = string.Join(',', rows[0]) + Environment.NewLine + "12,NA,25,cat,1" + Environment.NewLine + "13,1.5,1025,dog,0" + Environment.NewLine + "14,NA,5555,,5";
        Assert.AreEqual(expectedResult, observedResult);
    }

    [Test]
    public void TestIsRegressionProblem()
    {
        var encoder = new DatasetEncoder(new List<string> { "cat2" }, new List<string> { "id" }, new List<string> { "y" });
        //the target is not among the categorical features: so the target is a numerical feature and it is a regression problem
        Assert.IsTrue(encoder.IsRegressionProblem);

        encoder = new DatasetEncoder(new List<string> { "cat2", "y" }, new List<string> { "id" }, new List<string> { "y" });
        //the target is among the categorical features: it is a regression problem
        Assert.IsFalse(encoder.IsRegressionProblem);
    }
}
