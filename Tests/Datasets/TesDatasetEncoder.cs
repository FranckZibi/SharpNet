using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

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
        var testDatasetSample = new TestDatasetSample(new [] { "cat2" }, new[] { "id" }, new [] { "y" });
        var encoder = new DatasetEncoder(testDatasetSample);

        var rows = SimpleTestDataset();
        var df = encoder.NumericalEncoding(rows);
        var observedResult = encoder.NumericalDecoding(df, ',', "NA" );
        var expectedResult = string.Join(',', rows[0]) + Environment.NewLine + "12,NA,25,cat,1" + Environment.NewLine + "13,1.5,1025,dog,0" + Environment.NewLine + "14,NA,5555,,5";
        Assert.AreEqual(expectedResult, observedResult);
    }


    public class TestDatasetSample : AbstractDatasetSample
    {
        public TestDatasetSample(string[] categoricalFeatures, string[] idColumns, string[] targetLabels) : base(new HashSet<string>())
        {
            CategoricalFeatures = categoricalFeatures;
            IdColumns = idColumns;
            TargetLabels = targetLabels;
        }

        public override Objective_enum GetObjective() => Objective_enum.Regression;
        public override string[] CategoricalFeatures { get; }
        public override string[] IdColumns { get; }
        public override string[] TargetLabels { get; }
        public override IDataSet TestDataset() { throw new NotImplementedException(); }
        public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation() { throw new NotImplementedException(); }
        public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat_with_IdColumns) { throw new NotImplementedException(); }

        protected override EvaluationMetricEnum GetRankingEvaluationMetric() => throw new NotImplementedException();
    }
}
