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
    public static DataFrame SimpleTestDataFrame()
    {
        var content = new []
        {
             "12", " missing", "$25", "cat", "1" ,
            "13", " 1.5GBP ", "€ 1,025", "dog", "0",
            "14", "bad", "€ 5 555 euros", "     ", "5"
        };
        return DataFrame.New(content, new[] { "id", "num1", "num2", "cat2", "y" });
    }

    [Test]
    public void TestEncodingDecoding()
    {
        var testDatasetSample = new TestDatasetSample(new [] { "cat2" }, new[] { "id" }, new [] { "y" });
        var encoder = new DatasetEncoder(testDatasetSample);

        var df_raw = SimpleTestDataFrame();
        var df_encoded = encoder.NumericalEncoding(df_raw);
        var observedResult = encoder.NumericalDecoding(df_encoded, "NA" );
        Assert.AreEqual(df_raw.Columns, observedResult.Columns);
        Assert.AreEqual(observedResult.StringCpuTensor().Content.ToArray(), "12,NA,25,cat,1,13,1.5,1025,dog,0,14,NA,5555,,5".Split(","));
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
        public override DataSet TestDataset() { throw new NotImplementedException(); }
        public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation() { throw new NotImplementedException(); }
        public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat_with_IdColumns) { throw new NotImplementedException(); }

        protected override EvaluationMetricEnum GetRankingEvaluationMetric() => throw new NotImplementedException();
    }
}
