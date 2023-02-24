using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet;
using SharpNet.Datasets;

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

    //[TestCase(true)]
    [TestCase(false)]
    public void TestEncodingDecoding(bool standardizeDoubleValues)
    {
        var testDatasetSample = new TestDatasetSample(new [] { "cat2" }, "id", new [] { "y" });
        var encoder = new DatasetEncoder(testDatasetSample, standardizeDoubleValues, true);

        var df_raw = SimpleTestDataFrame();
        var df_encoded = encoder.Fit_Transform(df_raw);
        var observedResult = encoder.Inverse_Transform(df_encoded, "NA" );
        Assert.AreEqual(df_raw.Columns, observedResult.Columns);
        Assert.AreEqual(observedResult.StringCpuTensor().Content.ToArray(), "12,NA,25,cat,1,13,1.5,1025,dog,0,14,NA,5555,,5".Split(","));
    }


    public class TestDatasetSample : AbstractDatasetSample
    {
        public TestDatasetSample(string[] categoricalFeatures, string idColumn, string[] targetLabels) : base(new HashSet<string>())
        {
            CategoricalFeatures = categoricalFeatures;
            IdColumn = idColumn;
            TargetLabels = targetLabels;
        }

        public override Objective_enum GetObjective() => Objective_enum.Regression;
        public override string[] CategoricalFeatures { get; }
        public override string IdColumn { get; }
        public override string[] TargetLabels { get; }
        public override DataSet TestDataset() { throw new NotImplementedException(); }
        public override DataSet FullTrainingAndValidation() => throw new NotImplementedException();
        public override EvaluationMetricEnum GetRankingEvaluationMetric() => throw new NotImplementedException();
    }
}
