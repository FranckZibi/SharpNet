using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;

// ReSharper disable FieldCanBeMadeReadOnly.Local
// ReSharper disable MemberCanBePrivate.Local
// ReSharper disable ConvertToConstant.Local

namespace SharpNetTests.HyperParameters;

[TestFixture]

public class TestISample
{
    private class TestClass : AbstractDatasetSample
    {
        public TestClass() : base(new HashSet<string>()) { }
        public override Objective_enum GetObjective() => Objective_enum.Regression;

        public override string[] CategoricalFeatures => new string[0];
        public override string[] IdColumns => throw new NotImplementedException();
        public override string[] TargetLabels => throw new NotImplementedException();
        public override IDataSet TestDataset() { throw new NotImplementedException(); }
        public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()  { throw new NotImplementedException(); }
        public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat_with_IdColumns) { throw new NotImplementedException(); }
        protected override EvaluationMetricEnum GetRankingEvaluationMetric() => throw new NotImplementedException();
    }

    [Test]
    public void TestCopyWithNewPercentageInTrainingAndKFold()
    {
        var sample =  new TestClass();
        sample.PercentageInTraining = 0.5;
        sample.KFold = 3;
        sample.Train_XDatasetPath = "Train_XDatasetPath";
        sample.Train_YDatasetPath = "Train_YDatasetPath";
        sample.Validation_XDatasetPath = "Validation_XDatasetPath";
        sample.Validation_YDatasetPath = "Validation_YDatasetPath";
        sample.Test_XDatasetPath = "Test_XDatasetPath";
        sample.Test_YDatasetPath = "Test_YDatasetPath";

        var sample_0_75_5 = sample.CopyWithNewPercentageInTrainingAndKFold(0.75, 5);
        Assert.AreEqual(0.5, sample.PercentageInTraining, 1e-6);
        Assert.AreEqual(3, sample.KFold, 1e-6);
        Assert.AreEqual(0.75, sample_0_75_5.PercentageInTraining, 1e-6);
        Assert.AreEqual(5, sample_0_75_5.KFold, 1e-6);
        Assert.IsNull(sample_0_75_5.Train_XDatasetPath);
        Assert.IsNull(sample_0_75_5.Train_YDatasetPath);
        Assert.IsNull(sample_0_75_5.Validation_XDatasetPath);
        Assert.IsNull(sample_0_75_5.Validation_YDatasetPath);
        Assert.AreEqual("Test_XDatasetPath", sample_0_75_5.Test_XDatasetPath );
        Assert.AreEqual("Test_YDatasetPath", sample_0_75_5.Test_YDatasetPath );
    }


    [Test]
    public void TestCloneWithMultiSamples()
    {
        var modelSample = new LightGBMSample();
        modelSample.bagging_seed = 666;
        var datasetSample = new TestClass();
        datasetSample.Test_XDatasetPath = "a1";
        datasetSample.Test_YDatasetPath = "a2";
        var original = ModelAndDatasetPredictionsSample.New(modelSample, datasetSample);
        var cloned = (ModelAndDatasetPredictionsSample) original.Clone();
        ((LightGBMSample)cloned.ModelSample).bagging_seed = 667;
        cloned.DatasetSample.Test_XDatasetPath = "b1";
        cloned.DatasetSample.Test_YDatasetPath = "b2";
        Assert.AreEqual(666, modelSample.bagging_seed);
        Assert.AreEqual(667, ((LightGBMSample)cloned.ModelSample).bagging_seed);
        Assert.AreEqual("a1", datasetSample.Test_XDatasetPath);
        Assert.AreEqual("b1", cloned.DatasetSample.Test_XDatasetPath);
        Assert.AreEqual("a2", datasetSample.Test_YDatasetPath);
        Assert.AreEqual("b2", cloned.DatasetSample.Test_YDatasetPath);
    }
}
