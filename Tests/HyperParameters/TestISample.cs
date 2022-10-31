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
        public override string[] CategoricalFeatures => Array.Empty<string>();
        public override string[] IdColumns => throw new NotImplementedException();
        public override string[] TargetLabels => throw new NotImplementedException();
        public override DataSet TestDataset() => throw new NotImplementedException();
        public override DataSet FullTrainingAndValidation() => throw new NotImplementedException();
        public override EvaluationMetricEnum GetRankingEvaluationMetric() => throw new NotImplementedException();
    }

    [Test]
    public void TestCopyWithNewPercentageInTrainingAndKFold()
    {
        var sample =  new TestClass();
        sample.PercentageInTraining = 0.5;
        sample.KFold = 3;
        sample.Train_XDatasetPath_InTargetFormat = "Train_XDatasetPath_InTargetFormat";
        sample.Train_YDatasetPath_InTargetFormat = "Train_YDatasetPath_InTargetFormat";
        sample.Validation_XDatasetPath_InTargetFormat = "Validation_XDatasetPath_InTargetFormat";
        sample.Validation_YDatasetPath_InTargetFormat = "Validation_YDatasetPath_InTargetFormat";
        sample.Test_XDatasetPath_InTargetFormat = "Test_XDatasetPath_InTargetFormat";
        sample.Test_YDatasetPath_InTargetFormat = "Test_YDatasetPath_InTargetFormat";

        var sample_0_75_5 = sample.CopyWithNewPercentageInTrainingAndKFold(0.75, 5);
        Assert.AreEqual(0.5, sample.PercentageInTraining, 1e-6);
        Assert.AreEqual(3, sample.KFold, 1e-6);
        Assert.AreEqual(0.75, sample_0_75_5.PercentageInTraining, 1e-6);
        Assert.AreEqual(5, sample_0_75_5.KFold, 1e-6);
        Assert.IsNull(sample_0_75_5.Train_XDatasetPath_InTargetFormat);
        Assert.IsNull(sample_0_75_5.Train_YDatasetPath_InTargetFormat);
        Assert.IsNull(sample_0_75_5.Validation_XDatasetPath_InTargetFormat);
        Assert.IsNull(sample_0_75_5.Validation_YDatasetPath_InTargetFormat);
        Assert.AreEqual("Test_XDatasetPath_InTargetFormat", sample_0_75_5.Test_XDatasetPath_InTargetFormat);
        Assert.AreEqual("Test_YDatasetPath_InTargetFormat", sample_0_75_5.Test_YDatasetPath_InTargetFormat);
    }


    [Test]
    public void TestCloneWithMultiSamples()
    {
        var modelSample = new LightGBMSample();
        modelSample.bagging_seed = 666;
        var datasetSample = new TestClass();
        datasetSample.Test_XDatasetPath_InTargetFormat = "a1";
        datasetSample.Test_YDatasetPath_InTargetFormat = "a2";
        var original = ModelAndDatasetPredictionsSample.New(modelSample, datasetSample);
        var cloned = (ModelAndDatasetPredictionsSample) original.Clone();
        ((LightGBMSample)cloned.ModelSample).bagging_seed = 667;
        cloned.DatasetSample.Test_XDatasetPath_InTargetFormat = "b1";
        cloned.DatasetSample.Test_YDatasetPath_InTargetFormat = "b2";
        Assert.AreEqual(666, modelSample.bagging_seed);
        Assert.AreEqual(667, ((LightGBMSample)cloned.ModelSample).bagging_seed);
        Assert.AreEqual("a1", datasetSample.Test_XDatasetPath_InTargetFormat);
        Assert.AreEqual("b1", cloned.DatasetSample.Test_XDatasetPath_InTargetFormat);
        Assert.AreEqual("a2", datasetSample.Test_YDatasetPath_InTargetFormat);
        Assert.AreEqual("b2", cloned.DatasetSample.Test_YDatasetPath_InTargetFormat);
    }
}
