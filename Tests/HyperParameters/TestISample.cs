using System;
using System.Collections.Generic;
using NUnit.Framework;
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
        public override List<string> CategoricalFeatures() { return new List<string>(); }
        public override List<string> IdFeatures() { throw new NotImplementedException(); }
        public override List<string> TargetFeatures() { throw new NotImplementedException(); }
        public override IDataSet TestDataset() { throw new NotImplementedException(); }
        public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()  { throw new NotImplementedException(); }
        public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat) { throw new NotImplementedException(); }
    }

    [Test]
    public void TestCopyWithNewPercentageInTraining()
    {
        var sample =  new TestClass();
        sample.PercentageInTraining = 0.5;
        sample.Train_XDatasetPath = "Train_XDatasetPath";
        sample.Train_YDatasetPath = "Train_YDatasetPath";
        sample.Validation_XDatasetPath = "Validation_XDatasetPath";
        sample.Validation_YDatasetPath = "Validation_YDatasetPath";
        sample.Test_XDatasetPath = "Test_XDatasetPath";
        sample.Test_YDatasetPath = "Test_YDatasetPath";

        var sample_0_75 = sample.CopyWithNewPercentageInTraining(0.75);
        Assert.AreEqual(0.5, sample.PercentageInTraining, 1e-6);
        Assert.AreEqual(0.75, sample_0_75.PercentageInTraining, 1e-6);
        Assert.AreEqual("", sample_0_75.Train_XDatasetPath);
        Assert.AreEqual("", sample_0_75.Train_YDatasetPath);
        Assert.AreEqual("", sample_0_75.Validation_XDatasetPath);
        Assert.AreEqual("", sample_0_75.Validation_YDatasetPath);
        Assert.AreEqual("Test_XDatasetPath", sample_0_75.Test_XDatasetPath );
        Assert.AreEqual("Test_YDatasetPath", sample_0_75.Test_YDatasetPath );
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