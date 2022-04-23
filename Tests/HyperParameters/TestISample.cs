using System;
using System.Collections.Generic;
using NUnit.Framework;
using SharpNet.CPU;
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
        public override void SavePredictionsInTargetFormat(CpuTensor<float> predictionsInTargetFormat, string path) { throw new NotImplementedException(); }
        public override IDataSet TestDataset() { throw new NotImplementedException(); }
        public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()  { throw new NotImplementedException(); }
        public override CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(CpuTensor<float> predictionsInModelFormat) { throw new NotImplementedException(); }
        public override CpuTensor<float> PredictionsInTargetFormat_2_PredictionsInModelFormat(CpuTensor<float> predictionsInTargetFormat) { throw new NotImplementedException(); }
    }

    [Test]
    public void TestCopyWithNewPercentageInTraining()
    {
        var sample =  new TestClass();
        sample.PercentageInTraining = 0.5;
        sample.Train_XDatasetPath = "Train_XDatasetPath";
        sample.Train_YDatasetPath = "Train_YDatasetPath";
        sample.Train_PredictionsPath = "Train_PredictionsPath";
        sample.Validation_XDatasetPath = "Validation_XDatasetPath";
        sample.Validation_YDatasetPath = "Validation_YDatasetPath";
        sample.Validation_PredictionsPath = "Validation_PredictionsPath";
        sample.Test_DatasetPath = "Test_DatasetPath";
        sample.Test_PredictionsPath = "Test_PredictionsPath";

        var sample_0_75 = sample.CopyWithNewPercentageInTraining(0.75);
        Assert.AreEqual(0.5, sample.PercentageInTraining, 1e-6);
        Assert.AreEqual(0.75, sample_0_75.PercentageInTraining, 1e-6);
        Assert.AreEqual("", sample_0_75.Train_XDatasetPath);
        Assert.AreEqual("", sample_0_75.Train_YDatasetPath);
        Assert.AreEqual("", sample_0_75.Train_PredictionsPath);
        Assert.AreEqual("", sample_0_75.Validation_XDatasetPath);
        Assert.AreEqual("", sample_0_75.Validation_YDatasetPath);
        Assert.AreEqual("", sample_0_75.Validation_PredictionsPath);
        Assert.AreEqual("Test_DatasetPath", sample_0_75.Test_DatasetPath );
        Assert.AreEqual("Test_PredictionsPath", sample_0_75.Test_PredictionsPath);
    }


    [Test]
    public void TestCloneWithMultiSamples()
    {
        var modelSample = new LightGBMSample();
        modelSample.bagging_seed = 666;
        var datasetSample = new TestClass();
        datasetSample.Test_DatasetPath = "a";
        var original = new ModelAndDatasetSample(modelSample, datasetSample);
        var cloned = (ModelAndDatasetSample) original.Clone();
        ((LightGBMSample)cloned.ModelSample).bagging_seed = 667;
        cloned.DatasetSample.Test_DatasetPath = "b";
        Assert.AreEqual(666, modelSample.bagging_seed);
        Assert.AreEqual(667, ((LightGBMSample)cloned.ModelSample).bagging_seed);
        Assert.AreEqual("a", datasetSample.Test_DatasetPath);
        Assert.AreEqual("b", cloned.DatasetSample.Test_DatasetPath);
    }
}