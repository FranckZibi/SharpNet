using System.Collections.Generic;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using System;
using SharpNet.Models;

namespace SharpNet.HPO;

public class WeightsOptimizerDatasetSample : AbstractDatasetSample
{
    #region private fields
    private readonly AbstractDatasetSample _embeddedDatasetSample;
    #endregion

    public WeightsOptimizerDatasetSample(AbstractDatasetSample embeddedDatasetSample) : base(new HashSet<string>())
    {
        _embeddedDatasetSample = embeddedDatasetSample;
        Test_DatasetPath = embeddedDatasetSample.Test_DatasetPath;
        Train_XDatasetPath = embeddedDatasetSample.Train_XDatasetPath;
        Train_YDatasetPath = embeddedDatasetSample.Train_YDatasetPath;
        Validation_XDatasetPath = embeddedDatasetSample.Validation_XDatasetPath;
        Validation_YDatasetPath = embeddedDatasetSample.Validation_YDatasetPath;
    }

    public override (string train_PredictionsPath, float trainScore, string validation_PredictionsPath, float validationScore, string test_PredictionsPath) Fit(AbstractModel model, bool computeAndSavePredictions, bool computeValidationScore, bool saveTrainedModel)
    {
        var weightsOptimizerModel = (WeightsOptimizerModel)model;
        var res = ("", float.NaN, "", float.NaN, "");
        if (computeAndSavePredictions)
        {
            res = ComputeAndSavePredictions(model);
        }
        else if (computeValidationScore)
        {
            var validationScore = weightsOptimizerModel.Predictions().validationScore;
            res = ("", float.NaN, "", validationScore, "");
        }
        if (saveTrainedModel)
        {
            model.Save(model.WorkingDirectory, model.ModelName);
        }
        return res;
    }
    public override (string train_PredictionsPath, float trainScore, string validation_PredictionsPath, float validationScore, string test_PredictionsPath) ComputeAndSavePredictions(AbstractModel model)
    {
        var weightsOptimizerModel = (WeightsOptimizerModel)model;
        var (trainPredictions, trainScore, validationPredictions, validationScore, testPredictions) = weightsOptimizerModel.Predictions();

        Train_PredictionsPath = "";
        Validation_PredictionsPath = "";
        Test_PredictionsPath = "";

        if (!float.IsNaN(trainScore))
        {
            SaveTrainPredictions(model, trainPredictions, trainScore);
        }
        if (!float.IsNaN(validationScore))
        {
            SaveValidationPredictions(model, validationPredictions, validationScore);
        }
        if (testPredictions != null)
        {
            SaveTestPredictions(model, testPredictions);
        }

        return (Train_PredictionsPath, trainScore, Validation_PredictionsPath, validationScore, Test_PredictionsPath);
    }

    public override List<string> CategoricalFeatures()
    {
        return _embeddedDatasetSample.CategoricalFeatures();
    }
    public override IDataSet FullTraining()
    {
        return _embeddedDatasetSample.FullTraining();
    }
    public override CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(string dataframe_path)
    {
        return _embeddedDatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(dataframe_path);
    }

    public override (CpuTensor<float> trainPredictions, CpuTensor<float> validationPredictions, CpuTensor<float> testPredictions) LoadAllPredictions()
    {
        return _embeddedDatasetSample.LoadAllPredictions();
    }

    public override void ComputeAndSavePredictions(CpuTensor<float> predictionsInModelFormat, string path)
    {
        _embeddedDatasetSample.ComputeAndSavePredictions(predictionsInModelFormat, path);
    }
    protected override IDataSet TestDataset()
    {
        throw new NotImplementedException();
    }

    protected override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
    {
        throw new NotImplementedException();
    }
}