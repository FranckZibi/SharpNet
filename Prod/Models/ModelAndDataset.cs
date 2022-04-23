using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

public class ModelAndDataset
{
    #region public fields
    public IModel Model { get; }
    public ModelAndDatasetSample ModelAndDatasetSample { get; }
    #endregion

    #region constructor
    private ModelAndDataset(ModelAndDatasetSample modelAndDatasetSample, string workingDirectory, string modelName)
    {
        ModelAndDatasetSample = modelAndDatasetSample;
        Model = IModel.NewModel(ModelAndDatasetSample.ModelSample, workingDirectory, modelName);
    }
    public static ModelAndDataset NewUntrainedModelAndDataset(ModelAndDatasetSample modelAndDatasetSample, string workingDirectory)
    {
        return new ModelAndDataset(modelAndDatasetSample, workingDirectory, modelAndDatasetSample.ComputeHash());
    }
    public static ModelAndDataset LoadModelAndDataset(string workingDirectory, string modelName)
    {
        var start = Stopwatch.StartNew();
        var modelAndDatasetSample = ModelAndDatasetSample.LoadModelAndDatasetSample(workingDirectory, modelName);
        ISample.Log.Debug($"{nameof(ModelAndDatasetSample.LoadModelAndDatasetSample)} took {start.Elapsed.TotalSeconds}s");
        return new ModelAndDataset(modelAndDatasetSample, workingDirectory, modelName);
    }
    #endregion

    public (string train_PredictionsPath, float trainScore, string validation_PredictionsPath, float validationScore, string test_PredictionsPath) 
        Fit(bool computeAndSavePredictionsInTargetFormat, bool computeValidationScore, bool saveTrainedModel)
    {
        using var trainingAndValidation = DatasetSample.SplitIntoTrainingAndValidation();
        (DatasetSample.Train_XDatasetPath, DatasetSample.Train_YDatasetPath, DatasetSample.Validation_XDatasetPath, DatasetSample.Validation_YDatasetPath) = 
            Model.Fit(trainingAndValidation.Training, trainingAndValidation.Test);
        var res = ("", float.NaN, "", float.NaN, "");
        if (computeAndSavePredictionsInTargetFormat)
        {
            var start = Stopwatch.StartNew();
            res = ComputeAndSavePredictionsInTargetFormat(trainingAndValidation.Training, trainingAndValidation.Test);
            ISample.Log.Debug($"{nameof(ComputeAndSavePredictionsInTargetFormat)} took '{start.Elapsed.TotalSeconds}'s");
        }
        else if (computeValidationScore)
        {
            var validationScore = ComputeScoreAndPredictionsInTargetFormat(trainingAndValidation.Test).score;
            res = ("", float.NaN, "", validationScore, "");
        }
        if (saveTrainedModel)
        {
            Save(Model.WorkingDirectory, Model.ModelName);
        }
        return res;
    }
    public (string train_PredictionsPath, float trainScore, string validation_PredictionsPath, float validationScore, string test_PredictionsPath) 
        ComputeAndSavePredictionsInTargetFormat()
    {
        using var trainAndValidation = DatasetSample.SplitIntoTrainingAndValidation();
        return ComputeAndSavePredictionsInTargetFormat(trainAndValidation.Training, trainAndValidation.Test);
    }
    //public CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(string y_train_dataset)
    //{
    //    return ModelAndDatasetSample.DatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(y_train_dataset);
    //}
    public List<string> AllFiles()
    {
        var res = ModelAndDatasetSample.SampleFiles(Model.WorkingDirectory, Model.ModelName);
        res.AddRange(Model.ModelFiles());
        return res;
    }
    public void Save(string workingDirectory, string modelName)
    {
        var start = Stopwatch.StartNew();
        ModelAndDatasetSample.Save(workingDirectory, modelName);
        Model.Save(workingDirectory, modelName);
        ISample.Log.Debug($"{nameof(ModelAndDatasetSample)}.Save took '{start.Elapsed.TotalSeconds}'s");
    }
    public string ModelName => Model.ModelName;

    private (CpuTensor<float> predictionsInTargetFormat, string predictionPath) PredictWithPath(IDataSet dataset)
    {
        var (predictionsInModelFormat, predictionPath) = Model.PredictWithPath(dataset);
        var predictionsInTargetFormat = DatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(predictionsInModelFormat);
        return (predictionsInTargetFormat, predictionPath);
    }
    private CpuTensor<float> Predict(IDataSet dataset)
    {
        return PredictWithPath(dataset).predictionsInTargetFormat;
    }
    private AbstractDatasetSample DatasetSample => ModelAndDatasetSample.DatasetSample;
    private void SaveTrainPredictionsInTargetFormat(CpuTensor<float> trainPredictionsInModelFormat, float trainScore)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Training Dataset (score={trainScore})");
        DatasetSample.Train_PredictionsPath = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_predict_train_" + Math.Round(trainScore, 5) + ".csv");
        DatasetSample.SavePredictionsInTargetFormat(trainPredictionsInModelFormat, DatasetSample.Train_PredictionsPath);
    }
    private void SaveValidationPredictionsInTargetFormat(CpuTensor<float> validationPredictionsInTargetFormat, float validationScore)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Validation Dataset (score={validationScore})");
        DatasetSample.Validation_PredictionsPath = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_predict_valid_" + Math.Round(validationScore, 5) + ".csv");
        DatasetSample.SavePredictionsInTargetFormat(validationPredictionsInTargetFormat, DatasetSample.Validation_PredictionsPath);
    }
    private void SaveTestPredictionsInTargetFormat(CpuTensor<float> testPredictionsInTargetFormat)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Test Dataset");
        DatasetSample.Test_PredictionsPath = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_predict_test_.csv");
        DatasetSample.SavePredictionsInTargetFormat(testPredictionsInTargetFormat, DatasetSample.Test_PredictionsPath);
    }
    private (float score, CpuTensor<float> predictionsInTargetFormat) ComputeScoreAndPredictionsInTargetFormat(IDataSet dataset)
    {
        var start = Stopwatch.StartNew();

        var predictionsInTargetFormat = Predict(dataset);
        var predictionsInTargetFormatWithoutIndex = predictionsInTargetFormat.DropColumns(DatasetSample.IndexColumnsInPredictionsInTargetFormat());

        var true_predictionsInTargetFormat = DatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(dataset.Y);
        var true_predictionsInTargetFormatWithoutIndex = true_predictionsInTargetFormat.DropColumns(DatasetSample.IndexColumnsInPredictionsInTargetFormat());

        var predictionsScore = Model.ComputeScore(true_predictionsInTargetFormatWithoutIndex, predictionsInTargetFormatWithoutIndex);
        ISample.Log.Debug($"ComputeScoreAndPredictionsInModelFormat took {start.Elapsed.TotalSeconds}s");
        return (predictionsScore, predictionsInTargetFormat);
    }
    private (string train_PredictionsPath, float trainScore, string validation_PredictionsPath, float validationScore, string test_PredictionsPath) 
        ComputeAndSavePredictionsInTargetFormat(IDataSet trainDataset, IDataSet validationDatasetIfAny)
    {
        DatasetSample.Train_PredictionsPath = "";
        DatasetSample.Validation_PredictionsPath = "";
        DatasetSample.Test_PredictionsPath = "";

        ISample.Log.Debug($"Computing Model '{Model.ModelName}' predictions and score for Training Dataset");
        var (trainScore, trainPredictionsInTargetFormat) = ComputeScoreAndPredictionsInTargetFormat(trainDataset);
        ISample.Log.Info($"Model '{Model.ModelName}' score on training: {trainScore}");
        SaveTrainPredictionsInTargetFormat(trainPredictionsInTargetFormat, trainScore);

        float validationScore = float.NaN;
        if (validationDatasetIfAny != null)
        {
            ISample.Log.Debug($"Computing Model '{Model.ModelName}' predictions and score for Validation Dataset");
            (validationScore, var validationPredictionsInTargetFormat) = ComputeScoreAndPredictionsInTargetFormat(validationDatasetIfAny);
            ISample.Log.Info($"Model '{Model.ModelName}' score on Validation: {validationScore}");
            if (!float.IsNaN(validationScore))
            {
                SaveValidationPredictionsInTargetFormat(validationPredictionsInTargetFormat, validationScore);
            }
        }

        var testDatasetIfAny = DatasetSample.TestDataset();
        if (testDatasetIfAny != null)
        {
            ISample.Log.Debug($"Computing Model '{Model.ModelName}' predictions for Test Dataset");
            (var testPredictionsInTargetFormat, DatasetSample.Test_DatasetPath) = PredictWithPath(testDatasetIfAny);
            SaveTestPredictionsInTargetFormat(testPredictionsInTargetFormat);
            testDatasetIfAny.Dispose();
        }
        return (DatasetSample.Train_PredictionsPath, trainScore, DatasetSample.Validation_PredictionsPath, validationScore, DatasetSample.Test_PredictionsPath);
    }
}
