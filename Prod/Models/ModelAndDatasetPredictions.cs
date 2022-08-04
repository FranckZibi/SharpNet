using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

public class ModelAndDatasetPredictions
{
    #region public fields
    public IModel Model { get; }
    public ModelAndDatasetPredictionsSample ModelAndDatasetPredictionsSample { get; }
    #endregion

    #region constructor
    public ModelAndDatasetPredictions(ModelAndDatasetPredictionsSample modelAndDatasetPredictionsSample, string workingDirectory, string modelName)
    {
        ModelAndDatasetPredictionsSample = modelAndDatasetPredictionsSample;
        Model = IModel.NewModel(ModelAndDatasetPredictionsSample.ModelSample, workingDirectory, modelName);
    }
    public static ModelAndDatasetPredictions New(ModelAndDatasetPredictionsSample modelAndDatasetPredictionsSample, string workingDirectory)
    {
        return new ModelAndDatasetPredictions(modelAndDatasetPredictionsSample, workingDirectory, modelAndDatasetPredictionsSample.ComputeHash());
    }
    public static ModelAndDatasetPredictions Load(string workingDirectory, string modelName)
    {
        var start = Stopwatch.StartNew();
        var modelAndDatasetSample = ModelAndDatasetPredictionsSample.Load(workingDirectory, modelName);
        ISample.Log.Debug($"{nameof(ModelAndDatasetPredictionsSample.Load)} took {start.Elapsed.TotalSeconds}s");
        return new ModelAndDatasetPredictions(modelAndDatasetSample, workingDirectory, modelName);
    }
    #endregion


    public (string train_PredictionsPath, float trainScore, string validation_PredictionsPath, float validationScore, string test_PredictionsPath) 
        Fit(bool savePredictionsInTargetFormat, bool computeValidationScore, bool saveTrainedModel)
    {
        using var trainingAndValidation = DatasetSample.SplitIntoTrainingAndValidation();
        (DatasetSample.Train_XDatasetPath, DatasetSample.Train_YDatasetPath, DatasetSample.Train_XYDatasetPath, DatasetSample.Validation_XDatasetPath, DatasetSample.Validation_YDatasetPath, DatasetSample.Validation_XYDatasetPath) = 
            Model.Fit(trainingAndValidation.Training, trainingAndValidation.Test);
        var res = ("", float.NaN, "", float.NaN, "");
        if (savePredictionsInTargetFormat)
        {
            var start = Stopwatch.StartNew();
            res = SavePredictionsInTargetFormat(trainingAndValidation.Training, trainingAndValidation.Test);
            ISample.Log.Debug($"{nameof(SavePredictionsInTargetFormat)} took '{start.Elapsed.TotalSeconds}'s");
        }
        else if (computeValidationScore)
        {
            var validationScore = ComputePredictionsAndScore(trainingAndValidation.Test).score;
            res = ("", float.NaN, "", validationScore, "");
        }
        if (saveTrainedModel)
        {
            Save(Model.WorkingDirectory, Model.ModelName);
        }
        return res;
    }
    public (string train_PredictionsPath, float trainScore, string validation_PredictionsPath, float validationScore, string test_PredictionsPath) 
        SavePredictionsInTargetFormat()
    {
        using var trainAndValidation = DatasetSample.SplitIntoTrainingAndValidation();
        return SavePredictionsInTargetFormat(trainAndValidation.Training, trainAndValidation.Test);
    }
    //public CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(string y_train_dataset)
    //{
    //    return ModelAndDatasetSample.DatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(y_train_dataset);
    //}
    public List<string> AllFiles()
    {
        var res = ModelAndDatasetPredictionsSample.SampleFiles(Model.WorkingDirectory, Model.ModelName);
        res.AddRange(Model.ModelFiles());
        return res;
    }
    public void Save(string workingDirectory, string modelName)
    {
        var start = Stopwatch.StartNew();
        ModelAndDatasetPredictionsSample.Save(workingDirectory, modelName);
        Model.Save(workingDirectory, modelName);
        ISample.Log.Debug($"{nameof(ModelAndDatasetPredictionsSample)}.Save took '{start.Elapsed.TotalSeconds}'s");
    }
    public string Name => Model.ModelName;

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
    private PredictionsSample PredictionsSample => ModelAndDatasetPredictionsSample.PredictionsSample;
    private AbstractDatasetSample DatasetSample => ModelAndDatasetPredictionsSample.DatasetSample;
    public void SaveTrainPredictionsInTargetFormat(CpuTensor<float> trainPredictionsInModelFormat, float trainScore)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Training Dataset (score={trainScore})");
        PredictionsSample.Train_PredictionsPath = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_predict_train_" + ScoreToStringForPredictionFile(trainScore) + ".csv");
        DatasetSample.SavePredictionsInTargetFormat(trainPredictionsInModelFormat, PredictionsSample.Train_PredictionsPath);
    }
    public void SaveValidationPredictionsInTargetFormat(CpuTensor<float> validationPredictionsInTargetFormat, float validationScore)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Validation Dataset (score={validationScore})");
        PredictionsSample.Validation_PredictionsPath = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_predict_valid_" + ScoreToStringForPredictionFile(validationScore) + ".csv");
        DatasetSample.SavePredictionsInTargetFormat(validationPredictionsInTargetFormat, PredictionsSample.Validation_PredictionsPath);
    }
    public void SaveTestPredictionsInTargetFormat(CpuTensor<float> testPredictionsInTargetFormat, float testScore)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Test Dataset");
        PredictionsSample.Test_PredictionsPath = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_predict_test_" + ScoreToStringForPredictionFile(testScore) + ".csv");
        DatasetSample.SavePredictionsInTargetFormat(testPredictionsInTargetFormat, PredictionsSample.Test_PredictionsPath);
    }

    private static string ScoreToStringForPredictionFile(float score)
    {
        return float.IsNaN(score) ? "" : Math.Round(score, 5).ToString(CultureInfo.InvariantCulture);
    }

    private (CpuTensor<float> predictions, float score) ComputePredictionsAndScore(IDataSet dataset)
    {
        var start = Stopwatch.StartNew();

        var y_pred_InTargetFormat = Predict(dataset);
        var y_true_InTargetFormat = DatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(dataset.Y);
        var score = DatasetSample.ComputeScore(y_true_InTargetFormat, y_pred_InTargetFormat);
        ISample.Log.Debug($"{nameof(ComputePredictionsAndScore)} took {start.Elapsed.TotalSeconds}s");
        return (y_pred_InTargetFormat, score);
    }
    private (string train_PredictionsPath, float trainScore, string validation_PredictionsPath, float validationScore, string test_PredictionsPath) 
        SavePredictionsInTargetFormat(IDataSet trainDataset, IDataSet validationDatasetIfAny)
    {
        PredictionsSample.Train_PredictionsPath = null;
        PredictionsSample.Validation_PredictionsPath = null;
        PredictionsSample.Test_PredictionsPath = null;

        ISample.Log.Debug($"Computing Model '{Model.ModelName}' predictions and score for Training Dataset");
        var (trainPredictionsInTargetFormat, trainScore) = ComputePredictionsAndScore(trainDataset);
        ISample.Log.Info($"Model '{Model.ModelName}' score on Training: {trainScore}");
        DatasetSample.Train_XYDatasetPath = trainDataset.to_csv_in_directory(Model.RootDatasetPath, true, false);
        SaveTrainPredictionsInTargetFormat(trainPredictionsInTargetFormat, trainScore);

        float validationScore = float.NaN;
        if (validationDatasetIfAny != null)
        {
            ISample.Log.Debug($"Computing Model '{Model.ModelName}' predictions and score for Validation Dataset");
            (var validationPredictionsInTargetFormat, validationScore) = ComputePredictionsAndScore(validationDatasetIfAny);
            ISample.Log.Info($"Model '{Model.ModelName}' score on Validation: {validationScore}");
            DatasetSample.Validation_XYDatasetPath = validationDatasetIfAny.to_csv_in_directory(Model.RootDatasetPath, true, false);
            SaveValidationPredictionsInTargetFormat(validationPredictionsInTargetFormat, validationScore);
        }

        var testDatasetIfAny = DatasetSample.TestDataset();
        if (testDatasetIfAny != null)
        {
            ISample.Log.Debug($"Computing Model '{Model.ModelName}' predictions for Test Dataset");
            var (testPredictionsInTargetFormat, testScore) = ComputePredictionsAndScore(testDatasetIfAny);
            if (float.IsNaN(testScore))
            {
                DatasetSample.Test_XDatasetPath = testDatasetIfAny.to_csv_in_directory(Model.RootDatasetPath, false, false);
                DatasetSample.Test_YDatasetPath = DatasetSample.Test_XYDatasetPath = null;
            }
            else
            {
                ISample.Log.Info($"Model '{Model.ModelName}' score on Test: {testScore}");
                DatasetSample.Test_XYDatasetPath = testDatasetIfAny.to_csv_in_directory(Model.RootDatasetPath, true, false);
                DatasetSample.Test_YDatasetPath = DatasetSample.Test_XDatasetPath = null;
            }
            SaveTestPredictionsInTargetFormat(testPredictionsInTargetFormat, testScore);
            testDatasetIfAny.Dispose();
        }
        return (PredictionsSample.Train_PredictionsPath, trainScore, PredictionsSample.Validation_PredictionsPath, validationScore, PredictionsSample.Test_PredictionsPath);
    }

    public (CpuTensor<float> trainPredictionsInTargetFormatWithoutIndex, CpuTensor<float> validationPredictionsInTargetFormatWithoutIndex, CpuTensor<float> testPredictionsInTargetFormatWithoutIndex)
        LoadAllPredictionsInTargetFormat()
    {
        return
            (DatasetSample.LoadPredictionsInTargetFormat(PredictionsSample.Train_PredictionsPath),
                DatasetSample.LoadPredictionsInTargetFormat(PredictionsSample.Validation_PredictionsPath),
                DatasetSample.LoadPredictionsInTargetFormat(PredictionsSample.Test_PredictionsPath));
    }
}
