using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

public class ModelAndDatasetPredictions
{
    #region private fields and properties
    /// <summary>
    /// the embedded model  used when KFold is enabled,
    /// null if KFold is disabled
    /// </summary>
    private readonly IModel _embeddedModelForKFold;
    #endregion

    #region public fields and properties
    public IModel Model { get; }
    public ModelAndDatasetPredictionsSample ModelAndDatasetPredictionsSample { get; }
    #endregion

    #region constructor
    public ModelAndDatasetPredictions(ModelAndDatasetPredictionsSample modelAndDatasetPredictionsSample, string workingDirectory, string modelName)
    {
        ModelAndDatasetPredictionsSample = modelAndDatasetPredictionsSample;
        Model = IModel.NewModel(ModelAndDatasetPredictionsSample.ModelSample, workingDirectory, modelName);
        if (UseKFold)
        {
            _embeddedModelForKFold = Model;
            var datasetSample = ModelAndDatasetPredictionsSample.DatasetSample;
            if (datasetSample.PercentageInTraining < 1.0)
            {
                throw new ArgumentException($"PercentageInTraining must be 100% if KFold is enabled (found {datasetSample.PercentageInTraining}");
            }
            var kfoldSample = new KFoldSample(datasetSample.KFold, _embeddedModelForKFold.ModelName, _embeddedModelForKFold.WorkingDirectory, _embeddedModelForKFold.ModelSample, datasetSample.DatasetRowsInModelFormatMustBeMultipleOf());
            Model = new KFoldModel(kfoldSample, _embeddedModelForKFold.WorkingDirectory, _embeddedModelForKFold.ModelName + "_KFOLD");
        }
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


    /// <summary>
    /// 
    /// </summary>
    /// <param name="saveTrainValidAndTestsPredictions"></param>
    /// <param name="computeValidationRankingScore"></param>
    /// <param name="saveTrainedModel"></param>
    /// <returns>validation ranking score</returns>
    public IScore Fit(bool saveTrainValidAndTestsPredictions, bool computeValidationRankingScore, bool saveTrainedModel)
    {
        using var trainingAndValidation = DatasetSample.SplitIntoTrainingAndValidation();
        var validationDataSet = trainingAndValidation.Test;
        var trainDataset = trainingAndValidation.Training;
        (DatasetSample.Train_XDatasetPath, DatasetSample.Train_YDatasetPath, DatasetSample.Train_XYDatasetPath, DatasetSample.Validation_XDatasetPath, DatasetSample.Validation_YDatasetPath, DatasetSample.Validation_XYDatasetPath) = 
            Model.Fit(trainDataset, validationDataSet);
        IScore validationRankingScore = null;
        if (saveTrainValidAndTestsPredictions)
        {
            var start = Stopwatch.StartNew();
            validationRankingScore = ComputeAndSaveTrainValidAndTestPredictions(trainingAndValidation).validationRankingScore;
            ISample.Log.Debug($"{nameof(ComputeAndSaveTrainValidAndTestPredictions)} took '{start.Elapsed.TotalSeconds}'s");
        }
        else if (computeValidationRankingScore)
        {
            if (UseKFold)
            {
                Debug.Assert(validationDataSet == null);
                validationRankingScore = ((KFoldModel)Model).ComputeEvaluationMetricOnFullDataset(trainDataset, DatasetSample).validationScore;
            }
            else
            {
                validationRankingScore = ComputePredictionsAndRankingScore(validationDataSet).rankingScore;
            }
        }
        if (saveTrainedModel)
        {
            Save(Model.WorkingDirectory, Model.ModelName);
        }
        return validationRankingScore;
    }
    public (IScore trainRankingScore, IScore validationRankingScore)
        ComputeAndSaveTrainValidAndTestPredictions(ITrainingAndTestDataSet trainingAndValidation)
    {
        var trainDataset = trainingAndValidation.Training;
        var validationDataset = trainingAndValidation.Test;
        PredictionsSample.Train_PredictionsPath = null;
        PredictionsSample.Validation_PredictionsPath = null;
        PredictionsSample.Test_PredictionsPath = null;
        PredictionsSample.Train_PredictionsPath_InModelFormat = null;
        PredictionsSample.Validation_PredictionsPath_InModelFormat = null;
        PredictionsSample.Test_PredictionsPath_InModelFormat = null;
        const bool includeIdColumns = true;
        const bool overwriteIfExists = false;

        ISample.Log.Debug($"Computing Model '{Model.ModelName}' predictions and score for Training Dataset");
        var (trainPredictionsInTargetFormat, trainPredictionsInModelFormat_with_IdColumns, trainRankingScore) = ComputePredictionsAndRankingScore(trainDataset);
        DatasetSample.Train_XYDatasetPath = trainDataset.to_csv_in_directory(Model.RootDatasetPath, true, includeIdColumns, overwriteIfExists);
        IScore validationRankingScore = null;
        if (UseKFold)
        {
            Debug.Assert(validationDataset == null);
            var overfitScoreOnTraining = trainRankingScore;
            (trainRankingScore, validationRankingScore) = ((KFoldModel)Model).ComputeEvaluationMetricOnFullDataset(trainDataset, DatasetSample);
            ISample.Log.Debug($"KFold Model '{Model.ModelName}' Avg score on Training: {trainRankingScore} (with over fitting: {overfitScoreOnTraining})");
            ISample.Log.Info($"KFold Model '{Model.ModelName}' Avg score on Validation: {validationRankingScore}");
            DatasetSample.Validation_XYDatasetPath = DatasetSample.Train_XYDatasetPath;
            SaveTrainPredictionsInTargetFormat(trainPredictionsInTargetFormat, trainRankingScore);
            SaveValidationPredictionsInTargetFormat(trainPredictionsInTargetFormat, validationRankingScore);
            SaveTrainPredictionsInModelFormat(trainPredictionsInModelFormat_with_IdColumns, null);
            SaveValidationPredictionsInModelFormat(trainPredictionsInModelFormat_with_IdColumns, null);
        }
        else
        {
            ISample.Log.Debug($"Model '{Model.ModelName}' score on Training: {trainRankingScore}");
            SaveTrainPredictionsInTargetFormat(trainPredictionsInTargetFormat, trainRankingScore);
            SaveTrainPredictionsInModelFormat(trainPredictionsInModelFormat_with_IdColumns, null);
            if (validationDataset != null)
            {
                ISample.Log.Debug($"Computing Model '{Model.ModelName}' predictions and score for Validation Dataset");
                (var validationPredictionsInTargetFormat, var validationPredictionsInModelFormat_with_IdColumns, validationRankingScore) = ComputePredictionsAndRankingScore(validationDataset);
                ISample.Log.Info($"Model '{Model.ModelName}' score on Validation: {validationRankingScore}");
                DatasetSample.Validation_XYDatasetPath = validationDataset.to_csv_in_directory(Model.RootDatasetPath, true, includeIdColumns, overwriteIfExists);
                SaveValidationPredictionsInTargetFormat(validationPredictionsInTargetFormat, validationRankingScore);
                SaveValidationPredictionsInModelFormat(validationPredictionsInModelFormat_with_IdColumns, null);
            }
        }

        var testDatasetIfAny = DatasetSample.TestDataset();
        if (testDatasetIfAny != null)
        {
            ISample.Log.Debug($"Computing Model '{Model.ModelName}' predictions for Test Dataset");
            var (testPredictionsInTargetFormat, testPredictionsInModelFormat_with_IdColumns, testRankingScore) = ComputePredictionsAndRankingScore(testDatasetIfAny);
            if (testRankingScore == null)
            {
                DatasetSample.Test_XDatasetPath = testDatasetIfAny.to_csv_in_directory(Model.RootDatasetPath, false, includeIdColumns, overwriteIfExists);
                DatasetSample.Test_YDatasetPath = DatasetSample.Test_XYDatasetPath = null;
            }
            else
            {
                ISample.Log.Info($"Model '{Model.ModelName}' score on Test: {testRankingScore}");
                DatasetSample.Test_XYDatasetPath = testDatasetIfAny.to_csv_in_directory(Model.RootDatasetPath, true, includeIdColumns, overwriteIfExists);
                DatasetSample.Test_YDatasetPath = DatasetSample.Test_XDatasetPath = null;
            }
            SaveTestPredictionsInTargetFormat(testPredictionsInTargetFormat, testRankingScore);
            SaveTestPredictionsInModelFormat(testPredictionsInModelFormat_with_IdColumns, null);
            testDatasetIfAny.Dispose();
        }
        return (trainRankingScore, validationRankingScore);
    }


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

    public void SaveTrainPredictionsInTargetFormat(DataFrame trainPredictionsInTargetFormat, IScore trainScore)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Training Dataset (score={trainScore})");
        PredictionsSample.Train_PredictionsPath = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_predict_train_" + IScore.ToString(trainScore, 5) + ".csv");
        DatasetSample.SavePredictionsInTargetFormat(trainPredictionsInTargetFormat, PredictionsSample.Train_PredictionsPath);
    }
    public void SaveValidationPredictionsInTargetFormat(DataFrame validationPredictionsInTargetFormat, IScore validationScore)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Validation Dataset (score={validationScore})");
        PredictionsSample.Validation_PredictionsPath = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_predict_valid_" + IScore.ToString(validationScore, 5) + ".csv");
        DatasetSample.SavePredictionsInTargetFormat(validationPredictionsInTargetFormat, PredictionsSample.Validation_PredictionsPath);
    }
    public void SaveTestPredictionsInTargetFormat(DataFrame testPredictionsInTargetFormat, IScore testScore)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Test Dataset");
        PredictionsSample.Test_PredictionsPath = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_predict_test_" + IScore.ToString(testScore, 5) + ".csv");
        DatasetSample.SavePredictionsInTargetFormat(testPredictionsInTargetFormat, PredictionsSample.Test_PredictionsPath);
    }

    private void SaveTrainPredictionsInModelFormat(DataFrame trainPredictionsInModelFormat, IScore trainLoss)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions in Model Format for Training Dataset (loss={trainLoss})");
        PredictionsSample.Train_PredictionsPath_InModelFormat = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_modelformat_predict_train_" + IScore.ToString(trainLoss, 5) + ".csv");
        DatasetSample.SavePredictionsInModelFormat(trainPredictionsInModelFormat, PredictionsSample.Train_PredictionsPath_InModelFormat);
    }

    private void SaveValidationPredictionsInModelFormat(DataFrame validationPredictionsInModelFormat, IScore validationLoss)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions in Model Format for Validation Dataset (loss={validationLoss})");
        PredictionsSample.Validation_PredictionsPath_InModelFormat = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_modelformat_predict_valid_" + IScore.ToString(validationLoss, 5) + ".csv");
        DatasetSample.SavePredictionsInModelFormat(validationPredictionsInModelFormat, PredictionsSample.Validation_PredictionsPath_InModelFormat);
    }

    private void SaveTestPredictionsInModelFormat(DataFrame testPredictionsInModelFormat, IScore testLoss)
    {
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions in Model Format for Test Dataset");
        PredictionsSample.Test_PredictionsPath_InModelFormat = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_modelformat_predict_test_" + IScore.ToString(testLoss, 5) + ".csv");
        DatasetSample.SavePredictionsInModelFormat(testPredictionsInModelFormat, PredictionsSample.Test_PredictionsPath_InModelFormat);
    }


    public (DataFrame trainPredictionsInTargetFormatWithoutIndex, DataFrame validationPredictionsInTargetFormatWithoutIndex, DataFrame testPredictionsInTargetFormatWithoutIndex)
        LoadAllPredictionsInTargetFormat()
    {
        return
            (DatasetSample.LoadPredictionsInTargetFormat(PredictionsSample.Train_PredictionsPath),
                DatasetSample.LoadPredictionsInTargetFormat(PredictionsSample.Validation_PredictionsPath),
                DatasetSample.LoadPredictionsInTargetFormat(PredictionsSample.Test_PredictionsPath));
    }

    private (DataFrame predictionsInTargetFormat, DataFrame predictionsInModelFormat_with_IdColumns, IScore rankingScore) ComputePredictionsAndRankingScore(IDataSet dataset)
    {
        Debug.Assert(dataset != null);
        var start = Stopwatch.StartNew();
        var predictionsInModelFormat_with_IdColumns = Model.Predict(dataset, true, false);
        var y_pred_InTargetFormat = DatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(predictionsInModelFormat_with_IdColumns);
        IScore rankingScore = null;
        var y_true_InTargetFormat = dataset.Y_InTargetFormat();
        if (y_true_InTargetFormat != null)
        {
            rankingScore = DatasetSample.ComputeRankingEvaluationMetric(y_true_InTargetFormat, y_pred_InTargetFormat);
        }
        ISample.Log.Debug($"{nameof(ComputePredictionsAndRankingScore)} took {start.Elapsed.TotalSeconds}s");
        return (y_pred_InTargetFormat, predictionsInModelFormat_with_IdColumns, rankingScore);
    }

    private PredictionsSample PredictionsSample => ModelAndDatasetPredictionsSample.PredictionsSample;
    private AbstractDatasetSample DatasetSample => ModelAndDatasetPredictionsSample.DatasetSample;
    private bool UseKFold => DatasetSample.KFold >= 2;

}
