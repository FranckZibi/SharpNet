using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using DataSet = SharpNet.Datasets.DataSet;

namespace SharpNet.Models;

public sealed class ModelAndDatasetPredictions
{
    #region public fields and properties
    public Model Model { get; }

    public Model EmbeddedModel
    {
        get
        {
            if (Model is KFoldModel kfoldModel)
            {
                return kfoldModel.EmbeddedModel(0);
            }
            return Model;
        }
    }


    public ModelAndDatasetPredictionsSample ModelAndDatasetPredictionsSample { get; }
    #endregion

    #region constructor
    public ModelAndDatasetPredictions(ModelAndDatasetPredictionsSample modelAndDatasetPredictionsSample, string workingDirectory, string modelName)
        : this(modelAndDatasetPredictionsSample, Model.NewModel(modelAndDatasetPredictionsSample.ModelSample, modelAndDatasetPredictionsSample.DatasetSample, workingDirectory, modelName))
    {
    }

    private ModelAndDatasetPredictions(ModelAndDatasetPredictionsSample modelAndDatasetPredictionsSample, Model model)
    {
        ModelAndDatasetPredictionsSample = modelAndDatasetPredictionsSample;
        Model = model;
        if (modelAndDatasetPredictionsSample.DatasetSample.KFold >= 2 && model is not KFoldModel)
        {
            var withKFold = WithKFold(modelAndDatasetPredictionsSample.DatasetSample.KFold);
            ModelAndDatasetPredictionsSample = withKFold.ModelAndDatasetPredictionsSample;
            Model = withKFold.Model;
        }
    }

    public ModelAndDatasetPredictions WithKFold(int n_splits)
    {
        Debug.Assert(n_splits >= 2);
        var embeddedModel = EmbeddedModel;
        var kfoldSample = new KFoldSample(n_splits, embeddedModel.WorkingDirectory, embeddedModel.ModelName, embeddedModel.ModelSample.GetLoss(), DatasetSample.DatasetRowsInModelFormatMustBeMultipleOf());
        var modelAndDatasetPredictionsSample = ModelAndDatasetPredictionsSample
            .CopyWithNewModelSample(kfoldSample)
            .CopyWithNewPercentageInTrainingAndKFold(1.0, n_splits);
        
        var kfoldModelName = KFoldModel.EmbeddedModelNameToModelNameWithKfold(embeddedModel.ModelName, n_splits);
        var kfoldModel = new KFoldModel(kfoldSample, embeddedModel.WorkingDirectory, kfoldModelName, embeddedModel, DatasetSample);
        return new ModelAndDatasetPredictions(modelAndDatasetPredictionsSample, kfoldModel);
    }


    public ModelAndDatasetPredictions WithFullTrainingNoKFold()
    {
        var embeddedModel = EmbeddedModel;
        var modelAndDatasetPredictionsSample = ModelAndDatasetPredictionsSample
            .CopyWithNewModelSample(embeddedModel.ModelSample)
            .CopyWithNewPercentageInTrainingAndKFold(1.0, 1);
        return new ModelAndDatasetPredictions(modelAndDatasetPredictionsSample, embeddedModel.WorkingDirectory, embeddedModel.ModelName+"_FULL");
    }

    public static ModelAndDatasetPredictions New(ModelAndDatasetPredictionsSample modelAndDatasetPredictionsSample, string workingDirectory)
    {
        return new ModelAndDatasetPredictions(modelAndDatasetPredictionsSample, workingDirectory, modelAndDatasetPredictionsSample.ComputeHash());
    }
    public static ModelAndDatasetPredictions Load(string workingDirectory, string modelName)
    {
        var start = Stopwatch.StartNew();
        var modelAndDatasetSample = ModelAndDatasetPredictionsSample.Load(workingDirectory, modelName);
        ISample.Log.Debug($"{nameof(ModelAndDatasetPredictionsSample.Load)} of model '{modelName}' took {start.Elapsed.TotalSeconds}s");
        return new ModelAndDatasetPredictions(modelAndDatasetSample, workingDirectory, modelName);
    }
    #endregion

    public AbstractDatasetSample DatasetSample => ModelAndDatasetPredictionsSample.DatasetSample;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="computeAndSavePredictions"></param>
    /// <param name="computeValidationRankingScore"></param>
    /// <param name="saveTrainedModel"></param>
    /// <returns>validation ranking score</returns>
    public IScore Fit(bool computeAndSavePredictions, bool computeValidationRankingScore, bool saveTrainedModel)
    {
        var trainingAndValidation = DatasetSample.SplitIntoTrainingAndValidation();
        var validationDataSet = trainingAndValidation.Test;
        var trainDataset = trainingAndValidation.Training;
        (DatasetSample.Train_XDatasetPath_InModelFormat, DatasetSample.Train_YDatasetPath_InModelFormat, DatasetSample.Train_XYDatasetPath_InModelFormat, 
         DatasetSample.Validation_XDatasetPath_InModelFormat, DatasetSample.Validation_YDatasetPath_InModelFormat, DatasetSample.Validation_XYDatasetPath_InModelFormat,
         var trainScoreIfAvailable, var validationScoreIfAvailable) = Model.Fit(trainDataset, validationDataSet);
        IScore validationRankingScore = null;
        if (computeAndSavePredictions)
        {
            var start = Stopwatch.StartNew();
            (_,validationRankingScore) = ComputeAndSavePredictions(trainingAndValidation);
            ISample.Log.Debug($"{nameof(ComputeAndSavePredictions)} took '{start.Elapsed.TotalSeconds}'s");
        }
        else if (computeValidationRankingScore)
        {
            var start = Stopwatch.StartNew();
            validationRankingScore = Model.ComputePredictionsAndRankingScore(trainingAndValidation, DatasetSample, false).validationRankingScore_InTargetFormat;
            ISample.Log.Debug($"{nameof(Model.ComputePredictionsAndRankingScore)} took '{start.Elapsed.TotalSeconds}'s");
        }
        if (saveTrainedModel)
        {
            Save(Model.WorkingDirectory);
        }

        ISample.Log.Debug($"Model {Model.ModelName} scores: trainScore = {trainScoreIfAvailable} / validationScore  = {validationScoreIfAvailable} / validationRankingScore = {validationRankingScore}");
        return validationRankingScore;
    }
    public (IScore trainRankingScore, IScore validationRankingScore) 
        ComputeAndSavePredictions(ITrainingAndTestDataSet trainingAndValidation)
    {
        var trainDataset = trainingAndValidation.Training;
        var validationDataset = trainingAndValidation.Test;
        PredictionsSample.Train_PredictionsFileName = null;
        PredictionsSample.Validation_PredictionsFileName = null;
        PredictionsSample.Test_PredictionsFileName = null;
        PredictionsSample.Train_PredictionsFileName_InModelFormat = null;
        PredictionsSample.Validation_PredictionsFileName_InModelFormat = null;
        PredictionsSample.Test_PredictionsFileName_InModelFormat = null;
        const bool includeIdColumns = true;
        const bool overwriteIfExists = false;

        ISample.Log.Debug($"Computing Model '{Model.ModelName}' predictions for Training & Validation Dataset");
        DatasetSample.Train_XYDatasetPath_InTargetFormat = trainDataset.to_csv_in_directory(Model.RootDatasetPath, true, includeIdColumns, overwriteIfExists);
        var (trainPredictions_InTargetFormat, trainRankingScore_InTargetFormat, 
         trainPredictions_InModelFormat, trainLoss_InModelFormat,
         validationPredictions_InTargetFormat, validationRankingScore_InTargetFormat,
         validationPredictions_InModelFormat, validationLoss_InModelFormat) = 
            Model.ComputePredictionsAndRankingScore(trainingAndValidation, DatasetSample, true);
        if (validationPredictions_InModelFormat != null && validationDataset == null)
        {
            validationDataset = trainDataset;
        }

        if (trainRankingScore_InTargetFormat != null)
        {
            ISample.Log.Debug($"Model '{Model.ModelName}' score on Training: {trainRankingScore_InTargetFormat}");
        }
        SaveTrainPredictionsInModelFormat(trainPredictions_InModelFormat, trainLoss_InModelFormat);
        SaveTrainPredictionsInTargetFormat(trainPredictions_InTargetFormat, trainDataset, trainRankingScore_InTargetFormat);
        if (validationRankingScore_InTargetFormat != null)
        {
            ISample.Log.Info($"Model '{Model.ModelName}' score on Validation: {validationRankingScore_InTargetFormat}");
        }
        SaveValidationPredictionsInModelFormat(validationPredictions_InModelFormat, validationLoss_InModelFormat);
        SaveValidationPredictionsInTargetFormat(validationPredictions_InTargetFormat, validationDataset, validationRankingScore_InTargetFormat);

        var testDatasetIfAny = DatasetSample.TestDataset();
        if (testDatasetIfAny != null)
        {
            ISample.Log.Debug($"Computing Model '{Model.ModelName}' predictions for Test Dataset");
            var (testPredictionsInTargetFormat, testPredictionsInModelFormat, testRankingScore, testDatasetPath_InModelFormat) = DatasetSample.ComputePredictionsAndRankingScore(testDatasetIfAny, Model);
            if (testRankingScore == null)
            {
                DatasetSample.Test_XDatasetPath_InTargetFormat = testDatasetIfAny.to_csv_in_directory(Model.RootDatasetPath, false, includeIdColumns, overwriteIfExists);
                DatasetSample.Test_XDatasetPath_InModelFormat = testDatasetPath_InModelFormat;
                DatasetSample.Test_YDatasetPath_InTargetFormat = DatasetSample.Test_XYDatasetPath_InTargetFormat = null;
                DatasetSample.Test_YDatasetPath_InModelFormat = DatasetSample.Test_XYDatasetPath_InModelFormat = null;
            }
            else
            {
                ISample.Log.Info($"Model '{Model.ModelName}' score on Test: {testRankingScore}");
                DatasetSample.Test_XYDatasetPath_InTargetFormat = testDatasetIfAny.to_csv_in_directory(Model.RootDatasetPath, true, includeIdColumns, overwriteIfExists);
                DatasetSample.Test_XYDatasetPath_InModelFormat = testDatasetPath_InModelFormat;
                DatasetSample.Test_YDatasetPath_InTargetFormat = DatasetSample.Test_XDatasetPath_InTargetFormat = null;
                DatasetSample.Test_YDatasetPath_InModelFormat = DatasetSample.Test_XDatasetPath_InModelFormat = null;
            }
            SaveTestPredictionsInTargetFormat(testPredictionsInTargetFormat, testDatasetIfAny, testRankingScore);
            SaveTestPredictionsInModelFormat(testPredictionsInModelFormat, null);
            //testDatasetIfAny.Dispose();
        }
        return (trainRankingScore_InTargetFormat, validationRankingScore_InTargetFormat);
    }

    /// <summary>
    /// Compute and Save the Feature Importance for the current Model & Dataset
    /// </summary>
    /// <param name="computeFeatureImportanceForAllDatasetTypes">
    /// if true, it will try to compute Feature Importance for the Test, Validation & Train Dataset
    /// if false, it will stop asa a Feature Importance has been computed for anu DataSet
    /// </param>
    public void ComputeAndSaveFeatureImportance(bool computeFeatureImportanceForAllDatasetTypes = false)
    {
        foreach (var datasetType in new[] { AbstractDatasetSample.DatasetType.Test, AbstractDatasetSample.DatasetType.Validation, AbstractDatasetSample.DatasetType.Train })
        {
            var featureImportance_df = Model.ComputeFeatureImportance(DatasetSample, datasetType);
            if (featureImportance_df == null)
            {
                Model.Log.Info($"Failed to compute Feature Importance for {datasetType} Dataset");
                continue;
            }
            var featureImportance_path = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_feature_importance_" + datasetType + ".csv");
            featureImportance_df.to_csv(featureImportance_path);
            Model.Log.Info($"Feature Importance for {datasetType} Dataset has been saved to {featureImportance_path}");
            if (!computeFeatureImportanceForAllDatasetTypes)
            {
                //we have successfully computed Feature Importance for a DataSet , no need to compute Feature Importance for the remaining DataSet
                break;
            }
        }
    }
    public List<string> AllFiles()
    {
        var res = ModelAndDatasetPredictionsSample.SampleFiles(Model.WorkingDirectory, Model.ModelName);
        res.AddRange(Model.AllFiles());
        return res;
    }

    public void Save(string workingDirectory)
    {
        Save(workingDirectory, Model.ModelName);
    }

    public void Save(string workingDirectory, string modelName)
    {
        var start = Stopwatch.StartNew();
        ModelAndDatasetPredictionsSample.Save(workingDirectory, modelName);
        Model.Save(workingDirectory, modelName);
        ISample.Log.Debug($"{nameof(ModelAndDatasetPredictionsSample)}.Save took '{start.Elapsed.TotalSeconds}'s");
    }

    //private string Name => Model.ModelName;

    private void SaveTrainPredictionsInTargetFormat(DataFrame trainPredictionsInTargetFormat, DataSet xDataset, IScore trainScore)
    {
        if (trainPredictionsInTargetFormat == null)
        {
            return;
        }
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Training Dataset (score={trainScore})");
        var fileName = Model.ModelName + "_predict_train_" + IScore.ToString(trainScore, 5) + ".csv";
        PredictionsSample.Train_PredictionsFileName = fileName;
        DatasetSample.SavePredictionsInTargetFormat(trainPredictionsInTargetFormat, xDataset, Path.Combine(Model.WorkingDirectory, fileName));
    }

    private void SaveValidationPredictionsInTargetFormat(DataFrame validationPredictionsInTargetFormat, DataSet xDataset, IScore validationScore)
    {
        if (validationPredictionsInTargetFormat == null)
        {
            return;
        }
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Validation Dataset (score={validationScore})");
        var fileName = Model.ModelName + "_predict_valid_" + IScore.ToString(validationScore, 5) + ".csv";
        PredictionsSample.Validation_PredictionsFileName = fileName;
        DatasetSample.SavePredictionsInTargetFormat(validationPredictionsInTargetFormat, xDataset, Path.Combine(Model.WorkingDirectory, fileName));
    }

    /// <summary>
    /// save the predictions in the Challenge target format, adding an Id at left if needed
    /// </summary>
    /// <param name="testPredictionsInTargetFormat"></param>
    /// <param name="xDataset"></param>
    /// <param name="testScore"></param>
    private void SaveTestPredictionsInTargetFormat(DataFrame testPredictionsInTargetFormat, DataSet xDataset, IScore testScore)
    {
        if (testPredictionsInTargetFormat == null)
        {
            return;
        }
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions for Test Dataset");
        var fileName = Model.ModelName + "_predict_test_" + IScore.ToString(testScore, 5) + ".csv";
        PredictionsSample.Test_PredictionsFileName = fileName;
        DatasetSample.SavePredictionsInTargetFormat(testPredictionsInTargetFormat, xDataset, Path.Combine(Model.WorkingDirectory, fileName));
    }

    private PredictionsSample PredictionsSample => ModelAndDatasetPredictionsSample.PredictionsSample;

    private void SaveTrainPredictionsInModelFormat(DataFrame trainPredictionsInModelFormat, IScore trainLoss)
    {
        if (trainPredictionsInModelFormat == null)
        {
            return;
        }
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions in Model Format for Training Dataset (loss={trainLoss})");
        var fileName = Model.ModelName + "_modelformat_predict_train_" + IScore.ToString(trainLoss, 5) + ".csv";
        PredictionsSample.Train_PredictionsFileName_InModelFormat = fileName;
        DatasetSample.SavePredictionsInModelFormat(trainPredictionsInModelFormat, Path.Combine(Model.WorkingDirectory, fileName));
    }
    private void SaveValidationPredictionsInModelFormat(DataFrame validationPredictionsInModelFormat, IScore validationLoss)
    {
        if (validationPredictionsInModelFormat == null)
        {
            return;
        }
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions in Model Format for Validation Dataset (loss={validationLoss})");
        var fileName = Model.ModelName + "_modelformat_predict_valid_" + IScore.ToString(validationLoss, 5) + ".csv";
        PredictionsSample.Validation_PredictionsFileName_InModelFormat = fileName;
        DatasetSample.SavePredictionsInModelFormat(validationPredictionsInModelFormat, Path.Combine(Model.WorkingDirectory, fileName));
    }
    private void SaveTestPredictionsInModelFormat(DataFrame testPredictionsInModelFormat, IScore testLoss)
    {
        if (testPredictionsInModelFormat == null)
        {
            return;
        }
        ISample.Log.Debug($"Saving Model '{Model.ModelName}' predictions in Model Format for Test Dataset");
        var fileName = Model.ModelName + "_modelformat_predict_test_" + IScore.ToString(testLoss, 5) + ".csv";
        PredictionsSample.Test_PredictionsFileName_InModelFormat = fileName;
        DatasetSample.SavePredictionsInModelFormat(testPredictionsInModelFormat, Path.Combine(Model.WorkingDirectory, fileName));
    }
  
}
