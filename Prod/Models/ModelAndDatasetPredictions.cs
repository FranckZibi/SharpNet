using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using DataSet = SharpNet.Datasets.DataSet;

namespace SharpNet.Models;

public sealed class ModelAndDatasetPredictions : IDisposable    
{
    #region public fields and properties
    /// <summary>
    /// true if the 'this' object has created the 'ModelAndDatasetPredictionsSample' and will need to dispose it
    /// false if the 'this' object has received the 'ModelAndDatasetPredictionsSample' as a parameter (no need to dispose it)
    /// </summary>
    private readonly bool isOwnerOfModelAndDatasetPredictionsSample;
    public Model Model { get; private set; }
    private Model EmbeddedModel
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
    public ModelAndDatasetPredictionsSample ModelAndDatasetPredictionsSample { get; private set; }
    #endregion

    #region constructor
    public ModelAndDatasetPredictions(ModelAndDatasetPredictionsSample modelAndDatasetPredictionsSample, string workingDirectory, string modelName, bool isOwnerOfModelAndDatasetPredictionsSample)
        : this(modelAndDatasetPredictionsSample, modelAndDatasetPredictionsSample.ModelSample.NewModel(modelAndDatasetPredictionsSample.DatasetSample, workingDirectory, modelName))
    {
        this.isOwnerOfModelAndDatasetPredictionsSample = isOwnerOfModelAndDatasetPredictionsSample;
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
    private ModelAndDatasetPredictions WithKFold(int n_splits)
    {
        Debug.Assert(n_splits >= 2);
        var embeddedModel = EmbeddedModel;
        var kfoldSample = new KFoldSample(n_splits, embeddedModel.WorkingDirectory, embeddedModel.ModelName, embeddedModel.ModelSample.GetLoss(), embeddedModel.ModelSample.GetRankingEvaluationMetric(), DatasetSample.DatasetRowsInModelFormatMustBeMultipleOf());
        var modelAndDatasetPredictionsSample = ModelAndDatasetPredictionsSample
            .CopyWithNewModelSample(kfoldSample)
            .CopyWithNewPercentageInTrainingAndKFold(1.0, n_splits);
        
        var kfoldModelName = KFoldModel.EmbeddedModelNameToModelNameWithKfold(embeddedModel.ModelName, n_splits);
        var kfoldModel = new KFoldModel(kfoldSample, embeddedModel.WorkingDirectory, kfoldModelName, DatasetSample, embeddedModel.ModelSample);
        return new ModelAndDatasetPredictions(modelAndDatasetPredictionsSample, kfoldModel);
    }
    public static ModelAndDatasetPredictions Load(string workingDirectory, string modelName, bool useAllAvailableCores)
    {
        var start = Stopwatch.StartNew();
        var modelAndDatasetSample = ModelAndDatasetPredictionsSample.Load(workingDirectory, modelName, useAllAvailableCores);
        ISample.Log.Debug($"{nameof(ModelAndDatasetPredictionsSample.Load)} of model '{modelName}' took {start.Elapsed.TotalSeconds}s");
        return new ModelAndDatasetPredictions(modelAndDatasetSample, workingDirectory, modelName, true);
    }
    public static ModelAndDatasetPredictions LoadWithKFold(string workingDirectory, string modelName, int n_splits, bool useAllAvailableCores)
    {
        using var m = Load(workingDirectory, modelName, useAllAvailableCores);
        return m.WithKFold(n_splits);
    }
    public static ModelAndDatasetPredictions LoadWithNewPercentageInTrainingNoKFold(double newPercentageInTraining, string workingDirectory, string modelName, bool useAllAvailableCores)
    {
        using var m = Load(workingDirectory, modelName, useAllAvailableCores);
        var embeddedModel = m.EmbeddedModel;
        var modelAndDatasetPredictionsSample = m.ModelAndDatasetPredictionsSample
            .CopyWithNewModelSample(embeddedModel.ModelSample)
            .CopyWithNewPercentageInTrainingAndKFold(newPercentageInTraining, 1);
        var newModelName = (newPercentageInTraining >= 0.999)
            ? embeddedModel.ModelName + "_FULL"
            : modelAndDatasetPredictionsSample.ComputeHash();
        m.Dispose();
        return new ModelAndDatasetPredictions(modelAndDatasetPredictionsSample, embeddedModel.WorkingDirectory, newModelName, true);
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
    [SuppressMessage("ReSharper", "UnusedVariable")]
    public IScore Fit(bool computeAndSavePredictions, bool computeValidationRankingScore, bool saveTrainedModel)
    {
        using var trainingAndValidation = DatasetSample.SplitIntoTrainingAndValidation();
        var trainDataset = trainingAndValidation.Training;
        var validationDataset = trainingAndValidation.Test;
        (   DatasetSample.Train_XDatasetPath_InModelFormat,
            DatasetSample.Train_YDatasetPath_InModelFormat, 
            DatasetSample.Train_XYDatasetPath_InModelFormat, 
            DatasetSample.Validation_XDatasetPath_InModelFormat, 
            DatasetSample.Validation_YDatasetPath_InModelFormat, 
            DatasetSample.Validation_XYDatasetPath_InModelFormat,
            var trainLossIfAvailable, 
            var validationLossIfAvailable, 
            var trainRankingMetricIfAvailable, 
            var validationRankingMetricIfAvailable) 
            = Model.Fit(trainDataset, validationDataset, (saveModel, savePredictions, c,d, modelName) => Save(saveModel, savePredictions, trainDataset, validationDataset, modelName));
        var trainRankingScore = ExtractRankingScoreFromModelMetricsIfAvailable(trainLossIfAvailable, trainRankingMetricIfAvailable);
        var validationRankingScore = ExtractRankingScoreFromModelMetricsIfAvailable(validationLossIfAvailable, validationRankingMetricIfAvailable);
        if (computeAndSavePredictions)
        {
            var start = Stopwatch.StartNew();
            (_,validationRankingScore,_) = ComputeAndSavePredictions(trainDataset, validationDataset, Model.ModelName);
            ISample.Log.Debug($"{nameof(ComputeAndSavePredictions)} took '{start.Elapsed.TotalSeconds}'s");
        }
        else if (computeValidationRankingScore)
        {
            if (validationRankingScore != null)
            {
                ISample.Log.Debug($"No need to compute Validation Ranking score because it is already known {validationRankingScore}");
            }
            else
            {
                var start = Stopwatch.StartNew();
                var predictionsAndRankingScore = Model.ComputePredictionsAndRankingScore(trainingAndValidation.Training, trainingAndValidation.Test, DatasetSample, false);
                validationRankingScore = predictionsAndRankingScore.validationRankingScore_InTargetFormat;
                if (validationRankingScore == null && predictionsAndRankingScore.trainRankingScore_InTargetFormat != null)
                {
                    validationRankingScore = predictionsAndRankingScore.trainRankingScore_InTargetFormat;
                }
                ISample.Log.Debug($"{nameof(Model.ComputePredictionsAndRankingScore)} took '{start.Elapsed.TotalSeconds}'s");
            }
        }
        if (saveTrainedModel)
        {
            Save(Model.WorkingDirectory, Model.ModelName);
        }

        ISample.Log.Debug($"Model {Model.ModelName} losses: trainLoss = {trainLossIfAvailable} / validationLoss = {validationLossIfAvailable} / trainRankingScore = {trainRankingScore} / validationRankingScore = {validationRankingScore}");
        return validationRankingScore;
    }

    #region Contribution to Loss
    /// <summary>
    /// 
    /// </summary>
    /// <param name="computeAlsoRankingScore">
    /// if true:
    ///     we compute both the contribution to the model loss and the contribution to the ranking score
    /// else (false):
    ///     we compute only the contribution to the model loss and the contribution to the ranking score
    /// </param>
    /// <param name="maxGroupSize">
    /// max number of columns in a group.
    /// if 1:
    ///     a contribution will be computed for each column
    /// if int.MaxValue:
    ///     a contribution will be computed for each group
    /// </param>
    public void EstimateLossContribution(bool computeAlsoRankingScore = true, int maxGroupSize = int.MaxValue)
    {
        var sb = new StringBuilder();
        sb.Append("ModelLossMetric,TrainModelLoss,TrainLossContribution,ValidationModelLoss,ValidationLossContribution,RankingMetric,TrainRankingScore,TrainRankingScoreContribution,ValidationRankingScore,ValidationRankingScoreContribution,RandomizedColumnsCount,RandomizedColumnNames" + Environment.NewLine);

        using var trainingAndValidation = (AbstractTrainingAndTestDataset)DatasetSample.SplitIntoTrainingAndValidation();
        var (_, trainLoss, _, trainRankingScore, _) = DatasetSample.ComputePredictionsAndRankingScoreV2(trainingAndValidation.Training, Model, removeAllTemporaryFilesAtEnd:false, computeAlsoRankingScore: computeAlsoRankingScore);
        var (_, validationLoss, _, validationRankingScore, _) = DatasetSample.ComputePredictionsAndRankingScoreV2(trainingAndValidation.Test, Model, removeAllTemporaryFilesAtEnd: false, computeAlsoRankingScore: computeAlsoRankingScore);
        sb.Append(LossContributionLine(trainLoss, trainLoss, validationLoss, validationLoss, trainRankingScore, trainRankingScore, validationRankingScore, validationRankingScore, new List<string>(), "All Features") + Environment.NewLine);

        var r = new Random(0);
        
        var listOfGroupOfFeaturesToRandomize = GroupOfColumnsToShuffle(trainingAndValidation.Training, maxGroupSize);
        // in the first group, we'll randomize all features
        var allFeatures = listOfGroupOfFeaturesToRandomize.SelectMany(c => c).ToList();
        listOfGroupOfFeaturesToRandomize.Insert(0, allFeatures);
        for (var index = 0; index < listOfGroupOfFeaturesToRandomize.Count; index++)
        {
            var featuresToRandomize = listOfGroupOfFeaturesToRandomize[index];
            var groupName = index == 0 ? "No Feature" : string.Join("_", featuresToRandomize).Replace(',', ' ');
            ISample.Log.Info($"Computing Loss Contribution of feature(s): {string.Join(",", featuresToRandomize)}");
            var randomized = trainingAndValidation.WithRandomizeColumnDataSet(featuresToRandomize, r);
            const bool removeAllTemporaryFilesAtEnd = true;
            var (_, groupTrainLoss, _, groupTrainRankingScore, _) = DatasetSample.ComputePredictionsAndRankingScoreV2(randomized.Training, Model, removeAllTemporaryFilesAtEnd: removeAllTemporaryFilesAtEnd, computeAlsoRankingScore: computeAlsoRankingScore);
            var (_, groupValidationLoss, _, groupValidationRankingScore, _) = DatasetSample.ComputePredictionsAndRankingScoreV2(randomized.Test, Model, removeAllTemporaryFilesAtEnd: removeAllTemporaryFilesAtEnd, computeAlsoRankingScore: computeAlsoRankingScore);
            sb.Append(LossContributionLine(trainLoss, groupTrainLoss, validationLoss, groupValidationLoss, trainRankingScore, groupTrainRankingScore, validationRankingScore, groupValidationRankingScore, featuresToRandomize, groupName) + Environment.NewLine);
        }

        var outputPath = Path.Combine(Model.WorkingDirectory, Model.ModelName + "_LossContribution_" + DateTime.Now.Ticks + ".csv");
        ISample.Log.Info($"Contribution to loss written to file {outputPath}");
        File.WriteAllText(outputPath, sb.ToString());
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="dataset"></param>
    /// <param name="maxGroupSize">
    /// max number of columns in a group.
    /// if 1:
    ///     a contribution will be computed for each column
    /// if int.MaxValue:
    ///     a contribution will be computed for each group
    /// </param>
    /// <returns></returns>
    private static List<List<string>> GroupOfColumnsToShuffle(DataSet dataset, int maxGroupSize = int.MaxValue)
    {
        var stemmingColumnNameToColumnNames = new Dictionary<string, List<string>>();
        foreach (var c in dataset.ColumnNames)
        {
            if (Equals(dataset.IdColumn, c))
            {
                continue;
            }
            //we remove the digits at start and end of the column name to get a 'stem' column name
            // Ex: "12_r25" => "_r"
            var stem = c.Trim('1', '2', '3', '4', '5', '6', '7', '8', '9', '0');
            if (stem.Length == 0)
            {
                stem = c;
            }
            if (!stemmingColumnNameToColumnNames.TryGetValue(stem, out var list))
            {
                list = new List<string>();
                stemmingColumnNameToColumnNames[stem] = list;
            }
            list.Add(c);
            if (list.Count >= maxGroupSize)
            {
                stemmingColumnNameToColumnNames[stem + "_" + c] = list;
                stemmingColumnNameToColumnNames.Remove(stem);
            }
        }
        return stemmingColumnNameToColumnNames.Values.ToList();
    }
    private static float LossContribution(IScore baseScore, IScore updatedScore)
    {
        if (baseScore == null || updatedScore == null)
        {
            return 0;
        }
        if (baseScore.HigherIsBetter)
        {
            return updatedScore.Value - baseScore.Value;
        }
        return baseScore.Value - updatedScore.Value;
    }
    private static string ToString(IScore a)
    {
        if (a == null)
        {
            return "";
        }
        return a.Value.ToString(CultureInfo.InvariantCulture);
    }
    private string LossContributionLine(
        IScore baseTrainLoss, IScore trainLoss, IScore baseValidationLoss, IScore validationLoss,
        IScore baseTrainRankingScore, IScore trainRankingScore, IScore baseValidationRankingScore,
        IScore validationRankingScore,
        List<string> columnsToRandomize,
        string groupName)
    {
        //sb.Append("ModelLossMetricName,
        //TrainModelLoss,TrainContributionToLoss,ValidationModelLoss,//ValidationContributionToLoss,
        //RankingMetric,TrainRankingScore,TrainContributionToRankingScore,ValidationRankingScore,ValidationContributionToRankingScore,RandomizedColumnsCount,RandomizedColumnNames" + Environment.NewLine);
        return $"{Model.ModelSample.GetLoss()},"
               + $"{ToString(trainLoss)},"
               + $"{LossContribution(trainLoss, baseTrainLoss)},"
               + $"{ToString(validationLoss)},"
               + $"{LossContribution(validationLoss, baseValidationLoss)},"
               + $"{Model.ModelSample.GetRankingEvaluationMetric()},"
               + $"{ToString(trainRankingScore)},"
               + $"{LossContribution(trainRankingScore, baseTrainRankingScore)},"
               + $"{ToString(validationRankingScore)},"
               + $"{LossContribution(validationRankingScore, baseValidationRankingScore)},"
               + $"{columnsToRandomize.Count},"
               + $"{groupName}"
            ;

    }
    #endregion

    private List<string> Save(bool saveModel, bool savePredictions, DataSet trainDataset, DataSet validationDataset, string modelName)
    {
        List<string> savedFiles = new();
        if (saveModel)
        {
            savedFiles.AddRange(Save(Model.WorkingDirectory, modelName));
        }
        if (savePredictions)
        {
            savedFiles.AddRange(ComputeAndSavePredictions(trainDataset, validationDataset, modelName).savedFiles);
        }
        return savedFiles;
    }

    public (IScore trainRankingScore, IScore validationRankingScore, List<string> savedFiles) ComputeAndSavePredictions(DataSet trainDataset, DataSet validationDataset, string modelName)
    {
        PredictionsSample.Train_PredictionsFileName = null;
        PredictionsSample.Validation_PredictionsFileName = null;
        PredictionsSample.Test_PredictionsFileName = null;
        PredictionsSample.Train_PredictionsFileName_InModelFormat = null;
        PredictionsSample.Validation_PredictionsFileName_InModelFormat = null;
        PredictionsSample.Test_PredictionsFileName_InModelFormat = null;
        const bool includeIdColumns = true;
        const bool overwriteIfExists = false;
        List<string> savedFiles = new();

        ISample.Log.Debug($"Computing Model '{modelName}' predictions for Training & Validation Dataset");
        DatasetSample.Train_XYDatasetPath_InTargetFormat = trainDataset.to_csv_in_directory(Model.DatasetPath, true, includeIdColumns, overwriteIfExists);
        var (trainPredictions_InTargetFormat, trainRankingScore_InTargetFormat, 
         trainPredictions_InModelFormat, trainLoss_InModelFormat,
         validationPredictions_InTargetFormat, validationRankingScore_InTargetFormat,
         validationPredictions_InModelFormat, validationLoss_InModelFormat) = 
            Model.ComputePredictionsAndRankingScore(trainDataset, validationDataset, DatasetSample, true);
        if (validationPredictions_InModelFormat != null && validationDataset == null)
        {
            validationDataset = trainDataset;
        }

        if (trainRankingScore_InTargetFormat != null)
        {
            ISample.Log.Debug($"Model '{modelName}' score on Training: {trainRankingScore_InTargetFormat}");
        }
        savedFiles.AddRange(SaveTrainPredictionsInModelFormat(trainPredictions_InModelFormat, trainLoss_InModelFormat, modelName));
        savedFiles.AddRange(SaveTrainPredictionsInTargetFormat(trainPredictions_InTargetFormat, trainDataset, trainRankingScore_InTargetFormat, modelName));
        if (validationRankingScore_InTargetFormat != null)
        {
            ISample.Log.Info($"Model '{modelName}' score on Validation: {validationRankingScore_InTargetFormat}");
        }
        savedFiles.AddRange(SaveValidationPredictionsInModelFormat(validationPredictions_InModelFormat, validationLoss_InModelFormat, modelName));
        savedFiles.AddRange(SaveValidationPredictionsInTargetFormat(validationPredictions_InTargetFormat, validationDataset, validationRankingScore_InTargetFormat, modelName));

        var testDatasetIfAny = DatasetSample.TestDataset();
        if (testDatasetIfAny != null)
        {
            ISample.Log.Debug($"Computing Model '{modelName}' predictions for Test Dataset");
            var (testPredictionsInModelFormat, _,testPredictionsInTargetFormat, testRankingScore, testDatasetPath_InModelFormat) = DatasetSample.ComputePredictionsAndRankingScoreV2(testDatasetIfAny, Model, false);
            if (testRankingScore == null)
            {
                DatasetSample.Test_XDatasetPath_InTargetFormat = testDatasetIfAny.to_csv_in_directory(Model.DatasetPath, false, includeIdColumns, overwriteIfExists);
                DatasetSample.Test_XDatasetPath_InModelFormat = testDatasetPath_InModelFormat;
                DatasetSample.Test_YDatasetPath_InTargetFormat = DatasetSample.Test_XYDatasetPath_InTargetFormat = null;
                DatasetSample.Test_YDatasetPath_InModelFormat = DatasetSample.Test_XYDatasetPath_InModelFormat = null;
            }
            else
            {
                ISample.Log.Info($"Model '{modelName}' score on Test: {testRankingScore}");
                DatasetSample.Test_XYDatasetPath_InTargetFormat = testDatasetIfAny.to_csv_in_directory(Model.DatasetPath, true, includeIdColumns, overwriteIfExists);
                DatasetSample.Test_XYDatasetPath_InModelFormat = testDatasetPath_InModelFormat;
                DatasetSample.Test_YDatasetPath_InTargetFormat = DatasetSample.Test_XDatasetPath_InTargetFormat = null;
                DatasetSample.Test_YDatasetPath_InModelFormat = DatasetSample.Test_XDatasetPath_InModelFormat = null;
            }
            savedFiles.AddRange(SaveTestPredictionsInTargetFormat(testPredictionsInTargetFormat, testDatasetIfAny, testRankingScore, modelName));
            savedFiles.AddRange(SaveTestPredictionsInModelFormat(testPredictionsInModelFormat, null, modelName));
            //testDatasetIfAny.Dispose();
        }
        return (trainRankingScore_InTargetFormat, validationRankingScore_InTargetFormat, savedFiles);
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
    public List<string> Save(string workingDirectory, string modelName)
    {
        var res = new List<string>();
        var start = Stopwatch.StartNew();
        res.AddRange(ModelAndDatasetPredictionsSample.Save(workingDirectory, modelName));
        res.AddRange(Model.Save(workingDirectory, modelName));
        ISample.Log.Debug($"{nameof(ModelAndDatasetPredictionsSample)}.Save took '{start.Elapsed.TotalSeconds}'s");
        return res;
    }

    private List<string> SaveTrainPredictionsInTargetFormat(DataFrame trainPredictionsInTargetFormat, DataSet xDataset, IScore trainScore, string modelName)
    {
        if (trainPredictionsInTargetFormat == null)
        {
            return new List<string>();
        }
        ISample.Log.Debug($"Saving Model '{modelName}' predictions in Target Format for Training Dataset (score={trainScore})");
        var fileName = modelName + "_predict_train_" + IScore.ToString(trainScore, 5) + ".csv";
        PredictionsSample.Train_PredictionsFileName = fileName;
        var path = Path.Combine(Model.WorkingDirectory, fileName);
        DatasetSample.SavePredictionsInTargetFormat(trainPredictionsInTargetFormat, xDataset, path);
        return new List<string> { path };
    }
    private List<string> SaveValidationPredictionsInTargetFormat(DataFrame validationPredictionsInTargetFormat, DataSet xDataset, IScore validationScore, string modelName)
    {
        if (validationPredictionsInTargetFormat == null)
        {
            return new List<string>();
        }
        ISample.Log.Debug($"Saving Model '{modelName}' predictions in Target Format for Validation Dataset (score={validationScore})");
        var fileName = modelName + "_predict_valid_" + IScore.ToString(validationScore, 5) +".csv";
        PredictionsSample.Validation_PredictionsFileName = fileName;
        var path = Path.Combine(Model.WorkingDirectory, fileName);
        DatasetSample.SavePredictionsInTargetFormat(validationPredictionsInTargetFormat, xDataset, path);
        return new List<string> { path };

    }

    /// <summary>
    /// save the predictions in the Challenge target format, adding an Id at left if needed
    /// </summary>
    /// <param name="testPredictionsInTargetFormat"></param>
    /// <param name="xDataset"></param>
    /// <param name="testScore"></param>
    /// <param name="modelName"></param>
    private List<string> SaveTestPredictionsInTargetFormat(DataFrame testPredictionsInTargetFormat, DataSet xDataset, IScore testScore, string modelName)
    {
        if (testPredictionsInTargetFormat == null)
        {
            return new List<string>();
        }
        ISample.Log.Debug($"Saving Model '{modelName}' predictions in Target Format for Test Dataset");
        var fileName = modelName + "_predict_test_" + IScore.ToString(testScore, 5) +".csv";
        PredictionsSample.Test_PredictionsFileName = fileName;
        var path = Path.Combine(Model.WorkingDirectory, fileName);
        DatasetSample.SavePredictionsInTargetFormat(testPredictionsInTargetFormat, xDataset, path);
        return new List<string> { path };

    }
    private PredictionsSample PredictionsSample => ModelAndDatasetPredictionsSample.PredictionsSample;
    private List<string> SaveTrainPredictionsInModelFormat(DataFrame trainPredictionsInModelFormat, IScore trainLoss, string modelName)
    {
        if (trainPredictionsInModelFormat == null)
        {
            return new List<string>();
        }
        ISample.Log.Debug($"Saving Model '{modelName}' predictions in Model Format for Training Dataset (loss={trainLoss})");
        var fileName = modelName + "_modelformat_predict_train_" + IScore.ToString(trainLoss, 5) + ".csv";
        PredictionsSample.Train_PredictionsFileName_InModelFormat = fileName;
        var path = Path.Combine(Model.WorkingDirectory, fileName);
        DatasetSample.SavePredictionsInModelFormat(trainPredictionsInModelFormat, path);
        return new List<string> { path };

    }
    private List<string> SaveValidationPredictionsInModelFormat(DataFrame validationPredictionsInModelFormat, IScore validationLoss, string modelName)
    {
        if (validationPredictionsInModelFormat == null)
        {
            return new List<string>();
        }
        ISample.Log.Debug($"Saving Model '{modelName}' predictions in Model Format for Validation Dataset (loss={validationLoss})");
        var fileName = modelName + "_modelformat_predict_valid_" + IScore.ToString(validationLoss, 5) + ".csv";
        PredictionsSample.Validation_PredictionsFileName_InModelFormat = fileName;
        var path = Path.Combine(Model.WorkingDirectory, fileName);
        DatasetSample.SavePredictionsInModelFormat(validationPredictionsInModelFormat, path);
        return new List<string> { path };

    }
    private List<string> SaveTestPredictionsInModelFormat(DataFrame testPredictionsInModelFormat, IScore testLoss, string modelName)
    {
        if (testPredictionsInModelFormat == null)
        {
            return new List<string>();
        }
        ISample.Log.Debug($"Saving Model '{modelName}' predictions in Model Format for Test Dataset");
        var fileName = modelName + "_modelformat_predict_test_" + IScore.ToString(testLoss, 5) + ".csv";
        PredictionsSample.Test_PredictionsFileName_InModelFormat = fileName;
        var path = Path.Combine(Model.WorkingDirectory, fileName);
        DatasetSample.SavePredictionsInModelFormat(testPredictionsInModelFormat, path);
        return new List<string> { path };
    }
    /// <summary>
    /// try to use one of the metric computed when training the model as a ranking score
    /// </summary>
    /// <param name="modelMetrics">the metrics computed when training the model</param>
    /// <returns></returns>
    private IScore ExtractRankingScoreFromModelMetricsIfAvailable(params IScore[] modelMetrics)
    {
        return modelMetrics.FirstOrDefault(v => v != null && v.Metric == Model.ModelSample.GetRankingEvaluationMetric());
    }

    #region Dispose pattern
    private bool disposed = false;
    private void Dispose(bool disposing)
    {
        if (disposed)
        {
            return;
        }
        disposed = true;
        //Release Unmanaged Resources
        if (disposing)
        {
            //Release Managed Resources
            Model?.Dispose();
            Model = null;
            if (isOwnerOfModelAndDatasetPredictionsSample)
            {
                ModelAndDatasetPredictionsSample?.Dispose();
                ModelAndDatasetPredictionsSample = null;
            }
        }
    }
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    ~ModelAndDatasetPredictions()
    {
        Dispose(false);
    }
    #endregion
}
