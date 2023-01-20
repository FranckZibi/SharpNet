using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

/// <summary>
/// Train 'kfold' distinct models on a the training dataset.
/// Use Stacked Ensemble Learning to make predictions for the models 
/// </summary>
public class KFoldModel : Model
{
    #region private fields
    private readonly List<Model> _embeddedModels;
    #endregion
    private const string SuffixKfoldModel = "_KFOLD";

    #region constructors
    public KFoldModel(KFoldSample modelSample, string kfoldWorkingDirectory, string kfoldModelName, AbstractDatasetSample datasetSample) : base(modelSample, kfoldWorkingDirectory, kfoldModelName)
    {
        _embeddedModels = new();
        for (int i = 0; i < modelSample.n_splits; ++i)
        {
            _embeddedModels.Add(LoadEmbeddedModel(i, datasetSample));
        }
    }

    public KFoldModel(KFoldSample modelSample, string kfoldWorkingDirectory, string kfoldModelName, AbstractDatasetSample datasetSample, IModelSample embeddedModelSample) : base(modelSample, kfoldWorkingDirectory, kfoldModelName)
    {
        _embeddedModels = new();
        for (int i = 0; i < modelSample.n_splits; ++i)
        {
            _embeddedModels.Add(embeddedModelSample.NewModel(datasetSample, kfoldWorkingDirectory, KFoldModelNameEmbeddedModelName(ModelName, i)));
        }
    }

    //Model = new KFoldModel(kfoldSample, embeddedModel.WorkingDirectory, embeddedModel.ModelName + KFoldModel.SuffixKfoldModel);


    public IModelSample EmbeddedModelSample(int embeddedModelIndex)
    {
        return EmbeddedModel(embeddedModelIndex).ModelSample;
    }

    public Model EmbeddedModel(int embeddedModelIndex)
    {
        return _embeddedModels[embeddedModelIndex];
    }


    // ReSharper disable once UnusedParameter.Global
    public static string EmbeddedModelNameToModelNameWithKfold(string embeddedModelName, int n_splits)
    {
        return embeddedModelName + SuffixKfoldModel;
    }


    private static string KFoldModelNameEmbeddedModelName(string kfoldModelName, int index)
    {
        if (index < 0)
        {
            if (kfoldModelName.EndsWith(SuffixKfoldModel))
            {
                kfoldModelName = kfoldModelName.Substring(0, kfoldModelName.Length - SuffixKfoldModel.Length);
            }
            return kfoldModelName;
        }
        return kfoldModelName + "_kfold_" + index;
    }

    private Model LoadEmbeddedModel(int embeddedModelIndex, AbstractDatasetSample datasetSample)
    {
        var e = new Exception();
        //The initial directory of the embedded model may have changed, we'll check also in the KFold Model directory
        foreach (var directory in new[] { KFoldSample.EmbeddedModelWorkingDirectory, WorkingDirectory })
        {
            try { return GetEmbeddedModel(directory, embeddedModelIndex, datasetSample); }
            catch (Exception ex) { e = ex; }
        }
        throw e;
    }

    private Model GetEmbeddedModel(string directory, int embeddedModelIndex, AbstractDatasetSample datasetSample)
    {
        var embeddedModelName = KFoldModelNameEmbeddedModelName(ModelName, embeddedModelIndex);
        IModelSample embeddedModelSample;
        try
        {
            embeddedModelSample = IModelSample.LoadModelSample(directory, embeddedModelName, KFoldSample.Should_Use_All_Available_Cores);
        }
        catch
        {
            //we try to load the embedded model from its original name
            embeddedModelSample = IModelSample.LoadModelSample(directory, KFoldModelNameEmbeddedModelName(ModelName, -1), KFoldSample.Should_Use_All_Available_Cores);
        }
        return embeddedModelSample.NewModel(datasetSample, WorkingDirectory, embeddedModelName);
    }

    #endregion

    //public static void TrainEmbeddedModelWithKFold(string kfoldModelWorkingDirectory, string embeddedModelAndDatasetPredictionsWorkingDirectory, string embeddedModelAndDatasetPredictionsName, int n_splits)
    //{
    //    var embeddedModelAndDatasetPredictions = ModelAndDatasetPredictions.Load(embeddedModelAndDatasetPredictionsWorkingDirectory, embeddedModelAndDatasetPredictionsName);
    //    var datasetSample = embeddedModelAndDatasetPredictions.DatasetSample;

    //    //We first train on part of training dataset, then on full training dataset
    //    foreach(var useFullTraining in new[]{false, true})
    //    {
    //        var currentDatasetSample = useFullTraining
    //            ? datasetSample.CopyWithNewPercentageInTrainingAndKFold(1.0, datasetSample.KFold)
    //            : (AbstractDatasetSample)datasetSample.Clone();
    //        var kfoldSample = new KFoldSample(n_splits, embeddedModelAndDatasetPredictionsName, embeddedModelAndDatasetPredictionsWorkingDirectory, currentDatasetSample.DatasetRowsInModelFormatMustBeMultipleOf());
    //        var kfoldModelAndDatasetPredictionsSample = ModelAndDatasetPredictionsSample.New(kfoldSample, currentDatasetSample);
    //        var kfoldModelAndDatasetPredictions = ModelAndDatasetPredictions.New(kfoldModelAndDatasetPredictionsSample, kfoldModelWorkingDirectory);
    //        kfoldModelAndDatasetPredictions.Model.Use_All_Available_Cores();
    //        LogInfo($"Training '{kfoldModelAndDatasetPredictions.Name}' (based on model {embeddedModelAndDatasetPredictionsName}) using {Math.Round(100* currentDatasetSample.PercentageInTraining,0)}% of Training Dataset");
    //        kfoldModelAndDatasetPredictions.Fit(true, true, true);
    //    }
    //}

    public override (string train_XDatasetPath_InModelFormat, string train_YDatasetPath_InModelFormat, string train_XYDatasetPath_InModelFormat, string validation_XDatasetPath_InModelFormat, string validation_YDatasetPath_InModelFormat, string validation_XYDatasetPath_InModelFormat,
    IScore trainScoreIfAvailable, IScore validationScoreIfAvailable) 
        Fit(DataSet trainDataset, DataSet nullValidationDataset)
    {
        if (nullValidationDataset != null)
        {
            throw new ArgumentException($"Validation Dataset must be null for KFold");
        }

        const bool includeIdColumns = true;
        const bool overwriteIfExists = false;
        int n_splits = KFoldSample.n_splits;
        var foldedTrainAndValidationDataset = KFold(trainDataset, n_splits, KFoldSample.CountMustBeMultipleOf);
        var train_XYDatasetPath_InTargetFormat = trainDataset.to_csv_in_directory(RootDatasetPath, true, includeIdColumns, overwriteIfExists);

        List<IScore> allFoldTrainScoreIfAvailable = new();
        List<IScore> allFoldValidationScoreIfAvailable = new();

        for (int fold = 0; fold < n_splits; ++fold)
        {
            var embeddedModel = _embeddedModels[fold];
            LogDebug($"Training embedded model '{embeddedModel.ModelName}' on fold[{fold}/{n_splits}]");
            var trainAndValidation = foldedTrainAndValidationDataset[fold];
            var (_, _, _, _, _, _, foldTrainScoreIfAvailable, foldValidationScoreIfAvailable) = embeddedModel.Fit(trainAndValidation.Training, trainAndValidation.Test);
            if (foldTrainScoreIfAvailable != null)
            {
                allFoldTrainScoreIfAvailable.Add(foldTrainScoreIfAvailable);
            }

            // we retrieve (or recompute if required) the validation score 
            if (foldValidationScoreIfAvailable != null) //the validation score is already available
            {
                //LogDebug($"No need to recompute Validation Loss for fold[{fold}/{n_splits}] : it is already available");
                allFoldValidationScoreIfAvailable.Add(foldValidationScoreIfAvailable);
            }
            else
            {
                LogDebug($"Computing Validation Loss for fold[{fold}/{n_splits}]");
                var validationDataset = trainAndValidation.Test;
                var fold_y_pred = embeddedModel.Predict(validationDataset, false);
                foldValidationScoreIfAvailable = embeddedModel.ComputeLoss(validationDataset.Y_InModelFormat().FloatCpuTensor(), fold_y_pred.FloatCpuTensor());
            }
            LogDebug($"Validation Loss for fold[{fold}/{n_splits}] : {foldValidationScoreIfAvailable}");
        }
        var trainScoreIfAvailable = IScore.Average(allFoldTrainScoreIfAvailable);
        var validationScoreIfAvailable = IScore.Average(allFoldValidationScoreIfAvailable);
        return (null, null, train_XYDatasetPath_InTargetFormat, null, null, train_XYDatasetPath_InTargetFormat, trainScoreIfAvailable, validationScoreIfAvailable);
    }

    public override (DataFrame trainPredictions_InTargetFormat, IScore trainRankingScore_InTargetFormat,
        DataFrame trainPredictions_InModelFormat, IScore trainLoss_InModelFormat,
        DataFrame validationPredictions_InTargetFormat, IScore validationRankingScore_InTargetFormat,
        DataFrame validationPredictions_InModelFormat, IScore validationLoss_InModelFormat)
        ComputePredictionsAndRankingScore(ITrainingAndTestDataSet trainingAndValidation, AbstractDatasetSample datasetSample, bool computeTrainMetrics)
    {
        var trainDataset = trainingAndValidation.Training;
        Debug.Assert(trainingAndValidation.Test == null);
        int n_splits = KFoldSample.n_splits;
        var foldedTrainAndValidationDataset = KFold(trainDataset, n_splits, KFoldSample.CountMustBeMultipleOf);
        var validationIntervalForKfold = KFoldIntervals(n_splits, trainDataset.Count, KFoldSample.CountMustBeMultipleOf);

        DataFrame trainPredictions_InModelFormat = null;
        DataFrame validationPredictions_InModelFormat = null;

        for (int fold = 0; fold < n_splits; ++fold)
        {
            int idxStartValidation = validationIntervalForKfold[fold].Item1;
            int idxEndValidation = validationIntervalForKfold[fold].Item2;
            int countInValidation = idxEndValidation - idxStartValidation + 1;
            var embeddedModel = _embeddedModels[fold];
            LogDebug($"Computing embedded model '{embeddedModel.ModelName}' predictions on fold[{fold}/{n_splits}]");
            var trainAndValidation = foldedTrainAndValidationDataset[fold];
            var fold_trainPredictions_InModelFormat = computeTrainMetrics
                ?embeddedModel.Predict(trainAndValidation.Training, false)
                : null;
            var fold_validationPredictions_InModelFormat = embeddedModel.Predict(trainAndValidation.Test, false);

            if (validationPredictions_InModelFormat == null)
            {
                validationPredictions_InModelFormat = fold_validationPredictions_InModelFormat.ResizeWithNewNumberOfRows(trainDataset.Count);
                trainPredictions_InModelFormat = computeTrainMetrics
                    ?validationPredictions_InModelFormat.Clone()
                    :null;
                validationPredictions_InModelFormat.FloatTensor?.SetValue(0);
                trainPredictions_InModelFormat?.FloatTensor?.SetValue(0);
            }
            validationPredictions_InModelFormat.RowSlice(idxStartValidation, countInValidation, true).Add(fold_validationPredictions_InModelFormat);
            if (computeTrainMetrics)
            {
                if (idxStartValidation != 0)
                {
                    trainPredictions_InModelFormat.RowSlice(0, idxStartValidation, true).Add(fold_trainPredictions_InModelFormat.RowSlice(0, idxStartValidation, true));
                }
                if (idxEndValidation != trainDataset.Count-1)
                {
                    int rowsToCopy = trainDataset.Count - idxEndValidation - 1;
                    trainPredictions_InModelFormat.RowSlice(idxEndValidation+1, rowsToCopy, true).Add(fold_trainPredictions_InModelFormat.RowSlice(idxStartValidation, rowsToCopy, true));
                }
            }
        }
        Debug.Assert(validationPredictions_InModelFormat != null);


        var y_true_InModelFormat = trainDataset.Y_InModelFormat().FloatCpuTensor();
        var y_true_InTargetFormat = datasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame.New(y_true_InModelFormat));

        if (n_splits >= 2)
        {
            trainPredictions_InModelFormat?.Mult(1f / (n_splits-1));
        }

        IScore trainLoss_InModelFormat = null;
        DataFrame trainPredictions_InTargetFormat = null;
        IScore trainRankingScore_InTargetFormat = null;
        if (computeTrainMetrics)
        {
            trainLoss_InModelFormat = ComputeLoss(y_true_InModelFormat, trainPredictions_InModelFormat.FloatCpuTensor());
            trainPredictions_InTargetFormat = datasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(trainPredictions_InModelFormat);
            trainRankingScore_InTargetFormat = datasetSample.ComputeRankingEvaluationMetric(y_true_InTargetFormat, trainPredictions_InTargetFormat);
        }

        //validationPredictions_InModelFormat.Mult(1f / n_splits);
        var validationLoss_InModelFormat = ComputeLoss(y_true_InModelFormat, validationPredictions_InModelFormat.FloatCpuTensor());
        var validationPredictions_InTargetFormat = datasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(validationPredictions_InModelFormat);
        var validationRankingScore_InTargetFormat = datasetSample.ComputeRankingEvaluationMetric(y_true_InTargetFormat, validationPredictions_InTargetFormat);

        return 
            (trainPredictions_InTargetFormat, trainRankingScore_InTargetFormat,
            trainPredictions_InModelFormat, trainLoss_InModelFormat,
            validationPredictions_InTargetFormat, validationRankingScore_InTargetFormat,
            validationPredictions_InModelFormat, validationLoss_InModelFormat);
    }

    public override (DataFrame, string) PredictWithPath(DataSet dataset, bool removeAllTemporaryFilesAtEnd)
    {
        CpuTensor<float> res = null;
        Debug.Assert(KFoldSample.n_splits == _embeddedModels.Count);
        //each model weight
        var weight = 1.0f / KFoldSample.n_splits;
        var columnNames = new List<string>();
        foreach (var m in _embeddedModels)
        {
            var modelPrediction = m.Predict(dataset, removeAllTemporaryFilesAtEnd);
            columnNames = modelPrediction.Columns.ToList();
            if (res == null)
            {
                res = modelPrediction.FloatCpuTensor();
                res.Update_Multiplying_By_Alpha(weight);
            }
            else
            {
                res.AddTensor(weight, modelPrediction.FloatCpuTensor(), 1.0f);
            }
        }
        return (DataFrame.New(res, columnNames), "");
    }
    public override void Save(string workingDirectory, string modelName)
    {
        ModelSample.Save(workingDirectory, modelName);
        foreach (var embeddedModel in _embeddedModels)
        {
            embeddedModel.Save(workingDirectory, embeddedModel.ModelName);
        }
    }
    public override List<string> AllFiles()
    {
        List<string> res = new();
        res.AddRange(ModelSample.SampleFiles(WorkingDirectory, ModelName));
        foreach (var m in _embeddedModels)
        {
            res.AddRange(m.AllFiles());
        }
        return res;
    }

    public override int GetNumEpochs()
    {
        return _embeddedModels[0].GetNumEpochs();
    }
    public override string DeviceName()
    {
        return _embeddedModels[0].DeviceName();
    }
    public override double GetLearningRate()
    {
        return _embeddedModels[0].GetLearningRate();
    }
    public override int TotalParams()
    {
        return -1; //TODO
    }

    private KFoldSample KFoldSample => (KFoldSample)ModelSample;
    //TODO add tests
    private static List<Tuple<int, int>> KFoldIntervals(int n_splits, int count, int countMustBeMultipleOf)
    {
        Debug.Assert(n_splits >= 1);
        List<Tuple<int, int>> validationIntervalForKfold = new();
        int countByFold = count / n_splits;
        Debug.Assert(countByFold >= 1);
        while (validationIntervalForKfold.Count < n_splits)
        {
            var start = validationIntervalForKfold.Count == 0 ? 0 : validationIntervalForKfold.Last().Item2 + 1;
            var end = start + countByFold - 1;
            end -= (end - start + 1) % countMustBeMultipleOf;
            if (validationIntervalForKfold.Count == n_splits - 1)
            {
                end = count - 1;
            }
            validationIntervalForKfold.Add(Tuple.Create(start, end));
        }
        return validationIntervalForKfold;
    }
    private static List<TrainingAndTestDataset> KFold(DataSet dataset, int kfold, int countMustBeMultipleOf)
    {
        var validationIntervalForKfold = KFoldIntervals(kfold, dataset.Count, countMustBeMultipleOf);
        List<TrainingAndTestDataset> res = new();
        for (var index = 0; index < validationIntervalForKfold.Count; index++)
        {
            var intervalForValidation = validationIntervalForKfold[index];
            var training = dataset.SubDataSet(id => id < intervalForValidation.Item1 || id > intervalForValidation.Item2);
            var test = dataset.SubDataSet(id => id >= intervalForValidation.Item1 && id <= intervalForValidation.Item2);
            res.Add(new TrainingAndTestDataset(training, test, KFoldModelNameEmbeddedModelName(dataset.Name, index)));
        }
        return res;
    }


}
