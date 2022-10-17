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
public class KFoldModel : AbstractModel
{
    #region private fields
    private readonly List<IModel> _embeddedModels;
    #endregion

    #region constructors
    public KFoldModel(KFoldSample modelSample, string kfoldWorkingDirectory, string kfoldModelName) : base(modelSample, kfoldWorkingDirectory, kfoldModelName)
    {
        _embeddedModels = new();
        for (int i = 0; i < modelSample.n_splits; ++i)
        {
            IModel embeddedModel;
            try
            {
                embeddedModel = GetEmbeddedModel(KFoldSample.EmbeddedModelWorkingDirectory, i);
            }
            catch
            {
                //The initial directory of the embedded model may have been changed, we check in tje KFold Model directory
                embeddedModel = GetEmbeddedModel(WorkingDirectory, i);
            }
            _embeddedModels.Add(embeddedModel);
        }
    }


    private IModel GetEmbeddedModel(string embeddedModelWorkingDirectory, int embeddedModelIndex)
    {
        var embeddedModelName = EmbeddedModelName(ModelName, embeddedModelIndex);
        var embeddedModelSample = IModelSample.LoadModelSample(embeddedModelWorkingDirectory, embeddedModelName);
        return IModel.NewModel(embeddedModelSample, WorkingDirectory, embeddedModelName);
    }

    #endregion

    //public static void TrainEmbeddedModelWithKFold(string kfoldModelWorkingDirectory, string embeddedModelAndDatasetPredictionsWorkingDirectory, string embeddedModelAndDatasetPredictionsName, int n_splits)
    //{
    //    var embeddedModelAndDatasetPredictions = ModelAndDatasetPredictions.Load(embeddedModelAndDatasetPredictionsWorkingDirectory, embeddedModelAndDatasetPredictionsName);
    //    var datasetSample = embeddedModelAndDatasetPredictions.ModelAndDatasetPredictionsSample.DatasetSample;

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

    public override (string train_XDatasetPath, string train_YDatasetPath, string train_XYDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath, string validation_XYDatasetPath) 
        Fit(IDataSet trainDataset, IDataSet nullValidationDataset)
    {
        if (nullValidationDataset != null)
        {
            throw new ArgumentException($"Validation Dataset must be null for KFold");
        }

        const bool includeIdColumns = false;
        const bool overwriteIfExists = false;
        int n_splits = KFoldSample.n_splits;
        var foldedTrainAndValidationDataset = KFold(trainDataset, n_splits, KFoldSample.CountMustBeMultipleOf);

        var train_XYDatasetPath = trainDataset.to_csv_in_directory(RootDatasetPath, true, includeIdColumns, overwriteIfExists);

        for (int fold = 0; fold < n_splits; ++fold)
        {
            var embeddedModel = _embeddedModels[fold];
            LogDebug($"Training embedded model '{embeddedModel.ModelName}' on fold[{fold}/{n_splits}]");
            var trainAndValidation = foldedTrainAndValidationDataset[fold];
            embeddedModel.Fit(trainAndValidation.Training, trainAndValidation.Test);
            var validationDataset = trainAndValidation.Test;
            var fold_y_pred = embeddedModel.Predict(validationDataset, false, false);
            var foldValidationLoss = embeddedModel.ComputeLoss(validationDataset.Y_InModelFormat(fold_y_pred.Shape[1], false).FloatCpuTensor(), fold_y_pred.FloatCpuTensor());
            LogDebug($"Validation Loss for fold[{fold}/{n_splits}] : {foldValidationLoss}");
        }
        return (null, null, train_XYDatasetPath, null, null, train_XYDatasetPath);
    }

    public (IScore trainingScore, IScore validationScore) ComputeEvaluationMetricOnFullDataset(IDataSet trainDataset, AbstractDatasetSample datasetSample)
    {

        int n_splits = KFoldSample.n_splits;
        var foldedTrainAndValidationDataset = KFold(trainDataset, n_splits, KFoldSample.CountMustBeMultipleOf);

        List<IScore> trainingScore = new();
        List<IScore> validationScore = new();
        for (int fold = 0; fold < n_splits; ++fold)
        {
            var embeddedModel = _embeddedModels[fold];
            LogDebug($"Computing embedded model '{embeddedModel.ModelName}' predictions on fold[{fold}/{n_splits}]");
            var trainAndValidation = foldedTrainAndValidationDataset[fold];
            var train_predictionsInModelFormat_with_IdColumns = embeddedModel.Predict(trainAndValidation.Training, true, false);
            var y_pred_train_InTargetFormat = datasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(train_predictionsInModelFormat_with_IdColumns);
            var y_true_train_InTargetFormat = trainAndValidation.Training.Y_InTargetFormat(true);
            if (y_true_train_InTargetFormat != null)
            {
                var foldTrainingScore = datasetSample.ComputeRankingEvaluationMetric(y_true_train_InTargetFormat, y_pred_train_InTargetFormat);
                LogDebug($"Model '{embeddedModel.ModelName}' Training ranking score for fold[{fold}/{n_splits}] : {foldTrainingScore}");
                trainingScore.Add(foldTrainingScore);
            }

            var validation_predictionsInModelFormat_with_IdColumns = embeddedModel.Predict(trainAndValidation.Test, true, false);
            var y_pred_valid_InTargetFormat = datasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(validation_predictionsInModelFormat_with_IdColumns);
            var y_true_valid_InTargetFormat = trainAndValidation.Test.Y_InTargetFormat(true);
            if (y_true_valid_InTargetFormat != null)
            {
                var foldValidationScore = datasetSample.ComputeRankingEvaluationMetric(y_true_valid_InTargetFormat, y_pred_valid_InTargetFormat);
                LogDebug($"Model '{embeddedModel.ModelName}' Validation ranking score for fold[{fold}/{n_splits}] : {foldValidationScore}");
                validationScore.Add(foldValidationScore);
            }
        }
        return (IScore.Average(trainingScore), IScore.Average(validationScore));
    }


    public override DataFrame Predict(IDataSet dataset, bool addIdColumnsAtLeft, bool removeAllTemporaryFilesAtEnd)
    {
        CpuTensor<float> res = null;
        Debug.Assert(KFoldSample.n_splits == _embeddedModels.Count);
        //each model weight
        var weight = 1.0f / KFoldSample.n_splits;
        var columnNames = new List<string>();
        foreach (var m in _embeddedModels)
        {
            var modelPrediction = m.Predict(dataset, false, removeAllTemporaryFilesAtEnd);
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
        DataFrame df = DataFrame.New(res, columnNames);
        if (addIdColumnsAtLeft)
        {
            df = dataset.AddIdColumnsAtLeftIfNeeded(df);
        }
        return df;
    }
    public override void Save(string workingDirectory, string modelName)
    {
        ModelSample.Save(workingDirectory, modelName);
        foreach (var embeddedModel in _embeddedModels)
        {
            embeddedModel.Save(workingDirectory, embeddedModel.ModelName);
        }
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
    public override void Use_All_Available_Cores()
    {
        _embeddedModels.ForEach(m=>m.Use_All_Available_Cores());
    }

    public override int TotalParams()
    {
        return -1; //TODO
    }
    public override List<string> ModelFiles()
    {
        List<string> res = new();
        foreach (var m in _embeddedModels)
        {
            res.AddRange(m.ModelFiles());
        }
        return res;
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
    private static List<TrainingAndTestDataset> KFold(IDataSet dataset, int kfold, int countMustBeMultipleOf)
    {
        var validationIntervalForKfold = KFoldIntervals(kfold, dataset.Count, countMustBeMultipleOf);
        List<TrainingAndTestDataset> res = new();
        for (var index = 0; index < validationIntervalForKfold.Count; index++)
        {
            var intervalForValidation = validationIntervalForKfold[index];
            var training = dataset.SubDataSet(id => id < intervalForValidation.Item1 || id > intervalForValidation.Item2);
            var test = dataset.SubDataSet(id => id >= intervalForValidation.Item1 && id <= intervalForValidation.Item2);
            res.Add(new TrainingAndTestDataset(training, test, EmbeddedModelName(dataset.Name, index)));
        }
        return res;
    }

    private static string EmbeddedModelName(string kfoldModelName, int index)
    {
        return kfoldModelName + "_kfold_" + index;
    }
}
