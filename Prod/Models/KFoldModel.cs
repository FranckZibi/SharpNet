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
        var embeddedModelSample = modelSample.GetEmbeddedModelSample();
        _embeddedModels = new();
        for (int i = 0; i < modelSample.n_splits; ++i)
        {
            _embeddedModels.Add(IModel.NewModel((IModelSample)embeddedModelSample.Clone(), kfoldWorkingDirectory, EmbeddedModelName(kfoldModelName, i)));
        }
        Debug.Assert(KFoldSample.n_splits == _embeddedModels.Count);
    }
    #endregion

    public static void TrainEmbeddedModelWithKFold(string kfoldModelWorkingDirectory, string embeddedModelAndDatasetPredictionsWorkingDirectory, string embeddedModelAndDatasetPredictionsName, int n_splits)
    {
        var embeddedModelAndDatasetPredictions = ModelAndDatasetPredictions.Load(embeddedModelAndDatasetPredictionsWorkingDirectory, embeddedModelAndDatasetPredictionsName);
        var datasetSample = embeddedModelAndDatasetPredictions.ModelAndDatasetPredictionsSample.DatasetSample;

        //We first train on part of training dataset, then on full training dataset
        foreach(var useFullTraining in new[]{false, true})
        {
            var currentDatasetSample = useFullTraining
                ? datasetSample.CopyWithNewPercentageInTraining(1.0)
                : (AbstractDatasetSample)datasetSample.Clone();
            var kfoldSample = new KFoldSample(n_splits, embeddedModelAndDatasetPredictionsName, embeddedModelAndDatasetPredictionsWorkingDirectory, currentDatasetSample.DatasetRowsInModelFormatMustBeMultipleOf());
            var kfoldModelAndDatasetPredictionsSample = ModelAndDatasetPredictionsSample.New(kfoldSample, currentDatasetSample);
            var kfoldModelAndDatasetPredictions = ModelAndDatasetPredictions.New(kfoldModelAndDatasetPredictionsSample, kfoldModelWorkingDirectory);
            kfoldModelAndDatasetPredictions.Model.Use_All_Available_Cores();
            LogInfo($"Training '{kfoldModelAndDatasetPredictions.Name}' (based on model {embeddedModelAndDatasetPredictionsName}) using {Math.Round(100* currentDatasetSample.PercentageInTraining,0)}% of Training Dataset");
            kfoldModelAndDatasetPredictions.Fit(true, true, true);
        }
    }

    public override (string train_XDatasetPath, string train_YDatasetPath, string train_XYDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath, string validation_XYDatasetPath) 
        Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny)
    {
        int n_splits = KFoldSample.n_splits;
        var foldedTrainAndValidationDataset = KFold(trainDataset, n_splits, KFoldSample.CountMustBeMultipleOf);

        var train_XYDatasetPath = trainDataset.to_csv_in_directory(RootDatasetPath, true, false);
        string validation_XYDatasetPath = "";
        if (validationDatasetIfAny != null)
        {
            validation_XYDatasetPath = validationDatasetIfAny.to_csv_in_directory(RootDatasetPath, true, false);
        }

        for (int fold = 0; fold < n_splits; ++fold)
        {
            var embeddedModel = _embeddedModels[fold];
            LogInfo($"Training embedded model '{embeddedModel.ModelName}' on fold[{fold}/{n_splits}]");
            var trainAndValidation = foldedTrainAndValidationDataset[fold];
            embeddedModel.Fit(trainAndValidation.Training, trainAndValidation.Test);
            var validationDataset = validationDatasetIfAny ?? trainAndValidation.Test;
            var foldValidationPrediction = embeddedModel.Predict(validationDataset);
            var foldValidationScore = embeddedModel.ComputeScore(validationDataset.Y, foldValidationPrediction.FloatCpuTensor());
            LogInfo($"Validation score for fold[{fold}/{n_splits}] : {foldValidationScore}");
        }
        return (null, null, train_XYDatasetPath, null, null, validation_XYDatasetPath);
    }
    public override DataFrame Predict(IDataSet dataset)
    {
        CpuTensor<float> res = null;
        Debug.Assert(KFoldSample.n_splits == _embeddedModels.Count);
        //each model weight
        var weight = 1.0f / KFoldSample.n_splits;
        var featuresNames = new List<string>();
        var categoricalFeatures = new List<string>();
        foreach (var m in _embeddedModels)
        {
            var modelPrediction = m.Predict(dataset);
            featuresNames = modelPrediction.FeatureNames.ToList();
            categoricalFeatures = modelPrediction.CategoricalFeatures.ToList();
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
        return DataFrame.New(res, featuresNames, categoricalFeatures);
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
    private static string EmbeddedModelName(string modelName, int index)
    {
        return modelName + "_kfold_" + index;
    }
}
