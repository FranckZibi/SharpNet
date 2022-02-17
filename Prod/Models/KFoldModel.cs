using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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
    private readonly List<AbstractModel> _embeddedModels;
    #endregion

    #region constructors
    public KFoldModel(ISample sample, string workingDirectory, string modelName, int n_splits) : base(sample, workingDirectory, modelName+"_kfold")
    {
        Debug.Assert(n_splits>=2);
        _embeddedModels = new();
        for (int i = 0; i < n_splits; ++i)
        {
            _embeddedModels.Add(IModel.NewModel(sample, workingDirectory, ToModelNameForKFold(modelName,i,n_splits )));
        }
    }
    public static KFoldModel ValueOf(string workingDirectory, string modelName)
    {
        var modelNameForKFold = ToModelNameForKFold(modelName, 0);
        string[] files = Directory.GetFiles(workingDirectory, modelNameForKFold+"*.*");
        var errorMsg = $"can't load a {nameof(KFoldModel)} model from {workingDirectory} with name {modelName}";
        if (files.Length == 0)
        {
            throw new ArgumentException(errorMsg);
        }

        var splitted = files[0].Replace(modelNameForKFold, "").Split('_');
        if (splitted.Length < 2 || !int.TryParse(splitted[1], out var n_splits) || n_splits<2)
        {
            throw new ArgumentException(errorMsg);
        }

        List<AbstractModel> embeddedModels = new();
        for (int i = 0; i < n_splits; ++i)
        {
            var m = IModel.ValueOf(workingDirectory, ToModelNameForKFold(modelName, i, n_splits));
            embeddedModels.Add(m);
        }
        return new KFoldModel(embeddedModels, workingDirectory, modelName);
    }
    private KFoldModel(List<AbstractModel> embeddedModels, string workingDirectory, string modelName) : base(embeddedModels[0].Sample, workingDirectory, modelName)
    {
        _embeddedModels = embeddedModels;
    }
    #endregion
    public override void Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny)
    {
        if (validationDatasetIfAny != null)
        {
            throw new ArgumentException($"{nameof(validationDatasetIfAny)} must be empty : the full training dataset must be cotained in {nameof(trainDataset)}");
        }

        int n_splits = _embeddedModels.Count;
        var foldedTrainDataset = KFold(trainDataset, n_splits);
        for(int fold=0;fold<n_splits;++fold)
        {
            Log.Info($"Training on fold[{fold}/{n_splits}]");
            var model = _embeddedModels[fold];
            var trainAndValidation = foldedTrainDataset[fold];
            //Training on fold[0 / 4]
            model.Fit(trainAndValidation.Training, trainAndValidation.Test);
            var foldValidationPrediction = model.Predict(trainAndValidation.Test);
            var foldValidationScore = model.ComputeScore(trainAndValidation.Test.Y, foldValidationPrediction);
            Log.Info($"Validation score for fold[{fold}/{n_splits}] : {foldValidationScore}");
        }
    }
    public override CpuTensor<float> Predict(IDataSet dataset)
    {
        CpuTensor<float> res = null;
        //each model weight
        int n_splits = _embeddedModels.Count;
        var weight = 1.0f/n_splits;
        foreach (var m in _embeddedModels)
        {
            var modelPrediction = m.Predict(dataset);
            if (res == null)
            {
                res = modelPrediction;
                res.Update_Multiplying_By_Alpha(weight);
            }
            else
            {
                res.AddTensor(weight, modelPrediction, 1.0f);
            }
        }
        return res;
    }
    public override void Save(string workingDirectory, string modelName)
    {
        foreach (var m in _embeddedModels)
        {
            m.Save(m.WorkingDirectory, m.ModelName);
        }
    }

    public override int GetNumEpochs()
    {
        return _embeddedModels[0].GetNumEpochs();
    }

    public override string GetDeviceName()
    {
        return _embeddedModels[0].GetDeviceName();
    }

    public override double GetLearningRate()
    {
        return _embeddedModels[0].GetLearningRate();
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

    private static string ToModelNameForKFold(string modelName, int index, int n_splits)
    {
        return ToModelNameForKFold(modelName, index) + "_" + n_splits + "_split";
    }
    private static string ToModelNameForKFold(string modelName, int index)
    {
        return modelName + "_kfold_" + index;
    }

    //TODO add tests
    private static List<Tuple<int, int>> KFoldIntervals(int n_splits, int count)
    {
        Debug.Assert(n_splits >= 1);
        List<Tuple<int, int>> validationIntervalForKfold = new();
        int countByFold = count / n_splits;
        Debug.Assert(countByFold >= 1);
        while (validationIntervalForKfold.Count < n_splits)
        {
            var start = validationIntervalForKfold.Count == 0 ? 0 : validationIntervalForKfold.Last().Item2 + 1;
            var end = start + countByFold - 1;
            if (validationIntervalForKfold.Count == n_splits - 1)
            {
                end = count - 1;
            }
            validationIntervalForKfold.Add(Tuple.Create(start, end));
        }
        return validationIntervalForKfold;
    }
    private static List<TrainingAndTestDataLoader> KFold(IDataSet dataset, int kfold)
    {
        var validationIntervalForKfold = KFoldIntervals(kfold, dataset.Count);
        List<TrainingAndTestDataLoader> res = new();
        for (var index = 0; index < validationIntervalForKfold.Count; index++)
        {
            var intervalForValidation = validationIntervalForKfold[index];
            var training = dataset.SubDataSet(id => id < intervalForValidation.Item1 || id > intervalForValidation.Item2);
            var test = dataset.SubDataSet(id => id >= intervalForValidation.Item1 && id <= intervalForValidation.Item2);
            res.Add(new TrainingAndTestDataLoader(training, test, KFoldModel.ToModelNameForKFold(dataset.Name, index, validationIntervalForKfold.Count)));
        }
        return res;
    }
}
