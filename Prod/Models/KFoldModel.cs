//using System;
//using System.Collections.Generic;
//using System.Diagnostics;
//using System.Linq;
//using SharpNet.CPU;
//using SharpNet.Datasets;
//using SharpNet.HyperParameters;

//namespace SharpNet.Models;

///// <summary>
//    /// Train 'kfold' distinct models on a the training dataset.
//    /// Use Stacked Ensemble Learning to make predictions for the models 
//    /// </summary>
//public class KFoldModel : AbstractModel
//{
//    #region private fields
//    private readonly List<IModel> _embeddedModels;
//    #endregion

//    #region constructors
//    private static KFoldModel NewUntrainedModel(IModelSample embeddedModelSample, string workingDirectory, string embeddedModelName, int n_splits, bool useFullTrainingDataset, int countMustBeMultipleOf)
//    {
//        KFoldSample sample = new(n_splits, embeddedModelName, useFullTrainingDataset, countMustBeMultipleOf, embeddedModelSample.GetMetric(), embeddedModelSample.GetLoss());
//        var modelName = sample.ComputeHash();
//        Debug.Assert(sample.n_splits >= 2);
//        List<IModel> embeddedModels = new();
//        for (int i = 0; i < sample.n_splits; ++i)
//        {
//            embeddedModels.Add(NewUntrainedModel(embeddedModelSample, workingDirectory, EmbeddedModelName(modelName, i)));
//        }
//        return new KFoldModel(sample, embeddedModels, workingDirectory, modelName);

//    }

//    public static KFoldModel LoadTrainedKFoldModel(string workingDirectory, string modelName)
//    {
//        var sample = KFoldSample.LoadKFoldSample(workingDirectory, modelName);
//        List<IModel> embeddedModels = new();
//        for (int i = 0; i < sample.n_splits; ++i)
//        {
//            var m = LoadTrainedAbstractModel(workingDirectory, EmbeddedModelName(modelName, i));
//            embeddedModels.Add(m);
//        }
//        return new KFoldModel(sample, embeddedModels, workingDirectory, modelName);
//    }
//    private KFoldModel(KFoldSample modelSample, List<IModel> embeddedModels, string workingDirectory, string modelName) : base(modelSample, workingDirectory, modelName)
//    {
//        _embeddedModels = embeddedModels;
//    }
//    #endregion

//    public override (string train_XDatasetPath, string train_YDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath) Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny)
//    {
//        if (validationDatasetIfAny != null && KFoldSample.UseFullTrainingDataset)
//        {
//            throw new ArgumentException($"{nameof(validationDatasetIfAny)} must be empty when using full training dataset or training");
//        }

//        Debug.Assert(KFoldSample.n_splits == _embeddedModels.Count);
//        int n_splits = KFoldSample.n_splits;
//        var foldedTrainAndValidationDataset = KFold(trainDataset, n_splits, KFoldSample.CountMustBeMultipleOf);
//        for(int fold=0;fold<n_splits;++fold)
//        {
//            Log.Info($"Training model '{ModelName}' on fold[{fold}/{n_splits}]");
//            var model = _embeddedModels[fold];
//            var trainAndValidation = foldedTrainAndValidationDataset[fold];
//            model.Fit(trainAndValidation.Training, trainAndValidation.Test);
//            var validationDataset = validationDatasetIfAny ?? trainAndValidation.Test;
//            var foldValidationPrediction = model.Predict(validationDataset);
//            var foldValidationScore = model.ComputeScore(validationDataset.Y, foldValidationPrediction);
//            Log.Info($"Validation score for fold[{fold}/{n_splits}] : {foldValidationScore}");
//        }
//        return ("", "", "", "");
//    }
//    public override CpuTensor<float> Predict(IDataSet dataset)
//    {
//        CpuTensor<float> res = null;
//        Debug.Assert(KFoldSample.n_splits == _embeddedModels.Count);
//        //each model weight
//        var weight = 1.0f/ KFoldSample.n_splits;
//        foreach (var m in _embeddedModels)
//        {
//            var modelPrediction = m.Predict(dataset);
//            if (res == null)
//            {
//                res = modelPrediction;
//                res.Update_Multiplying_By_Alpha(weight);
//            }
//            else
//            {
//                res.AddTensor(weight, modelPrediction, 1.0f);
//            }
//        }
//        return res;
//    }
//    public override void Save(string workingDirectory, string modelName)
//    {
//        ModelSample.Save(workingDirectory, modelName);
//        foreach (var m in _embeddedModels)
//        {
//            m.Save(m.WorkingDirectory, m.ModelName);
//        }
//    }

//    public override int GetNumEpochs()
//    {
//        return _embeddedModels[0].GetNumEpochs();
//    }
//    public override string DeviceName()
//    {
//        return _embeddedModels[0].DeviceName();
//    }
//    public override double GetLearningRate()
//    {
//        return _embeddedModels[0].GetLearningRate();
//    }
//    public override int TotalParams()
//    {
//        return -1; //TODO
//    }
//    public override List<string> ModelFiles()
//    {
//        List<string> res = new();
//        foreach (var m in _embeddedModels)
//        {
//            res.AddRange(m.ModelFiles());
//        }
//        return res;
//    }
//    // ReSharper disable once UnusedMember.Global
//    public static void TrainEmbeddedModelWithKFold(string workingDirectory, string embeddedModelName, int n_splits, int countMustBeMultipleOf = 1)
//    {
//        var modelAndDataset = ModelAndDataset.LoadModelAndDataset(workingDirectory, embeddedModelName);
//        var modelSample = modelAndDataset.ModelAndDatasetSample.ModelSample;
//        var datasetSample = modelAndDataset.ModelAndDatasetSample.DatasetSample;
//        var backupPath = (datasetSample.Train_XDatasetPath, datasetSample.Train_YDatasetPath, datasetSample.Validation_XDatasetPath, datasetSample.Validation_YDatasetPath, datasetSample.Test_DatasetPath);
//        modelSample.Use_All_Available_Cores();

//        //var model007 = KFoldModel.LoadTrainedModel(workingDirectory, "C1610E14E8");
//        //datasetSample.ComputePredictions(model007); return;

//        //We first train on part of training dataset
//        var kfoldModelWithPartOfTrainingDataset = NewUntrainedModel(modelSample, workingDirectory, embeddedModelName, n_splits, false, countMustBeMultipleOf);
//        Log.Info($"Training '{kfoldModelWithPartOfTrainingDataset.ModelName}' (based on model {embeddedModelName}) on part of Training Dataset");
//        datasetSample.Fit(kfoldModelWithPartOfTrainingDataset, true, true, true);
//        (datasetSample.Train_XDatasetPath, datasetSample.Train_YDatasetPath, datasetSample.Validation_XDatasetPath,  datasetSample.Validation_YDatasetPath, datasetSample.Test_DatasetPath) = backupPath;
//        datasetSample.Save(workingDirectory, kfoldModelWithPartOfTrainingDataset.ModelName + "_1");

//        //We then train on full training dataset
//        var kfoldModelWithFullTrainingDataset = NewUntrainedModel(modelSample, workingDirectory, embeddedModelName, n_splits, true, countMustBeMultipleOf);
//        Log.Info($"Training '{kfoldModelWithFullTrainingDataset.ModelName}' (based on model {embeddedModelName}) on full of Training Dataset");
//        using var fullTraining = datasetSample.FullTraining();
//        kfoldModelWithFullTrainingDataset.Fit(fullTraining, null);
//        datasetSample.ComputeAndSavePredictions(kfoldModelWithFullTrainingDataset);
//        kfoldModelWithFullTrainingDataset.Save(kfoldModelWithFullTrainingDataset.WorkingDirectory, kfoldModelWithFullTrainingDataset.ModelName);
//        (datasetSample.Train_XDatasetPath, datasetSample.Train_YDatasetPath, datasetSample.Validation_XDatasetPath, datasetSample.Validation_YDatasetPath, datasetSample.Test_DatasetPath) = backupPath;
//        datasetSample.Save(workingDirectory, kfoldModelWithFullTrainingDataset.ModelName + "_1");
//    }

//    private KFoldSample KFoldSample => (KFoldSample)ModelSample;
//    private static string EmbeddedModelName(string modelName, int index)
//    {
//        return modelName+"_kfold_"+index;
//    }
//    //TODO add tests
//    private static List<Tuple<int, int>> KFoldIntervals(int n_splits, int count, int countMustBeMultipleOf)
//    {
//        Debug.Assert(n_splits >= 1);
//        List<Tuple<int, int>> validationIntervalForKfold = new();
//        int countByFold = count / n_splits;
//        Debug.Assert(countByFold >= 1);
//        while (validationIntervalForKfold.Count < n_splits)
//        {
//            var start = validationIntervalForKfold.Count == 0 ? 0 : validationIntervalForKfold.Last().Item2 + 1;
//            var end = start + countByFold - 1;
//            end -= (end - start + 1) % countMustBeMultipleOf;
//            if (validationIntervalForKfold.Count == n_splits - 1)
//            {
//                end = count - 1;
//            }
//            validationIntervalForKfold.Add(Tuple.Create(start, end));
//        }
//        return validationIntervalForKfold;
//    }
//    private static List<TrainingAndTestDataset> KFold(IDataSet dataset, int kfold, int countMustBeMultipleOf)
//    {
//        var validationIntervalForKfold = KFoldIntervals(kfold, dataset.Count, countMustBeMultipleOf);
//        List<TrainingAndTestDataset> res = new();
//        for (var index = 0; index < validationIntervalForKfold.Count; index++)
//        {
//            var intervalForValidation = validationIntervalForKfold[index];
//            var training = dataset.SubDataSet(id => id < intervalForValidation.Item1 || id > intervalForValidation.Item2);
//            var test = dataset.SubDataSet(id => id >= intervalForValidation.Item1 && id <= intervalForValidation.Item2);
//            res.Add(new TrainingAndTestDataset(training, test, EmbeddedModelName(dataset.Name, index)));
//        }
//        return res;
//    }
//}
