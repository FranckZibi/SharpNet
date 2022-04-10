//using System;
//using System.Collections.Generic;
//using System.Diagnostics;
//using System.IO;
//using System.Linq;
//using JetBrains.Annotations;
//using SharpNet.CPU;
//using SharpNet.HyperParameters;
//using SharpNet.Models;

//namespace SharpNet.HPO;

//public class WeightsOptimizer /*: AbstractModel*/
//{
//    #region private fields
//    private readonly List<ModelAndDataset> _embeddedModels = new();
//    private readonly List<CpuTensor<float>> _originalTrainPredictions = new();
//    private readonly List<CpuTensor<float>> _originalValidationPredictions = new();
//    private readonly List<CpuTensor<float>> _originalTestPredictions = new();
//    private readonly CpuTensor<float> _perfectTrainPredictions;
//    private readonly CpuTensor<float> _perfectValidationPredictions;
//    private WeightsOptimizerSample WeightsOptimizerSample { get; }
//    #endregion


//    private IModel FirstModel => _embeddedModels[0].Model;
//    #region constructor
//    public WeightsOptimizer(WeightsOptimizerSample weightsOptimizerSample, [NotNull] string workingDirectory, string modelName) /*: base(weightsOptimizerSample, workingDirectory, modelName)*/
//    {
//        if (!Directory.Exists(workingDirectory))
//        {
//            Directory.CreateDirectory(workingDirectory);
//        }

//        WeightsOptimizerSample = weightsOptimizerSample;

//        var workingDirectoryAndModelNames = WeightsOptimizerSample.GetWorkingDirectoryAndModelNames();
//        for (int i = 0; i < workingDirectoryAndModelNames.Count; ++i)
//        {
//            var (embeddedModelWorkingDirectory, embeddedModelName) = workingDirectoryAndModelNames[i];
//            var embeddedModel = ModelAndDataset.LoadTrainedModelAndDataset(embeddedModelWorkingDirectory, embeddedModelName);
//            var embeddedModelSample = embeddedModel.ModelAndDatasetSample;
//            _embeddedModels.Add(embeddedModel);
//            if (i == 0)
//            {
//                _datasetSample = embeddedModelSample.DatasetSample;
//                _perfectTrainPredictions = embeddedModel.PredictionsInModelFormat_2_PredictionsInTargetFormat(_datasetSample.Train_YDatasetPath);
//                _perfectValidationPredictions = embeddedModel.PredictionsInModelFormat_2_PredictionsInTargetFormat(_datasetSample.Validation_YDatasetPath);
//            }

//            var (trainPredictions, validationPredictions, testPredictions) = 
//                embeddedModelSample.DatasetSample.LoadAllPredictions();

//            _originalTrainPredictions.Add(trainPredictions);
//            _originalValidationPredictions.Add(validationPredictions);
//            _originalTestPredictions.Add(testPredictions);
//        }

//        //_perfect_validation_predictions = trainedModels[0].Perfect_Validation_Predictions_if_any();
//        Debug.Assert(_perfectValidationPredictions != null);
//        //_perfect_train_predictions_if_any = trainedModels[0].Perfect_Train_Predictions_if_any();

//        //we load the train predictions done by the source models (if any)
//        _originalTrainPredictions.RemoveAll(t => t == null);
//        Debug.Assert(_originalTrainPredictions.Count == 0 || _originalTrainPredictions.Count == _originalValidationPredictions.Count);
//        Debug.Assert(SameShape(_originalTrainPredictions));

//        //we load the validation predictions done by the source models
//        _originalValidationPredictions.RemoveAll(t => t == null);
//        Debug.Assert(_originalValidationPredictions.All(t => t != null));
//        Debug.Assert(SameShape(_originalValidationPredictions));


//        //we load the test predictions done by the source models (if any)
//        _originalTestPredictions.RemoveAll(t => t == null);
//        Debug.Assert(_originalTestPredictions.Count == 0 || _originalTestPredictions.Count == _originalValidationPredictions.Count);
//        Debug.Assert(SameShape(_originalTestPredictions));
//    }
//    #endregion

//    public static void SearchForBestWeights(List<Tuple<string, string>> workingDirectoryAndModelNames, string workingDirectory, string csvPath)
//    {
//        Utils.ConfigureGlobalLog4netProperties(workingDirectory, "log");
//        Utils.ConfigureThreadLog4netProperties(workingDirectory, "log");

//        var sample = new WeightsOptimizerSample(workingDirectoryAndModelNames, MetricEnum.DEFAULT, LossFunctionEnum.DEFAULT);
//        var weightsOptimizer = new WeightsOptimizer(sample, workingDirectory, sample.ComputeHash());
//        var metric = weightsOptimizer._embeddedModels[0].Model.ModelSample.GetMetric();
//        var loss = weightsOptimizer._embeddedModels[0].Model.ModelSample.GetLoss();
//        sample.Metric = metric;
//        sample.Loss = loss;

//        var searchSpace = new Dictionary<string, object>();
//        for (int i = 0; i < weightsOptimizer._originalTrainPredictions.Count; ++i)
//        {
//            searchSpace["w_" + i.ToString("D2")] = AbstractHyperParameterSearchSpace.Range(0.0f, 1.0f);
//            IModel.Log.Info($"Original validation score of model#{i} ({workingDirectoryAndModelNames[i].Item2}) : {weightsOptimizer._embeddedModels[i].ComputeScore(weightsOptimizer._perfectValidationPredictions, weightsOptimizer._originalValidationPredictions[i])}");
//            if (weightsOptimizer._originalTrainPredictions[i] != null)
//            {
//                IModel.Log.Info($"Original train score of model#{i} ({workingDirectoryAndModelNames[i].Item2}) : {weightsOptimizer._embeddedModels[i].ComputeScore(weightsOptimizer._perfectTrainPredictions, weightsOptimizer._originalTrainPredictions[i])}");
//            }
//        }

//        sample.SetEqualWeights();
//        IModel.Log.Info($"Validation score if Equal Weights: {weightsOptimizer.Predictions().validationScore}");

//        var hpo = new BayesianSearchHPO(searchSpace, () => new ModelAndDatasetSample(new WeightsOptimizerSample(weightsOptimizer.WeightsOptimizerSample.WorkingDirectoryAndModelNames, metric, loss), new WeightsOptimizerDatasetSample(weightsOptimizer._datasetSample)), workingDirectory);
//        float bestScoreSoFar = float.NaN;
//        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetSample)t, workingDirectory, csvPath, ref bestScoreSoFar));
//    }
//    //public override (string train_XDatasetPath, string train_YDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath) Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny)
//    //{
//    //    return ("", "", "", "");
//    //}
//    //public override CpuTensor<float> Predict(IDataSet dataset)
//    //{
//    //    var predictions = _embeddedModels.Select(m => m.Model.Predict(dataset)).ToList();
//    //    return WeightsOptimizerSample.ApplyWeights(predictions);
//    //}
//    public (CpuTensor<float> trainPredictions, float trainScore, CpuTensor<float> validationPredictions, float validationScore, CpuTensor<float> testPredictions) 
//        Predictions()
//    {
//        var trainPredictions = WeightsOptimizerSample.ApplyWeights(_originalTrainPredictions);
//        float trainScore = (trainPredictions != null && _perfectTrainPredictions != null)
//                ? _embeddedModels[0].Model.ComputeScore(_perfectTrainPredictions, trainPredictions)
//                : float.NaN;

//        var validationPredictions = WeightsOptimizerSample.ApplyWeights(_originalValidationPredictions);
//        float validationScore = (validationPredictions != null && _perfectValidationPredictions != null)
//            ? _embeddedModels[0].Model.ComputeScore(_perfectValidationPredictions, validationPredictions)
//            : float.NaN;

//        var testPredictions = WeightsOptimizerSample.ApplyWeights(_originalTestPredictions);

//        return (trainPredictions, trainScore, validationPredictions, validationScore, testPredictions);
//    }
//    //public override void Save(string workingDirectory, string modelName)
//    //{
//    //    throw new NotImplementedException();
//    //}
//    //public override int GetNumEpochs()
//    //{
//    //    return _embeddedModels.Select(m => m.GetNumEpochs()).Sum();
//    //}
//    //public override string DeviceName()
//    //{
//    //    var distinct = new HashSet<string>(_embeddedModels.Select(m => m.DeviceName()));
//    //    return string.Join(" / ", distinct);
//    //}
//    //public override int TotalParams()
//    //{
//    //    return _embeddedModels.Select(m=>m.TotalParams()).Sum();
//    //}
//    //public override double GetLearningRate()
//    //{
//    //    return _embeddedModels[0].GetLearningRate();
//    //}
//    //public override List<string> ModelFiles()
//    //{
//    //    return new List<string>();
//    //}

//    private static bool SameShape(IList<CpuTensor<float>> tensors)
//    {
//        return tensors.All(t => t.SameShape(tensors[0]));
//    }
//}
