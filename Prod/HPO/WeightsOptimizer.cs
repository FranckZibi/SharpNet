using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.HPO;

public class WeightsOptimizer /*: AbstractModel*/
{
    #region private fields
    private readonly List<ModelAndDatasetPredictions> _embeddedModelsAndDataset = new();
    private readonly List<DataFrame> _y_preds_train_InModelFormat = new();
    private readonly List<DataFrame> _y_preds_valid_InModelFormat = new();
    private readonly List<DataFrame> _y_preds_test_InModelFormat = new();
    private AbstractDatasetSample DatasetSample => _embeddedModelsAndDataset[0].ModelAndDatasetPredictionsSample.DatasetSample;

    private readonly DataFrame _y_true_train_InTargetFormat;
    private readonly DataFrame _y_true_valid_InTargetFormat;
    private readonly DataFrame _y_true_test_InTargetFormat;
    #endregion


    //private IModel FirstModel => _embeddedModelsAndDataset[0].Model;
    private AbstractDatasetSample FirstDatasetSample => _embeddedModelsAndDataset[0].ModelAndDatasetPredictionsSample.DatasetSample;


    #region constructor

    private WeightsOptimizer(IReadOnlyList<Tuple<string, string>> workingDirectoryAndModelNames, [NotNull] string workingDirectory)
    {
        if (!Directory.Exists(workingDirectory))
        {
            Directory.CreateDirectory(workingDirectory);
        }
        
        for (int i = 0; i < workingDirectoryAndModelNames.Count; ++i)
        {
            var (embeddedModelWorkingDirectory, embeddedModelName) = workingDirectoryAndModelNames[i];
            var embeddedModelAndDatasetPredictions = ModelAndDatasetPredictions.Load(embeddedModelWorkingDirectory, embeddedModelName);
            var datasetSample = embeddedModelAndDatasetPredictions.ModelAndDatasetPredictionsSample.DatasetSample;
            _embeddedModelsAndDataset.Add(embeddedModelAndDatasetPredictions);

            var (trainPredictions, validationPredictions, testPredictions) = embeddedModelAndDatasetPredictions.LoadAllPredictionsInModelFormat();

            if (validationPredictions == null)
            {
                validationPredictions = trainPredictions;
            }

            _y_preds_train_InModelFormat.Add(trainPredictions);
            _y_preds_valid_InModelFormat.Add(validationPredictions);
            _y_preds_test_InModelFormat.Add(testPredictions);

            if (i == 0)
            {
                var trainAndValidation = datasetSample.LoadTrainingAndValidationDataset_Encoded_InTargetFormat();
                var trainDataset = trainAndValidation.Training;
                var validationDataset = trainAndValidation.Test;
                var testDataset = datasetSample.LoadTestDataset_Encoded_InTargetFormat().Training;
                _y_true_train_InTargetFormat = trainDataset?.Y_InTargetFormat(true);
                _y_true_valid_InTargetFormat = validationDataset?.Y_InTargetFormat(true);
                if (_y_true_valid_InTargetFormat == null)
                {
                    _y_true_valid_InTargetFormat = _y_true_train_InTargetFormat;
                }
                _y_true_test_InTargetFormat = testDataset?.Y_InTargetFormat(true);
            }
        }

        //Debug.Assert(_perfectValidationPredictions_InTargetFormat != null);

        //we load the train predictions done by the source models (if any)
        _y_preds_train_InModelFormat.RemoveAll(t => t == null);
        Debug.Assert(_y_preds_train_InModelFormat.Count == 0 || _y_preds_train_InModelFormat.Count == _y_preds_valid_InModelFormat.Count);
        Debug.Assert(SameShape(_y_preds_train_InModelFormat));

        //we load the validation predictions done by the source models
        _y_preds_valid_InModelFormat.RemoveAll(t => t == null);
        Debug.Assert(_y_preds_valid_InModelFormat.All(t => t != null));
        Debug.Assert(SameShape(_y_preds_valid_InModelFormat));

        //we load the test predictions done by the source models (if any)
        _y_preds_test_InModelFormat.RemoveAll(t => t == null);
        //Debug.Assert(_TestPredictions_INModelFormat.Count == 0 || _TestPredictions_INModelFormat.Count == _ValidationPredictions_InModelFormat.Count);
        Debug.Assert(SameShape(_y_preds_test_InModelFormat));
    }
    #endregion



    private static bool SameShape(IList<DataFrame> tensors)
    {
        return tensors.All(t => t.Shape.SequenceEqual(tensors[0].Shape));
    }
    public static void SearchForBestWeights(List<Tuple<string, string>> workingDirectoryAndModelNames, string workingDirectory, string csvPath)
    {
        Utils.ConfigureGlobalLog4netProperties(workingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(workingDirectory, "log");

        var weightsOptimizer = new WeightsOptimizer(workingDirectoryAndModelNames, workingDirectory);
        var firstDatasetSample = weightsOptimizer.FirstDatasetSample;

        var searchSpace = new Dictionary<string, object>();
        for (int i = 0; i < weightsOptimizer._y_preds_train_InModelFormat.Count; ++i)
        {
            searchSpace["w_" + i.ToString("D2")] = AbstractHyperParameterSearchSpace.Range(0.0f, 1.0f);
            var y_pred_valid_InModelFormat = weightsOptimizer._y_preds_valid_InModelFormat[i];
            var y_pred_valid_InTargetFormat = weightsOptimizer.DatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(y_pred_valid_InModelFormat);
            var y_prev_valid_score = firstDatasetSample.ComputeRankingEvaluationMetric(weightsOptimizer._y_true_valid_InTargetFormat, y_pred_valid_InTargetFormat);
            IModel.Log.Info($"Original validation score of model#{i} ({workingDirectoryAndModelNames[i].Item2}) :{y_prev_valid_score}");
            var y_pred_train_InModelFormat = weightsOptimizer._y_preds_train_InModelFormat[i];
            if (y_pred_train_InModelFormat != null)
            {
                var y_pred_train_InTargetFormat = weightsOptimizer.DatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(y_pred_train_InModelFormat);
                IModel.Log.Info($"Original train score of model#{i} ({workingDirectoryAndModelNames[i].Item2}) : {firstDatasetSample.ComputeRankingEvaluationMetric(weightsOptimizer._y_true_train_InTargetFormat, y_pred_train_InTargetFormat)}");
            }
        }
        var equalWeightedModelSample = new WeightedModelSample(workingDirectoryAndModelNames);
        equalWeightedModelSample.SetEqualWeights();
        var validationScore = weightsOptimizer.ComputePredictionsAndRankingScore(weightsOptimizer._y_true_valid_InTargetFormat, weightsOptimizer._y_preds_valid_InModelFormat, equalWeightedModelSample).metricScore;
        IModel.Log.Info($"Validation score if Equal Weights: {validationScore}");

        var hpo = new BayesianSearchHPO(searchSpace, () => new WeightedModelSample(workingDirectoryAndModelNames), workingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => weightsOptimizer.TrainWithHyperParameters((WeightedModelSample)t, workingDirectory, csvPath, ref bestScoreSoFar));
    }


    private (DataFrame predictionsInTargetFormat, IScore metricScore) ComputePredictionsAndRankingScore(DataFrame y_true_InTargetFormat, List<DataFrame> all_y_pred_InModelFormat, WeightedModelSample weightedModelSample)
    {
        var idxOfIdColumns = all_y_pred_InModelFormat[0].ColumnNamesToIndexes(FirstDatasetSample.IdColumns);
        var y_pred_InModelFormat = weightedModelSample.ApplyWeights(all_y_pred_InModelFormat, idxOfIdColumns);
        var y_pred_InTargetFormat = DatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(y_pred_InModelFormat);
        var metricScore = FirstDatasetSample.ComputeRankingEvaluationMetric(y_true_InTargetFormat, y_pred_InTargetFormat);
        return (y_pred_InTargetFormat, metricScore);
    }

    private IScore TrainWithHyperParameters([NotNull] WeightedModelSample weightedModelSample, string workingDirectory, [CanBeNull] string resumeCsvPathIfAny, ref IScore bestScoreSoFar)
    {
        var sw = Stopwatch.StartNew();
        var (weightedModelValidationPrediction, validationScore) = ComputePredictionsAndRankingScore(_y_true_valid_InTargetFormat, _y_preds_valid_InModelFormat, weightedModelSample);
        Debug.Assert(validationScore != null);

        if (validationScore.IsBetterThan(bestScoreSoFar))
        {
            var modelAndDatasetPredictionsSample = ModelAndDatasetPredictionsSample.New(weightedModelSample, (AbstractDatasetSample)_embeddedModelsAndDataset[0].ModelAndDatasetPredictionsSample.DatasetSample.Clone());
            var modelAndDatasetPredictions = new ModelAndDatasetPredictions(modelAndDatasetPredictionsSample, workingDirectory, modelAndDatasetPredictionsSample.ComputeHash());
            IModel.Log.Info($"{nameof(WeightedModel)} {modelAndDatasetPredictions.Name} has new best score: {validationScore} (was: {bestScoreSoFar})");
            bestScoreSoFar = validationScore;

            IScore trainScore = null;
            if (_y_preds_train_InModelFormat != null)
            {
                (var weightedModelTrainPrediction, trainScore) = ComputePredictionsAndRankingScore(_y_true_train_InTargetFormat, _y_preds_train_InModelFormat, weightedModelSample);
                if (trainScore != null)
                {
                    IModel.Log.Info($"{nameof(WeightedModel)} {modelAndDatasetPredictions.Name} train score: {trainScore}");
                }
                modelAndDatasetPredictions.SaveTrainPredictionsInTargetFormat(weightedModelTrainPrediction, trainScore);
            }

            if (_y_preds_test_InModelFormat != null)
            {
                var (weightedModelTestPrediction, testScore)  = ComputePredictionsAndRankingScore(_y_true_test_InTargetFormat, _y_preds_test_InModelFormat, weightedModelSample);
                if (testScore != null)
                {
                    IModel.Log.Info($"{nameof(WeightedModel)} {modelAndDatasetPredictions.Name} test score: {testScore}");
                }
                modelAndDatasetPredictions.SaveTestPredictionsInTargetFormat(weightedModelTestPrediction, testScore);
            }

            modelAndDatasetPredictions.SaveValidationPredictionsInTargetFormat(weightedModelValidationPrediction, validationScore);
            modelAndDatasetPredictions.Save(workingDirectory, modelAndDatasetPredictions.Name);


            //var trainScore = modelAndDataset.ComputeAndSavePredictionsInTargetFormat().trainScore;
            if (!string.IsNullOrEmpty(resumeCsvPathIfAny))
            {
                var trainingTimeInSeconds = sw.Elapsed.TotalSeconds;
                modelAndDatasetPredictions.Model.AddResumeToCsv(trainingTimeInSeconds, trainScore, validationScore, resumeCsvPathIfAny);
            }
        }
        return validationScore;
    }
}
