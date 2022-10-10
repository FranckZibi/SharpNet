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
    private readonly List<DataFrame> _embeddedModelsTrainPredictions = new();
    private readonly List<DataFrame> _embeddedModelsValidationPredictions = new();
    private readonly List<DataFrame> _embeddedModelsTestPredictions = new();
    
    private readonly DataFrame _perfectTrainPredictions;
    private readonly DataFrame _perfectValidationPredictions;
    private readonly DataFrame _perfectTestPredictions;
    #endregion


    //private IModel FirstModel => _embeddedModelsAndDataset[0].Model;
    private AbstractDatasetSample FirstDatasetSample => _embeddedModelsAndDataset[0].ModelAndDatasetPredictionsSample.DatasetSample;


    #region constructor

    private WeightsOptimizer(List<Tuple<string, string>> workingDirectoryAndModelNames, [NotNull] string workingDirectory)
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

            var (trainPredictions, validationPredictions, testPredictions) = embeddedModelAndDatasetPredictions.LoadAllPredictionsInTargetFormat();

            _embeddedModelsTrainPredictions.Add(trainPredictions);
            _embeddedModelsValidationPredictions.Add(validationPredictions);
            _embeddedModelsTestPredictions.Add(testPredictions);

            if (i == 0)
            {
                _perfectTrainPredictions = datasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(datasetSample.Train_YDatasetPath);
                _perfectValidationPredictions = datasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(datasetSample.Validation_YDatasetPath);
                _perfectTestPredictions = datasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(datasetSample.Test_YDatasetPath);
            }
        }

        Debug.Assert(_perfectValidationPredictions != null);

        //we load the train predictions done by the source models (if any)
        _embeddedModelsTrainPredictions.RemoveAll(t => t == null);
        Debug.Assert(_embeddedModelsTrainPredictions.Count == 0 || _embeddedModelsTrainPredictions.Count == _embeddedModelsValidationPredictions.Count);
        Debug.Assert(DataFrame.SameShape(_embeddedModelsTrainPredictions));

        //we load the validation predictions done by the source models
        _embeddedModelsValidationPredictions.RemoveAll(t => t == null);
        Debug.Assert(_embeddedModelsValidationPredictions.All(t => t != null));
        Debug.Assert(DataFrame.SameShape(_embeddedModelsValidationPredictions));

        //we load the test predictions done by the source models (if any)
        _embeddedModelsTestPredictions.RemoveAll(t => t == null);
        Debug.Assert(_embeddedModelsTestPredictions.Count == 0 || _embeddedModelsTestPredictions.Count == _embeddedModelsValidationPredictions.Count);
        Debug.Assert(DataFrame.SameShape(_embeddedModelsTestPredictions));
    }
    #endregion

    public static void SearchForBestWeights(List<Tuple<string, string>> workingDirectoryAndModelNames, string workingDirectory, string csvPath)
    {
        Utils.ConfigureGlobalLog4netProperties(workingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(workingDirectory, "log");

        var weightsOptimizer = new WeightsOptimizer(workingDirectoryAndModelNames, workingDirectory);
        var firstDatasetSample = weightsOptimizer.FirstDatasetSample;

        var searchSpace = new Dictionary<string, object>();
        for (int i = 0; i < weightsOptimizer._embeddedModelsTrainPredictions.Count; ++i)
        {
            searchSpace["w_" + i.ToString("D2")] = AbstractHyperParameterSearchSpace.Range(0.0f, 1.0f);
            IModel.Log.Info($"Original validation score of model#{i} ({workingDirectoryAndModelNames[i].Item2}) :{firstDatasetSample.ComputeRankingEvaluationMetric(weightsOptimizer._perfectValidationPredictions, weightsOptimizer._embeddedModelsValidationPredictions[i])}");
            if (weightsOptimizer._embeddedModelsTrainPredictions[i] != null)
            {
                IModel.Log.Info($"Original train score of model#{i} ({workingDirectoryAndModelNames[i].Item2}) : {firstDatasetSample.ComputeRankingEvaluationMetric(weightsOptimizer._perfectTrainPredictions, weightsOptimizer._embeddedModelsTrainPredictions[i])}");
            }
        }
        var equalWeightedModelSample = new WeightedModelSample(workingDirectoryAndModelNames);
        equalWeightedModelSample.SetEqualWeights();
        var validationScore = weightsOptimizer.ComputePredictionsAndRankingEvaluationMetric(weightsOptimizer._perfectValidationPredictions, weightsOptimizer._embeddedModelsValidationPredictions, equalWeightedModelSample).metricScore;
        IModel.Log.Info($"Validation score if Equal Weights: {validationScore}");

        var hpo = new BayesianSearchHPO(searchSpace, () => new WeightedModelSample(workingDirectoryAndModelNames), workingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => weightsOptimizer.TrainWithHyperParameters((WeightedModelSample)t, workingDirectory, csvPath, ref bestScoreSoFar));
    }

    private (DataFrame predictionsInTargetFormat, IScore metricScore) ComputePredictionsAndRankingEvaluationMetric(DataFrame true_predictions_in_target_format, List<DataFrame> t, WeightedModelSample weightedModelSample)
    {
        var idxOfIdColumns = true_predictions_in_target_format.ColumnNamesToIndexes(FirstDatasetSample.IdColumns());
        var weightedModelPredictions = weightedModelSample.ApplyWeights(t, idxOfIdColumns);
        var metricScore = FirstDatasetSample.ComputeRankingEvaluationMetric(true_predictions_in_target_format, weightedModelPredictions);
        return (weightedModelPredictions, metricScore);
    }

    private IScore TrainWithHyperParameters([NotNull] WeightedModelSample weightedModelSample, string workingDirectory, [CanBeNull] string resumeCsvPathIfAny, ref IScore bestScoreSoFar)
    {
        var sw = Stopwatch.StartNew();
        var (weightedModelValidationPrediction, validationScore) = ComputePredictionsAndRankingEvaluationMetric(_perfectValidationPredictions, _embeddedModelsValidationPredictions, weightedModelSample);
        Debug.Assert(validationScore != null);

        if (validationScore.IsBetterThan(bestScoreSoFar))
        {
            var modelAndDatasetPredictionsSample = ModelAndDatasetPredictionsSample.New(weightedModelSample, (AbstractDatasetSample)_embeddedModelsAndDataset[0].ModelAndDatasetPredictionsSample.DatasetSample.Clone());
            var modelAndDatasetPredictions = new ModelAndDatasetPredictions(modelAndDatasetPredictionsSample, workingDirectory, modelAndDatasetPredictionsSample.ComputeHash());
            IModel.Log.Info($"{nameof(WeightedModel)} {modelAndDatasetPredictions.Name} has new best score: {validationScore} (was: {bestScoreSoFar})");
            bestScoreSoFar = validationScore;

            IScore trainScore = null;
            if (_embeddedModelsTrainPredictions != null)
            {
                (var weightedModelTrainPrediction, trainScore) = ComputePredictionsAndRankingEvaluationMetric(_perfectTrainPredictions, _embeddedModelsTrainPredictions, weightedModelSample);
                if (trainScore != null)
                {
                    IModel.Log.Info($"{nameof(WeightedModel)} {modelAndDatasetPredictions.Name} train score: {trainScore}");
                }
                modelAndDatasetPredictions.SaveTrainPredictionsInTargetFormat(weightedModelTrainPrediction, trainScore);
            }

            if (_embeddedModelsTestPredictions != null)
            {
                var (weightedModelTestPrediction, testScore)  = ComputePredictionsAndRankingEvaluationMetric(_perfectTestPredictions, _embeddedModelsTestPredictions, weightedModelSample);
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
