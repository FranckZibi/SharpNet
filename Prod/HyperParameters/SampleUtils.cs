﻿using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.Models;

namespace SharpNet.HyperParameters;

public static class SampleUtils
{
    /// <summary>
    /// Train the model (in modelAndDatasetSample) using the dataset (in modelAndDatasetSample)
    /// If validation score of the trained model is better then 'bestScoreSoFar'
    ///     update 'bestScoreSoFar'
    ///     save the model (and associated predictions) in disk
    /// Else
    ///     remove all files associated with the model
    /// If 'resumeCsvPathIfAny' is not null
    ///     store train statistics of the model in CSV file 'resumeCsvPathIfAny'
    /// </summary>
    /// <param name="modelAndDatasetPredictionsSample">the 'model sample' and dataset to use for training the model</param>
    /// <param name="workingDirectory">the directory where the 'model sample' and 'dataset description' is located</param>
    /// <param name="bestScoreSoFar">the best score associated with the best sample found so far for the model</param>
    /// <returns>the score of the ranking evaluation metric for the validation dataset</returns>
    public static IScore TrainWithHyperParameters([NotNull] ModelAndDatasetPredictionsSample modelAndDatasetPredictionsSample, string workingDirectory, ref IScore bestScoreSoFar)
    {
        var modelAndDataset = ModelAndDatasetPredictions.New(modelAndDatasetPredictionsSample, workingDirectory);
        var model = modelAndDataset.Model;
        var validationRankingScore = modelAndDataset.Fit(false, true, false);

        Debug.Assert(validationRankingScore != null);

        if (validationRankingScore.IsBetterThan(bestScoreSoFar))
        {
            Model.Log.Info($"Model '{model.ModelName}' has new best score: {validationRankingScore} (was: {bestScoreSoFar})");
            bestScoreSoFar = validationRankingScore;
            var datasetSample = modelAndDataset.DatasetSample;
            if (datasetSample.MinimumScoreToSaveModel == null || bestScoreSoFar.IsBetterThan(datasetSample.MinimumScoreToSaveModel))
            {
                var trainAndValidation = datasetSample.SplitIntoTrainingAndValidation();
                modelAndDataset.ComputeAndSavePredictions(trainAndValidation);
                modelAndDataset.Save(workingDirectory);
                var modelAndDatasetPredictionsSampleOnFullDataset = modelAndDatasetPredictionsSample.CopyWithNewPercentageInTrainingAndKFold(1.0, 1);
                var modelAndDatasetOnFullDataset = new ModelAndDatasetPredictions(modelAndDatasetPredictionsSampleOnFullDataset, workingDirectory, model.ModelName+"_FULL");
                Model.Log.Info($"Retraining Model '{model.ModelName}' on full Dataset no KFold (Model on full Dataset name: {modelAndDatasetOnFullDataset.Model.ModelName})");
                modelAndDatasetOnFullDataset.Fit(true, true, true);
                //modelAndDatasetOnFullDataset.ComputeAndSaveFeatureImportance();
            }
            else
            {
                Model.Log.Info($"No interest to save the Model because best score {bestScoreSoFar} is better than threshold {datasetSample.MinimumScoreToSaveModel}");
            }
        }
        else
        {
            Model.Log.Debug($"Removing all model '{model.ModelName}' files because of low score ({validationRankingScore})");
            modelAndDataset.AllFiles().ForEach(path => Utils.TryDelete(path));
        }

        return validationRankingScore;
    }
}
