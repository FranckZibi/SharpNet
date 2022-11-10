using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using JetBrains.Annotations;
using SharpNet.Datasets;
using SharpNet.HPO;
using SharpNet.LightGBM;
using SharpNet.Models;

namespace SharpNet.HyperParameters;

public static class SampleUtils
{


    public static (ISample bestSample, IScore bestScore) LaunchLightGBMHPO([NotNull] AbstractDatasetSample datasetSample, [NotNull] string workingDirectory, int num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{ "KFold", new[] { 3 } },


            { "boosting", new[] { "gbdt", "dart" } },
            
            //dart mode
            //{"drop_rate", new[]{0.05, 0.1, 0.2}},
            { "max_drop", new[] { 40, 50, 60 } },
            { "skip_drop", AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f) },

            //related to LightGBM model
            //{ "num_iterations", AbstractHyperParameterSearchSpace.Range(min_num_iterations, 3*min_num_iterations) },
            { "num_iterations", num_iterations },
            { "verbosity", "0" },
            { "num_threads", 1 },
            { "learning_rate", AbstractHyperParameterSearchSpace.Range(0.01f, 0.2f) },
            { "extra_trees", false },
            { "early_stopping_round", num_iterations / 10 },
            { "bagging_fraction", new[] { 0.9f, 1.0f } },
            { "bagging_freq", new[] { 0, 1 } },
            { "colsample_bytree", AbstractHyperParameterSearchSpace.Range(0.3f, 1.0f) },
            //{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
            { "lambda_l1", AbstractHyperParameterSearchSpace.Range(0f, 2f) },
            { "lambda_l2", AbstractHyperParameterSearchSpace.Range(0f, 2f) },
            { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            { "max_depth", new[] { 10, 20, 50, 100, 255 } },
            //{ "min_data_in_bin", AbstractHyperParameterSearchSpace.Range(3, 100) },
            { "min_data_in_bin", new[] { 3, 10, 100, 150 } },
            //{ "min_data_in_leaf", AbstractHyperParameterSearchSpace.Range(20, 200) },
            //{ "min_data_in_leaf", new[]{10, 20, 30} },
            //{ "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(5, 100) },
            //{ "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) },
        };

        datasetSample.FillWithDefaultLightGBMHyperParameterValues(searchSpace);

        if (!Directory.Exists(workingDirectory))
        {
            Directory.CreateDirectory(workingDirectory);
        }
        var dataDirectory = Path.Combine(workingDirectory, "Data");
        if (!Directory.Exists(dataDirectory))
        {
            Directory.CreateDirectory(dataDirectory);
        }

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), datasetSample), workingDirectory); IScore bestScoreSoFar = null;
        hpo.Process(t => TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, workingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
        return (hpo.BestSampleFoundSoFar, hpo.ScoreOfBestSampleFoundSoFar);
    }


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
                modelAndDataset.Save(workingDirectory, model.ModelName);
                var modelAndDatasetPredictionsSampleOnFullDataset = modelAndDatasetPredictionsSample.CopyWithNewPercentageInTrainingAndKFold(1.0, 1);
                var modelAndDatasetOnFullDataset = new ModelAndDatasetPredictions(modelAndDatasetPredictionsSampleOnFullDataset, workingDirectory, model.ModelName+"_FULL");
                Model.Log.Info($"Retraining Model '{model.ModelName}' on full Dataset no KFold (Model on full Dataset name: {modelAndDatasetOnFullDataset.Model.ModelName})");
                modelAndDatasetOnFullDataset.Fit(true, true, true);
                //modelAndDatasetOnFullDataset.ComputeAndSaveFeatureImportance();
            }
            else
            {
                Model.Log.Info($"No interest to save the Model because best score {bestScoreSoFar} is lower than threshold {datasetSample.MinimumScoreToSaveModel}");
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
