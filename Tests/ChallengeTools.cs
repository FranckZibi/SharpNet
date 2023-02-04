using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.Datasets;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Models;
using SharpNet.TextPreprocessing;

// ReSharper disable ConvertToConstant.Local
// ReSharper disable ConditionIsAlwaysTrueOrFalse

namespace SharpNetTests;

[TestFixture]
public class ChallengeTools
{
    /// <summary>
    ///  compute feature importance of a Model
    /// </summary>
    [Test, Explicit]
    public void ComputeAndSaveFeatureImportance()
    {
        const string workingDirectory = @"C:\Projects\Challenges\CFM84\";
        const string modelName = "64C740FBCE";
        Utils.ConfigureGlobalLog4netProperties(workingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        Utils.ConfigureThreadLog4netProperties(workingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        using var m = ModelAndDatasetPredictions.Load(workingDirectory, modelName, true);
        m.ComputeAndSaveFeatureImportance();
    }

    /// <summary>
    /// normalize all CSV files in directory 'directory' and put the normalized files in sub directory 'subDirectory'
    /// </summary>
    /// <param name="directory"></param>
    /// <param name="hasHeader"></param>
    /// <param name="removeAccentedCharacters"></param>
    [TestCase(@"C:\Projects\Challenges\KaggleDays\Data", true, true), Explicit]
    public void NormalizeAllCsvInDirectory(string directory, bool hasHeader, bool removeAccentedCharacters)
    {
        Utils.ConfigureGlobalLog4netProperties(Path.Combine(directory), $"{nameof(NormalizeAllCsvInDirectory)}");
        Utils.ConfigureThreadLog4netProperties(Path.Combine(directory), $"{nameof(NormalizeAllCsvInDirectory)}");
        DataFrame.NormalizeAllCsvInDirectory(directory, hasHeader, removeAccentedCharacters);
    }


    [Test, Explicit]
    public void TfIdfEncode()
    {
        string[] csvFiles = { @"C:\Projects\Challenges\KaggleDays\Data\search_train.csv", @"C:\Projects\Challenges\KaggleDays\Data\search_test.csv" };
        string columnToEncode = "keyword";


        //string[] csvFiles = { @"C:\Projects\Challenges\KaggleDays\Data\item_info.csv" };
        //string columnToEncode = "name";

        int embeddingDim = 300;
        bool hasHeader = true;
        bool isNormalized = true;
        var keepEncodedColumnName = false;
        var reduceEmbeddingDimIfNeeded = false;
        var norm = TfIdfEncoding.TfIdfEncoding_norm.L2;
        var scikitLearnCompatibilityMode = false;

        string directory = Path.GetDirectoryName(csvFiles[0]) ?? "";
        Utils.ConfigureGlobalLog4netProperties(directory, $"{nameof(TfIdfEncode)}");
        Utils.ConfigureThreadLog4netProperties(directory, $"{nameof(TfIdfEncode)}");
        DataFrame.TfIdfEncode(csvFiles, hasHeader, isNormalized, columnToEncode, embeddingDim, 
            keepEncodedColumnName, reduceEmbeddingDimIfNeeded, norm, scikitLearnCompatibilityMode);
    }

    ///// <summary>
    ///// encode the string column 'columnToEncode' using Tf*Idf with 'embeddingDim' words and return a new DataFrame with this encoding
    ///// </summary>
    ///// <param name="columnToEncode"></param>
    ///// <param name="embeddingDim">the number of dimension for the encoding.
    ///// Only the top frequent 'embeddingDim' words will be considered for the encoding.
    ///// The other will be discarded</param>
    ///// <param name="keepEncodedColumnName">
    ///// Each new feature will have in its name the associated word for the TfIdf encoding</param>
    ///// <param name="reduceEmbeddingDimIfNeeded"></param>
    ///// <param name="norm"></param>
    ///// <param name="scikitLearnCompatibilityMode"></param>
    ///// <returns></returns>
    //public DataFrame TfIdfEncode(string columnToEncode, int embeddingDim, bool keepEncodedColumnName = false, bool reduceEmbeddingDimIfNeeded = false, TfIdfEncoding.TfIdfEncoding_norm norm = TfIdfEncoding.TfIdfEncoding_norm.L2, bool scikitLearnCompatibilityMode = false)
    //{
    //    return TfIdfEncoding.Encode(new[] { this }, columnToEncode, embeddingDim, keepEncodedColumnName, reduceEmbeddingDimIfNeeded, norm, scikitLearnCompatibilityMode)[0];
    //}



    /// <summary>
    /// Stack several trained models together to compute new predictions
    /// (through a new LightGBM model that will be trained to do the stacking)
    /// </summary>
    [TestCase(100,0), Explicit]
    public void StackedEnsemble(int num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        //const string workingDirectory = @"C:/Projects/Challenges/WasYouStayWorthItsPrice/submission";
        const string workingDirectory = @"C:\Projects\Challenges\KaggleDays\aaa7\";
        var modelName = new[]
        {
            "9F587BDFA9_KFOLD",
            "DEBB5D22D9_KFOLD",
           
        };
        const bool use_features_in_secondary = true;
        const int cv = 2;

        Console.WriteLine($"Performing Stacked Ensemble Training with {modelName.Length} models in directory {workingDirectory}");

        var workingDirectoryAndModelNames = modelName.Select(m => Tuple.Create(workingDirectory, m, m + "_FULL")).ToList();
        var datasetSample = StackingCVClassifierDatasetSample.New(workingDirectoryAndModelNames, workingDirectory, use_features_in_secondary, cv);

        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            { "KFold", cv },

            //high priority
            { "bagging_fraction", new[]{/*0.8f,*/ 0.9f /*, 1.0f*/} },
            { "bagging_freq", new[]{0, 1} },
            { "boosting", new []{/*"gbdt",*/ "dart"}},
            { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.3f, 1.0f)},
            { "early_stopping_round", num_iterations/10 },
            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.005f, 0.2f)},
            { "max_depth", new[]{10, 20, 50, 100 /*, 255*/} },
            { "min_data_in_leaf", new[]{20, 50 /*,100*/} },
            { "num_iterations", num_iterations },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 50) },
            { "num_threads", 1},
            { "verbosity", "0" },

            ////medium priority
            { "drop_rate", new[]{0.05, 0.1, 0.2}},                               //specific to dart mode
            { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "min_data_in_bin", new[]{3, 10, 100, 150}  },
            { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            { "max_drop", new[]{40, 50, 60}},                                   //specific to dart mode
            { "skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},  //specific to dart mode

            ////low priority
            //{ "extra_trees", new[] { true , false } }, //low priority 
            ////{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)}, //very low priority
            //{ "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) }, //low priority
            //{ "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },
        };

        var hpoWorkingDirectory = Path.Combine(workingDirectory, "hpo");
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), datasetSample), hpoWorkingDirectory); IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, hpoWorkingDirectory, true, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    /// <summary>
    /// retrain some models 
    /// </summary>
    [Test, Explicit]
    public void Retrain(int? n_splits = 3, double?percentageInTraining = null, bool retrainOnFullDataset = true)
    {
        if (n_splits.HasValue && percentageInTraining.HasValue)
        {
            throw new ArgumentException($"at most one of the 2 parameters {nameof(n_splits)} and {nameof(percentageInTraining)} can be specified");
        }
        if (n_splits.HasValue && n_splits.Value < 2)
        {
            throw new ArgumentException($"When specified, {nameof(n_splits)} must be at least 2");
        }
        if (percentageInTraining.HasValue && (percentageInTraining.Value > 1|| percentageInTraining.Value < 0) )
        {
            throw new ArgumentException($"When specified, {nameof(percentageInTraining)} must be between 0 and 1");
        }

        const string workingDirectory = @"C:\Projects\Challenges\CFM84\dump";
        Utils.ConfigureGlobalLog4netProperties(workingDirectory, $"{nameof(Retrain)}");
        Utils.ConfigureThreadLog4netProperties(workingDirectory, $"{nameof(Retrain)}");
        foreach (var modelName in new[]
                 {
                     "6C5DF90DD8",
                 })
        {


            var sw = Stopwatch.StartNew();
            ISample.Log.Info($"Retraining model '{modelName}' with {nameof(n_splits)}={n_splits}, {nameof(percentageInTraining)}={percentageInTraining} and {nameof(retrainOnFullDataset)}={retrainOnFullDataset}");

            if (n_splits.HasValue)
            {
                var swKfold = Stopwatch.StartNew();
                using var mKFold = ModelAndDatasetPredictions.LoadWithKFold(workingDirectory, modelName, n_splits.Value, true);
                ISample.Log.Info($"Training Model '{mKFold.Model.ModelName}' (= Model '{modelName}' with KFold={n_splits})");
                mKFold.Fit(true, true, true);
                mKFold.Save(workingDirectory);
                ISample.Log.Info($"Model '{mKFold.Model.ModelName}' trained in {swKfold.Elapsed.TotalSeconds}");
            }
            if (percentageInTraining.HasValue)
            {
                var swPercentageInTraining = Stopwatch.StartNew();
                using var modelAndDataset = ModelAndDatasetPredictions.LoadWithNewPercentageInTrainingNoKFold(percentageInTraining.Value, workingDirectory, modelName, true);
                Model.Log.Info($"Training Model '{modelAndDataset.Model.ModelName}' (= Model '{modelName}' with {Math.Round(100* percentageInTraining.Value,1)}% in training no KFold)");
                modelAndDataset.Fit(true, true, true);
                ISample.Log.Info($"Model '{modelAndDataset.Model.ModelName}' trained in {swPercentageInTraining.Elapsed.TotalSeconds}");
            }
            if (retrainOnFullDataset)
            {
                var swRetrainOnFullDataset = Stopwatch.StartNew();
                using var modelAndDatasetOnFullDataset = ModelAndDatasetPredictions.LoadWithNewPercentageInTrainingNoKFold(1.0, workingDirectory, modelName, true); ;
                Model.Log.Info($"Training Model '{modelAndDatasetOnFullDataset.Model.ModelName}' (= Model '{modelName}' on full Dataset no KFold)");
                modelAndDatasetOnFullDataset.Fit(true, true, true);
                ISample.Log.Info($"Model '{modelAndDatasetOnFullDataset.Model.ModelName}' trained in {swRetrainOnFullDataset.Elapsed.TotalSeconds}");
            }
            ISample.Log.Info($"Model {modelName} retrained in {sw.Elapsed.TotalSeconds}");
            //KaggleDaysDatasetSample.Enrich(@"C:\Projects\Challenges\KaggleDays\catboost\a_KFOLD_modelformat_predict_test_.csv"); return;
        }
    }
}