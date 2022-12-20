using System;
using System.Collections.Generic;
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
        const string workingDirectory = @"C:\Projects\Challenges\WasYouStayWorthItsPrice\sub2\";
        const string modelName = "D875D0F56C_KFOLD_FULL";
        SharpNet.Utils.ConfigureGlobalLog4netProperties(workingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        SharpNet.Utils.ConfigureThreadLog4netProperties(workingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        var m = ModelAndDatasetPredictions.Load(workingDirectory, modelName);
        m.ComputeAndSaveFeatureImportance();
    }

    /// <summary>
    /// normalize all CSV files in directory 'directory' and put the normalized files in sub directory 'subDirectory'
    /// </summary>
    /// <param name="directory"></param>
    /// <param name="hasHeader"></param>
    /// <param name="removeAccentedCharacters"></param>
    [TestCase(@"C:\Projects\Challenges\perfs\raw", true, true, "normalize"), Explicit]
    public void NormalizeAllCsvInDirectory(string directory, bool hasHeader, bool removeAccentedCharacters)
    {
        const string subDirectory = nameof(NormalizeAllCsvInDirectory);
        SharpNet.Utils.ConfigureGlobalLog4netProperties(directory, $"{nameof(NormalizeAllCsvInDirectory)}");
        SharpNet.Utils.ConfigureThreadLog4netProperties(directory, $"{nameof(NormalizeAllCsvInDirectory)}");
        foreach (var file in Directory.GetFiles(directory, "*.csv"))
        {
            DataFrame.Normalize(file, hasHeader, removeAccentedCharacters, subDirectory);
        }
    }


    [TestCase(new []{ @"C:\Projects\Challenges\perfs\raw\normalize\search_train.csv", @"C:\Projects\Challenges\perfs\raw\normalize\search_test.csv" }, true, true, "keyword", 300), Explicit]
    [TestCase(new []{ @"C:\Projects\Challenges\perfs\raw\normalize\item_info.csv"}, true, true, "name", 300)]
    public void TfIdfEncode(string[] csvFiles, bool hasHeader, bool isNormalized, string columnToEncode, int embeddingDim)
    {
        var keepEncodedColumnName = false;
        var reduceEmbeddingDimIfNeeded = false;
        var norm = TfIdfEncoding.TfIdfEncoding_norm.L2;
        var scikitLearnCompatibilityMode = false;


        const string subDirectory = nameof(TfIdfEncode);
        string directory = Path.GetDirectoryName(csvFiles[0])??"";
        string targetDirectory = Path.Combine(directory, subDirectory);
        if (!Directory.Exists(targetDirectory))
        {
            Directory.CreateDirectory(targetDirectory);
        }
        SharpNet.Utils.ConfigureGlobalLog4netProperties(directory, $"{nameof(TfIdfEncode)}");
        SharpNet.Utils.ConfigureThreadLog4netProperties(directory, $"{nameof(TfIdfEncode)}");

        var dfs = new List<DataFrame>();
        List<string> columnContent = new();
        foreach (var fileName in csvFiles)
        {
            ISample.Log.Info($"loading CSV file {fileName}");
            var df = DataFrame.read_string_csv(fileName, hasHeader, isNormalized);
            ISample.Log.Info($"extracting content of column {columnToEncode} in CSV file {fileName}");
            columnContent.AddRange(df.StringColumnContent(columnToEncode));
            dfs.Add(df);
        }
        columnContent.Sort();

        var df_ColumnToEncode = DataFrame.New(columnContent.ToArray(), new[] { columnToEncode });

        ISample.Log.Info($"Encoding column {columnToEncode}");
        var encoded_column_df = df_ColumnToEncode
            .TfIdfEncode(columnToEncode, embeddingDim, true, reduceEmbeddingDimIfNeeded, norm, scikitLearnCompatibilityMode)
            .AverageBy(columnToEncode);

        var encoded_column_df_path = Path.Combine(targetDirectory, columnToEncode+"_"+DateTime.Now.Ticks+".csv");
        ISample.Log.Info($"Encoded column file {encoded_column_df_path}");
        encoded_column_df.to_csv(encoded_column_df_path, ',', hasHeader);

        for (var index = 0; index < dfs.Count; index++)
        {
            var targetPath = Path.Combine(targetDirectory, Path.GetFileName(csvFiles[index]));
            ISample.Log.Info($"Creating encoded DataFrame for {csvFiles[index]} and saving it to {targetPath}");
            var df = dfs[index];
            var df2 = df.LeftJoinWithoutDuplicates(encoded_column_df, columnToEncode);
            if (!keepEncodedColumnName)
            {
                df2 = df2.Drop(columnToEncode);
            }
            df2.to_csv(targetPath, ',', hasHeader);
        }
        ISample.Log.Info($"All CSV files have been encoded and saved to directory {targetDirectory}");

        //foreach (var file in Directory.GetFiles(directory, "*.csv"))
        //{
        //    DataFrame.Normalize(file, hasHeader, removeAccentedCharacters, subDirectory);
        //}
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
        const string workingDirectory = @"C:\Projects\Challenges\Natixis70\aaa7\";
        var modelName = new[]
        {
            ////Natixis70
            "35CB39CB45_KFOLD", //LightGBM
            "CAD2F81CEB_KFOLD", //LightGBM
            "B222453FF6_KFOLD", //CatBoost
            "7598B6F2E4_KFOLD", //CatBoost

            //WasYouStayWorthItsPrice
            //"F2CC39BB32_KFOLD", //1D-CNN    F2CC39BB32_KFOLD_FULL_predict_test_0.31976.csv
            //"D875D0F56C_KFOLD", //LightGBM  D875D0F56C_KFOLD_FULL_predict_test_0.3204.csv
            //"3F2CB236DB_KFOLD", //CatBoost  3F2CB236DB_KFOLD_FULL_predict_test_0.31759.csv
            //"0EF01A90D8_KFOLD", //Deep Learning
            //"3580990008_KFOLD",
            //"395B343296_KFOLD", //LightGBM GBDT
            ////"48F31E6543_KFOLD",
            ////"56E668E7DB_KFOLD",
            //"66B4F3653A_KFOLD",
            //"8CF93D9FA0_KFOLD",
            //"90840F212D_KFOLD",
            ////"90DAFFB8FC_KFOLD",
            //"E72AD5B74B_KFOLD"
        };
        const bool use_features_in_secondary = true;
        const int cv = 2;

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

        if (!Directory.Exists(workingDirectory))
        {
            Directory.CreateDirectory(workingDirectory);
        }
        var dataDirectory = Path.Combine(workingDirectory, "Data");
        if (!Directory.Exists(dataDirectory))
        {
            Directory.CreateDirectory(dataDirectory);
        }

        var hpoWorkingDirectory = Path.Combine(workingDirectory, "hpo");
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), datasetSample), hpoWorkingDirectory); IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, hpoWorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    /// <summary>
    /// retrain some models 
    /// </summary>
    [Test, Explicit]
    public void Retrain(int n_splits = 3, bool retrainOnFullDataset = true)
    {

        foreach (var modelName in new[]
                {
                    "EE8A305CBE_KFOLD",
                   
                })
        {

            const string workingDirectory = @"C:\Projects\Challenges\KaggleDays";

            SharpNet.Utils.ConfigureGlobalLog4netProperties(workingDirectory, $"{nameof(Retrain)}");
            SharpNet.Utils.ConfigureThreadLog4netProperties(workingDirectory, $"{nameof(Retrain)}");

            var m = ModelAndDatasetPredictions.Load(workingDirectory, modelName);
            //var embeddedModel = mKfold.Model;
            var mKfoldModelAndDatasetPredictionsSample = m.ModelAndDatasetPredictionsSample;
            if (m.Model is not KFoldModel)
            {
                var embeddedModel = m.Model;
                var kfoldSample = new KFoldSample(n_splits, workingDirectory, embeddedModel.ModelName, embeddedModel.ModelSample.GetLoss(), mKfoldModelAndDatasetPredictionsSample.DatasetSample.DatasetRowsInModelFormatMustBeMultipleOf());
                var sample = new ModelAndDatasetPredictionsSample(new ISample[]
                {
                    kfoldSample,
                    mKfoldModelAndDatasetPredictionsSample.DatasetSample.CopyWithNewPercentageInTrainingAndKFold(1.0, kfoldSample.n_splits),
                    mKfoldModelAndDatasetPredictionsSample.PredictionsSample
                });
                m = new ModelAndDatasetPredictions(sample, workingDirectory, embeddedModel.ModelName + KFoldModel.SuffixKfoldModel);
            }


            var kfoldModelName = m.Model.ModelName;
            if (n_splits >= 2)
            {
                m.Model.Use_All_Available_Cores();
                m.Fit(true, true, true);
                m.Save(workingDirectory, kfoldModelName);
            }

            if (retrainOnFullDataset)
            {
                var sampleFullDataset = new ModelAndDatasetPredictionsSample(new []
                {
                    ((KFoldModel)m.Model).EmbeddedModelSample(0).Clone(),
                    mKfoldModelAndDatasetPredictionsSample.DatasetSample.CopyWithNewPercentageInTrainingAndKFold(1.0, 1),
                    new PredictionsSample()
                });
                var modelAndDatasetOnFullDataset = new ModelAndDatasetPredictions(sampleFullDataset, workingDirectory, kfoldModelName + "_FULL");
                Model.Log.Info($"Retraining Model '{modelName}' on full Dataset no KFold (Model on full Dataset name: {modelAndDatasetOnFullDataset.Model.ModelName})");
                modelAndDatasetOnFullDataset.Model.Use_All_Available_Cores();
                modelAndDatasetOnFullDataset.Fit(true, true, true);
            }
        }
    }
}