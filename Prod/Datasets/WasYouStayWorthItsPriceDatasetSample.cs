using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.CatBoost;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Models;
using SharpNet.Networks;
using SharpNet.TextPreprocessing;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets;


public class KaggleDaysDatasetSample : AbstractDatasetSample
{
    #region private fields
    private const string NAME = "KaggleDays";
    private static readonly DataFrame xytrain_string_df;
    private static readonly DataFrame xtest_string_df;
    private static readonly ConcurrentDictionary<string, Tuple<DataSetV2, DataSetV2, DatasetEncoder>> CacheDataset = new();
    #endregion


    // ReSharper disable once UnusedMember.Global
    public static void CreateEnrichedDataSet()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, $"{nameof(CreateEnrichedDataSet)}");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, $"{nameof(CreateEnrichedDataSet)}");
        var sw = Stopwatch.StartNew();

        /*
        #region Creating Embedding file using Tf Idf
        var reviewsTfIdfEncodedRawFile = Path.Combine(WorkingDirectory, "reviews_tfidf_encoded_normalized_" + DateTime.Now.Ticks + ".csv");
        var rawReviewsFile = Path.Combine(DataDirectory, "reviews.csv");
        var review_file = DataFrame.read_string_csv(rawReviewsFile)["listing_id", "renters_comments"]
            .RenameInPlace("listing_id", "id");
        //review_file = review_file.TfIdfEncode("property_4", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true)
        //            .TfIdfEncode("property_5", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true)
        //            .TfIdfEncode("property_7", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true);
        var encoded_review_df = review_file
            .TfIdfEncode("renters_comments", 200, norm:TfIdfEncoding.TfIdfEncoding_norm.L2, scikitLearnCompatibilityMode:false)
            .AverageBy("id");
        encoded_review_df.to_csv(reviewsTfIdfEncodedRawFile);
        Model.Log.Info($"elapsed fo encoding: {sw.Elapsed.Seconds}s");
        #endregion
        */
        #region Embedding done by Sentence Transformers 
        var reviewsTfIdfEncodedRawFile = Path.Combine(DataDirectory, "reviews_embedded_transformers_avg.csv");
        #endregion

        var fullReviewsForEmbeddingDim = DataFrame.read_csv(reviewsTfIdfEncodedRawFile, true, c => (c == "listing_id" ? typeof(string) : typeof(float)))
            .RenameInPlace("listing_id", "id");
        //fullReviewsForEmbeddingDim = fullReviewsForEmbeddingDim.ReduceFloatDimension(TOTAL_Reviews_EmbeddingDim / 4);

        var trainDf = DataFrame.read_string_csv(RawTrainFile);
        var testDf = DataFrame.read_string_csv(RawTestFile);
        var res = new List<DataFrame> { trainDf, testDf };

        res[0] = res[0].LeftJoinWithoutDuplicates(fullReviewsForEmbeddingDim, "id");
        res[1] = res[1].LeftJoinWithoutDuplicates(fullReviewsForEmbeddingDim, "id");
        res[0].to_csv(RawTrainFile + FILE_EXT);
        res[1].to_csv(RawTestFile + FILE_EXT);

        Model.Log.Info($"elapsed total: {sw.Elapsed.Seconds}s");

    }


    static WasYouStayWorthItsPriceDatasetSample()
    {
        xytrain_string_df = DataFrame.read_string_csv(XYTrainRawFile);
        xtest_string_df = DataFrame.read_string_csv(XTestRawFile);
    }
    private WasYouStayWorthItsPriceDatasetSample() : base(new HashSet<string>())
    {
    }

    #region Hyper-Parameters
    // ReSharper disable once UnusedMember.Global
    // ReSharper disable once MemberCanBePrivate.Global
    // ReSharper disable once FieldCanBeMadeReadOnly.Global
    public string WasYouStayWorthItsPriceDatasetSample_Version = "v1";
    /// <summary>
    /// the embedding dim to use to enrich the dataset with the reviews
    /// </summary>
    // ReSharper disable once MemberCanBePrivate.Global
    public int Reviews_EmbeddingDim = TOTAL_Reviews_EmbeddingDim;
    #endregion

    public override int NumClass => 7;

    public const int TOTAL_Reviews_EmbeddingDim = 200; //Tf Idf
    //public const int TOTAL_Reviews_EmbeddingDim = 384; // Sentence Transformers

    // ReSharper disable once UnusedMember.Global
    public static void TrainNetwork(int numEpochs = 15, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"KFold", 2},
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            {"InitialLearningRate", AbstractHyperParameterSearchSpace.Range(0.003f, 0.2f, AbstractHyperParameterSearchSpace.range_type.normal)},

            //dataset 
            //{"StandardizeDoubleValues", new[]{true, false} },
            //{"Reviews_EmbeddingDim", new[]{0, 100, TOTAL_Reviews_EmbeddingDim}},
            
            // Optimizer 
            {"OptimizerType", "AdamW"},
            //{"AdamW_L2Regularization", AbstractHyperParameterSearchSpace.Range(0.003f, 0.01f)},
            {"AdamW_L2Regularization", 0.004},

            //{"LossFunction", "Rmse"}, //Mse , Mae
            //{"LossFunction", "BinaryCrossentropy"},
            {"LossFunction", "CategoricalCrossentropy"},

            // Learning Rate Scheduler
            {"LearningRateSchedulerType", new[]{ "OneCycle"}},

            { "EmbeddingDim", new[]{10} },
            //{ "EmbeddingDim", 10 },

            //{"dropout_top", 0.1},
            //{"dropout_mid", 0.3},
            //{"dropout_bottom", 0},

            //run on GPU
            {"NetworkSample_1DCNN_UseGPU", true},

            {"BatchSize", new[]{256} },

            //{"two_stage", new[]{true,false } },
            //{"Use_ConcatenateLayer", new[]{true,false } },
            //{"Use_AddLayer", new[]{true,false } },

            {"two_stage", true },
            {"Use_ConcatenateLayer", false },
            {"Use_AddLayer", true },


            {"NumEpochs", numEpochs},
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new NetworkSample_1DCNN(), new WasYouStayWorthItsPriceDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    // ReSharper disable once UnusedMember.Global
    public static void LaunchCatBoostHPO(int iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        // ReSharper disable once ConvertToConstant.Local
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"KFold", 2},
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            //related to CatBoost model
            { "logging_level", "Silent"},
            { "allow_writing_files",false},
            { "thread_count",1},
            { "iterations", iterations },
            { "od_type", "Iter"},
            { "od_wait",iterations/10},
            { "depth", AbstractHyperParameterSearchSpace.Range(2, 7) },
            //{ "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.10f)},
            { "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
            //{ "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
            //{ "l2_leaf_reg",AbstractHyperParameterSearchSpace.Range(0f, 10f)},
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new WasYouStayWorthItsPriceDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    public const string FILE_EXT = "_tfidf_l2_norm_scikit_stem_allstopwords.csv";
    //public const string FILE_EXT = "_sentence_transformers.csv";
    //public const string FILE_EXT = "_sentence_transformers_reduce_192.csv";
    //public const string FILE_EXT = "_sentence_transformers_reduce_96.csv";

    // ReSharper disable once UnusedMember.Global
    public static (ISample bestSample, IScore bestScore) LaunchLightGBMHPO(int min_num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"KFold", 2},
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            //related to Dataset 
            {"Reviews_EmbeddingDim", TOTAL_Reviews_EmbeddingDim},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            //!D {"KFold", new[]{2}},
            

            {"boosting", new []{"gbdt", "dart"}},
            //dart mode
            //{"drop_rate", new[]{0.05, 0.1, 0.2}},
            {"max_drop", new[]{40, 50, 60}},
            {"skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},
            
            //related to LightGBM model
            //{ "metric", "multi_logloss" },
            //{ "objective", "multiclass" },
            //{ "num_iterations", AbstractHyperParameterSearchSpace.Range(min_num_iterations, 3*min_num_iterations) },
            { "num_iterations", min_num_iterations },
            //{ "num_class", numClasses },
            { "verbosity", "0" },
            { "num_threads", 1},
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.2f)},
            { "extra_trees", false },
            { "early_stopping_round", min_num_iterations/10 },
            { "bagging_fraction", new[]{0.9f, 1.0f} },
            { "bagging_freq", new[]{0, 1} },
            { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.3f, 1.0f)},
            //{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            { "max_depth", new[]{10, 20, 50, 100, 255} },
            //{ "min_data_in_bin", AbstractHyperParameterSearchSpace.Range(3, 100) },
            { "min_data_in_bin", new[]{3, 10, 100, 150}  },
            //{ "min_data_in_leaf", AbstractHyperParameterSearchSpace.Range(20, 200) },
            //{ "min_data_in_leaf", new[]{10, 20, 30} },
            //{ "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(5, 100) },
            //{ "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) },
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), new WasYouStayWorthItsPriceDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
        return (hpo.BestSampleFoundSoFar, hpo.ScoreOfBestSampleFoundSoFar);
    }

    public override Objective_enum GetObjective() => Objective_enum.Classification;
    public override EvaluationMetricEnum GetRankingEvaluationMetric() => EvaluationMetricEnum.F1Micro;
    public override IScore MinimumScoreToSaveModel => new Score(0.32f, GetRankingEvaluationMetric());
    public override string[] CategoricalFeatures => new[] { "host_2", "host_3", "host_4", "host_5", "property_10", "property_15", "property_4", "property_5", "property_7" };
    public override string[] IdColumns => new[] { "id" };
    public override string[] TargetLabels => new[] { "max_rating_class" };
    public override DataSet TestDataset()
    {
        return LoadAndEncodeDataset_If_Needed().testDataset;
    }

    public override DataSetV2 FullTrainingAndValidation()
    {
        return LoadAndEncodeDataset_If_Needed().fullTrainingAndValidation;
    }

    private (DataSetV2 fullTrainingAndValidation, DataSetV2 testDataset) LoadAndEncodeDataset_If_Needed()
    {
        var key = ComputeHash();
        if (CacheDataset.TryGetValue(key, out var result))
        {
            DatasetEncoder = result.Item3;
            return (result.Item1, result.Item2);
        }
        DatasetEncoder = new DatasetEncoder(this, StandardizeDoubleValues);

        var xyTrain = UpdateFeatures(xytrain_string_df.Clone());
        var xtest = UpdateFeatures(xtest_string_df.Clone());
        DatasetEncoder.Fit(xyTrain);
        DatasetEncoder.Fit(xtest);

        var xTrain_Encoded = DatasetEncoder.Transform(xyTrain.Drop(TargetLabels));
        var yTrain_Encoded = DatasetEncoder.Transform(xyTrain[TargetLabels]);
        var xtest_Encoded = DatasetEncoder.Transform(xtest);

        var fullTrainingAndValidation = new DataSetV2(this, xTrain_Encoded, yTrain_Encoded, false);
        var testDataset = new DataSetV2(this, xtest_Encoded, null, false);

        CacheDataset.TryAdd(key, Tuple.Create(fullTrainingAndValidation, testDataset, DatasetEncoder));
        return (fullTrainingAndValidation, testDataset);
    }

    private DataFrame UpdateFeatures(DataFrame x)
    {
        var columnToDrop = new List<string>();
        columnToDrop.AddRange(TfIdfEncoding.ColumnToRemoveToFitEmbedding(x, "renters_comments", Reviews_EmbeddingDim, true));
        if (columnToDrop.Count == 0)
        {
            return x;
        }
        return x.DropIgnoreErrors(columnToDrop.ToArray());
    }

    private static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    private static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    private static string XYTrainRawFile => Path.Combine(DataDirectory, "train.csv" + FILE_EXT); //!D
    private static string XTestRawFile => Path.Combine(DataDirectory, "test.csv" + FILE_EXT); //!D
    private static string RawTrainFile => Path.Combine(DataDirectory, "train.csv");
    private static string RawTestFile => Path.Combine(DataDirectory, "test.csv");


}


public class WasYouStayWorthItsPriceDatasetSample : AbstractDatasetSample
{
    #region private fields
    private const string NAME = "WasYouStayWorthItsPrice";
    private static readonly DataFrame xytrain_string_df;
    private static readonly DataFrame xtest_string_df;
    private static readonly ConcurrentDictionary<string, Tuple<DataSetV2, DataSetV2, DatasetEncoder>> CacheDataset = new();
    #endregion


    // ReSharper disable once UnusedMember.Global
    public static void CreateEnrichedDataSet()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, $"{nameof(CreateEnrichedDataSet)}");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, $"{nameof(CreateEnrichedDataSet)}");
        var sw = Stopwatch.StartNew();

        /*
        #region Creating Embedding file using Tf Idf
        var reviewsTfIdfEncodedRawFile = Path.Combine(WorkingDirectory, "reviews_tfidf_encoded_normalized_" + DateTime.Now.Ticks + ".csv");
        var rawReviewsFile = Path.Combine(DataDirectory, "reviews.csv");
        var review_file = DataFrame.read_string_csv(rawReviewsFile)["listing_id", "renters_comments"]
            .RenameInPlace("listing_id", "id");
        //review_file = review_file.TfIdfEncode("property_4", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true)
        //            .TfIdfEncode("property_5", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true)
        //            .TfIdfEncode("property_7", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true);
        var encoded_review_df = review_file
            .TfIdfEncode("renters_comments", 200, norm:TfIdfEncoding.TfIdfEncoding_norm.L2, scikitLearnCompatibilityMode:false)
            .AverageBy("id");
        encoded_review_df.to_csv(reviewsTfIdfEncodedRawFile);
        Model.Log.Info($"elapsed fo encoding: {sw.Elapsed.Seconds}s");
        #endregion
        */
        #region Embedding done by Sentence Transformers 
        var reviewsTfIdfEncodedRawFile = Path.Combine(DataDirectory, "reviews_embedded_transformers_avg.csv");
        #endregion

        var fullReviewsForEmbeddingDim = DataFrame.read_csv(reviewsTfIdfEncodedRawFile, true, c => (c == "listing_id" ? typeof(string) : typeof(float)))
            .RenameInPlace("listing_id", "id");
        //fullReviewsForEmbeddingDim = fullReviewsForEmbeddingDim.ReduceFloatDimension(TOTAL_Reviews_EmbeddingDim / 4);

        var trainDf = DataFrame.read_string_csv(RawTrainFile);
        var testDf = DataFrame.read_string_csv(RawTestFile);
        var res = new List<DataFrame> { trainDf, testDf };

        res[0] = res[0].LeftJoinWithoutDuplicates(fullReviewsForEmbeddingDim, "id");
        res[1] = res[1].LeftJoinWithoutDuplicates(fullReviewsForEmbeddingDim, "id");
        res[0].to_csv(RawTrainFile + FILE_EXT);
        res[1].to_csv(RawTestFile + FILE_EXT);

        Model.Log.Info($"elapsed total: {sw.Elapsed.Seconds}s");

    }


    static WasYouStayWorthItsPriceDatasetSample()
    {
        xytrain_string_df = DataFrame.read_string_csv(XYTrainRawFile);
        xtest_string_df = DataFrame.read_string_csv(XTestRawFile);
    }
    private WasYouStayWorthItsPriceDatasetSample() : base(new HashSet<string>())
    {
    }

    #region Hyper-Parameters
    // ReSharper disable once UnusedMember.Global
    // ReSharper disable once MemberCanBePrivate.Global
    // ReSharper disable once FieldCanBeMadeReadOnly.Global
    public string WasYouStayWorthItsPriceDatasetSample_Version = "v1";
    /// <summary>
    /// the embedding dim to use to enrich the dataset with the reviews
    /// </summary>
    // ReSharper disable once MemberCanBePrivate.Global
    public int Reviews_EmbeddingDim = TOTAL_Reviews_EmbeddingDim;
    #endregion

    public override int NumClass => 7;

    public const int TOTAL_Reviews_EmbeddingDim = 200; //Tf Idf
    //public const int TOTAL_Reviews_EmbeddingDim = 384; // Sentence Transformers

    // ReSharper disable once UnusedMember.Global
    public static void TrainNetwork(int numEpochs = 15, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"KFold", 2},
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            {"InitialLearningRate", AbstractHyperParameterSearchSpace.Range(0.003f, 0.2f, AbstractHyperParameterSearchSpace.range_type.normal)},

            //dataset 
            //{"StandardizeDoubleValues", new[]{true, false} },
            //{"Reviews_EmbeddingDim", new[]{0, 100, TOTAL_Reviews_EmbeddingDim}},
            
            // Optimizer 
            {"OptimizerType", "AdamW"},
            //{"AdamW_L2Regularization", AbstractHyperParameterSearchSpace.Range(0.003f, 0.01f)},
            {"AdamW_L2Regularization", 0.004},

            //{"LossFunction", "Rmse"}, //Mse , Mae
            //{"LossFunction", "BinaryCrossentropy"},
            {"LossFunction", "CategoricalCrossentropy"},

            // Learning Rate Scheduler
            {"LearningRateSchedulerType", new[]{ "OneCycle"}},

            { "EmbeddingDim", new[]{10} },
            //{ "EmbeddingDim", 10 },

            //{"dropout_top", 0.1},
            //{"dropout_mid", 0.3},
            //{"dropout_bottom", 0},

            //run on GPU
            {"NetworkSample_1DCNN_UseGPU", true},

            {"BatchSize", new[]{256} },

            //{"two_stage", new[]{true,false } },
            //{"Use_ConcatenateLayer", new[]{true,false } },
            //{"Use_AddLayer", new[]{true,false } },

            {"two_stage", true },
            {"Use_ConcatenateLayer", false },
            {"Use_AddLayer", true },


            {"NumEpochs", numEpochs},
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new NetworkSample_1DCNN(), new WasYouStayWorthItsPriceDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    // ReSharper disable once UnusedMember.Global
    public static void LaunchCatBoostHPO(int iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        // ReSharper disable once ConvertToConstant.Local
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"KFold", 2},
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            //related to CatBoost model
            { "logging_level", "Silent"},
            { "allow_writing_files",false},
            { "thread_count",1},
            { "iterations", iterations },
            { "od_type", "Iter"},
            { "od_wait",iterations/10},
            { "depth", AbstractHyperParameterSearchSpace.Range(2, 7) },
            //{ "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.10f)},
            { "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
            //{ "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
            //{ "l2_leaf_reg",AbstractHyperParameterSearchSpace.Range(0f, 10f)},
        };
        
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new WasYouStayWorthItsPriceDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }
    

    public const string FILE_EXT = "_tfidf_l2_norm_scikit_stem_allstopwords.csv";
    //public const string FILE_EXT = "_sentence_transformers.csv";
    //public const string FILE_EXT = "_sentence_transformers_reduce_192.csv";
    //public const string FILE_EXT = "_sentence_transformers_reduce_96.csv";

    // ReSharper disable once UnusedMember.Global
    public static (ISample bestSample, IScore bestScore) LaunchLightGBMHPO(int min_num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"KFold", 2},
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            //related to Dataset 
            {"Reviews_EmbeddingDim", TOTAL_Reviews_EmbeddingDim},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            //!D {"KFold", new[]{2}},
            

            {"boosting", new []{"gbdt", "dart"}},
            //dart mode
            //{"drop_rate", new[]{0.05, 0.1, 0.2}},
            {"max_drop", new[]{40, 50, 60}},
            {"skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},
            
            //related to LightGBM model
            //{ "metric", "multi_logloss" },
            //{ "objective", "multiclass" },
            //{ "num_iterations", AbstractHyperParameterSearchSpace.Range(min_num_iterations, 3*min_num_iterations) },
            { "num_iterations", min_num_iterations },
            //{ "num_class", numClasses },
            { "verbosity", "0" },
            { "num_threads", 1},
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.2f)},
            { "extra_trees", false },
            { "early_stopping_round", min_num_iterations/10 },
            { "bagging_fraction", new[]{0.9f, 1.0f} },
            { "bagging_freq", new[]{0, 1} },
            { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.3f, 1.0f)},
            //{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            { "max_depth", new[]{10, 20, 50, 100, 255} },
            //{ "min_data_in_bin", AbstractHyperParameterSearchSpace.Range(3, 100) },
            { "min_data_in_bin", new[]{3, 10, 100, 150}  },
            //{ "min_data_in_leaf", AbstractHyperParameterSearchSpace.Range(20, 200) },
            //{ "min_data_in_leaf", new[]{10, 20, 30} },
            //{ "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(5, 100) },
            //{ "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) },
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), new WasYouStayWorthItsPriceDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
        return (hpo.BestSampleFoundSoFar, hpo.ScoreOfBestSampleFoundSoFar);
    }

    public override Objective_enum GetObjective() => Objective_enum.Classification;
    public override EvaluationMetricEnum GetRankingEvaluationMetric() => EvaluationMetricEnum.F1Micro;
    public override  IScore MinimumScoreToSaveModel => new Score(0.32f, GetRankingEvaluationMetric());
    public override string[] CategoricalFeatures => new [] { "host_2", "host_3", "host_4", "host_5", "property_10", "property_15", "property_4", "property_5", "property_7"};
    public override string[] IdColumns => new [] { "id" };
    public override string[] TargetLabels => new[] { "max_rating_class" };
    public override DataSet TestDataset()
    {
        return LoadAndEncodeDataset_If_Needed().testDataset;
    }

    public override DataSetV2 FullTrainingAndValidation()
    {
        return LoadAndEncodeDataset_If_Needed().fullTrainingAndValidation;
    }

    private (DataSetV2 fullTrainingAndValidation, DataSetV2 testDataset) LoadAndEncodeDataset_If_Needed()
    {
        var key = ComputeHash();
        if ( CacheDataset.TryGetValue(key, out var result))
        {
            DatasetEncoder = result.Item3;
            return (result.Item1, result.Item2);
        }
        DatasetEncoder = new DatasetEncoder(this, StandardizeDoubleValues);
        
        var xyTrain = UpdateFeatures(xytrain_string_df.Clone());
        var xtest = UpdateFeatures(xtest_string_df.Clone());
        DatasetEncoder.Fit(xyTrain);
        DatasetEncoder.Fit(xtest);

        var xTrain_Encoded = DatasetEncoder.Transform(xyTrain.Drop(TargetLabels));
        var yTrain_Encoded = DatasetEncoder.Transform(xyTrain[TargetLabels]);
        var xtest_Encoded = DatasetEncoder.Transform(xtest);

        var fullTrainingAndValidation = new DataSetV2(this, xTrain_Encoded, yTrain_Encoded, false);
        var testDataset = new DataSetV2(this, xtest_Encoded, null, false);

        CacheDataset.TryAdd(key, Tuple.Create(fullTrainingAndValidation, testDataset, DatasetEncoder));
        return (fullTrainingAndValidation, testDataset);
    }

    private DataFrame UpdateFeatures(DataFrame x)
    {
        var columnToDrop = new List<string>();
        columnToDrop.AddRange(TfIdfEncoding.ColumnToRemoveToFitEmbedding(x, "renters_comments", Reviews_EmbeddingDim, true));
        if (columnToDrop.Count == 0)
        {
            return x;
        }
        return x.DropIgnoreErrors(columnToDrop.ToArray());
    }
   
    private static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    private static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    private static string XYTrainRawFile => Path.Combine(DataDirectory, "train.csv"+ FILE_EXT); //!D
    private static string XTestRawFile => Path.Combine(DataDirectory, "test.csv"+ FILE_EXT); //!D
    private static string RawTrainFile => Path.Combine(DataDirectory, "train.csv");
    private static string RawTestFile => Path.Combine(DataDirectory, "test.csv");

    
}
