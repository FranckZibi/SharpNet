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


    static KaggleDaysDatasetSample()
    {
        xytrain_string_df = DataFrame.read_string_csv(XYTrainRawFile);
        xtest_string_df = DataFrame.read_string_csv(XTestRawFile);
    }
    private KaggleDaysDatasetSample() : base(new HashSet<string>())
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

    private const int TOTAL_Reviews_EmbeddingDim = 200; //Tf Idf
    //public const int TOTAL_Reviews_EmbeddingDim = 384; // Sentence Transformers

    // ReSharper disable once UnusedMember.Global
    private const string FILE_EXT = "_tfidf_l2_norm_scikit_stem_allstopwords.csv";

    public override Objective_enum GetObjective() => Objective_enum.Classification;
    public override EvaluationMetricEnum GetRankingEvaluationMetric() => EvaluationMetricEnum.Mse;
    //public override IScore MinimumScoreToSaveModel => new Score(0.32f, GetRankingEvaluationMetric());
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


    // ReSharper disable once UnusedMember.Global
    public static void TrainNetwork(int numEpochs = 15, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"KFold", 2},
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            //uncomment appropriate one
            //{"LossFunction", "Rmse"},                     //for Regression Tasks: Rmse, Mse, Mae, etc.
            //{"LossFunction", "BinaryCrossentropy"},       //for binary classification
            //{"LossFunction", "CategoricalCrossentropy"},  //for multi class classification

            // Optimizer 
            { "OptimizerType", new[] { "AdamW", "SGD", "Adam" /*, "VanillaSGD", "VanillaSGDOrtho"*/ } },
            { "AdamW_L2Regularization", new[] { 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 } },
            { "SGD_usenesterov", new[] { true, false } },
            { "lambdaL2Regularization", new[] { 0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 } },

            // Learning Rate
            { "InitialLearningRate", AbstractHyperParameterSearchSpace.Range(1e-5f, 1f, AbstractHyperParameterSearchSpace.range_type.normal) },
            // Learning Rate Scheduler
            { "LearningRateSchedulerType", new[] { "CyclicCosineAnnealing", "OneCycle", "Linear" } },
            { "EmbeddingDim", new[] { 0, 4, 8, 12 } },
            //{"weight_norm", new[]{true, false}},
            //{"leaky_relu", new[]{true, false}},
            { "dropout_top", new[] { 0, 0.1, 0.2 } },
            { "dropout_mid", new[] { 0, 0.3, 0.5 } },
            { "dropout_bottom", new[] { 0, 0.2, 0.4 } },
            { "BatchSize", new[] { 256, 512, 1024, 2048 } },
            { "NumEpochs", new[] { numEpochs } },

        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new NetworkSample_1DCNN(), new KaggleDaysDatasetSample()), WorkingDirectory);
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

            //uncomment appropriate one
            //{"loss_function", "RMSE"},          //for Regression Tasks: RMSE, etc.
            //{"loss_function", "Logloss"},     //for binary classification
            //{"loss_function", "MultiClass"},  //for multi class classification

            { "logging_level", "Silent"},
            { "allow_writing_files",false},
            { "thread_count",1},
            { "iterations", iterations },
            { "od_type", "Iter"},
            { "od_wait",iterations/10},
            { "depth", AbstractHyperParameterSearchSpace.Range(2, 10) },
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 1.00f)},
            { "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
            { "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
            { "l2_leaf_reg",AbstractHyperParameterSearchSpace.Range(0, 10)},
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new KaggleDaysDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    // ReSharper disable once UnusedMember.Global
    public static (ISample bestSample, IScore bestScore) LaunchLightGBMHPO(int num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
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
            

            //uncomment appropriate one
            //{"objective", "regression"},      //for Regression Tasks
            //{"objective", "binary"},          //for binary classification
            //{"objective", "multiclass"},      //for multi class classification
            //{"num_class", number_of_class },  //for multi class classification

            //high priority
            { "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
            { "bagging_freq", new[]{0, 1} },
            { "boosting", new []{"gbdt", "dart"}},
            { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.3f, 1.0f)},
            { "early_stopping_round", num_iterations/10 },
            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.005f, 0.2f)},
            { "max_depth", new[]{10, 20, 50, 100, 255} },
            { "min_data_in_leaf", new[]{20, 50 /*,100*/} },
            { "num_iterations", num_iterations },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 50) },
            { "num_threads", 1},
            { "verbosity", "0" },

            //medium priority
            { "drop_rate", new[]{0.05, 0.1, 0.2}},                               //specific to dart mode
            { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "min_data_in_bin", new[]{3, 10, 100, 150}  },
            { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            { "max_drop", new[]{40, 50, 60}},                                   //specific to dart mode
            { "skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},  //specific to dart mode

            //low priority
            { "extra_trees", new[] { true , false } }, //low priority 
            //{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)}, //very low priority
            { "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) }, //low priority
            { "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), new KaggleDaysDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
        return (hpo.BestSampleFoundSoFar, hpo.ScoreOfBestSampleFoundSoFar);
    }
}
