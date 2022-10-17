using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpNet.CatBoost;
using SharpNet.Data;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;

namespace SharpNet.Datasets;

public class WasYouStayWorthItsPriceDatasetSample : AbstractDatasetSample
{
    #region private fields
    private const string NAME = "WasYouStayWorthItsPrice";
    private static readonly object LockObject = new();
    private static DatasetEncoder _trainTestEncoder;
    private static InMemoryDataSet _fullTrainDatasetEncoded;
    private static InMemoryDataSet _fullTestDatasetEncoded;
    private static DataFrame _fullReviewEncoded;
    #endregion

    private WasYouStayWorthItsPriceDatasetSample() : base(new HashSet<string>())
    {
        //var s2 = Stopwatch.StartNew();
        lock (LockObject)
        {
            if (_trainTestEncoder == null)
            {
                _trainTestEncoder = new DatasetEncoder(this);
                _fullTrainDatasetEncoded = _trainTestEncoder.NumericalEncoding(XYTrainRawFile);
                _fullTestDatasetEncoded = _trainTestEncoder.NumericalEncoding(XTestRawFile, null);
            }
            WasYouStayWorthItsPriceDatasetSample_NumClasses = _trainTestEncoder.NumClasses();
            if (!File.Exists(ReviewsTfIdfEncodedRawFile))
            {
                var encoded_review_df = DataFrame.read_string_csv(ReviewsRawFile)["listing_id", "renters_comments"]
                    .RenameInPlace("listing_id", "id")
                    .TfIdfEncoding("renters_comments", 1000)
                    .AverageBy("id");
                encoded_review_df.to_csv(ReviewsTfIdfEncodedRawFile);
            }
            if (_fullReviewEncoded == null)
            {
                _fullReviewEncoded = _trainTestEncoder.NumericalEncoding(ReviewsTfIdfEncodedRawFile, null).XDataFrame;
            }
        }


        //_fullTrainDatasetEncoded.to_csv(@"C:\Projects\Challenges\WasYouStayWorthItsPrice\yx_train.csv", GetSeparator(), true, true);
        //_fullTestDatasetEncoded.to_csv(@"C:\Projects\Challenges\WasYouStayWorthItsPrice\x_test.csv", GetSeparator(), false, true);

        //_reviewsEncoder = new DatasetEncoder(new List<string>{"id", "listing_id", "renters_comments"}, new List<string>{"id"}, new List<string>());
        //_reviewEncoded = _reviewsEncoder.NumericalEncoding(ReviewsRawFile);
        //_xTrainEncoded.T
    }

    #region Hyper-Parameters
    // ReSharper disable once UnusedMember.Global
    // ReSharper disable once MemberCanBePrivate.Global
    // ReSharper disable once FieldCanBeMadeReadOnly.Global
    public int WasYouStayWorthItsPriceDatasetSample_NumClasses;
    /// <summary>
    /// the embedding dim to use to enrich the dataset with the reviews
    /// </summary>
    // ReSharper disable once MemberCanBePrivate.Global
    public int Reviews_EmbeddingDim = 200;
    #endregion

    public static void WeightOptimizer()
    {
        const string submissionWorkingDirectory = @"C:\Projects\Challenges\WasYouStayWorthItsPrice\submission";
        List<Tuple<string, string>> workingDirectoryAndModelNames = new()
        {
            Tuple.Create(submissionWorkingDirectory, "56E668E7DB_FULL"),
            Tuple.Create(submissionWorkingDirectory, "8CF93D9FA0_FULL"),
            Tuple.Create(submissionWorkingDirectory, "90840F212D_FULL"),
            Tuple.Create(submissionWorkingDirectory, "E72AD5B74B_FULL"),
            Tuple.Create(submissionWorkingDirectory, "66B4F3653A_FULL"),
            Tuple.Create(submissionWorkingDirectory, "395B343296_FULL"),
        };
        var searchWorkingDirectory = Path.Combine(submissionWorkingDirectory, "search");
        WeightsOptimizer.SearchForBestWeights(workingDirectoryAndModelNames, searchWorkingDirectory, null);
    }
    // ReSharper disable once UnusedMember.Global
    public static void LaunchCatBoostHPO()
    {
        // ReSharper disable once ConvertToConstant.Local
        var iterations = 100;
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            //{"KFold", new[]{1,3}},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            //related to CatBoost model
            { "loss_function", "MultiClass"},
            { "logging_level", "Silent"},
            { "allow_writing_files",false},
            { "thread_count",1},
            { "iterations", iterations },
            { "od_type", "Iter"},
            { "od_wait",iterations/10},
            { "depth", AbstractHyperParameterSearchSpace.Range(2, 5) },
            //{ "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.10f)},
            { "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
            //{ "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
            //{ "l2_leaf_reg",AbstractHyperParameterSearchSpace.Range(0f, 10f)},
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new WasYouStayWorthItsPriceDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        var csvPath = Path.Combine(DataDirectory, "Tests_" + NAME + ".csv");
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, csvPath, ref bestScoreSoFar));
    }
    public static (ISample bestSample, IScore bestScore) LaunchLightGBMHPO(int min_num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        // ReSharper disable once ConvertToConstant.Local

        var numClasses = new WasYouStayWorthItsPriceDatasetSample().WasYouStayWorthItsPriceDatasetSample_NumClasses;

        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            //{"KFold", new[]{1,3}},
            {"KFold", new[]{1}},

            {"boosting", new []{"gbdt", "dart"}},
            //dart mode
            //{"drop_rate", new[]{0.05, 0.1, 0.2}},
            {"max_drop", new[]{40, 50, 60}},
            {"skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},
            //{"xgboost_dart_mode", new[]{false, true}},
            //{"uniform_drop", new[]{false, true}},
            //{"drop_seed", 4},

            //related to LightGBM model
            { "metric", "multi_logloss" },
            { "objective", "multiclass" },
            //{ "num_iterations", AbstractHyperParameterSearchSpace.Range(min_num_iterations, 3*min_num_iterations) },
            { "num_iterations", min_num_iterations },
            { "num_class", numClasses },
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
        var csvPath = Path.Combine(DataDirectory, "Tests_" + NAME + ".csv");
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, csvPath, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
        return (hpo.BestSampleFoundSoFar, hpo.ScoreOfBestSampleFoundSoFar);
    }

    public override Objective_enum GetObjective() => Objective_enum.Classification;
    protected override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        return EvaluationMetricEnum.F1Micro;
    }
    public override string[] CategoricalFeatures => new [] { "host_2", "host_3", "host_4", "host_5", "property_10", "property_15", "property_4", "property_5", "property_7"};
    public override string[] IdColumns => new [] { "id" };
    public override string[] TargetLabels => new[] { "max_rating_class" };
    public override IDataSet TestDataset()
    {
        return EnrichIfNeeded(_fullTestDatasetEncoded);
    }
    public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
    {
        var enrichFullTrain = EnrichIfNeeded(_fullTrainDatasetEncoded);
        int rowsForTraining = (int)(PercentageInTraining * enrichFullTrain.Count + 0.1);
        rowsForTraining -= rowsForTraining % DatasetRowsInModelFormatMustBeMultipleOf();
        return enrichFullTrain.IntSplitIntoTrainingAndValidation(rowsForTraining);

    }
    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat_with_IdColumns)
    {
        var cpuTensor_InTargetFormat = predictionsInModelFormat_with_IdColumns.Drop(IdColumns).FloatCpuTensor().ArgMax();
        var probaDataFrame_InTargetFormat = DataFrame.New(cpuTensor_InTargetFormat, TargetLabels);
        return DataFrame.MergeHorizontally(predictionsInModelFormat_with_IdColumns[IdColumns], probaDataFrame_InTargetFormat);
    }
    public override void SavePredictionsInTargetFormat(DataFrame encodedPredictionsInTargetFormat, string path)
    {
        var df_decoded = _trainTestEncoder.NumericalDecoding(encodedPredictionsInTargetFormat);
        df_decoded.to_csv(path, GetSeparator());
    }

    private static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    private static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    private static string XYTrainRawFile => Path.Combine(DataDirectory, "train.csv");
    private static string XTestRawFile => Path.Combine(DataDirectory, "test.csv");
    private static string ReviewsRawFile => Path.Combine(DataDirectory, "reviews.csv");
    private static string ReviewsTfIdfEncodedRawFile => Path.Combine(DataDirectory, "reviews_tfidf_encoded.csv");
    private InMemoryDataSet EnrichIfNeeded(InMemoryDataSet dataset)
    {
        if (Reviews_EmbeddingDim > 0)
        {
            var fullReviewsForEmbeddingDim = _fullReviewEncoded[_fullReviewEncoded.Columns.Take(IdColumns.Length + Reviews_EmbeddingDim).ToArray()];
            var xEnriched = dataset.XDataFrame.LeftJoinWithoutDuplicates(fullReviewsForEmbeddingDim, "id");
            return DatasetEncoder.NewInMemoryDataSet(xEnriched, dataset.Y, this);
        }
        else
        {
            return dataset;
        }
    }
}
