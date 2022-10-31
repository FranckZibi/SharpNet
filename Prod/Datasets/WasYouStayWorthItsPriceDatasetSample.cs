using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.CatBoost;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Models;
using SharpNet.TextPreprocessing;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets;

public class WasYouStayWorthItsPriceDatasetSample : AbstractDatasetSample
{
    #region private fields
    private const string NAME = "WasYouStayWorthItsPrice";
    private static readonly object LockObject = new();
    private static DatasetEncoder _trainTestEncoder;
    private static InMemoryDataSetV2 _fullTrainDatasetEncoded;
    private static InMemoryDataSetV2 _fullTestDatasetEncoded;
    #endregion


    // ReSharper disable once UnusedMember.Global
    public static void CreateEnrichedDataSet()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, $"{nameof(CreateEnrichedDataSet)}");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, $"{nameof(CreateEnrichedDataSet)}");
        var sw = Stopwatch.StartNew();

        var reviewsTfIdfEncodedRawFile  = Path.Combine(WorkingDirectory, "reviews_tfidf_encoded_normalized_"+DateTime.Now.Ticks+".csv");

        //res = TfIdfEncoding.Encode(res, "property_4", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true);
        //res = TfIdfEncoding.Encode(res, "property_5", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true);
        //res = TfIdfEncoding.Encode(res, "property_7", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true);
        var review_file = DataFrame.read_string_csv(RawReviewsFile)["listing_id", "renters_comments"]
            .RenameInPlace("listing_id", "id");
        var encoded_review_df = review_file
            .TfIdfEncode("renters_comments", 200, norm:TfIdfEncoding.TfIdfEncoding_norm.L2, scikitLearnCompatibilityMode:false)
            .AverageBy("id");


        encoded_review_df.to_csv(reviewsTfIdfEncodedRawFile);
        Model.Log.Info($"elapsed fo encoding: {sw.Elapsed.Seconds}s");


        var fullReviewsForEmbeddingDim = DataFrame.read_csv(reviewsTfIdfEncodedRawFile, true, c => (c == "id" ? typeof(string) : typeof(float)));

        var trainDf = DataFrame.read_string_csv(RawTrainFile);
        var testDf = DataFrame.read_string_csv(RawTestFile);
        var res = new List<DataFrame> { trainDf, testDf };

        res[0] = res[0].LeftJoinWithoutDuplicates(fullReviewsForEmbeddingDim, "id");
        res[1] = res[1].LeftJoinWithoutDuplicates(fullReviewsForEmbeddingDim, "id");
        res[0].to_csv(RawTrainFile + FILE_EXT);
        res[1].to_csv(RawTestFile + FILE_EXT);

        Model.Log.Info($"elapsed total: {sw.Elapsed.Seconds}s");

    }

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

    public override int NumClass => WasYouStayWorthItsPriceDatasetSample_NumClasses;


    // ReSharper disable once UnusedMember.Global
    //public static void WeightOptimizer()
    //{
    //    const string submissionWorkingDirectory = @"C:\Projects\Challenges\WasYouStayWorthItsPrice\submission2";
    //    List<Tuple<string, string, string>> workingDirectoryAndModelNames = new()
    //    {
    //        //Tuple.Create(submissionWorkingDirectory, "3580990008"),
    //        //Tuple.Create(submissionWorkingDirectory, "48F31E6543"),
    //        //Tuple.Create(submissionWorkingDirectory, "56E668E7DB"),
    //        //Tuple.Create(submissionWorkingDirectory, "8CF93D9FA0"),
    //        //Tuple.Create(submissionWorkingDirectory, "90840F212D"),
    //        //Tuple.Create(submissionWorkingDirectory, "E72AD5B74B"),
    //        //Tuple.Create(submissionWorkingDirectory, "66B4F3653A"),
    //        //Tuple.Create(submissionWorkingDirectory, "395B343296"),


        //        Tuple.Create(submissionWorkingDirectory, "3580990008_KFOLD", "3580990008_KFOLD_FULL"),
        //        Tuple.Create(submissionWorkingDirectory, "395B343296_KFOLD", "395B343296_KFOLD_FULL"),
        //        Tuple.Create(submissionWorkingDirectory, "48F31E6543_KFOLD", "48F31E6543_KFOLD_FULL"),
        //        Tuple.Create(submissionWorkingDirectory, "56E668E7DB_KFOLD", "56E668E7DB_KFOLD_FULL"),
        //        Tuple.Create(submissionWorkingDirectory, "66B4F3653A_KFOLD", "66B4F3653A_KFOLD_FULL"),
        //        Tuple.Create(submissionWorkingDirectory, "8CF93D9FA0_KFOLD", "8CF93D9FA0_KFOLD_FULL"),
        //        Tuple.Create(submissionWorkingDirectory, "90840F212D_KFOLD", "90840F212D_KFOLD_FULL"),
        //        Tuple.Create(submissionWorkingDirectory, "90DAFFB8FC_KFOLD", "90DAFFB8FC_KFOLD_FULL"),
        //        Tuple.Create(submissionWorkingDirectory, "E72AD5B74B_KFOLD", "E72AD5B74B_KFOLD_FULL"),
        //    };
        //    var searchWorkingDirectory = Path.Combine(submissionWorkingDirectory, "search");
        //    WeightsOptimizer.SearchForBestWeights(workingDirectoryAndModelNames, searchWorkingDirectory, null);
        //}
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
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar));
    }


    public static void ComputeAndSaveFeatureImportance()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        var m = ModelAndDatasetPredictions.Load(WorkingDirectory, "CF1B33A251");
        m.ComputeAndSaveFeatureImportance();
        return;
    }

    public const string FILE_EXT = "_tfidf_l2_norm_scikit_stem_allstopwords.csv";

    // ReSharper disable once UnusedMember.Global
    public static (ISample bestSample, IScore bestScore) LaunchLightGBMHPO(int min_num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        var datasetSample = new WasYouStayWorthItsPriceDatasetSample();
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"Reviews_EmbeddingDim", 200},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            {"KFold", new[]{2}},
            

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

        datasetSample.FillWithDefaultLightGBMHyperParameterValues(searchSpace);
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
        return SelectFeatures(_fullTestDatasetEncoded);
    }

    public override DataSet FullTrainingAndValidation()
    {
        return SelectFeatures(_fullTrainDatasetEncoded);
    }

    protected override DatasetEncoder DatasetEncoder => _trainTestEncoder;
    
    private static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    private static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    private static string XYTrainRawFile => Path.Combine(DataDirectory, "train.csv"+ FILE_EXT); //!D
    private static string XTestRawFile => Path.Combine(DataDirectory, "test.csv"+ FILE_EXT); //!D
    private static string RawTrainFile => Path.Combine(DataDirectory, "train.csv");
    private static string RawTestFile => Path.Combine(DataDirectory, "test.csv");
    private static string RawReviewsFile => Path.Combine(DataDirectory, "reviews.csv");
    private InMemoryDataSetV2 SelectFeatures(InMemoryDataSetV2 dataset)
    {
        var df = dataset.XDataFrame;
        var columnToDrop = new List<string>();
        columnToDrop.AddRange(TfIdfEncoding.ColumnToRemoveToFitEmbedding(df, "renters_comments", Reviews_EmbeddingDim, true));
        if (columnToDrop.Count == 0)
        {
            return dataset;
        }
        var xUpdated = df.DropIgnoreErrors(columnToDrop.ToArray());
        return DatasetEncoder.NewInMemoryDataSetV2(xUpdated, dataset.YDataFrame, this);
    }

    // ReSharper disable once UnusedMember.Global
    public static void Retrain()
    {

        foreach(var modelName in new[]
                {
                    "3580990008",
                    "395B343296",
                    "48F31E6543",
                    "56E668E7DB",
                    "66B4F3653A",
                    "8CF93D9FA0",
                    "90840F212D",
                    "90DAFFB8FC",
                    "E72AD5B74B"
                })
        {

            const string workingDirectory = @"C:\Projects\Challenges\WasYouStayWorthItsPrice\submission3";

            Utils.ConfigureGlobalLog4netProperties(workingDirectory, $"{nameof(Retrain)}");
            Utils.ConfigureThreadLog4netProperties(workingDirectory, $"{nameof(Retrain)}");

            var m = ModelAndDatasetPredictions.Load(workingDirectory, modelName);
            //var embeddedModel = mKfold.Model;
            var mKfoldModelAndDatasetPredictionsSample = m.ModelAndDatasetPredictionsSample;
            if (m.Model is not KFoldModel)
            {
                var embeddedModel = m.Model;
                var kfoldSample = new KFoldSample(3, workingDirectory, embeddedModel.ModelSample.GetLoss(), mKfoldModelAndDatasetPredictionsSample.DatasetSample.DatasetRowsInModelFormatMustBeMultipleOf());
                var sample = new ModelAndDatasetPredictionsSample(new ISample[]
                {
                    kfoldSample,
                    mKfoldModelAndDatasetPredictionsSample.DatasetSample.CopyWithNewPercentageInTrainingAndKFold(1.0, kfoldSample.n_splits),
                    mKfoldModelAndDatasetPredictionsSample.PredictionsSample
                });
                m = new ModelAndDatasetPredictions(sample, workingDirectory, embeddedModel.ModelName + KFoldModel.SuffixKfoldModel);
            }

            m.Model.Use_All_Available_Cores();
            m.Fit(true, true, true);

            var kfoldModelName = m.Model.ModelName;
            m.Save(workingDirectory, kfoldModelName);
            var modelAndDatasetPredictionsSampleOnFullDataset = mKfoldModelAndDatasetPredictionsSample.CopyWithNewPercentageInTrainingAndKFold(1.0, 1);
            var modelAndDatasetOnFullDataset = new ModelAndDatasetPredictions(modelAndDatasetPredictionsSampleOnFullDataset, workingDirectory, kfoldModelName + "_FULL");
            Model.Log.Info($"Retraining Model '{kfoldModelName}' on full Dataset no KFold (Model on full Dataset name: {modelAndDatasetOnFullDataset.Model.ModelName})");
            modelAndDatasetOnFullDataset.Model.Use_All_Available_Cores();
            modelAndDatasetOnFullDataset.Fit(true, true, true);
        }
    }
}
