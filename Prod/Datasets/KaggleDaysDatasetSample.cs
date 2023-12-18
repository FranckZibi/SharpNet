using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using SharpNet.CatBoost;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Networks;

// ReSharper disable UnusedParameter.Global

// ReSharper disable UnusedMember.Local

namespace SharpNet.Datasets;

public class KaggleDaysDatasetSample : AbstractDatasetSample
{
    #region private fields
    private const string NAME = "KaggleDays";
    private static DataFrame xytrain_string_df;
    private static DataFrame xtest_string_df;
    #endregion
  
    private readonly object lockObject = new();
    public KaggleDaysDatasetSample() : base(new HashSet<string>())
    {
        lock (lockObject)
        {
            if (xytrain_string_df == null)
            {
                var sw = Stopwatch.StartNew();
                ISample.Log.Info($"Loading file {XYTrainRawFile}");
                xytrain_string_df = DataFrame.read_string_csv(XYTrainRawFile, true, true);
                ISample.Log.Info($"Loading file {XTestRawFile}");
                xtest_string_df = DataFrame.read_string_csv(XTestRawFile, true, true);
                GC.Collect();
                ISample.Log.Debug($"Loading files took {sw.ElapsedMilliseconds / 1000.0} seconds");
            }
        }
    }


    private static string[] SplitProducts(string str)
    {
        if (string.IsNullOrEmpty(str) || string.IsNullOrEmpty(str.Trim()))
        {
            return new string[0];
        }
        return str.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries).ToArray();
    }

    //public static void Enrich(string predictions)
    //{
    //    var preds = File.ReadAllLines(predictions).Skip(1).Select(float.Parse).ToArray();
    //    var idFile = Utils.ReadCsv(Path.Combine(DataDirectory, "search_test_v2.csv")).Skip(1).ToArray();

    //    Dictionary<string, List<Tuple<string,float>>> losses = new ();
    //    for (int i = 0; i < idFile.Length; i++)
    //    {
    //        var sessionId = idFile[i][0];
    //        var productId = idFile[i][1];
    //        var pred = preds[i];
    //        if (!losses.ContainsKey(sessionId))
    //        {
    //            losses[sessionId] = new List<Tuple<string, float>>();
    //        }
    //        losses[sessionId].Add(Tuple.Create(productId, pred));
    //    }

    //    var session_ids = DataFrame.read_string_csv(Path.Combine(DataDirectory, "sample_submission.csv")).StringColumnContent("session_id");
    //    var linePred =new List<string>();
    //    linePred.Add("session_id,best_products");
    //    foreach (var sessionId in session_ids)
    //    {
    //        if (!losses.ContainsKey(sessionId))
    //        {
    //            linePred.Add(sessionId+",");
    //        }
    //        else
    //        {
    //            string[] sorted = losses[sessionId].OrderBy(t => t.Item2).ToArray().Take(12).Select(t=>t.Item1).ToArray();
    //            linePred.Add(sessionId + ","+string.Join(' ', sorted));
    //        }
    //    }
    //    File.WriteAllLines(predictions+"_final.csv", linePred);
    //}

    // ReSharper disable once UnusedMember.Global
    private static void Create_search_train_v2()
    {
        var search_train_path = Path.Combine(DataDirectory, "search_train.csv");
        var allLines = Utils.ReadCsv(search_train_path);
        //session_id,search_results,keyword,best_products
        var rand = new Random(0);
        var sb = new StringBuilder();
        sb.Append("session_id,product_id,keyword,rank,count,y");
        foreach (var line in allLines.Skip(1))
        {
            string[] allProducts = SplitProducts(line[1]);
            var productToRank = new Dictionary<string, int>();
            for (var rank = 0; rank < allProducts.Length; rank++)
            {
                var product = allProducts[rank];
                productToRank[product] = rank;
            }

            Utils.Shuffle(allProducts, rand);
            var best_products = SplitProducts(line[3]);
            foreach (var s in best_products)
            {
                sb.Append(Environment.NewLine+ line[0] + "," + s + "," + line[2] + "," + productToRank[s] + ","+ allProducts.Length + ",1");
            }
            var bad_products = Utils.Without(allProducts, best_products);
            foreach (var s in bad_products)
            //foreach (var s in bad_products.Take(2* best_products.Length))
            {
                sb.Append(Environment.NewLine + line[0] + "," + s + "," + line[2] + "," + productToRank[s] + "," + allProducts.Length + ",0");
            }
        }
        File.WriteAllText(Path.Combine(DataDirectory, "search_train_v2.csv"), sb.ToString());
    }

    private static void Create_search_test_v2()
    {
        var search_train_path = Path.Combine(DataDirectory, "search_test.csv");
        var allLines = Utils.ReadCsv(search_train_path);
        var sb = new StringBuilder();
        sb.Append("session_id,product_id,keyword,rank,count");
        foreach (var line in allLines.Skip(1))
        {
            var allProducts = SplitProducts(line[1]);
            var productToRank = new Dictionary<string, int>();
            for (var rank = 0; rank < allProducts.Length; rank++)
            {
                var product = allProducts[rank];
                productToRank[product] = rank;
            }

            foreach (var s in allProducts)
            {
                sb.Append(Environment.NewLine + line[0] + "," + s +"," + line[2] + "," + productToRank[s] + "," + allProducts.Length);
            }
        }
        File.WriteAllText(Path.Combine(DataDirectory, "search_test_v2.csv"), sb.ToString());
    }

    private static readonly HashSet<string> StringColumns = new()
    {
        "action_type",
        "search_results","keyword","best_products",
        "session_id" , "product_id", "channel_grouping", "country", "region", "device_category", "category", "name", "market", "subbrand"
    };

    private static Type GetColumnType(string columnName)
    {
        return StringColumns.Contains(columnName) ? typeof(string) : typeof(float);

    }
    private static void Create_search_train_test_v3()
    {
        ISample.Log.Info("starting Create_search_train_test_v3");
        var user_info = DataFrame.read_csv(Path.Combine(DataDirectory, "user_info.csv"), true, GetColumnType);
        var item_info = DataFrame.read_csv(Path.Combine(DataDirectory, "item_info.csv"), true, GetColumnType);

        var before_search = DataFrame.read_csv(Path.Combine(DataDirectory, "before_search_v2.csv"), true, GetColumnType);


        //var name = DataFrame.read_csv(Path.Combine(DataDirectory, "tfidf_for_name_embedded_distiluse-base-multilingual-cased-v1.csv"), true, c => c == "name" ? typeof(string) : typeof(float));
        //name = name[name.Columns.Take(1 + embeddingDim_name).ToArray()];
        //var keyword = DataFrame.read_csv(Path.Combine(DataDirectory, "tfidf_for_keyword_embedded_distiluse-base-multilingual-cased-v1.csv"), true, c => c == "keyword" ? typeof(string) : typeof(float));
        //keyword = keyword[keyword.Columns.Take(1 + embeddingDim_keyword).ToArray()];

        var name = DataFrame.read_csv(Path.Combine(DataDirectory, "tfidf_for_name.csv"), true, c => c == "name" ? typeof(string) : typeof(float));
        var keyword = DataFrame.read_csv(Path.Combine(DataDirectory, "tfidf_for_keyword.csv"), true, c => c == "keyword" ? typeof(string) : typeof(float));


        ISample.Log.Info("Finished loading ref dataset");

        var train = DataFrame.read_csv(Path.Combine(DataDirectory, "search_train_v2.csv"), true, GetColumnType);
        var train_v3 = train
            .LeftJoinWithoutDuplicates(before_search, new[] { "session_id", "product_id" })
            .LeftJoinWithoutDuplicates(user_info, new[] { "session_id"})
            .LeftJoinWithoutDuplicates(item_info, new[] { "product_id"})
            .LeftJoinWithoutDuplicates(keyword, new[] { "keyword"})
            .LeftJoinWithoutDuplicates(name, new []{"name"})
            ;
        train_v3 = train_v3.DropIgnoreErrors("product_id", "keyword", "name");
        train_v3.to_csv(Path.Combine(DataDirectory, "search_train_v3.csv"));
        ISample.Log.Info("Finished building train dataset ref dataset");

        var test = DataFrame.read_csv(Path.Combine(DataDirectory, "search_test_v2.csv"), true, GetColumnType);
        var test_v3 = test
            .LeftJoinWithoutDuplicates(before_search, new[] { "session_id", "product_id" })
            .LeftJoinWithoutDuplicates(user_info, new[] { "session_id"})
            .LeftJoinWithoutDuplicates(item_info, new[] { "product_id" })
            .LeftJoinWithoutDuplicates(keyword, new[] { "keyword" })
            .LeftJoinWithoutDuplicates(name, new[] { "name" })
            ;
        test_v3 = test_v3.DropIgnoreErrors("product_id", "keyword", "name");
        test_v3.to_csv(Path.Combine(DataDirectory, "search_test_v3.csv"));
        ISample.Log.Info("Finished building test dataset ref dataset");

    }
    private static void AddBeforeSearchDataV2()
    {
        var before_search_path = Path.Combine(DataDirectory, "before_search.csv");
        Dictionary<string, Dictionary<string, HashSet<string>>> before_search_to_line = new();
        foreach (var l in Utils.ReadCsv(before_search_path).Skip(1))
        {
            var session_id = l[0];
            var search_results = l[1];
            var action_type = l[2];
            if (!before_search_to_line.ContainsKey(session_id))
            {
                before_search_to_line[session_id] = new Dictionary<string, HashSet<string>>();
            }

            var s0 = before_search_to_line[session_id];
            if (!s0.ContainsKey(search_results))
            {
                s0[search_results] = new HashSet<string>();
            }
            s0[search_results].Add(action_type);

        }

        var categories = "ADD_TO_CART,CHECKOUT,ITEM_DETAILS,ITEM_LIST,PURCHASE,REMOVE_FROM_CART".Split(',');
        var sb = new StringBuilder();
        sb.AppendFormat("session_id,product_id," + string.Join(',', categories));
        foreach (var (session_id, data) in before_search_to_line)
        {
            foreach (var (product_id, values) in data)
            {
                sb.Append(Environment.NewLine + session_id + "," + product_id);
                foreach (var h in categories)
                {
                    sb.Append("," + (values.Contains(h) ? 1 : 0));
                }
            }
        }
        File.WriteAllText(Path.Combine(DataDirectory, "before_search_v2.csv"), sb.ToString());
    }

    const int embeddingDim_keyword = 100;
    const int embeddingDim_name = 100;

    // ReSharper disable once UnusedMember.Global
    public static void CreateEnrichedDataSet()
    {

        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, $"{nameof(CreateEnrichedDataSet)}");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, $"{nameof(CreateEnrichedDataSet)}");

        DataFrame.NormalizeAllCsvInDirectory(DataDirectory, true, true);

        var search_train_path = Path.Combine(DataDirectory, "search_train.csv");
        var search_test_path = Path.Combine(DataDirectory, "search_test.csv");
        var item_info_path = Path.Combine(DataDirectory, "item_info.csv");

        foreach (var path in new[] { search_train_path, search_test_path })
        {
            var df1 =DataFrame.read_string_csv(path, true, true);
            df1["keyword"].UpdateInPlace(c => c?.ToLower());
            df1.to_csv(path);
        }
        var df2 = DataFrame.read_string_csv(item_info_path, true, true);
        df2["name"].UpdateInPlace(c => c?.ToLower());
        df2.to_csv(item_info_path);

        //TfIdf Encoding
        const bool hasHeader = true;
        const bool isNormalized = true;
        DataFrame.TfIdfEncode(new[] { search_train_path, search_test_path }, hasHeader, isNormalized, "keyword", embeddingDim_keyword, false);
        DataFrame.TfIdfEncode(new[] { item_info_path }, hasHeader, isNormalized, "name", embeddingDim_name, false);
        AddBeforeSearchDataV2();
        Create_search_train_v2();
        Create_search_test_v2();
        Create_search_train_test_v3();
    }


    #region Hyper-Parameters
    /// <summary>
    /// the embedding dim to use to enrich the dataset with the reviews
    /// </summary>
    // ReSharper disable once MemberCanBePrivate.Global
    public string ToRemove = "product_id,subbrand,keyword,name,market";
    #endregion

    public override int NumClass => TargetLabelDistinctValues.Length;
    public override string[] TargetLabelDistinctValues => new[] { "y" };
    public override Objective_enum GetObjective() => Objective_enum.Classification;
    public override string[] CategoricalFeatures => new[] { "channel_grouping", "country", "region", "device_category", "category", "name", "market" };

    public override int[] X_Shape(int batchSize) => throw new NotImplementedException(); //!D TODO
    public override int[] Y_Shape(int batchSize) => throw new NotImplementedException(); //!D TODO

    public override string IdColumn => "session_id";
    public override string[] TargetLabels => new[] { "y" };
    public override DataSet TestDataset()
    {
        return LoadAndEncodeDataset_If_Needed().testDataset;
    }

    


    public override DataFrameDataSet FullTrainingAndValidation()
    {
        return LoadAndEncodeDataset_If_Needed().fullTrainingAndValidation;
    }


    private static readonly Dictionary<string, Tuple<DataFrameDataSet, DataFrameDataSet, DatasetEncoder>> CacheDataset = new();

    private (DataFrameDataSet fullTrainingAndValidation, DataFrameDataSet testDataset) LoadAndEncodeDataset_If_Needed()
    {
        var sw = Stopwatch.StartNew();
        var key = ComputeHash();
        lock(CacheDataset)
        {
            if (CacheDataset.TryGetValue(key, out var result))
            {
                DatasetEncoder = result.Item3;
                return (result.Item1, result.Item2);
            }
            ISample.Log.Debug($"Loading Encoded Dataset for key '{key}'");
            DatasetEncoder = new DatasetEncoder(this, StandardizeDoubleValues, true);
            var xyTrain = UpdateFeatures(xytrain_string_df);
            var xtest = UpdateFeatures(xtest_string_df);
            DatasetEncoder.Fit(xyTrain);
            DatasetEncoder.Fit(xtest);
            var xTrain_Encoded = DatasetEncoder.Transform(xyTrain.Drop(TargetLabels));
            var yTrain_Encoded = DatasetEncoder.Transform(xyTrain[TargetLabels]);
            var xtest_Encoded = DatasetEncoder.Transform(xtest);
            var fullTrainingAndValidation = new DataFrameDataSet(this, xTrain_Encoded, yTrain_Encoded, xytrain_string_df.StringColumnContent(IdColumn));
            var testDataset = new DataFrameDataSet(this, xtest_Encoded, null, xtest_string_df.StringColumnContent(IdColumn));
            CacheDataset[key] = Tuple.Create(fullTrainingAndValidation, testDataset, DatasetEncoder);
            ISample.Log.Debug($"Loading Encoded Dataset for key '{key}' took {sw.Elapsed.TotalSeconds}s");
            return (fullTrainingAndValidation, testDataset);
        }
    }

    private DataFrame UpdateFeatures(DataFrame x)
    {
        var columnToDrop = ToRemove.Split(',').ToArray();
        //columnToDrop.AddRange(TfIdfEncoding.ColumnToRemoveToFitEmbedding(x, "renters_comments", Reviews_EmbeddingDim, true));
        return x.DropIgnoreErrors(columnToDrop.ToArray());
    }

    private static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    private static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    //private static string XYTrainRawFile => Path.Combine(DataDirectory, "search_train_v2.csvv4.csv");
    private static string XYTrainRawFile => Path.Combine(DataDirectory, "search_train_v3.csv");
    //private static string XTestRawFile => Path.Combine(DataDirectory, "search_test_v2.csvv4.csv");
    private static string XTestRawFile => Path.Combine(DataDirectory, "search_test_v3.csv");


    // ReSharper disable once UnusedMember.Global
    public static void TrainNetwork(int numEpochs = 15, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"KFold", 2},
            //{nameof(AbstractDatasetSample.PercentageInTraining), 0.8}, //will be automatically set to 1 if KFold is enabled

            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},

            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW"} },
            //{ nameof(NetworkSample.OptimizerType), "AdamW" },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] {0.001f, 0.01f } },
            //{ nameof(NetworkSample.AdamW_L2Regularization), AbstractHyperParameterSearchSpace.Range(0.001f/4, 0.001f*4, AbstractHyperParameterSearchSpace.range_type.normal) },
            //{ nameof(NetworkSample.AdamW_L2Regularization), 0.001f },
            //{ nameof(NetworkSample.SGD_usenesterov), new[] { true, false } },
            //{ nameof(NetworkSample.lambdaL2Regularization), new[] { 0, 1e-5, 1e-4, 1e-3, } },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), AbstractHyperParameterSearchSpace.Range(0.001f/4, 0.001f*4, AbstractHyperParameterSearchSpace.range_type.normal) },
            //{ nameof(NetworkSample.InitialLearningRate), 0.001f },
            // Learning Rate Scheduler
            { nameof(NetworkSample.LearningRateSchedulerType), new[] { "OneCycle" } },
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            //{ "EmbeddingDim", new[] { 0, 4, 8, 12 } },
            { "EmbeddingDim", new[]{4, 8} },
            //{"weight_norm", new[]{true, false}},
            //{"leaky_relu", new[]{true, false}},
            { "dropout_top", new[] { 0, 0.1, 0.2 } },
            { "dropout_mid", new[] { 0, 0.3, 0.5 } },
            { "dropout_bottom", new[] { 0, 0.2, 0.4 } },
            { nameof(NetworkSample.BatchSize), new[] {1024, 2048, 4096} },
            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },


            { "hidden_size", new[]{512, 1024,2048} },
            { "channel_1", 128},
            { "channel_2", 256},
            { "channel_3", 256},

    };

        //var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new NetworkSample_1DCNN(), new KaggleDaysDatasetSample()), WorkingDirectory);
        var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new NetworkSample_1DCNN(), new KaggleDaysDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, true, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Global
    public static void LaunchCatBoostHPO(int iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        // ReSharper disable once ConvertToConstant.Local
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"KFold", 2},
            //{nameof(AbstractDatasetSample.PercentageInTraining), 0.8}, //will be automatically set to 1 if KFold is enabled

            //related to model
            {"loss_function", nameof(CatBoostSample.loss_function_enum.Logloss)},
            {"eval_metric", nameof(CatBoostSample.metric_enum.Logloss)},
            { "logging_level", "Silent"},
            { "allow_writing_files",false},
            { "thread_count",1},
            { "iterations", iterations },
            { "od_type", "Iter"},
            { "od_wait",iterations/10},
            { "depth", 7 /* AbstractHyperParameterSearchSpace.Range(7, 8)*/ },
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.3f, 0.7f)}, //0.5204
            { "random_strength",AbstractHyperParameterSearchSpace.Range(0.01f, 0.1f, AbstractHyperParameterSearchSpace.range_type.normal)}, //0.07
            { "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
            { "l2_leaf_reg", 2 /*AbstractHyperParameterSearchSpace.Range(0, 10)*/},
        };


        var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new KaggleDaysDatasetSample()), WorkingDirectory);
        //var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new KaggleDaysDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, true, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    // ReSharper disable once UnusedMember.Global
    public static (ISample bestSample, IScore bestScore) LaunchLightGBMHPO(int num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {

            //related to Dataset 
            {"KFold", 2},
            //{nameof(AbstractDatasetSample.PercentageInTraining), 0.8}, //will be automatically set to 1 if KFold is enabled
            {"objective", nameof(LightGBMSample.objective_enum.binary)},
            {"metric", ""}, //same as objective
            { "num_threads", -1},
            { "boosting", "gbdt"},
            {"objective", "binary"},          //for binary classification
            { "learning_rate", 0.15},
            { "colsample_bytree",0.9},
            { "bagging_fraction", 0.7},
            { "bagging_freq", 10},
            { "verbosity", "0" },
            { "max_depth", 11},
            { "num_iterations", num_iterations },
        };


        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), new KaggleDaysDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, true, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
        return (hpo.BestSampleFoundSoFar, hpo.ScoreOfBestSampleFoundSoFar);
    }
}
