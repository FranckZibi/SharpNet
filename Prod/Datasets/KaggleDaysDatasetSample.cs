using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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


    //public string ToRemove = "product_id,subbrand,keyword";
    public string ToRemove = "product_id,subbrand,keyword,market";


    #region private fields
    private const string NAME = "KaggleDays";
    private static readonly DataFrame xytrain_string_df;
    private static readonly DataFrame xtest_string_df;
    private static readonly ConcurrentDictionary<string, Tuple<DataSetV2, DataSetV2, DatasetEncoder>> CacheDataset = new();
    #endregion
    /*
    static KaggleDaysDatasetSample()
    {
        xytrain_string_df = DataFrame.read_string_csv(XYTrainRawFile);
        xtest_string_df = DataFrame.read_string_csv(XTestRawFile);
    }
    */

    public static string[] RemoveComma(string[] str)
    {
        return str.Select(s => s.Replace(',', ' ').Replace(';', ' ').Replace('\"', ' ')).ToArray();
    }

    public static string[] SplitProducts(string str)
    {
        if (string.IsNullOrEmpty(str) || string.IsNullOrEmpty(str.Trim()))
        {
            return new string[0];
        }
        return str.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries).ToArray();
    }


    public static string ExtraFeatures(string session_id, string search_results, Dictionary<string, Dictionary<string, HashSet<string>>> before_search_to_line)
    {


        string extraFeartures = "";

        foreach (var id in new[] { "ADD_TO_CART", "CHECKOUT", "ITEM_DETAILS", "ITEM_LIST", "PURCHASE", "REMOVE_FROM_CART" })
        {

            if (before_search_to_line.ContainsKey(session_id)
                && before_search_to_line[session_id].ContainsKey(search_results)
                && before_search_to_line[session_id][search_results].Contains(id))
            {
                extraFeartures += "1,";
            }
            else
            {
                extraFeartures += "0,";
            }
        }

        return extraFeartures.TrimEnd(',');

    }


    public static void Enrich(string predictions)
    {
        var preds = File.ReadAllLines(predictions).Skip(1).Select(float.Parse).ToArray();

        var idFile = Utils.ReadCsv(Path.Combine(DataDirectory, "idsContent.csv")).Skip(1).ToArray();

        Dictionary<string, List<string>> sessionIdToProducts = new();


        Dictionary<string, List<Tuple<string,float>>> losses = new ();
        for (int i = 0; i < idFile.Length; i++)
        {
            var sessonId = idFile[i][0];
            var productId = idFile[i][1];
            var pred = preds[i];
            if (!losses.ContainsKey(sessonId))
            {
                losses[sessonId] = new List<Tuple<string, float>>();
            }
            losses[sessonId].Add(Tuple.Create(productId, pred));
        }


        var sesssion_ids = DataFrame.read_string_csv(Path.Combine(DataDirectory, "sample_submission.csv")).StringColumnContent("session_id");
        var linePred =new List<string>();
        linePred.Add("session_id,best_products");
        foreach (var sess in sesssion_ids)
        {
            if (!losses.ContainsKey(sess))
            {
                linePred.Add(sess+",");
            }
            else
            {
                string[] sorted = losses[sess].OrderByDescending(t => t.Item2).ToArray().Take(12).Select(t=>t.Item1).ToArray();
                linePred.Add(sess + ","+string.Join(' ', sorted));
            }
        }
        File.WriteAllLines(predictions+"_final.csv", linePred);





        //session_id,best_products
        //52764a1dcd6addc9,4efa378890596517 5d0ea89215a8e8fa dfc2aededec87b26 706fd4eac2f5cb32 58e4e987cc9899bf
        //20ed7cda57581baa,09db50dfc9abbf12 6028359e23034acd df90d7b1179e43fe b9b3c024a75ae0b1 74073f15a63d290e 9fa83166f46410ea 510336c108f32187 7246c26b86ec2d36 7bba2d1762a9ffb8 8f175349022ed02c 5ed1f985367f50b9 3d12fd27b15931cb 3d16fecdce91d7a6 df2ac739d8c0228d bfb6af14a7b03a34 b4efdc17770199d1 ea909cb1eb52d3de 7b490bbfeb01900d 02373c1c21513129 f36e046a71211ca7 ca56dd967fb4b154 8e994c2785b3c95a ad0d0fbdb67d36e8 78d8db418009f3e3 d5ed5e63fc8b56d3 af610ab7d6aa8554 7afefad841749aa7 49f9e4334c17dbef a025d9984761bccc 0d809eebb5e37fc5 c38fcc88485b051e f82f2f0282cf5900 f92fb63e11131150

    }


    // ReSharper disable once UnusedMember.Global
    public static void CreateEnrichedDataSet()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, $"{nameof(CreateEnrichedDataSet)}");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, $"{nameof(CreateEnrichedDataSet)}");
        //var sw = Stopwatch.StartNew();

        int tfidf_count = 300;
        /*
          var file1 = DataFrame.read_string_csv(Path.Combine(DataDirectory, "search_test_v2.csv"))["keyword"];
          var file2 = DataFrame.read_string_csv(Path.Combine(DataDirectory, "search_train_v2.csv"))["keyword"];
          var file = DataFrame.MergeVertically(file1, file2);

          //review_file = review_file.TfIdfEncode("property_4", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true)
          //            .TfIdfEncode("property_5", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true)
          //            .TfIdfEncode("property_7", 20, keepEncodedColumnName: true, reduceEmbeddingDimIfNeeded: true);
          var encoded_review_df = file
              .TfIdfEncode("keyword", tfidf_count, norm:TfIdfEncoding.TfIdfEncoding_norm.L2, scikitLearnCompatibilityMode:false, keepEncodedColumnName:true)
              .AverageBy("keyword");
          var reviewsTfIdfEncodedRawFile = Path.Combine(DataDirectory, "tfidf.csv");
          encoded_review_df.to_csv(reviewsTfIdfEncodedRawFile);
          */

        var user_info_header = "session_id,channel_grouping,country,region,device_category";
        var item_info_header = "product_id,category,name,market,subbrand,price";




        //product_id,category,name,market,subbrand,price

        //var allItems = File.ReadAllLines(item_info_path);
        //var item_info_header = allItems[0];
        var item_info_path = Path.Combine(DataDirectory, "item_info_v2.csv");
        Dictionary<string, string> product_id_to_line = new();
        foreach (var l in File.ReadAllLines(item_info_path).Skip(1))
        {
            var id = l.Split(',')[0];
            product_id_to_line[id] = l;
        }

        var user_info_path = Path.Combine(DataDirectory, "user_info_v2.csv");
        Dictionary<string, string> session_id_to_line = new();
        foreach (var l in File.ReadAllLines(user_info_path).Skip(1))
        {
            var id = l.Split(',')[0];
            session_id_to_line[id] = l;
        }

        var tfidf_path = Path.Combine(DataDirectory, "tfidf.csv");
        Dictionary<string, string> keyword_to_line = new();
        var tfidf_header = File.ReadAllLines(tfidf_path)[0];
        foreach (var l in File.ReadAllLines(tfidf_path).Skip(1))
        {
            var id = l.Split(',')[0];
            keyword_to_line[id] = l;
        }


        //Sep=,
        //session_id,search_results,action_type
        //82aff0d6665cd26b,9fc0cc21322f9fc3,ITEM_LIST

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

        ////product_id,category,name,market,subbrand,price
        //var user_info_path = Path.Combine(DataDirectory, "user_info.csv");
        //var allUsers = System.IO.File.ReadAllLines(user_info_path);
        //Dictionary<string, string> session_id_to_line = new();
        //var user_info_header = allUsers[0];
        //var use_info_lines = new List<string>();
        //use_info_lines.Add(user_info_header);
        //foreach (string[] splitted in Utils.ReadCsv(user_info_path).Skip(1))
        //{
        //    var splitted2 = RemoveComma(splitted);
        //    var session_id = splitted[0];
        //    session_id_to_line[session_id] = string.Join(',', splitted2);
        //    use_info_lines.Add(string.Join(',', splitted2));

        //}
        //File.WriteAllText(Path.Combine(DataDirectory, "user_info_v2.csv"), string.Join(Environment.NewLine, use_info_lines));

        



        //session_id,search_results,keyword,best_products


        foreach (var token in new[] { "search_train_v2", "search_test_v2" })
        {

            var search_train_path = Path.Combine(DataDirectory, token+".csv");
            var search_train = Utils.ReadCsv(search_train_path).ToList();

            var rand = new Random();
            List<string> newLines = new();

            if (token == "search_test_v2")
            {
                newLines.Add(user_info_header + ",ADD_TO_CART,CHECKOUT,ITEM_DETAILS,ITEM_LIST,PURCHASE,REMOVE_FROM_CART," + item_info_header + ","+ tfidf_header);
            }
            else
            {
                newLines.Add(user_info_header + ",ADD_TO_CART,CHECKOUT,ITEM_DETAILS,ITEM_LIST,PURCHASE,REMOVE_FROM_CART," + item_info_header + ","+ tfidf_header+",ok");
            }


            var search_train_header = (token == "search_train_v2")
                        ?"session_id,search_results,keyword,best_products"
                        : "session_id,search_results,keyword";
            var search_train_lines = new List<string>();
            search_train_lines.Add(search_train_header);
            for (int row = 1; row < search_train.Count; ++row)
            {
                search_train[row] = RemoveComma(search_train[row]);
                var session_id = search_train[row][0];
                var search_results = SplitProducts(search_train[row][1]);
                search_train_lines.Add(string.Join(',', search_train[row]));

                Utils.Shuffle(search_results, rand);


                var keyword = search_train[row][2];

                string keywords = "";
                if (keyword_to_line.ContainsKey(keyword))
                {
                    keywords = keyword_to_line[keyword];
                }
                else
                {
                    keywords = "keyword,"+string.Join(',', Enumerable.Repeat(0, tfidf_count).Select(t=>t.ToString()));

                }




                if (token == "search_test_v2")
                {
                    foreach (var s in search_results)
                    {

                        

                        var line = session_id_to_line[session_id] + "," + ExtraFeatures(session_id, s, before_search_to_line) + ","+ product_id_to_line[s] + "," + keywords;
                        newLines.Add(line);
                    }

                }
                else
                {
                    var best_products = new HashSet<string>(SplitProducts(search_train[row][3]));
                    foreach (var s in best_products)
                    {
                        var line = session_id_to_line[session_id] + "," + ExtraFeatures(session_id, s, before_search_to_line) + "," + product_id_to_line[s] + "," + keywords + ",1";
                        newLines.Add(line);
                    }

                    search_results = Utils.Without(search_results, best_products).ToArray();

                    foreach (var s in search_results.Take(best_products.Count))
                    {
                        var line = session_id_to_line[session_id] + "," + ExtraFeatures(session_id, s, before_search_to_line) + "," + product_id_to_line[s] + "," + keywords + ",0";
                        newLines.Add(line);
                    }

                }

            }
            File.WriteAllLines(search_train_path + "v5.csv", newLines);
            //File.WriteAllText(Path.Combine(DataDirectory, token + "_v3.csv"), string.Join(Environment.NewLine, search_train_lines));
        }

        return;

        //session_id,search_results,keyword,best_products
        //e00f4cc4c9ffc860,c2318f182e808960 304ea1877587818c 7fbdef73858d43fc 269b512402d656ef e59afed6684f043c 7c10e75637637cdc 2268c99d06e5dd7e 4d4140667796ed29 83d86674a2c6bb4e 38d3340da2ff724a 11695f0f12630727 926aac3ef31cfa81,Rare,c2318f182e808960
    }


    private KaggleDaysDatasetSample() : base(new HashSet<string>())
    {
    }

    #region Hyper-Parameters
    // ReSharper disable once UnusedMember.Global
    // ReSharper disable once MemberCanBePrivate.Global
    // ReSharper disable once FieldCanBeMadeReadOnly.Global
    public string KaggleDaysDatasetSample_Version = "v1";
    /// <summary>
    /// the embedding dim to use to enrich the dataset with the reviews
    /// </summary>
    // ReSharper disable once MemberCanBePrivate.Global
    #endregion

    public override int NumClass => 1;


    public override Objective_enum GetObjective() => Objective_enum.Classification;
    public override EvaluationMetricEnum GetRankingEvaluationMetric() => EvaluationMetricEnum.BinaryCrossentropy;
    //public override IScore MinimumScoreToSaveModel => new Score(0.32f, GetRankingEvaluationMetric());
    public override string[] CategoricalFeatures => new[] { "channel_grouping", "country", "region", "device_category", "category", "name", "market" };
    public override string[] IdColumns => new[] { "session_id" };
    public override string[] TargetLabels => new[] { "ok" };
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
        var columnToDrop = ToRemove.Split(',').ToArray();
        //columnToDrop.AddRange(TfIdfEncoding.ColumnToRemoveToFitEmbedding(x, "renters_comments", Reviews_EmbeddingDim, true));
        return x.DropIgnoreErrors(columnToDrop.ToArray());
    }

    private static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    private static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    private static string XYTrainRawFile => Path.Combine(DataDirectory, "search_train_v2.csvv3.csv");
    private static string XTestRawFile => Path.Combine(DataDirectory, "search_test_v2.csvv3.csv");


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
            //{"KFold", 2},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            
            { "num_threads", 1},
            { "verbosity", "0" },
            { "early_stopping_round", num_iterations/10 },
            { "num_iterations", num_iterations },

            //uncomment appropriate one
            //{"objective", "regression"},      //for Regression Tasks
            {"objective", "binary"},          //for binary classification
            //{"objective", "multiclass"},      //for multi class classification
            //{"num_class", number_of_class },  //for multi class classification

            //high priority
            { "bagging_fraction", 0.8},
            //{ "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
            { "bagging_freq", new[]{0, 1} },
            
            
            //{ "boosting", new []{"gbdt", "dart"}},
            { "boosting", "gbdt"},
            //?D { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.3f, 1.0f)},

            //{ "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            //{ "learning_rate",AbstractHyperParameterSearchSpace.Range(0.005f, 0.2f)},
            { "learning_rate", new[]{0.001, 0.01, 0.1}},
            //{ "max_depth", new[]{10, 20, 50, 100, 255} },
            { "max_depth", new[]{10, 20} },
            //{ "min_data_in_leaf", new[]{20, 50 /*,100*/} },
            //{ "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 50) },
            { "num_leaves", new[]{3,10,20} },

            ////medium priority
            //{ "drop_rate", new[]{0.05, 0.1, 0.2}},                               //specific to dart mode
            //{ "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            //{ "min_data_in_bin", new[]{3, 10, 100, 150}  },
            //{ "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            //{ "max_drop", new[]{40, 50, 60}},                                   //specific to dart mode
            //{ "skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},  //specific to dart mode

            ////low priority
            //{ "extra_trees", new[] { true , false } }, //low priority 
            ////{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)}, //very low priority
            //{ "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) }, //low priority
            //{ "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), new KaggleDaysDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
        return (hpo.BestSampleFoundSoFar, hpo.ScoreOfBestSampleFoundSoFar);
    }
}
