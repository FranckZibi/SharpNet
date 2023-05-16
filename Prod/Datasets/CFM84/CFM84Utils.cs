using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using log4net;
using SharpNet.CatBoost;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Networks;

namespace SharpNet.Datasets.CFM84;

public static class CFM84Utils
{
    public const string NAME = "CFM84";


    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(CFM84Utils));
    #endregion

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    // ReSharper disable once MemberCanBePrivate.Global

    public static string XTrainPath => Path.Combine(DataDirectory, "input_training.csv");
    public static string YTrainPath => Path.Combine(DataDirectory, "output_training_gmEd6Zt.csv");
    public static string XTestPath => Path.Combine(DataDirectory, "input_test.csv");
    public static string YTestRandomPath => Path.Combine(DataDirectory, "output_test_random.csv");
    public static string StatPath => Path.Combine(DataDirectory, "cfm84_stats.csv");
    public static readonly string[] TargetLabelDistinctValues = new[] { "0", "-1", "1" };




    // ReSharper disable once UnusedMember.Global
    public static void AverageTestPredictions(double[] weights, params string[] predictionsPaths)
    {
        var outputTestRandom_df = DataFrame.read_float_csv(YTestRandomPath);
        var id_df = outputTestRandom_df["ID"].Clone();
        var avg = DataFrame.WeightedSum(weights, predictionsPaths.Select(p => DataFrame.read_float_csv(p)).ToArray());
        Debug.Assert(id_df.Shape[0] == avg.Shape[0]);
        var floatTensorArgMax = avg.FloatTensor.ArgMax().AsFloatCpuSpan.ToArray().Select(f=>Utils.NearestInt(f));
        var strContent = floatTensorArgMax.Select(i => TargetLabelDistinctValues[i]).ToArray();
        var reod_df = DataFrame.New(strContent, new List<string>() { "reod" });
        var pred_df = DataFrame.MergeHorizontally(id_df, reod_df);
        var avgFileName = "avg_"+string.Join("_", predictionsPaths.Select(Path.GetFileNameWithoutExtension))+".csv";
        var avgPath = Path.Combine(Path.GetDirectoryName(predictionsPaths[0])??"", avgFileName);
        pred_df.to_csv(avgPath);
        Log.Info($"Average predictions stored in {avgPath}");
    }

    public static float NormalizeReturn(float r)
    {
        if (float.IsNaN(r) || Math.Abs(r)>10000)
        {
            return float.NaN;
        }

        return r/10000;
    }

    // ReSharper disable once UnusedMember.Global
    public static void BuildStatNormalizeFile()
    {
        var train = DataFrame.read_csv(XTrainPath, true, ColumnNameToType, true);
        var return_columns = train.Columns.Where(c => c.StartsWith("r")).ToArray();
        train.UpdateColumnsInPlace(NormalizeReturn, return_columns);

        train = train.sort_values("day");
        train.to_csv(XTrainPath.Replace(".csv","")+"_normalized.csv");

        //we re order the y train dataset so it has the same order as the x train dataset
        var y_train_dico = CFM84Utils.PredToDico(YTrainPath);
        var y_train_ordered = new List<string>();
        foreach(var sorted_id in train.StringColumnContent("ID"))
        {
            y_train_ordered.Add(sorted_id);
            y_train_ordered.Add(y_train_dico[sorted_id]);
        }
        DataFrame.New(y_train_ordered.ToArray(), new []{ "ID", "reod"}).to_csv(YTrainPath.Replace(".csv", "") + "_normalized.csv");
        var test = DataFrame.read_csv(XTestPath, true, ColumnNameToType, true);
        test.UpdateColumnsInPlace(NormalizeReturn, return_columns);
        test.to_csv(XTestPath.Replace(".csv", "") + "_normalized.csv");
    }

    public static Dictionary<string, string> PredToDico(string path)
    {
        var res = new Dictionary<string, string>();
        var y_pred_df = DataFrame.read_string_csv(path, true, true);
        var y_ID = y_pred_df.StringColumnContent("ID");
        var y_reod = y_pred_df.StringColumnContent("reod");
        for (int i = 0; i < y_ID.Length; ++i)
        {
            res.Add(y_ID[i], y_reod[i]);
        }
        return res;
    }

    public static void Run()
    {
        //Misc.CreateAllFiles(); return;
        //Misc.YCNG(); return;
        //BuildStatNormalizeFile();
        //Misc.BuildStatFile(); return;

        //using var m = ModelAndDatasetPredictions.Load(@"C:\Projects\Challenges\CFM84\dump\", "FB801BE40C", true);
        //m.EstimateLossContribution(computeAlsoRankingScore: true, maxGroupSize: 5000);

        ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "dump"), "E1B434F23C", null, 0.99, false);


        //AverageTestPredictions(new []{0.5,0.5}, @"C:\Projects\Challenges\CFM84\submit\9DE295AB09_modelformat_predict_test_.csv",  @"C:\Projects\Challenges\CFM84\submit\CF37BAB198_modelformat_predict_test_.csv");
        //AverageTestPredictions(new[] { 0.6, 0.4 }, @"C:\Projects\Challenges\CFM84\avg\26A5D7BA9E_modelformat_predict_test_.csv", @"C:\Projects\Challenges\CFM84\avg\CFB81C1361_modelformat_predict_test_.csv");
        //LaunchLightGBMHPO(100);
        //LaunchCatBoostHPO(2000);
        //LaunchNeuralNetworkHPO(10);
        //FitDistribution();
        //FixDistribution();
    }


    public static Type ColumnNameToType(string columnName)
    {
        if (columnName.Equals("ID") || columnName.Equals("equity") || columnName.Equals("reod"))
        {
            return typeof(string);
        }
        if (columnName.Equals("day"))
        {
            return typeof(int);
        }
        return typeof(float);
    }

    // ReSharper disable once UnusedMember.Global
    public static void LaunchCatBoostHPO(int iterations = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        // ReSharper disable once ConvertToConstant.Local
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"KFold", 5},
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            //uncomment appropriate one
            {"loss_function", "MultiClass"},  //for multi class classification
            
            {"use_r_day_equity", new []{true , false}},                 //0.48218 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_day_market", new []{/*true ,*/ false}},         //0.48194 Valid Accuracy (False) vs 0.47576 Valid Accuracy (True) , but much better result in training
            {"use_r_dataset_equity", new []{true, false}},              //0.48294 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_day_equity", new []{true, false}},              //0.48194 Valid Accuracy (False) vs 0.47949 Valid Accuracy (True) 
            {"use_r_day_market", new []{true /*, false*/}},             //0.47531 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_market_correl_r_day_equity", new []{true, false}},    //0.48109 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_dataset_equity", new []{true, false}},          //0.48194 Valid Accuracy (False) vs 0.48116 Valid Accuracy (True)
            {"rr_count", new[]{0,1,2} },


            //{"use_r_dataset", new []{/*true ,*/ false}}, //must be false
            //{"use_vol_r_dataset", new []{/*true,*/ false}}, //must be false



            //{"grow_policy", new []{ "SymmetricTree", "Depthwise" /*, "Lossguide"*/}},


            { "logging_level", nameof(CatBoostSample.logging_level_enum.Verbose)},
            { "allow_writing_files",false},
            { "thread_count",1},
            { "iterations", iterations },
            //{ "use_best_model",true},
            //{ "od_type", "Iter"},
            //{ "od_wait",iterations/10},
            { "depth", AbstractHyperParameterSearchSpace.Range(4, 8) }, //no need to go more than 8

            { "use_best_model",false},

            { "depth", AbstractHyperParameterSearchSpace.Range(4, 8) }, //no need to go more than 8
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.10f)},
            //{ "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
            { "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
            { "l2_leaf_reg",AbstractHyperParameterSearchSpace.Range(0, 15)},
        };


        //best params:
        //searchSpace["depth"] = 6;
        //searchSpace["learning_rate"] = 0.08989349;
        //searchSpace["random_strength"] = 0.00037680444;
        //searchSpace["bagging_temperature"] = 0.9402393;
        //searchSpace["l2_leaf_reg"] = 9;
        //searchSpace["iterations"] = 10000;

        /*
        rr_count = 1
        use_market_correl_r_day_equity = True
        use_r_dataset_equity = True
        use_r_day_equity = True
        use_r_day_market = True
        use_vol_r_dataset_equity = True
        use_vol_r_day_equity = True
        use_vol_r_day_market = False
        */


        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new CFM84DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }
    
    // ReSharper disable once UnusedMember.Global
    public static (ISample bestSample, IScore bestScore) LaunchLightGBMHPO(int num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled


            {"use_r_day_equity", new []{true , false}},                 //0.48218 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_day_market", new []{true , false}},             //0.48194 Valid Accuracy (False) vs 0.47576 Valid Accuracy (True) , but much better result in training
            {"use_r_dataset_equity", new []{true, false}},              //0.48294 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_day_equity", new []{true, false}},              //0.48194 Valid Accuracy (False) vs 0.47949 Valid Accuracy (True) 
            {"use_r_day_market", new []{true /*, false*/}},             //0.47531 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_market_correl_r_day_equity", new []{true, false}},    //0.48109 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_dataset_equity", new []{true, false}},          //0.48194 Valid Accuracy (False) vs 0.48116 Valid Accuracy (True)
            {"rr_count", AbstractHyperParameterSearchSpace.Range(0, 3)},

            {"use_r_dataset", new []{/*true ,*/ false}}, //must be false
            {"use_vol_r_dataset", new []{/*true,*/ false}}, //must be false


            { "num_threads", -1},
            { "verbosity", "0" },
            { "early_stopping_round", num_iterations/10 },
            { "num_iterations", num_iterations },

            {"objective", "multiclass"},
            {"num_class", TargetLabelDistinctValues.Length},


            { "colsample_bytree",0.9},

            //{ "bagging_fraction", 0.8},
            { "bagging_fraction", new[]{0.8f, 1.0f} },
            { "bagging_freq", new[]{/*0,*/ 10} },

            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},

            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.2f)},
            { "max_depth", new[]{10, 20, 50} },
            { "min_data_in_leaf", new[]{20, 100 , 200} },


            //{ "boosting", new []{"gbdt", "dart"}},
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 50) },

/*
            //high priority
            { "bagging_fraction", 0.8},
            //{ "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
            { "bagging_freq", new[]{0, 10} },
            
            
            //{ "boosting", new []{"gbdt", "dart"}},
            { "boosting", "dart"},
            { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.6f, 1.0f)},

            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            //{ "learning_rate",AbstractHyperParameterSearchSpace.Range(0.005f, 0.2f)},
            
            { "learning_rate", new[]{0.001, 0.01, 0.1}},

            { "max_depth", new[]{10, 20, 50} },

            { "min_data_in_leaf", new[]{20, 50 ,100} },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 50) },
            //{ "num_leaves", new[]{3,15,50} },

            ////medium priority
            //{ "drop_rate", new[]{0.05, 0.1, 0.2}},                               //specific to dart mode
            { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            //{ "min_data_in_bin", new[]{3, 10, 100, 150}  },
            //{ "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            
            //{ "max_drop", new[]{40, 50, 60}},                                   //specific to dart mode

            //{ "skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},  //specific to dart mode

            ////low priority
            //{ "extra_trees", new[] { true , false } }, //low priority 
            ////{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)}, //very low priority
            //{ "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) }, //low priority
            //{ "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) }
*/
        };

        

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), new CFM84DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
        return (hpo.BestSampleFoundSoFar, hpo.ScoreOfBestSampleFoundSoFar);
    }


    // ReSharper disable once UnusedMember.Global
    public static void LaunchNeuralNetworkHPO(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //Dataset specific
            //{ "KFold", 3 },
            {"PercentageInTraining", new[]{0.8}},
            
            {"use_r_day_equity", new []{true , false}},                 //0.48218 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_day_market", new []{/*true , */ false}},             //0.48194 Valid Accuracy (False) vs 0.47576 Valid Accuracy (True) , but much better result in training
            {"use_r_dataset_equity", new []{true, false}},              //0.48294 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_day_equity", new []{true, false}},              //0.48194 Valid Accuracy (False) vs 0.47949 Valid Accuracy (True) 
            {"use_r_day_market", new []{true /*, false*/}},             //0.47531 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_market_correl_r_day_equity", new []{true, false}},    //0.48109 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_dataset_equity", new []{true, false}},          //0.48194 Valid Accuracy (False) vs 0.48116 Valid Accuracy (True)
            {"rr_count", new[]{0,1,2}},

            {"LossFunction", "CategoricalCrossentropy"},  //for multi class classification
            
            {"fillna_with_0", true},  //NaN are not supported in Neural Networks

            // Optimizer 
            { "OptimizerType", new[] { "AdamW" } },
            //{ "OptimizerType", "SGD" },
            { "AdamW_L2Regularization", new[] { 0.01 } },
            //{ "SGD_usenesterov", new[] { true, false } },
            //{ "lambdaL2Regularization", 0},

            // Learning Rate
            //{ "InitialLearningRate", AbstractHyperParameterSearchSpace.Range(1e-5f, 1f, AbstractHyperParameterSearchSpace.range_type.normal) },
            { "InitialLearningRate", new[]{ 0.001 } },
            // Learning Rate Scheduler
            ////{ "LearningRateSchedulerType", new[] { "CyclicCosineAnnealing", "OneCycle", "Linear" } },
            { "LearningRateSchedulerType", "CyclicCosineAnnealing"},
            //{ "EmbeddingDim", new[] { 4} },
            //{"weight_norm", new[]{true, false}},
            //{"leaky_relu", new[]{true, false}},
            //{ "dropout_top", new[] { 0, 0.1, 0.2 } },
            //{ "dropout_mid", new[] { 0, 0.3, 0.5 } },
            //{ "dropout_bottom", new[] { 0, 0.2, 0.4 } },
            
            { "BatchSize", new[] { 1024,2048,4096 } },
            { "NumEpochs", numEpochs },


            { "hidden_size", 1024 },
            { "channel_1", 256},
            { "channel_2", 512},
            { "channel_3", 512},

        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new NetworkSample_1DCNN(), new CFM84DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, false, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

}