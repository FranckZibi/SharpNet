using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using log4net;
using SharpNet.CatBoost;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Networks;

namespace SharpNet.Datasets.QRT97;

public static class QRT97Utils
{
    public const string NAME = "QRT97";


    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(QRT97Utils));
    #endregion

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    // ReSharper disable once MemberCanBePrivate.Global

    public static string XTrainPath => Path.Combine(DataDirectory, "X_train_NHkHMNU.csv");
    public static string YTrainPath => Path.Combine(DataDirectory, "y_train_ZAN5mwg.csv");
    public static string XTestPath => Path.Combine(DataDirectory, "X_test_final.csv");
    public static string OutputTestRandomPath => Path.Combine(DataDirectory, "y_test_random_final.csv");


    public static float NormalizeReturn(float r)
    {
        if (float.IsNaN(r) || Math.Abs(r) > 10000)
        {
            return float.NaN;
        }

        return r / 10000;
    }
    public static Dictionary<string, string> PredToDico(string path)
    {
        var res = new Dictionary<string, string>();
        var y_pred_df = DataFrame.read_string_csv(path, true, true);
        var y_ID = y_pred_df.StringColumnContent("ID");
        var y_reod = y_pred_df.StringColumnContent("TARGET");
        for (int i = 0; i < y_ID.Length; ++i)
        {
            res.Add(y_ID[i], y_reod[i]);
        }
        return res;
    }



    public static void BuildStatNormalizeFile()
    {
        var train = DataFrame.read_csv(XTrainPath, true, ColumnNameToType, true);
        var return_columns = train.Columns.Where(c => c.StartsWith("r")).ToArray();
        train.UpdateColumnsInPlace(NormalizeReturn, return_columns);

        train = train.sort_values("DAY_ID");
        train.to_csv(XTrainPath.Replace(".csv", "") + "_normalized.csv");

        //we re order the y train dataset so it has the same order as the x train dataset
        var y_train_dico = QRT97Utils.PredToDico(YTrainPath);
        var y_train_ordered = new List<string>();
        foreach (var sorted_id in train.StringColumnContent("ID"))
        {
            y_train_ordered.Add(sorted_id);
            y_train_ordered.Add(y_train_dico[sorted_id]);
        }
        DataFrame.New(y_train_ordered.ToArray(), new[] { "ID", "TARGET" }).to_csv(YTrainPath.Replace(".csv", "") + "_normalized.csv");
        var test = DataFrame.read_csv(XTestPath, true, ColumnNameToType, true);
        test.UpdateColumnsInPlace(NormalizeReturn, return_columns);
        test.to_csv(XTestPath.Replace(".csv", "") + "_normalized.csv");
    }




    public static void Run()
    {
        //BuildStatNormalizeFile();
        //LaunchLightGBMHPO(50, 2000);
        LaunchCatBoostHPO(50, 2000);
    }


    public static Type ColumnNameToType(string columnName)
    {
        if (columnName.Equals("ID") || columnName.Equals("COUNTRY") )
        {
            return typeof(string);
        }
        if (columnName.Equals("DAY_ID"))
        {
            return typeof(int);
        }
        return typeof(float);
    }

    public static void LaunchCatBoostHPO(int iterations_min = 10, int iterations_max = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        // ReSharper disable once ConvertToConstant.Local
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled


            {"loss_function", "MAE"},
            { "logging_level", nameof(CatBoostSample.logging_level_enum.Verbose)},
            { "allow_writing_files",false},
            { "thread_count",1},
            { "iterations", AbstractHyperParameterSearchSpace.Range(iterations_min, iterations_max)},
            //{ "od_type", "Iter"},
            //{ "od_wait",iterations/10},
            { "depth", AbstractHyperParameterSearchSpace.Range(2, 10) },
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 1.00f)},
            { "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
            { "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
            { "l2_leaf_reg",AbstractHyperParameterSearchSpace.Range(0, 10)},
            //{"grow_policy", new []{ "SymmetricTree", "Depthwise" /*, "Lossguide"*/}},

      };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new QRT97DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = true;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }
    
    public static (ISample bestSample, IScore bestScore) LaunchLightGBMHPO(int num_iterations_min = 100, int num_iterations_max = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
          
            { "num_threads", -1},
            { "verbosity", "0" },
            {"objective", "regression_l1"},      //for Regression Tasks
            
            //high priority
            { "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
            { "bagging_freq", new[]{0, 10} },
            { "boosting", new []{"gbdt", "dart"}},
            { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.3f, 1.0f)},
            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.005f, 0.2f)},
            { "max_depth", new[]{10, 20, 50, 100, 255} },
            { "min_data_in_leaf", new[]{2,5,20, 50 /*,100*/} },
            { "num_iterations", AbstractHyperParameterSearchSpace.Range(num_iterations_min , num_iterations_max)},
            //{ "early_stopping_round", num_iterations/10 },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 50) },

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

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), new QRT97DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = true;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
        return (hpo.BestSampleFoundSoFar, hpo.ScoreOfBestSampleFoundSoFar);
    }


    public static void LaunchNeuralNetworkHPO(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //Dataset specific
            //{ "KFold", 3 },
            {"PercentageInTraining", new[]{0.8}},
            
           
            {"LossFunction", "Mae"},                     //for Regression Tasks: Rmse, Mse, Mae, etc.
            
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

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new NetworkSample_1DCNN(), new QRT97DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, false, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

}