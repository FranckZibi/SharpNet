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
using SharpNet.Svm;

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
    public static string YTestRandomPath => Path.Combine(DataDirectory, "y_test_random_final.csv");

    public static Dictionary<string, double> IdToPrediction(string path)
    {
        var res = new Dictionary<string, double>();
        var y_pred_df = DataFrame.read_csv(path, true, ColumnNameToType);
        var y_ID = y_pred_df.StringColumnContent("ID");
        var y_reod = y_pred_df.FloatColumnContent("TARGET");
        for (int i = 0; i < y_ID.Length; ++i)
        {
            res.Add(y_ID[i], y_reod[i]);
        }
        return res;
    }


    public static void ReorderColumns()
    {
        for(int i=0;i<2;++i)
        {
            var datasetPath = (i == 0) ? XTrainPath : XTestPath;
            var dataset = DataFrame.read_csv(datasetPath, true, ColumnNameToType, true); ;
            var originalIdOrder = dataset.StringColumnContent("ID");
            var idToRow = new Dictionary<string, int>();
            for (int j = 0; j < originalIdOrder.Length; ++j)
            {
                idToRow[originalIdOrder[j]] = j;
            }
            var country_FR = dataset.StringColumnContent("COUNTRY").Select(c => c == "FR").ToArray();
            var dataset_FR = dataset.Filter(country_FR);
            dataset_FR = dataset_FR["ID", "DAY_ID", "COUNTRY", "FR_NET_EXPORT", "FR_CONSUMPTION", "FR_DE_EXCHANGE", "FR_GAS", "FR_COAL", "FR_HYDRO", "FR_NUCLEAR", "FR_SOLAR", "FR_WINDPOW", "FR_RESIDUAL_LOAD", "FR_RAIN", "FR_WIND", "FR_TEMP", "DE_NET_EXPORT", "DE_CONSUMPTION", "DE_GAS", "DE_COAL", "DE_HYDRO", "DE_NUCLEAR", "DE_SOLAR", "DE_WINDPOW", "DE_RESIDUAL_LOAD", "DE_RAIN", "DE_WIND", "DE_TEMP", "GAS_RET", "COAL_RET", "CARBON_RET", "DE_LIGNITE"];
            var country_DE = country_FR.Select(c => !c).ToArray();
            var dataset_DE = dataset.Filter(country_DE);
            dataset_DE = dataset_DE["ID", "DAY_ID", "COUNTRY", "DE_NET_EXPORT", "DE_CONSUMPTION", "DE_FR_EXCHANGE", "DE_GAS", "DE_COAL", "DE_HYDRO", "DE_NUCLEAR", "DE_SOLAR", "DE_WINDPOW", "DE_RESIDUAL_LOAD", "DE_RAIN", "DE_WIND", "DE_TEMP", "FR_NET_EXPORT", "FR_CONSUMPTION", "FR_GAS", "FR_COAL", "FR_HYDRO", "FR_NUCLEAR", "FR_SOLAR", "FR_WINDPOW", "FR_RESIDUAL_LOAD", "FR_RAIN", "FR_WIND", "FR_TEMP", "GAS_RET", "COAL_RET", "CARBON_RET", "DE_LIGNITE"];
            var newColumnNames = new[]
            {
                "ID", "DAY_ID", "COUNTRY",
                "NET_EXPORT", "CONSUMPTION", "EXCHANGE", "GAS", "COAL", "HYDRO", "NUCLEAR", "SOLAR", "WINDPOW", "RESIDUAL_LOAD", "RAIN", "WIND", "TEMP",
                "OTHER_NET_EXPORT", "OTHER_CONSUMPTION",                   "OTHER_GAS", "OTHER_COAL", "OTHER_HYDRO", "OTHER_NUCLEAR", "OTHER_SOLAR", "OTHER_WINDPOW", "OTHER_RESIDUAL_LOAD", "OTHER_RAIN", "OTHER_WIND", "OTHER_TEMP",
                "GAS_RET", "COAL_RET", "CARBON_RET", "DE_LIGNITE"
            };
            dataset_FR.SetColumnNames(newColumnNames);
            dataset_DE.SetColumnNames(newColumnNames);
            var merged = DataFrame.MergeVertically(dataset_FR, dataset_DE);

            var invalidIdOrder = merged.StringColumnContent("ID");
            var targetRowToSrcRow = new int[invalidIdOrder.Length];
            for (int srcRow = 0; srcRow < invalidIdOrder.Length; ++srcRow)
            {
                var id = invalidIdOrder[srcRow];
                var targetRow = idToRow[id];
                targetRowToSrcRow[targetRow] = srcRow;
            }
            merged = merged.ReorderRows(targetRowToSrcRow);
            merged.to_csv(datasetPath.Replace(".csv", "") + "_reordered.csv");
        }
    }


    public static void BuildStatNormalizeFile()
    {
        var train = DataFrame.read_csv(XTrainPath, true, ColumnNameToType, true);
        var test = DataFrame.read_csv(XTestPath, true, ColumnNameToType, true);

        foreach (var colDesc in train.ColumnsDesc.Where(c => c.Item2 == DataFrame.FLOAT_TYPE_IDX))
        {
            train.UpdateColumnsInPlace(f => QRT97DatasetSample.NormalizeReturn(f, colDesc.Item1), colDesc.Item1);
            test.UpdateColumnsInPlace(f => QRT97DatasetSample.NormalizeReturn(f, colDesc.Item1), colDesc.Item1);
        }
        train = train.sort_values("DAY_ID");
        train.to_csv(XTrainPath.Replace(".csv", "") + "_normalized.csv");
        test.to_csv(XTestPath.Replace(".csv", "") + "_normalized.csv");

        var ids = train.StringColumnContent("ID");
        var targets = new float[ids.Length];
        //we re order the y train dataset so it has the same order as the x train dataset
        var idToPrediction = IdToPrediction(YTrainPath);
        if (idToPrediction.Count != ids.Length)
        {
            throw new Exception("y_train_dico.Count != ids.Length");
        }
        for (int i = 0;i < ids.Length; ++i)
        {
            targets[i] = (float)idToPrediction[ids[i]];
        }
        var y_train_df = DataFrame.MergeHorizontally(DataFrame.New(ids, new[] { "ID" }), DataFrame.New(targets, new[] { "TARGET" }));
        y_train_df.to_csv(YTrainPath.Replace(".csv", "") + "_normalized.csv");
    }




    public static void Run()
    {
        //ChallengeTools.Retrain(@"C:\Projects\Challenges\QRT97\dump", "E8A881A122", null, 0.8, false);
        //ChallengeTools.ComputeAndSaveFeatureImportance(@"C:\Projects\Challenges\QRT97\Submit", "9B9DFA97F2_KFOLD_FULL");
        //ChallengeTools.EstimateLossContribution(@"C:\Projects\Challenges\QRT97\dump", "D885B11678");
        //ChallengeTools.Retrain(@"C:\Projects\Challenges\QRT97\dump", "9B9DFA97F2_KFOLD_FULL", null, 0.8, false);


        //BuildStatNormalizeFile();
        //ReorderColumns();
        //LaunchLightGBMHPO(50, 2000);
        LaunchCatBoostHPO(500, 1000);
        //LaunchSvmHPO();
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
            //{"KFold", 3},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            //{"fillna_with_0", true},  //NaN are not supported in Device (GPU)
            {"use_DAY_ID", false},
            {"use_COUNTRY", true},


            {"loss_function", "RMSE"},
            { "logging_level", nameof(CatBoostSample.logging_level_enum.Verbose)},
            { "allow_writing_files",false},
            { "thread_count",1},

            { "task_type","GPU"},

            { "iterations", AbstractHyperParameterSearchSpace.Range(iterations_min, iterations_max)},
            //{ "od_type", "Iter"},
            //{ "od_wait",iterations/10},
            { "depth", AbstractHyperParameterSearchSpace.Range(2, 10) },
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 1.00f)},
            { "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
            { "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
            { "l2_leaf_reg",AbstractHyperParameterSearchSpace.Range(0, 20)},
            //{"grow_policy", new []{ "SymmetricTree", "Depthwise" /*, "Lossguide"*/}},
            //{"boosting_type", new []{ "Ordered", "Plain"}},
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
            {"KFold", 5},
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
          
            
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



    public static (ISample bestSample, IScore bestScore) LaunchSvmHPO(int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            {"KFold", 5},
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            
            //{"n_fold_svm", 5},
            {"use_DAY_ID", false},
            {"use_COUNTRY", new []{true, false}},

      
            {"svm_type", new[]{"epsilon_SVR", "nu_SVR"}},      //for Regression Tasks
            //{"svm_type", "nu_SVR"},       //for Regression Tasks

            { "kernel_type", "radial_basis_function" },
            //{ "kernel_type", new[]{ "linear", "polynomial", "radial_basis_function" /*, "sigmoid"*/ } },
            
            { "nu", AbstractHyperParameterSearchSpace.Range(0.0f, 1.0f) },
            { "cost", new[]{ 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 100 } },
            { "gamma", new[]{ 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 100 } },
            { "epsilon_SVR", new[]{ 0.1, 0.05, 0.01,0.001,0.0001} },
            
            
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new SVMSample(), new QRT97DatasetSample()), WorkingDirectory);
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