using System.Collections.Generic;
using System.IO;
using SharpNet.CatBoost;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;

namespace SharpNet.Datasets.Natixis70
{
    public static class Natixis70Utils
    {
        public const string NAME = "Natixis70";

        public static readonly string[] MarketNames = { "VIX", "V2X", "EURUSD", "EURUSDV1M", "USGG10YR", "USGG2YR", "GDBR10YR", "GDBR2YR", "SX5E", "SPX", "SRVIX", "CVIX", "MOVE" };
        public const string PredictionHeader = ",Diff_VIX_1d,Diff_VIX_1w,Diff_VIX_2w,Diff_V2X_1d,Diff_V2X_1w,Diff_V2X_2w,Diff_EURUSD_1d,Diff_EURUSD_1w,Diff_EURUSD_2w,Diff_EURUSDV1M_1d,Diff_EURUSDV1M_1w,Diff_EURUSDV1M_2w,Diff_USGG10YR_1d,Diff_USGG10YR_1w,Diff_USGG10YR_2w,Diff_USGG2YR_1d,Diff_USGG2YR_1w,Diff_USGG2YR_2w,Diff_GDBR10YR_1d,Diff_GDBR10YR_1w,Diff_GDBR10YR_2w,Diff_GDBR2YR_1d,Diff_GDBR2YR_1w,Diff_GDBR2YR_2w,Diff_SX5E_1d,Diff_SX5E_1w,Diff_SX5E_2w,Diff_SPX_1d,Diff_SPX_1w,Diff_SPX_2w,Diff_SRVIX_1d,Diff_SRVIX_1w,Diff_SRVIX_2w,Diff_CVIX_1d,Diff_CVIX_1w,Diff_CVIX_2w,Diff_MOVE_1d,Diff_MOVE_1w,Diff_MOVE_2w";
        public const int EmbeddingDimension = 768;
        public static readonly string[] HorizonNames = { "1d", "1w", "2w" };

        #region load of datasets
        public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
        public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
        // ReSharper disable once MemberCanBePrivate.Global
        #endregion

        // ReSharper disable once UnusedMember.Global
        public static void LaunchLightGBMHPO()
        {
            // ReSharper disable once ConvertToConstant.Local
            var num_iterations = 1000;

            num_iterations = 10;

            var searchSpace = new Dictionary<string, object>
            {
                //related to Dataset 
                { "TryToPredictAllHorizonAtTheSameTime", false},
                { "MergeHorizonAndMarketIdInSameFeature",new[]{true/*, false*/} },
                //{ "Normalization",new[] { "NONE", "DIVIDE_BY_ABS_MEAN"} },

                //related to LightGBM model
                { "metric", "rmse" },
                { "objective", "regression" },
                { "verbosity", "0" },
                { "num_threads", 1},
                { "num_iterations", num_iterations },
                { "early_stopping_round", num_iterations/10 },

                { "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
                //{ "bagging_fraction",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
                { "bagging_freq", new[]{0, 1} },
                { "colsample_bytree", AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
                { "colsample_bynode", AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
                { "extra_trees", new[] { true /*, false*/ } }, //bad: false
                { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 1f)},
                { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 1f)},
                { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.03f, 0.07f)}, //for 1.000 trees
                //{ "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.03f)}, //for 10.000trees
                { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
                { "max_depth", new[]{10, 20, 50, 100, 255} },
                { "min_data_in_bin", AbstractHyperParameterSearchSpace.Range(3, 100) },
                { "min_data_in_leaf", AbstractHyperParameterSearchSpace.Range(20, 200) },
                { "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },
                { "num_leaves", AbstractHyperParameterSearchSpace.Range(5, 60) },
                { "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) },
            };

            var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), new Natixis70DatasetSample()), WorkingDirectory);
            IScore bestScoreSoFar = null;
            hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar)
            );
        }

        // ReSharper disable once UnusedMember.Global
        public static void LaunchCatBoostHPO()
        {
            // ReSharper disable once ConvertToConstant.Local
            var iterations = 1000;
            var searchSpace = new Dictionary<string, object>
            {
                //related to Dataset 
                { "TryToPredictAllHorizonAtTheSameTime", false},
                { "MergeHorizonAndMarketIdInSameFeature",new[]{true/*, false*/} },
                //{ "Normalization",new[] { "NONE", "DIVIDE_BY_ABS_MEAN"} },

                //related to CatBoost model
                { "eval_metric","RMSE"},
                { "loss_function","RMSE"},
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

            var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new Natixis70DatasetSample()), WorkingDirectory);
            IScore bestScoreSoFar = null;
            hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar));
        }

        // ReSharper disable once UnusedMember.Global
        //public static void SearchForBestWeights()
        //{
        //    WeightsOptimizer.SearchForBestWeights(
        //        new List<Tuple<string, string>>
        //        {
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "9736A5F52A"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "6301C10A9E"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "C8909AE935"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "D805551FDC"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "94648F9CA7"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "32AB0D5D2F"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "FD056E8CA9"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "60E67A6BCF"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "0F24432913"),
        //        },
        //        Path.Combine(WorkingDirectory, nameof(WeightsOptimizer)),
        //        Path.Combine(DataDirectory, "Tests_" + NAME + ".csv"));
        //}

        //public static void SearchForBestWeights_full_Dataset()
        //{
        //    WeightsOptimizer.SearchForBestWeights(
        //        new List<Tuple<string, string>>
        //        {
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "41C776CB10"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "D324191822"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "E9F2139538"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "FC18503756"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "2DAA3D22BD"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "832172A5DB"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "89D2FB42ED"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "22FD7C720F"),
        //            Tuple.Create(Path.Combine(WorkingDirectory, "aaa3"), "604A1690F4"),
        //        },
        //        Path.Combine(WorkingDirectory, nameof(WeightsOptimizer)),
        //        Path.Combine(DataDirectory, "Tests_" + NAME + ".csv"));
        //}
    }
}
