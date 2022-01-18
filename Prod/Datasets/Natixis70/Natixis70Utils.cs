using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.CPU;
using SharpNet.HPO;
using SharpNet.LightGBM;
using SharpNet.MathTools;
using SharpNet.Networks;

namespace SharpNet.Datasets.Natixis70
{
    public static class Natixis70Utils
    {
        public static readonly string[] MARKET_NAMES = { "VIX", "V2X", "EURUSD", "EURUSDV1M", "USGG10YR", "USGG2YR", "GDBR10YR", "GDBR2YR", "SX5E", "SPX", "SRVIX", "CVIX",  "MOVE"};
        public const string PREDICTION_HEADER =  ",Diff_VIX_1d,Diff_VIX_1w,Diff_VIX_2w,Diff_V2X_1d,Diff_V2X_1w,Diff_V2X_2w,Diff_EURUSD_1d,Diff_EURUSD_1w,Diff_EURUSD_2w,Diff_EURUSDV1M_1d,Diff_EURUSDV1M_1w,Diff_EURUSDV1M_2w,Diff_USGG10YR_1d,Diff_USGG10YR_1w,Diff_USGG10YR_2w,Diff_USGG2YR_1d,Diff_USGG2YR_1w,Diff_USGG2YR_2w,Diff_GDBR10YR_1d,Diff_GDBR10YR_1w,Diff_GDBR10YR_2w,Diff_GDBR2YR_1d,Diff_GDBR2YR_1w,Diff_GDBR2YR_2w,Diff_SX5E_1d,Diff_SX5E_1w,Diff_SX5E_2w,Diff_SPX_1d,Diff_SPX_1w,Diff_SPX_2w,Diff_SRVIX_1d,Diff_SRVIX_1w,Diff_SRVIX_2w,Diff_CVIX_1d,Diff_CVIX_1w,Diff_CVIX_2w,Diff_MOVE_1d,Diff_MOVE_1w,Diff_MOVE_2w";
        public const int EMBEDDING_DIMENSION = 768;
        public static readonly string[] HORIZON_NAMES = { "1d", "1w", "2w" };
        public static DoubleAccumulator[] Y_RAW_statistics { get; }
        public static DoubleAccumulator[] Y_RAW_abs_statistics { get; }

        public static void LaunchLightGBMHPO()
        {
            Utils.ConfigureGlobalLog4netProperties(Natixis70Utils.NatixisLogDirectory, "Natixis70");
            Utils.ConfigureThreadLog4netProperties(Natixis70Utils.NatixisLogDirectory, "Natixis70");

            const int num_iterations = 1000;
            var searchSpace = new Dictionary<string, object>
            {
                { "num_threads", new[]{1} }, //single core
                { "device_type", new[] { "cpu" } },
                { "num_iterations", new[]{ num_iterations } },
                { "early_stopping_round", new[] {num_iterations/10 } },

                { "MergeHorizonAndMarketIdInSameFeature",new[]{true, false} },
                { "Normalization",new[] { "NONE", "DIVIDE_BY_ABS_MEAN"} },
                { "extra_trees", new[] { true, false } },
                { "min_sum_hessian_in_leaf", new[] { 1e-3, 0.01, 0.1 } },
                { "min_data_in_bin", new[] { 3 ,10 } },
                { "path_smooth", new[] { 0 , 0.1, 1 } },
                { "learning_rate",new[]{ /*0.04, */ 0.05, /*0.06,*/ 0.1 } },
                { "num_leaves", new[]{25, /*50,*/ 100} }, //bad: 15
                { "lambda_l1", new[]{0, /*0.01, 0.1*/ 1}},
                { "lambda_l2", new[]{0, /*0.01, 0.1*/ 1}},
                //{ "max_bin", new[]{10,50,255} },
                { "min_data_in_leaf", new[]{20,50} },
            //{ "max_depth", new[]{-1, 10, 20, 50, 100} },
            //{ "bagging_fraction", new[]{0.8, 1} },
            //{ "bagging_freq", new[]{0, 1} },
            //{ "colsample_bytree", new[]{0.7, 0.8, 0.9, 1.0}}, //bad: 0.5
            //{ "colsample_bynode", new[]{0.5,1.0}},
        };

            var hpo = new RandomSearchHPO<Natixis70HyperParameters>(searchSpace, DefaultParametersLightGBM, t => t.ToBeCalledAfterEachConstruction(), t => t.IsValid(), HyperParameterSearchSpace.RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING);
            hpo.Process(Utils.CoreCount, t => TrainWithHyperParameters(t, ""), LightGBMModel.Log.Info);
        }

        static Natixis70Utils()
        {
            var cpuTensor = Dataframe.Load(YTrainRawFileIfAny, true, ',').Drop(new[] { "" }).Tensor;
            Y_RAW_statistics = ExtractColumnStatistic(cpuTensor, false);
            Y_RAW_abs_statistics = ExtractColumnStatistic(cpuTensor, true);
        }
        private static Natixis70HyperParameters DefaultParametersLightGBM()
        {
            var hyperParameters = new Natixis70HyperParameters();
            hyperParameters.TryToPredictAllHorizonAtTheSameTime = false;
            hyperParameters.verbosity = 0;
            hyperParameters.objective = Parameters.objective_enum.regression;
            hyperParameters.metric = "rmse";
            hyperParameters.Update_categorical_feature_field();
            hyperParameters.num_threads = 1;
            hyperParameters.device_type = Parameters.device_type_enum.cpu;
            hyperParameters.num_iterations = 1000;

            //to optimize in HPO
            hyperParameters.bagging_fraction = 1.0;
            hyperParameters.bagging_freq = 0;
            hyperParameters.colsample_bytree = 1;
            hyperParameters.extra_trees = true;
            hyperParameters.lambda_l1 = 0.0;
            hyperParameters.lambda_l2 = 0.0;
            hyperParameters.learning_rate = 0.05;
            hyperParameters.max_bin = 255;
            hyperParameters.max_depth = 50;
            hyperParameters.min_data_in_leaf = 50;
            hyperParameters.num_leaves = 100;

            return hyperParameters;

        }
        private static CpuTensor<float> UnnormalizeYIfNeeded(CpuTensor<float> y, Natixis70HyperParameters hyperParameters)
        {
            if (hyperParameters.Normalization == Natixis70HyperParameters.normalize_enum.NONE)
            {
                return y; //no need to unnormalize y
            }

            var ySpan = y.AsReadonlyFloatCpuContent;
            var Yunnormalized = new CpuTensor<float>(y.Shape);
            var YunnormalizedSpan = Yunnormalized.AsFloatCpuSpan;

            int index = 0;
            for (int row = 0; row < Yunnormalized.Shape[0]; ++row)
            {
                var horizonId = hyperParameters.RowToHorizonId(row);
                var marketId = hyperParameters.RowToMarketId(row);
                //we load the row 'row' in 'y' tensor
                for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (HORIZON_NAMES.Length - 1) : marketId); ++currentMarketId)
                {
                    for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (HORIZON_NAMES.Length - 1) : horizonId); ++currentHorizonId)
                    {
                        int rawColIndex = 1 + HORIZON_NAMES.Length * currentMarketId + currentHorizonId;
                        var yValue = ySpan[index];
                        if (hyperParameters.Normalization == Natixis70HyperParameters.normalize_enum.MINUS_MEAN_DIVIDE_BY_VOL)
                        {
                            var colStatistics = Y_RAW_statistics[rawColIndex - 1];
                            var yUnnormalizedValue = (float)(yValue * colStatistics.Volatility + colStatistics.Average);
                            YunnormalizedSpan[index] = yUnnormalizedValue;
                        }
                        else if (hyperParameters.Normalization == Natixis70HyperParameters.normalize_enum.DIVIDE_BY_ABS_MEAN)
                        {
                            var absColStatistics = Y_RAW_abs_statistics[rawColIndex - 1];
                            var yUnnormalizedValue = (float)(yValue * absColStatistics.Average);
                            YunnormalizedSpan[index] = yUnnormalizedValue;
                        }

                        ++index;
                    }
                }
            }
            return Yunnormalized;
        }
        private static double TrainWithHyperParameters(Natixis70HyperParameters hyperParameters, string modelPrefix)
        {
            var model = new LightGBMModel(hyperParameters, NatixisLogDirectory, modelPrefix);
            Utils.ConfigureThreadLog4netProperties(Natixis70Utils.NatixisLogDirectory, "Natixis70");


            using var fullTraining  = Natixis70DataSet.ValueOf(XTrainRawFile, YTrainRawFileIfAny, false, LightGBMModel.Log.Info, hyperParameters);
            using var Test = Natixis70DataSet.ValueOf(XTestRawFile, null, false, LightGBMModel.Log.Info, hyperParameters);

            int rowsInTrainingSet = (int)(hyperParameters.PercentageInTraining * fullTraining.Count + 0.1);
            rowsInTrainingSet -= rowsInTrainingSet % hyperParameters.RawCountToCount(1);
            var trainAndValidation = fullTraining.SplitIntoTrainingAndValidation(rowsInTrainingSet);

            var sw = Stopwatch.StartNew();
            model.Train(trainAndValidation.Training, trainAndValidation.Test);

            var (_, validationRmse) = model.CreateModelResults(
                hyperParameters.SavePredictions,
                y => UnnormalizeYIfNeeded(y, hyperParameters),
                sw.Elapsed.TotalSeconds,
                hyperParameters.X_Shape(1)[1],
                trainAndValidation.Training,
                trainAndValidation.Test,
                Test);
            return validationRmse;
        }
        /// <summary>
        /// return the statistics (average/volatility) of each column of the matrix 'y'
        /// </summary>
        /// <param name="y">a 2D tensor</param>
        /// <param name="useAbsValues"></param>
        /// <returns></returns>
        private static DoubleAccumulator[] ExtractColumnStatistic(CpuTensor<float> y, bool useAbsValues)
        {
            Debug.Assert(y.Shape.Length == 2); //only works for matrices
            var ySpan = y.AsReadonlyFloatCpuContent;

            var result = new List<DoubleAccumulator>();
            while (result.Count < y.Shape[1])
            {
                result.Add(new DoubleAccumulator());
            }

            for (int i = 0; i < ySpan.Length; ++i)
            {
                var yValue = ySpan[i];
                if (useAbsValues)
                {
                    yValue = Math.Abs(yValue);
                }
                result[i % y.Shape[1]].Add(yValue, 1);
            }
            return result.ToArray();
        }
        private static string NatixisLogDirectory => Path.Combine(NetworkConfig.DefaultLogDirectory, "Natixis70");
        private static string NatixisDataDirectory => Path.Combine(NetworkConfig.DefaultLogDirectory, "Natixis70", "Data");
        private static string XTrainRawFile => Path.Combine(Natixis70Utils.NatixisDataDirectory, "x_train_ACFqOMF.csv");
        private static string YTrainRawFileIfAny => Path.Combine(Natixis70Utils.NatixisDataDirectory, "y_train_HNMbC27.csv");
        private static string XTestRawFile => Path.Combine(Natixis70Utils.NatixisDataDirectory, "x_test_pf4T2aK.csv");
        

    }
}
