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
        private static DoubleAccumulator[] Y_RAW_statistics { get; }
        private static DoubleAccumulator[] Y_RAW_abs_statistics { get; }

        public static readonly string[] MarketNames = { "VIX", "V2X", "EURUSD", "EURUSDV1M", "USGG10YR", "USGG2YR", "GDBR10YR", "GDBR2YR", "SX5E", "SPX", "SRVIX", "CVIX",  "MOVE"};
        public const string PredictionHeader =  ",Diff_VIX_1d,Diff_VIX_1w,Diff_VIX_2w,Diff_V2X_1d,Diff_V2X_1w,Diff_V2X_2w,Diff_EURUSD_1d,Diff_EURUSD_1w,Diff_EURUSD_2w,Diff_EURUSDV1M_1d,Diff_EURUSDV1M_1w,Diff_EURUSDV1M_2w,Diff_USGG10YR_1d,Diff_USGG10YR_1w,Diff_USGG10YR_2w,Diff_USGG2YR_1d,Diff_USGG2YR_1w,Diff_USGG2YR_2w,Diff_GDBR10YR_1d,Diff_GDBR10YR_1w,Diff_GDBR10YR_2w,Diff_GDBR2YR_1d,Diff_GDBR2YR_1w,Diff_GDBR2YR_2w,Diff_SX5E_1d,Diff_SX5E_1w,Diff_SX5E_2w,Diff_SPX_1d,Diff_SPX_1w,Diff_SPX_2w,Diff_SRVIX_1d,Diff_SRVIX_1w,Diff_SRVIX_2w,Diff_CVIX_1d,Diff_CVIX_1w,Diff_CVIX_2w,Diff_MOVE_1d,Diff_MOVE_1w,Diff_MOVE_2w";
        public const int EmbeddingDimension = 768;
        public static readonly string[] HorizonNames = { "1d", "1w", "2w" };


        public static void LaunchLightGBMHPO()
        {
            Utils.ConfigureGlobalLog4netProperties(Natixis70Utils.NatixisLogDirectory, "Natixis70");
            Utils.ConfigureThreadLog4netProperties(Natixis70Utils.NatixisLogDirectory, "Natixis70");
            int coreCount = Utils.CoreCount;

            // ReSharper disable once ConvertToConstant.Local
            var num_iterations = 1000;

            // number of parallel threads in each single training
            // ReSharper disable once ConvertToConstant.Local
            int numThreadsForEachModelTraining = 1;//single core

            var searchSpace = new Dictionary<string, object>
            {
                { "num_threads", new[] { numThreadsForEachModelTraining } },
                { "device_type", new[] { "cpu" } },
                { "num_iterations", new[]{ num_iterations } },
                { "early_stopping_round", new[] {num_iterations/10 } },

                { "MergeHorizonAndMarketIdInSameFeature",new[]{true/*, false*/} },
                //{ "Normalization",new[] { "NONE", "DIVIDE_BY_ABS_MEAN"} },
                { "extra_trees", new[] { true, false } },
                { "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) },

                { "bagging_fraction",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
                { "bagging_freq", new[]{0, 1} },
                { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
                { "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},

                { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.03f, 0.1f)},
                
                { "num_leaves", AbstractHyperParameterSearchSpace.Range(10, 100) },
                { "min_data_in_leaf", new[]{20,50} },
                //{ "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },

                { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 1f)},
                { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 1f)},

                { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
                { "min_data_in_bin", AbstractHyperParameterSearchSpace.Range(3, 50) },

                { "max_depth", new[]{-1, 10, 20, 50, 100} },
            };

            if (coreCount % numThreadsForEachModelTraining != 0)
            {
                throw new ArgumentException($"invalid number of threads {numThreadsForEachModelTraining} : core count {coreCount} must be a multiple of it");
            }
            int numModelTrainingInParallel = coreCount / numThreadsForEachModelTraining;

            //var hpo = new RandomSearchHPO<Natixis70HyperParameters>(searchSpace, DefaultParametersLightGBM, t => t.ToBeCalledAfterEachConstruction(), t => t.IsValid(), AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING, LightGBMModel.Log.Info);
            var hpo = new BayesianSearchHPO<Natixis70HyperParameters>(searchSpace, DefaultParametersLightGBM, t => t.ToBeCalledAfterEachConstruction(), t => t.IsValid(), NatixisLogDirectory, AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION.FULLY_RANDOM, 
                numModelTrainingInParallel, 
                1* numModelTrainingInParallel,
                5000*1* numModelTrainingInParallel,
                LightGBMModel.Log.Info, 
                10_000);

            // ReSharper disable once UselessBinaryOperation
            hpo.Process(numModelTrainingInParallel, t => TrainWithHyperParameters(t, ""));
        }

        static Natixis70Utils()
        {
            var cpuTensor = Dataframe.Load(YTrainRawFile, true, ',').Drop(new[] { "" }).Tensor;
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
                for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (HorizonNames.Length - 1) : marketId); ++currentMarketId)
                {
                    for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (HorizonNames.Length - 1) : horizonId); ++currentHorizonId)
                    {
                        int rawColIndex = 1 + HorizonNames.Length * currentMarketId + currentHorizonId;
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
        private static float TrainWithHyperParameters(Natixis70HyperParameters sample, string modelPrefix)
        {
            var model = new LightGBMModel(sample, NatixisLogDirectory, modelPrefix);
            Utils.ConfigureThreadLog4netProperties(NatixisLogDirectory, "Natixis70");

            using var xFullTraining = Load_X(XTrainRawFile, sample);
            using var yFullTraining = Load_Y(YTrainRawFile, sample);
            using var xTest = Load_X(XTestRawFile, sample);
            using var fullTraining  = NewDataSet(xFullTraining, yFullTraining, false, sample);
            using var test = NewDataSet(xTest, null, false, sample);

            int rowsInTrainingSet = (int)(sample.PercentageInTraining * fullTraining.Count + 0.1);
            rowsInTrainingSet -= rowsInTrainingSet % sample.RawCountToCount(1);
            using var trainAndValidation = fullTraining.SplitIntoTrainingAndValidation(rowsInTrainingSet);

            var sw = Stopwatch.StartNew();
            model.Train(trainAndValidation.Training, trainAndValidation.Test);

            var validationRmse = model.CreateModelResults(
                sample.SavePredictions,
                y => UnnormalizeYIfNeeded(y, sample),
                sw.Elapsed.TotalSeconds,
                sample.X_Shape(1)[1],
                trainAndValidation.Training,
                trainAndValidation.Test,
                test);
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
        #region load of datasets
        private static string NatixisLogDirectory => Path.Combine(NetworkConfig.DefaultLogDirectory, "Natixis70");
        private static string NatixisDataDirectory => Path.Combine(NetworkConfig.DefaultLogDirectory, "Natixis70", "Data");
        private static string XTrainRawFile => Path.Combine(Natixis70Utils.NatixisDataDirectory, "x_train_ACFqOMF.csv");
        private static string YTrainRawFile => Path.Combine(Natixis70Utils.NatixisDataDirectory, "y_train_HNMbC27.csv");
        private static string XTestRawFile => Path.Combine(Natixis70Utils.NatixisDataDirectory, "x_test_pf4T2aK.csv");
        private static InMemoryDataSet NewDataSet(CpuTensor<float> x, CpuTensor<float> yIfAny, bool useBackgroundThreadToLoadNextMiniBatch, Natixis70HyperParameters hyperParameters)
        {
            return new InMemoryDataSet(
                x,
                yIfAny,
                hyperParameters.IsTryingToPredictErrors ? "Natixis70Errors" : "Natixis70",
                Objective_enum.Regression,
                null,
                new[] { "NONE" },
                hyperParameters.ComputeFeatureNames(),
                useBackgroundThreadToLoadNextMiniBatch);
        }

        /// <summary>
        /// Load the content of the file  'xRawFile' in a CpuTensor and return it
        /// The returned CpuTensor must be manually disposed
        /// </summary>
        /// <param name="xRawFile"></param>
        /// <param name="hyperParameters"></param>
        /// <returns></returns>
        private static CpuTensor<float> Load_X(string xRawFile, Natixis70HyperParameters hyperParameters)
        {
            //We load 'xRaw'
            var xRawDataframe = Dataframe.Load(xRawFile, true, ',');
            var xRaw = xRawDataframe.Tensor;
            Debug.Assert(xRaw.Shape[1] == EmbeddingDimension);
            int count = xRaw.Shape[0];
            var xRawSpan = xRaw.AsReadonlyFloatCpuContent;
            var x = new CpuTensor<float>(hyperParameters.X_Shape(count));
            var xSpan = x.AsFloatCpuSpan;
            int xSpanIndex = 0;
            var divider = hyperParameters.RawCountToCount(1);

            for (int row = 0; row < x.Shape[0]; row++)
            {
                int rawRow = row / divider;
                int horizonId = hyperParameters.RowToHorizonId(row);
                int marketId = hyperParameters.RowToMarketId(row);

                //we load the row 'row' in 'x' tensor
                for (int col = 0; col < xRaw.Shape[1]; ++col)
                {
                    xSpan[xSpanIndex++] = xRawSpan[rawRow * xRaw.Shape[1] + col];
                }

                if (hyperParameters.MergeHorizonAndMarketIdInSameFeature)
                {
                    Debug.Assert(marketId >= 0);
                    Debug.Assert(horizonId >= 0);
                    xSpan[xSpanIndex++] = marketId * Natixis70Utils.HorizonNames.Length + horizonId;
                }
                else
                {
                    if (marketId >= 0)
                    {
                        xSpan[xSpanIndex++] = marketId;
                    }
                    if (horizonId >= 0)
                    {
                        xSpan[xSpanIndex++] = horizonId;
                    }
                }
            }
            Debug.Assert(xSpanIndex == x.Count);
            xRaw.Dispose();
            return x;
        }


        /// <summary>
        /// Load the content of the file  'yRawFileIfAny' in a CpuTensor and return it
        /// The returned CpuTensor must be manually disposed
        /// </summary>
        /// <param name="yRawFileIfAny"></param>
        /// <param name="hyperParameters"></param>
        /// <returns></returns>
        private static CpuTensor<float> Load_Y(string yRawFileIfAny, Natixis70HyperParameters hyperParameters)
        {
            Debug.Assert(File.Exists(yRawFileIfAny));
            var yRawDataframe = Dataframe.Load(yRawFileIfAny, true, ',');
            var yRaw = yRawDataframe.Tensor;
            var yRawSpan = yRaw.AsReadonlyFloatCpuContent;
            var y = new CpuTensor<float>(hyperParameters.Y_Shape(yRaw.Shape[0]));
            var ySpan = y.AsFloatCpuSpan;
            int ySpanIndex = 0;
            var divider = hyperParameters.RawCountToCount(1);

            for (int row = 0; row < y.Shape[0]; row++)
            {
                int rawRow = row / divider;
                int horizonId = hyperParameters.RowToHorizonId(row);
                int marketId = hyperParameters.RowToMarketId(row);
            
                //we load the row 'row' in 'y' tensor
                for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (HorizonNames.Length - 1) : marketId); ++currentMarketId)
                {
                    for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (HorizonNames.Length - 1) : horizonId); ++currentHorizonId)
                    {
                        int rawColIndex = 1 + HorizonNames.Length * currentMarketId + currentHorizonId;
                        var yRawValue = yRawSpan[rawRow * yRaw.Shape[1] + rawColIndex];
                        if (hyperParameters.Normalization == Natixis70HyperParameters.normalize_enum.MINUS_MEAN_DIVIDE_BY_VOL)
                        {
                            var colStatistics = Y_RAW_statistics[rawColIndex - 1];
                            yRawValue = (float)((yRawValue - colStatistics.Average) / colStatistics.Volatility);
                        }
                        else if (hyperParameters.Normalization == Natixis70HyperParameters.normalize_enum.DIVIDE_BY_ABS_MEAN)
                        {
                            var absColStatistics = Y_RAW_abs_statistics[rawColIndex - 1];
                            yRawValue = (float)(yRawValue / absColStatistics.Average);
                        }
                        ySpan[ySpanIndex++] = yRawValue;
                    }
                }
            }
            Debug.Assert(ySpanIndex == y.Count);
            yRaw.Dispose();
            return y;
        }
        #endregion
    }
}
