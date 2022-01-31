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
            Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, "Natixis70");
            Utils.ConfigureThreadLog4netProperties(WorkingDirectory, "Natixis70");
            int coreCount = Utils.CoreCount;

            // ReSharper disable once ConvertToConstant.Local
            var num_iterations = 1000;

            var searchSpace = new Dictionary<string, object>
            {
                { "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
                //{ "bagging_fraction",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
                { "bagging_freq", new[]{0, 1} },

                { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
                { "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},

                { "device_type", new[] { "cpu" } },

                { "early_stopping_round", new[] {num_iterations/10 } },
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
                { "MergeHorizonAndMarketIdInSameFeature",new[]{true/*, false*/} },

                //{ "Normalization",new[] { "NONE", "DIVIDE_BY_ABS_MEAN"} },
                { "num_iterations", new[]{ num_iterations } },
                { "num_leaves", AbstractHyperParameterSearchSpace.Range(5, 60) },
                { "num_threads", new[] { 1 } },

                { "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) },

            };

            var hpo = new BayesianSearchHPO<Natixis70HyperParameters>(searchSpace,
                DefaultParametersLightGBM, t => t.PostBuild(), 
                0,  //no time limit
                AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING, 
                WorkingDirectory, 
                Natixis70HyperParameters.CategoricalHyperParameters());

            // ReSharper disable once UselessBinaryOperation
            hpo.Process(t => TrainWithHyperParameters(t, ""));
        }


        private static string DatasetHPOPath => Path.Combine(WorkingDirectory, "DatasetHPO");

        public static void DatasetHPO(string dataframePath, 
            IList<string> categoricalFeatures,
            Parameters.boosting_enum boosting, 
            int num_iterations)
        {
            if (!Directory.Exists(DatasetHPOPath))
            {
                Directory.CreateDirectory(DatasetHPOPath);
            }


            LightGBMModel.Log.Info($"Performing HPO for Dataframe {dataframePath}");

            //We load the dataset
            var dataframe = Dataframe.Load(dataframePath, true, ',');
            var xRawDataframe = dataframe.Drop(new[] { "y" });
            var yRawDataframe = dataframe.Keep(new[] { "y" });
            var fullTraining = new InMemoryDataSet(
                xRawDataframe.Tensor,
                yRawDataframe.Tensor,
                Path.GetFileNameWithoutExtension(dataframePath),
                Objective_enum.Regression,
                null,
                null,
                xRawDataframe.FeatureNames,
                false);
            const double percentageInTraining = 0.67;
            int rowsInTrainingSet = (int)(percentageInTraining * fullTraining.Count + 0.1);
            var trainAndValidation = fullTraining.SplitIntoTrainingAndValidation(rowsInTrainingSet);
            LightGBMModel.Log.Info($"Training Dataset: {trainAndValidation.Training} , Validation Dataset: {trainAndValidation.Test}");

            var searchSpace = new Dictionary<string, object>
            {
                { "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
                { "bagging_freq", new[]{0, 1} },
                { "boosting", new[] { boosting.ToString() } },

                { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
                { "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},


                { "device_type", new[] { "cpu" } },
                //{ "early_stopping_round", new[] {num_iterations/10 } },
                { "extra_trees", new[] { true , false } },

                { "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) },

                { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 1f)},
                { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 1f)},
                { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.0001f, 0.5f)},
                
                { "min_data_in_leaf", AbstractHyperParameterSearchSpace.Range(3, 200) },
                { "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },

                { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
                { "max_depth", new[]{10, 20, 50, 100, 255} },
                { "min_data_in_bin", AbstractHyperParameterSearchSpace.Range(3, 100) },

                { "num_iterations", new[]{ num_iterations } },
                { "num_threads", new[] { 1 } },
                { "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 200) },
            };

            var categoricalFeaturesFieldValue = (categoricalFeatures.Count >= 1) ? ("name:" + string.Join(',', categoricalFeatures)) : "";
            //var hpo = new RandomSearchHPO<ParametersWithPercentageInTraining>(searchSpace,
            //    () => new ParametersWithPercentageInTraining(),
            //    t =>
            //    {
            //        t.categorical_feature = categoricalFeaturesFieldValue;
            //        return t.ValidLightGBMHyperParameters();
            //    },
            //    AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION.FULLY_RANDOM,
            //    WorkingDirectory,
            //    0);

            var hpo = new BayesianSearchHPO<Parameters>(searchSpace,
                () => new Parameters(),
                t =>
                {
                    t.categorical_feature = categoricalFeaturesFieldValue;
                    return t.ValidLightGBMHyperParameters();
                },
                0, // no time limit
                AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION.FULLY_RANDOM, 
                WorkingDirectory, 
                new HashSet<string>(categoricalFeatures));

            // ReSharper disable once UselessBinaryOperation
            var minValidationScore = float.MaxValue;

         
            hpo.Process(t => DatasetHPO(t, trainAndValidation.Training, trainAndValidation.Test, ref minValidationScore));
        }
        private static float DatasetHPO(Parameters sample, IDataSet trainDataset, IDataSet validationDataset, ref float minValidationScore)
        {
            var model = new LightGBMModel(sample, DatasetHPOPath, "");
            var sw = Stopwatch.StartNew();
            model.Train(trainDataset, validationDataset);

            var minValidationScoreValue = minValidationScore;
            var validationRmse = model.CreateModelResults(
                (_,_)=> {},
                y => y,
                validationScore => validationScore < minValidationScoreValue,
                sw.Elapsed.TotalSeconds,
                trainDataset.X_if_available.Shape[1],
                trainDataset,
                validationDataset,
                null);

            if (validationRmse < minValidationScore)
            {
                minValidationScore = validationRmse;
            }

            return validationRmse;
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

            if (!hyperParameters.PostBuild())
            {
                throw new Exception($"invalid hyperParameters {hyperParameters}");
            }


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
            var model = new LightGBMModel(sample, WorkingDirectory, modelPrefix);

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
                validationScore => validationScore< Min_Validation_Score_To_Save_Predictions,
                sw.Elapsed.TotalSeconds,
                sample.X_Shape(1)[1],
                trainAndValidation.Training,
                trainAndValidation.Test,
                test);
            return validationRmse;
        }

        private const double Min_Validation_Score_To_Save_Predictions = 19.5;


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
        public static string WorkingDirectory => Path.Combine(NetworkConfig.DefaultLogDirectory, "Natixis70");
        private static string NatixisDataDirectory => Path.Combine(NetworkConfig.DefaultLogDirectory, "Natixis70", "Data");
        private static string XTrainRawFile => Path.Combine(NatixisDataDirectory, "x_train_ACFqOMF.csv");
        private static string YTrainRawFile => Path.Combine(NatixisDataDirectory, "y_train_HNMbC27.csv");
        private static string XTestRawFile => Path.Combine(NatixisDataDirectory, "x_test_pf4T2aK.csv");
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
