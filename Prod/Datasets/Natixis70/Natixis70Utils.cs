using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.CPU;
using SharpNet.HPO;
using SharpNet.LightGBM;
using SharpNet.MathTools;
using SharpNet.Models;

namespace SharpNet.Datasets.Natixis70
{
    public static class Natixis70Utils
    {
        public static DoubleAccumulator[] Y_RAW_statistics { get; }
        public static DoubleAccumulator[] Y_RAW_abs_statistics { get; }

        public static readonly string[] MarketNames = { "VIX", "V2X", "EURUSD", "EURUSDV1M", "USGG10YR", "USGG2YR", "GDBR10YR", "GDBR2YR", "SX5E", "SPX", "SRVIX", "CVIX",  "MOVE"};
        public const string PredictionHeader =  ",Diff_VIX_1d,Diff_VIX_1w,Diff_VIX_2w,Diff_V2X_1d,Diff_V2X_1w,Diff_V2X_2w,Diff_EURUSD_1d,Diff_EURUSD_1w,Diff_EURUSD_2w,Diff_EURUSDV1M_1d,Diff_EURUSDV1M_1w,Diff_EURUSDV1M_2w,Diff_USGG10YR_1d,Diff_USGG10YR_1w,Diff_USGG10YR_2w,Diff_USGG2YR_1d,Diff_USGG2YR_1w,Diff_USGG2YR_2w,Diff_GDBR10YR_1d,Diff_GDBR10YR_1w,Diff_GDBR10YR_2w,Diff_GDBR2YR_1d,Diff_GDBR2YR_1w,Diff_GDBR2YR_2w,Diff_SX5E_1d,Diff_SX5E_1w,Diff_SX5E_2w,Diff_SPX_1d,Diff_SPX_1w,Diff_SPX_2w,Diff_SRVIX_1d,Diff_SRVIX_1w,Diff_SRVIX_2w,Diff_CVIX_1d,Diff_CVIX_1w,Diff_CVIX_2w,Diff_MOVE_1d,Diff_MOVE_1w,Diff_MOVE_2w";
        public const int EmbeddingDimension = 768;
        public static readonly string[] HorizonNames = { "1d", "1w", "2w" };

        #region load of datasets
        public static string WorkingDirectory => Path.Combine(Utils.LocalApplicationFolderPath, "SharpNet", "Natixis70");
        public static string NatixisDatasetDirectory => Path.Combine(WorkingDirectory, "Dataset");
        // ReSharper disable once MemberCanBePrivate.Global
        public static string XTrainRawFile => Path.Combine(WorkingDirectory, "Data", "x_train_ACFqOMF.csv");
        // ReSharper disable once MemberCanBePrivate.Global
        public static string YTrainRawFile => Path.Combine(WorkingDirectory, "Data", "y_train_HNMbC27.csv");
        public static string XTestRawFile => Path.Combine(WorkingDirectory, "Data", "x_test_pf4T2aK.csv");
        #endregion


        // ReSharper disable once UnusedMember.Global
        public static void LaunchLightGBMHPO()
        {
            Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, "Natixis70");
            Utils.ConfigureThreadLog4netProperties(WorkingDirectory, "Natixis70");

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

            var hpo = new BayesianSearchHPO(searchSpace, DefaultParametersLightGBM, WorkingDirectory);
            hpo.Process(t => TrainWithHyperParameters((Natixis70_LightGBM_HyperParameters)t, ""));
        }

        private static string DatasetHPOPath => Path.Combine(WorkingDirectory, "DatasetHPO");

        // ReSharper disable once UnusedMember.Global
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

            var hpo = new BayesianSearchHPO(searchSpace,
                () =>
                {
                    var sample = new Parameters();
                    sample.categorical_feature = categoricalFeaturesFieldValue;
                    return sample;
                }, 
                WorkingDirectory, AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION.FULLY_RANDOM);

            // ReSharper disable once UselessBinaryOperation
            var minValidationScore = float.MaxValue;

         
            hpo.Process(t => DatasetHPO((Parameters)t, trainAndValidation.Training, trainAndValidation.Test, ref minValidationScore));
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

        public static void SearchForBestWeights()
        {
            var workingDirectory = Path.Combine(Utils.LocalApplicationFolderPath, "SharpNet", "Natixis70", "submit");
            List<TrainedModel> trainedModels = new();
            foreach (var modelName in new[]
                     {
                         "A8A78BE573",
                         "12ECEF9CB3",
                         "C24CF9FE92",
                         "9CBFCA0006",
                         "E1373D2F46",
                         "D939AAE448",
                         "0BCE79DE2D",
                         "3150BA17E9",
                         "05A621ABC5",
                         "0A1FC3DB8C"
                     })
            {
                trainedModels.Add(TrainedModel.ValueOf(workingDirectory, modelName));
            }
            var optimizeWeights = new WeightsOptimizer(Path.Combine(WorkingDirectory, nameof(WeightsOptimizer)), trainedModels);
            optimizeWeights.Run();
        }





        static Natixis70Utils()
        {
            var cpuTensor = Dataframe.Load(YTrainRawFile, true, ',').Drop(new[] { "" }).Tensor;
            Y_RAW_statistics = ExtractColumnStatistic(cpuTensor, false);
            Y_RAW_abs_statistics = ExtractColumnStatistic(cpuTensor, true);
        }
        private static Natixis70_LightGBM_HyperParameters DefaultParametersLightGBM()
        {
            var sample = new Natixis70_LightGBM_HyperParameters();
            sample.DatasetHyperParameters.TryToPredictAllHorizonAtTheSameTime = false;

            var lightGbmParameters = sample.LightGBMParameters;

            lightGbmParameters.verbosity = 0;
            lightGbmParameters.objective = Parameters.objective_enum.regression;
            lightGbmParameters.metric = "rmse";
            lightGbmParameters.num_threads = 1;
            lightGbmParameters.device_type = Parameters.device_type_enum.cpu;
            lightGbmParameters.num_iterations = 1000;

            //to optimize in HPO
            lightGbmParameters.bagging_fraction = 1.0;
            lightGbmParameters.bagging_freq = 0;
            lightGbmParameters.colsample_bytree = 1;
            lightGbmParameters.extra_trees = true;
            lightGbmParameters.lambda_l1 = 0.0;
            lightGbmParameters.lambda_l2 = 0.0;
            lightGbmParameters.learning_rate = 0.05;
            lightGbmParameters.max_bin = 255;
            lightGbmParameters.max_depth = 50;
            lightGbmParameters.min_data_in_leaf = 50;
            lightGbmParameters.num_leaves = 100;

            if (!sample.PostBuild())
            {
                throw new Exception($"invalid hyperParameters {lightGbmParameters}");
            }


            return sample;

        }
        private static CpuTensor<float> UnnormalizeYIfNeeded(CpuTensor<float> y, Natixis70DatasetHyperParameters sample)
        {
            if (sample.Normalization == Natixis70DatasetHyperParameters.normalize_enum.NONE)
            {
                return y; //no need to unnormalize y
            }

            var ySpan = y.AsReadonlyFloatCpuContent;
            var Yunnormalized = new CpuTensor<float>(y.Shape);
            var YunnormalizedSpan = Yunnormalized.AsFloatCpuSpan;

            int index = 0;
            for (int row = 0; row < Yunnormalized.Shape[0]; ++row)
            {
                var horizonId = sample.RowToHorizonId(row);
                var marketId = sample.RowToMarketId(row);
                //we load the row 'row' in 'y' tensor
                for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (HorizonNames.Length - 1) : marketId); ++currentMarketId)
                {
                    for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (HorizonNames.Length - 1) : horizonId); ++currentHorizonId)
                    {
                        int rawColIndex = 1 + HorizonNames.Length * currentMarketId + currentHorizonId;
                        var yValue = ySpan[index];
                        if (sample.Normalization == Natixis70DatasetHyperParameters.normalize_enum.MINUS_MEAN_DIVIDE_BY_VOL)
                        {
                            var colStatistics = Y_RAW_statistics[rawColIndex - 1];
                            var yUnnormalizedValue = (float)(yValue * colStatistics.Volatility + colStatistics.Average);
                            YunnormalizedSpan[index] = yUnnormalizedValue;
                        }
                        else if (sample.Normalization == Natixis70DatasetHyperParameters.normalize_enum.DIVIDE_BY_ABS_MEAN)
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
        private static float TrainWithHyperParameters(Natixis70_LightGBM_HyperParameters sample, string modelPrefix)
        {
            var model = new LightGBMModel(sample.LightGBMParameters, WorkingDirectory, modelPrefix);
            var datasetSample = sample.DatasetHyperParameters;

            using var fullTraining = datasetSample.NewDataSet(XTrainRawFile, YTrainRawFile);
            using var test = datasetSample.NewDataSet(XTestRawFile, null);

            int rowsInTrainingSet = (int)(datasetSample.PercentageInTraining * fullTraining.Count + 0.1);
            rowsInTrainingSet -= rowsInTrainingSet % datasetSample.RawCountToCount(1);
            using var trainAndValidation = fullTraining.SplitIntoTrainingAndValidation(rowsInTrainingSet);

            var sw = Stopwatch.StartNew();
            model.Train(trainAndValidation.Training, trainAndValidation.Test);

            var validationRmse = model.CreateModelResults(
                datasetSample.SavePredictions,
                y => UnnormalizeYIfNeeded(y, datasetSample),
                validationScore => validationScore< Min_Validation_Score_To_Save_Predictions,
                sw.Elapsed.TotalSeconds,
                datasetSample.X_Shape(1)[1],
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

    }

}
