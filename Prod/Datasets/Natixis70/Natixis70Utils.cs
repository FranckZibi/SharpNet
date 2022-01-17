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
        public static readonly string[] MARKET_NAMES = new[]{ "VIX", "V2X", "EURUSD", "EURUSDV1M", "USGG10YR", "USGG2YR", "GDBR10YR", "GDBR2YR", "SX5E", "SPX", "SRVIX", "CVIX",  "MOVE"};
        public const string PREDICTION_HEADER =  ",Diff_VIX_1d,Diff_VIX_1w,Diff_VIX_2w,Diff_V2X_1d,Diff_V2X_1w,Diff_V2X_2w,Diff_EURUSD_1d,Diff_EURUSD_1w,Diff_EURUSD_2w,Diff_EURUSDV1M_1d,Diff_EURUSDV1M_1w,Diff_EURUSDV1M_2w,Diff_USGG10YR_1d,Diff_USGG10YR_1w,Diff_USGG10YR_2w,Diff_USGG2YR_1d,Diff_USGG2YR_1w,Diff_USGG2YR_2w,Diff_GDBR10YR_1d,Diff_GDBR10YR_1w,Diff_GDBR10YR_2w,Diff_GDBR2YR_1d,Diff_GDBR2YR_1w,Diff_GDBR2YR_2w,Diff_SX5E_1d,Diff_SX5E_1w,Diff_SX5E_2w,Diff_SPX_1d,Diff_SPX_1w,Diff_SPX_2w,Diff_SRVIX_1d,Diff_SRVIX_1w,Diff_SRVIX_2w,Diff_CVIX_1d,Diff_CVIX_1w,Diff_CVIX_2w,Diff_MOVE_1d,Diff_MOVE_1w,Diff_MOVE_2w";
        public const int EMBEDDING_DIMENSION = 768;
        public static readonly string[] HORIZON_NAMES = { "1d", "1w", "2w" };
        public static string NatixisLogDirectory => Path.Combine(NetworkConfig.DefaultLogDirectory, "Natixis70");

        private static string NatixisDataDirectory => Path.Combine(NetworkConfig.DefaultLogDirectory, "Natixis70", "Data");
        //public static string NatixisLightGBMDataDirectory => Path.Combine(NetworkConfig.DefaultLogDirectory, "Natixis70", "Data", " LightGBM");

        public static string XTrainRawFile => Path.Combine(Natixis70Utils.NatixisDataDirectory, "x_train_ACFqOMF.csv");
        public static string YTrainRawFileIfAny => Path.Combine(Natixis70Utils.NatixisDataDirectory, "y_train_HNMbC27.csv");
        public static string XTestRawFile => Path.Combine(Natixis70Utils.NatixisDataDirectory, "x_test_pf4T2aK.csv");



        public static DoubleAccumulator[] Y_RAW_statistics { get; }


        static Natixis70Utils()
        {
            Y_RAW_statistics = ExtractColumnStatistic(Dataframe.Load(YTrainRawFileIfAny, true, ',').Drop(new[]{""}).Tensor);
        }

        public static Natixis70HyperParameters DefaultParametersLightGBM()
        {
            var hyperParameters = new Natixis70HyperParameters();
            hyperParameters.TryToPredictAllHorizonAtTheSameTime = false;
            hyperParameters.verbosity = 0;
            hyperParameters.objective = Parameters.objective_enum.regression;
            hyperParameters.metric = "rmse";
            var categoricalFeaturesFieldValue = (hyperParameters.CategoricalFeatures().Count >= 1) ? ("name:" + string.Join(',', hyperParameters.CategoricalFeatures())) : "";
            hyperParameters.categorical_feature = categoricalFeaturesFieldValue;
            hyperParameters.num_threads = 1;

            //to optimize in HPO
            hyperParameters.learning_rate = 0.01;
            hyperParameters.num_iterations = 1000;
            hyperParameters.num_leaves = 200;
            hyperParameters.bagging_fraction = 0.8;
            hyperParameters.colsample_bytree = 1.0;
            hyperParameters.lambda_l1 = 0.005;
            hyperParameters.lambda_l2 = 0.5;
            hyperParameters.device_type = Parameters.device_type_enum.cpu;

            return hyperParameters;

        }

        public static void ProcessLightGBMSearchSpace(int nbCpuCores, IDictionary<string, object> searchSpace, string searchSpaceName)
        {
            var hpo = new RandomGridSearchHPO<Natixis70HyperParameters>(searchSpace, DefaultParametersLightGBM);
            hpo.Process(nbCpuCores,  
                t =>
                {
                    var (trainRmse, validationRmse) = TrainWithHyperParameters(t, searchSpaceName);
                    return validationRmse;
                },
                LightGBMModel.Log.Info
                );
        }

        private static CpuTensor<float> UnnormalizeYIfNeeded(CpuTensor<float> y, Natixis70HyperParameters hyperParameters)
        {
            if (!hyperParameters.NormalizeAllLabels)
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
                for (int currMarketId = (marketId < 0 ? 0 : marketId); currMarketId <= (marketId < 0 ? (HORIZON_NAMES.Length - 1) : marketId); ++currMarketId)
                {
                    for (int currHorizonId = (horizonId < 0 ? 0 : horizonId); currHorizonId <= (horizonId < 0 ? (HORIZON_NAMES.Length - 1) : horizonId); ++currHorizonId)
                    {
                        int rawColIndex = 1 + HORIZON_NAMES.Length * currMarketId + currHorizonId;
                        var colStatistics = Y_RAW_statistics[rawColIndex-1];
                        var yValue = ySpan[index];
                        var yUnnormalizedValue = (float)( yValue*colStatistics.Volatility+ colStatistics.Average );
                        YunnormalizedSpan[index] = yUnnormalizedValue;
                        ++index;
                    }
                }
            }
            return Yunnormalized;
        }

        private static (double,double) TrainWithHyperParameters(Natixis70HyperParameters hyperParameters, string modelPrefix)
        {

            var model = new LightGBMModel(hyperParameters, NatixisLogDirectory, modelPrefix);

            //Utils.ConfigureGlobalLog4netProperties(NatixisLogDirectory, Path.GetFileNameWithoutExtension(model.ModelFileName));
            //Utils.ConfigureThreadLog4netProperties(NatixisLogDirectory, Path.GetFileNameWithoutExtension(model.ModelFileName));
            Utils.ConfigureThreadLog4netProperties(Natixis70Utils.NatixisLogDirectory, "Natixis70");


            using var fullTraining  = Natixis70DataSet.ValueOf(XTrainRawFile, YTrainRawFileIfAny, false, LightGBMModel.Log.Info, hyperParameters);
            using var Test = Natixis70DataSet.ValueOf(XTestRawFile, null, false, LightGBMModel.Log.Info, hyperParameters);

            int rowsInTrainingSet = (int)(hyperParameters.PercentageInTraining * fullTraining.Count + 0.1);
            rowsInTrainingSet -= rowsInTrainingSet % hyperParameters.RawCountToCount(1);
            var trainAndValidation = fullTraining.SplitIntoTrainingAndValidation(rowsInTrainingSet);

            var sw = Stopwatch.StartNew();
            model.Train(trainAndValidation.Training, trainAndValidation.Test);

            var (trainRmse, validationRmse) = model.CreateModelResults(
                hyperParameters.SavePredictions,
                y => UnnormalizeYIfNeeded(y, hyperParameters),
                sw.Elapsed.TotalSeconds,
                hyperParameters.X_Shape(1)[1],
                trainAndValidation.Training,
                trainAndValidation.Test,
                Test);
            return (trainRmse, validationRmse);
        }

        /// <summary>
        /// return the statistics (average/volatility) of each column of the matrix 'y'
        /// </summary>
        /// <param name="y">a 2D tensor</param>
        /// <returns></returns>
        private static DoubleAccumulator[] ExtractColumnStatistic(CpuTensor<float> y)
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
                result[i % y.Shape[1]].Add(ySpan[i], 1);
            }
            return result.ToArray();
        }



    }
}
