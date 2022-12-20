using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.MathTools;
using SharpNet.Networks;

namespace SharpNet.Datasets.Natixis70;

[SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
public class Natixis70DatasetSample : AbstractDatasetSample
{
    public const string NAME = "Natixis70";
    public static readonly string[] MarketNames = { "VIX", "V2X", "EURUSD", "EURUSDV1M", "USGG10YR", "USGG2YR", "GDBR10YR", "GDBR2YR", "SX5E", "SPX", "SRVIX", "CVIX", "MOVE" };
    public const string PredictionHeader = ",Diff_VIX_1d,Diff_VIX_1w,Diff_VIX_2w,Diff_V2X_1d,Diff_V2X_1w,Diff_V2X_2w,Diff_EURUSD_1d,Diff_EURUSD_1w,Diff_EURUSD_2w,Diff_EURUSDV1M_1d,Diff_EURUSDV1M_1w,Diff_EURUSDV1M_2w,Diff_USGG10YR_1d,Diff_USGG10YR_1w,Diff_USGG10YR_2w,Diff_USGG2YR_1d,Diff_USGG2YR_1w,Diff_USGG2YR_2w,Diff_GDBR10YR_1d,Diff_GDBR10YR_1w,Diff_GDBR10YR_2w,Diff_GDBR2YR_1d,Diff_GDBR2YR_1w,Diff_GDBR2YR_2w,Diff_SX5E_1d,Diff_SX5E_1w,Diff_SX5E_2w,Diff_SPX_1d,Diff_SPX_1w,Diff_SPX_2w,Diff_SRVIX_1d,Diff_SRVIX_1w,Diff_SRVIX_2w,Diff_CVIX_1d,Diff_CVIX_1w,Diff_CVIX_2w,Diff_MOVE_1d,Diff_MOVE_1w,Diff_MOVE_2w";
    public const int EmbeddingDimension = 768;
    public static readonly string[] HorizonNames = { "1d", "1w", "2w" };

    #region private fields
    private static readonly ConcurrentDictionary<string, Tuple<DataSetV2, DataSetV2, DatasetEncoder>> CacheDataset = new();
    private static DoubleAccumulator[] YStatsInTargetFormat { get; }
    private static DoubleAccumulator[] YAbsStatsInTargetFormat { get; }
    #endregion

    #region constructors
    static Natixis70DatasetSample()
    {
        var cpuTensor = DataFrame.read_float_csv(YTrainRawFile).Drop("").FloatCpuTensor();
        YStatsInTargetFormat = ExtractColumnStatistic(cpuTensor, false);
        YAbsStatsInTargetFormat = ExtractColumnStatistic(cpuTensor, true);
    }
    public Natixis70DatasetSample() : base(new HashSet<string>())
    {
    }
    #endregion

    #region Hyper-parameters
    /// <summary>
    /// true if we want to predict all horizons (1d / 1w / 2w) at the same time
    /// false if we want to predict each horizon separately
    /// </summary>
    public bool TryToPredictAllHorizonAtTheSameTime = false;
    /// <summary>
    /// true if we want to predict all markets (VIX, EURUSD, etc.) at the same time
    /// false if we want to predict each market separately 
    /// </summary>
    // ReSharper disable once Member6CanBePrivate.Global
    public bool TryToPredictAllMarketsAtTheSameTime = false;
    // ReSharper disable once MemberCanBePrivate.Global
    public bool MergeHorizonAndMarketIdInSameFeature = false;
    /// <summary>
    /// normalize all label:
    ///     y = (y -average(y)) / volatility(y)
    /// </summary>
    // ReSharper disable once MemberCanBePrivate.Global
    public normalize_enum Normalization = normalize_enum.NONE;
    public enum normalize_enum { NONE, MINUS_MEAN_DIVIDE_BY_VOL, DIVIDE_BY_ABS_MEAN };
    #endregion

    //public override IScore MinimumScoreToSaveModel => new Score(20, GetRankingEvaluationMetric());
    public CpuTensor<float> PredictionsInTargetFormat_2_PredictionsInModelFormat(CpuTensor<float> predictionsInTargetFormat)
    {
        if (predictionsInTargetFormat == null)
        {
            return null;
        }
        var predictionsInModelFormat = new CpuTensor<float>(YShapeInModelFormat(predictionsInTargetFormat.Shape[0]));
        var predictionsInTargetFormatSpan = predictionsInTargetFormat.AsReadonlyFloatCpuContent;
        var predictionsInModelFormatSpan = predictionsInModelFormat.AsFloatCpuSpan;
        int predictionsInModelFormatSpanIndex = 0;
        var divider = RowsInTargetFormatToRowsInModelFormat(1);

        for (int rowInModelFormat = 0; rowInModelFormat < predictionsInModelFormat.Shape[0]; rowInModelFormat++)
        {
            int rowInTargetFormat = rowInModelFormat / divider;
            int horizonId = RowInModelFormatToHorizonId(rowInModelFormat);
            int marketId = RowInModelFormatToMarketId(rowInModelFormat);

            //we load the row 'rowInModelFormat' of 'predictionsInModelFormat' tensor
            for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (HorizonNames.Length - 1) : marketId); ++currentMarketId)
            {
                for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (HorizonNames.Length - 1) : horizonId); ++currentHorizonId)
                {
                    int colIndexInTargetFormat = 1 + HorizonNames.Length * currentMarketId + currentHorizonId;
                    var yUnnormalizedValue = predictionsInTargetFormatSpan[rowInTargetFormat * predictionsInTargetFormat.Shape[1] + colIndexInTargetFormat];
                    var yNormalizedValue = yUnnormalizedValue;
                    if (Normalization == normalize_enum.MINUS_MEAN_DIVIDE_BY_VOL)
                    {
                        var colStatistics = YStatsInTargetFormat[colIndexInTargetFormat - 1];
                        yNormalizedValue = (float)((yUnnormalizedValue - colStatistics.Average) / colStatistics.Volatility);
                    }
                    else if (Normalization == normalize_enum.DIVIDE_BY_ABS_MEAN)
                    {
                        var absColStatistics = YAbsStatsInTargetFormat[colIndexInTargetFormat - 1];
                        yNormalizedValue = (float)(yUnnormalizedValue / absColStatistics.Average);
                    }
                    predictionsInModelFormatSpan[predictionsInModelFormatSpanIndex++] = yNormalizedValue;
                }
            }
        }
        Debug.Assert(predictionsInModelFormatSpanIndex == predictionsInModelFormat.Count);
        return predictionsInModelFormat;
    }
    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat)
    {
        if (predictionsInModelFormat == null)
        {
            return null;
        }
        if (predictionsInModelFormat.Shape[1] == 39)
        {
            //var cpuTensor = CpuTensor<float>.AddIndexInFirstColumn(predictionsInModelFormat.FloatCpuTensor(), 0);
            //return DataFrame.New(cpuTensor, Utils.Join(IdColumns, TargetLabels));
            return predictionsInModelFormat;
        }

        var predictionsInModelFormatSpan = predictionsInModelFormat.FloatCpuTensor().AsReadonlyFloatCpuContent;
        var predictionsInModelFormatSpanIndex = 0;

        // the predictions in the target format for the Natixis70 Challenge
        var predictionsInTargetFormat = new CpuTensor<float>(YShapeInTargetFormat(predictionsInModelFormat.Shape[0]));
        var predictionsInTargetFormatSpan = predictionsInTargetFormat.AsFloatCpuSpan;
        var divider = RowsInTargetFormatToRowsInModelFormat(1);

        for (int rowInModelFormat = 0; rowInModelFormat < predictionsInModelFormat.Shape[0]; ++rowInModelFormat)
        {
            var rowInTargetFormat = rowInModelFormat / divider;
            //the first element of each row in target format is the index of this row (starting from 0)
            //predictionsInTargetFormatSpan[rowInTargetFormat * predictionsInTargetFormat.Shape[1]] = rowInTargetFormat;
            int horizonId = RowInModelFormatToHorizonId(rowInModelFormat);
            int marketId = RowInModelFormatToMarketId(rowInModelFormat);
            //we load the row 'rowInModelFormat' in 'predictionsInTargetFormat' tensor
            for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (HorizonNames.Length - 1) : marketId); ++currentMarketId)
            {
                for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (HorizonNames.Length - 1) : horizonId); ++currentHorizonId)
                {
                    int colIndexInTargetFormat = HorizonNames.Length * currentMarketId + currentHorizonId;
                    var yNormalizedValue = predictionsInModelFormatSpan[predictionsInModelFormatSpanIndex++];
                    var yUnnormalizedValue = yNormalizedValue;

                    if (Normalization == normalize_enum.MINUS_MEAN_DIVIDE_BY_VOL)
                    {
                        var colStatistics = YStatsInTargetFormat[colIndexInTargetFormat - 1];
                        yUnnormalizedValue = (float)(yNormalizedValue * colStatistics.Volatility + colStatistics.Average);
                    }
                    else if (Normalization == normalize_enum.DIVIDE_BY_ABS_MEAN)
                    {
                        var absColStatistics = YAbsStatsInTargetFormat[colIndexInTargetFormat - 1];
                        yUnnormalizedValue = (float)(yNormalizedValue * absColStatistics.Average);
                    }
                    predictionsInTargetFormatSpan[rowInTargetFormat * predictionsInTargetFormat.Shape[1] + colIndexInTargetFormat] = yUnnormalizedValue;
                }
            }
        }
        Debug.Assert(predictionsInTargetFormat.Shape.Length == 2);
        Debug.Assert(predictionsInTargetFormat.Shape[1] == 39);
        return DataFrame.New(predictionsInTargetFormat, TargetLabels);

    }
    public override bool FixErrors()
    {
        if (!base.FixErrors())
        {
            return false;
        }
        if (MergeHorizonAndMarketIdInSameFeature)
        {
            if (TryToPredictAllMarketsAtTheSameTime || TryToPredictAllHorizonAtTheSameTime)
            {
                return false;
            }
        }
        return true;
    }
    //public override (CpuTensor<float> trainPredictionsInTargetFormatWithoutIndex, CpuTensor<float> validationPredictionsInTargetFormatWithoutIndex, CpuTensor<float> testPredictionsInTargetFormatWithoutIndex) 
    //    LoadAllPredictionsInTargetFormatWithoutIndex()
    //{
    //    return LoadAllPredictionsInTargetFormatWithoutIndex(true, true, ',');
    //}
    public override string[] CategoricalFeatures
    {
        get
        {
            var categoricalFeatures = new List<string>();
            if (MergeHorizonAndMarketIdInSameFeature)
            {
                Debug.Assert(!TryToPredictAllMarketsAtTheSameTime);
                Debug.Assert(!TryToPredictAllHorizonAtTheSameTime);
                categoricalFeatures.Add("marketIdhorizonId");
            }
            else
            {
                if (!TryToPredictAllMarketsAtTheSameTime)
                {
                    categoricalFeatures.Add("marketId");
                }
                if (!TryToPredictAllHorizonAtTheSameTime)
                {
                    categoricalFeatures.Add("horizonId");
                }
            }
            return categoricalFeatures.ToArray();
        }

    }
    public override string[] IdColumns => new[] { "" };
    public override string[] TargetLabels => PredictionHeader.Trim(',').Split(',');

    public string[] TargetLabelsInModelFormat
    {
        get
        {
            if (TryToPredictAllHorizonAtTheSameTime)
            {
                if (TryToPredictAllMarketsAtTheSameTime)
                {
                    return TargetLabels;
                }
                else
                {
                    return HorizonNames;
                }
            }
            else
            {
                if (TryToPredictAllMarketsAtTheSameTime)
                {
                    return TargetLabels.Where(s => s.EndsWith("_1d")).Select(s => s.Replace("_1d", "")).ToArray();
                }
                else
                {
                    return new [] { "y" };
                }
            }
        }
    }
    
    

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
        DatasetEncoder = new DatasetEncoder(this, StandardizeDoubleValues, true);

        var xTrain_InTargetFormat = DataFrame.read_float_csv(XTrainRawFile);
        var yTrain_InTargetFormat = DataFrame.read_float_csv(YTrainRawFile);
        var xTest_InTargetFormat = DataFrame.read_float_csv(XTestRawFile);

        DatasetEncoder.Fit(xTrain_InTargetFormat);
        DatasetEncoder.Fit(yTrain_InTargetFormat);
        DatasetEncoder.Fit(xTest_InTargetFormat);
        
        var xTrain_Encoded_InTargetFormat = DatasetEncoder.Transform(xTrain_InTargetFormat);
        //var yTrain_Encoded_InTargetFormat = DatasetEncoder.Transform(yTrain_InTargetFormat);
        var xTest_Encoded_InTargetFormat = DatasetEncoder.Transform(xTest_InTargetFormat);

        var xTrain_Encoded_InModelFormat = Load_XInModelFormat(xTrain_Encoded_InTargetFormat);
        var yTrain_Encoded_InModelFormat = Load_YInModelFormat(yTrain_InTargetFormat);
        var xTest_Encoded_InModelFormat = Load_XInModelFormat(xTest_Encoded_InTargetFormat);

        DatasetEncoder.FitMissingCategoricalColumns(xTrain_Encoded_InModelFormat, xTest_Encoded_InModelFormat);

        var fullTrainingAndValidation = new DataSetV2(this, xTrain_Encoded_InModelFormat, yTrain_Encoded_InModelFormat, false);
        var testDataset = new DataSetV2(this, xTest_Encoded_InModelFormat, null, false);
        CacheDataset.TryAdd(key, Tuple.Create(fullTrainingAndValidation, testDataset, DatasetEncoder));
        return (fullTrainingAndValidation, testDataset);
    }
    
    /// <summary>
    /// path to the test dataset in LightGBM compatible format
    /// </summary>
    /// <returns></returns>
    private DataFrame Load_XInModelFormat(DataFrame xInTargetFormatDataFrame)
    {
        var xInTargetFormat = xInTargetFormatDataFrame.FloatCpuTensor();
        Debug.Assert(xInTargetFormat.Shape[1] == EmbeddingDimension);
        int rowsInTargetFormat = xInTargetFormat.Shape[0];
        var xInTargetFormatSpan = xInTargetFormat.AsReadonlyFloatCpuContent;
        var xInModelFormat = new CpuTensor<float>(XShapeInModelFormat(rowsInTargetFormat));
        var xInModelFormatSpan = xInModelFormat.AsFloatCpuSpan;
        int xInModelFormatSpanIndex = 0;
        var divider = RowsInTargetFormatToRowsInModelFormat(1);

        for (int rowInModelFormat = 0; rowInModelFormat < xInModelFormat.Shape[0]; rowInModelFormat++)
        {
            int rowInTargetFormat = rowInModelFormat / divider;
            int horizonId = RowInModelFormatToHorizonId(rowInModelFormat);
            int marketId = RowInModelFormatToMarketId(rowInModelFormat);

            //we load the row 'row' in 'xInModelFormat' tensor
            for (int colInTargetFormat = 0; colInTargetFormat < xInTargetFormat.Shape[1]; ++colInTargetFormat)
            {
                xInModelFormatSpan[xInModelFormatSpanIndex++] = xInTargetFormatSpan[rowInTargetFormat * xInTargetFormat.Shape[1] + colInTargetFormat];
            }

            if (MergeHorizonAndMarketIdInSameFeature)
            {
                Debug.Assert(marketId >= 0);
                Debug.Assert(horizonId >= 0);
                xInModelFormatSpan[xInModelFormatSpanIndex++] = marketId * HorizonNames.Length + horizonId;
            }
            else
            {
                if (marketId >= 0)
                {
                    xInModelFormatSpan[xInModelFormatSpanIndex++] = marketId;
                }
                if (horizonId >= 0)
                {
                    xInModelFormatSpan[xInModelFormatSpanIndex++] = horizonId;
                }
            }
        }
        Debug.Assert(xInModelFormatSpanIndex == xInModelFormat.Count);
        xInTargetFormat.Dispose();

        return DataFrame.New(xInModelFormat, ColumnsInModelFormat);
    }
    /// <summary>
    /// Load the content of the file 'yFileInTargetFormat' in a CpuTensor (in model format) and return it
    /// </summary>
    /// <param name="yInTargetFormatDataFrame"></param>
    /// <returns></returns>
    private DataFrame Load_YInModelFormat(DataFrame yInTargetFormatDataFrame)
    {
        var yInTargetFormat = yInTargetFormatDataFrame.FloatCpuTensor();
        var yInModelFormat = PredictionsInTargetFormat_2_PredictionsInModelFormat(yInTargetFormat);
        yInTargetFormat.Dispose();

        return DataFrame.New(yInModelFormat, TargetLabelsInModelFormat);
    }


    public int marketIdEmbeddingSize = 100;
    public int marketIdhorizonIdEmbeddingSize = 100;
    public int horizonIdEmbeddingSize = 10;


    public override int EmbeddingForColumn(string columnName, int defaultEmbeddingSize)
    {
        switch(columnName )
        {
            case "marketId": return marketIdEmbeddingSize;
            case "horizonId": return horizonIdEmbeddingSize;
            case "marketIdhorizonId": return marketIdhorizonIdEmbeddingSize;
        }
        return defaultEmbeddingSize;
    }



    public override int DatasetRowsInModelFormatMustBeMultipleOf()
    {
        return RowsInTargetFormatToRowsInModelFormat(1);
    }
    public override EvaluationMetricEnum GetRankingEvaluationMetric() => EvaluationMetricEnum.Rmse;
    public override Objective_enum GetObjective() => Objective_enum.Regression;
    public string[] ColumnsInModelFormat
    {
        get
        {
            var result = Enumerable.Range(0, EmbeddingDimension).Select(i => "embed_" + i).ToList();
            result.AddRange(CategoricalFeatures);
            return result.ToArray();
        }
    }
    public override void SavePredictionsInTargetFormat(DataFrame y_pred_InTargetFormat, DataSet xDataset, string path)
    {
        if (y_pred_InTargetFormat == null)
        {
            return;
        }

        //we add the Id Column as 1st column
        var id_df = DataFrame.New(Enumerable.Range(0, y_pred_InTargetFormat.Shape[0]).ToArray(), IdColumns);
        y_pred_InTargetFormat = DataFrame.MergeHorizontally(id_df, y_pred_InTargetFormat);

        y_pred_InTargetFormat.to_csv(path, GetSeparator());
    }

    #region load of datasets
    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    // ReSharper disable once MemberCanBePrivate.Global
    #endregion

    
    private int[] YShapeInModelFormat(int rowsInTargetFormat)
    {
        int yColCountInModelFormat = 1;
        if (TryToPredictAllHorizonAtTheSameTime)
        {
            yColCountInModelFormat *= HorizonNames.Length;
        }
        if (TryToPredictAllMarketsAtTheSameTime)
        {
            yColCountInModelFormat *= MarketNames.Length;
        }
        return new[] { RowsInTargetFormatToRowsInModelFormat(rowsInTargetFormat), yColCountInModelFormat };
    }
    private int[] YShapeInTargetFormat(int rowsInModelFormat)
    {
        var divider = RowsInTargetFormatToRowsInModelFormat(1);
        Debug.Assert(rowsInModelFormat % divider == 0);
        var rowsInTargetFormat = rowsInModelFormat / divider;
        return new[] { rowsInTargetFormat, MarketNames.Length * HorizonNames.Length };
    }
    private int RowsInTargetFormatToRowsInModelFormat(int rowsInTargetFormat)
    {
        int rowsInModelFormat = rowsInTargetFormat;
        if (!TryToPredictAllHorizonAtTheSameTime)
        {
            rowsInModelFormat *= HorizonNames.Length;
        }
        if (!TryToPredictAllMarketsAtTheSameTime)
        {
            rowsInModelFormat *= MarketNames.Length;
        }
        return rowsInModelFormat;
    }
    private int[] XShapeInModelFormat(int rowsInTargetFormat)
    {
        return new[] { RowsInTargetFormatToRowsInModelFormat(rowsInTargetFormat), ColumnsInModelFormat.Length };
    }
    /// <summary>
    /// return the horizonId associated with row 'rowInModelFormat', or -1 if the row is associated with all horizon ids
    /// </summary>
    /// <param name="rowInModelFormat"></param>
    /// <returns></returns>
    private int RowInModelFormatToHorizonId(int rowInModelFormat)
    {
        if (TryToPredictAllHorizonAtTheSameTime)
        {
            return -1;
        }
        return rowInModelFormat % HorizonNames.Length;
    }
    /// <summary>
    /// return the marketId associated with row 'rowInModelFormat', or -1 if the row is associated with all market ids
    /// </summary>
    /// <param name="rowInModelFormat"></param>
    /// <returns></returns>
    private int RowInModelFormatToMarketId(int rowInModelFormat)
    {
        if (TryToPredictAllMarketsAtTheSameTime)
        {
            return -1;
        }
        if (!TryToPredictAllHorizonAtTheSameTime)
        {
            rowInModelFormat /= HorizonNames.Length;
        }

        return rowInModelFormat % MarketNames.Length;
    }
    private const string FILE_SUFFIX = "";
    //private const string FILE_SUFFIX = "_small";
    private static string XTrainRawFile => Path.Combine(DataDirectory, "x_train_ACFqOMF" + FILE_SUFFIX + ".csv");
    private static string XTestRawFile => Path.Combine(DataDirectory, "x_test_pf4T2aK" + FILE_SUFFIX + ".csv");
    private static string YTrainRawFile => Path.Combine(WorkingDirectory, "Data", "y_train_HNMbC27" + FILE_SUFFIX + ".csv");
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

    #region HPO
    // ReSharper disable once UnusedMember.Global
    public static void LaunchLightGBMHPO(int num_iterations = 1000, float maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
            {
                //related to Dataset 
                {"KFold", 2},
                //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
                { "TryToPredictAllHorizonAtTheSameTime", false},
                { "MergeHorizonAndMarketIdInSameFeature",new[]{true/*, false*/} },
                //{ "Normalization",new[] { "NONE", "DIVIDE_BY_ABS_MEAN"} },

                { "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
                { "bagging_freq", new[]{0, 1} },
                { "boosting", new []{"gbdt", "dart"}},
                { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.3f, 1.0f)},
                { "early_stopping_round", num_iterations/10 },
                { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
                { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.03f, 0.2f)}, //for 1.000 trees
                //{ "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.03f)}, //for 10.000trees
                { "max_depth", new[]{10, 20, 50, 100, 255} },
                { "num_iterations", num_iterations },
                { "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 50) },
                { "num_threads", 1},
                { "verbosity", "0" },

            };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), new Natixis70DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }
    // ReSharper disable once UnusedMember.Global
    public static void LaunchCatBoostHPO(int iterations = 1000, float maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
            {
                //related to Dataset 
                { "TryToPredictAllHorizonAtTheSameTime", false},
                { "MergeHorizonAndMarketIdInSameFeature",new[]{true/*, false*/} },
                //{ "Normalization",new[] { "NONE", "DIVIDE_BY_ABS_MEAN"} },
                //{"KFold", 2},
                {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

                //related to CatBoost model
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
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
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

    public static void TrainNetwork(int numEpochs = 15, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            {"InitialLearningRate", AbstractHyperParameterSearchSpace.Range(0.003f, 0.2f, AbstractHyperParameterSearchSpace.range_type.normal)},
            {"StandardizeDoubleValues", false},
            //dataset 
            //{"Reviews_EmbeddingDim", new[]{0, 100, TOTAL_Reviews_EmbeddingDim}},
            
            {"LossFunction", "Mse"},


            // Optimizer 
            {"OptimizerType", "AdamW"},
            //{"AdamW_L2Regularization", AbstractHyperParameterSearchSpace.Range(0.003f, 0.01f)},
            {"AdamW_L2Regularization", 0.004},

            // Learning Rate Scheduler
            {"LearningRateSchedulerType", new[]{ "CyclicCosineAnnealing"}},
            {"OneCycle_PercentInAnnealing", 0.5},

            { "EmbeddingDim", new[]{10} },
            //{ "EmbeddingDim", 10 },

            //{"dropout_top", 0.1},
            //{"dropout_mid", 0.3},
            //{"dropout_bottom", 0},

            //run on GPU
            {"NetworkSample_1DCNN_UseGPU", true},

            {"BatchSize", new[]{256} },

            //{"two_stage", new[]{true,false } },
            //{"Use_ConcatenateLayer", new[]{true,false } },
            //{"Use_AddLayer", new[]{true,false } },
            {"hidden_size", 2048 },

            {"two_stage", true },
            {"Use_ConcatenateLayer", false },
            {"Use_AddLayer", true },


            {"NumEpochs", numEpochs},
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new NetworkSample_1DCNN(), new Natixis70DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    #endregion


}
