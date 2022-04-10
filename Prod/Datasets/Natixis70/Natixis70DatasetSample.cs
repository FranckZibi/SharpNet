using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.HyperParameters;
using SharpNet.MathTools;

namespace SharpNet.Datasets.Natixis70;

[SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
public class Natixis70DatasetSample : AbstractDatasetSample
{
    #region private fields
    private static readonly ConcurrentDictionary<string, CpuTensor<float>> CacheDataset = new();
    private static DoubleAccumulator[] YStatsInTargetFormat { get; }
    private static DoubleAccumulator[] YAbsStatsInTargetFormat { get; }
    //private static readonly ConcurrentDictionary<string, CpuTensor<float>> PredictionsInModelFormat_2_PredictionsInTargetFormat_Cache = new();
    #endregion

    #region constructors
    static Natixis70DatasetSample()
    {
        var cpuTensor = Dataframe.Load(YTrainRawFile, true, ',').Drop(new[] { "" }).Tensor;
        YStatsInTargetFormat = ExtractColumnStatistic(cpuTensor, false);
        YAbsStatsInTargetFormat = ExtractColumnStatistic(cpuTensor, true);
    }
    public Natixis70DatasetSample() : base(new HashSet<string>())
    {
    }
    public static Natixis70DatasetSample ValueOfNatixis70DatasetSample(string workingDirectory, string sampleName)
    {
        return (Natixis70DatasetSample)ISample.LoadConfigIntoSample(() => new Natixis70DatasetSample(), workingDirectory, sampleName);
    }
    #endregion

    #region Hyper-parameters
    /// <summary>
    /// true if we want to predict all horizons (1d / 1w / 2w) at the same time
    /// false if we want to predict each horizon separately
    /// </summary>
    public bool TryToPredictAllHorizonAtTheSameTime = true;
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
    public double PercentageInTraining = 0.8;
    #endregion

    public override CpuTensor<float> PredictionsInTargetFormat_2_PredictionsInModelFormat(CpuTensor<float> predictionsInTargetFormat)
    {
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
            for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : marketId); ++currentMarketId)
            {
                for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : horizonId); ++currentHorizonId)
                {
                    int colIndexInTargetFormat = 1 + Natixis70Utils.HorizonNames.Length * currentMarketId + currentHorizonId;
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
    public override CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(CpuTensor<float> predictionsInModelFormat)
    {
        if (predictionsInModelFormat.Shape[1] == 39)
        {
            return CpuTensor<float>.AddIndexInFirstColumn(predictionsInModelFormat, 0);
        }

        var predictionsInModelFormatSpan = predictionsInModelFormat.AsReadonlyFloatCpuContent;
        var predictionsInModelFormatSpanIndex = 0;

        // the predictions in the target format for the Natixis70 Challenge
        var predictionsInTargetFormat = new CpuTensor<float>(YShapeInTargetFormat(predictionsInModelFormat.Shape[0]));
        var predictionsInTargetFormatSpan = predictionsInTargetFormat.AsFloatCpuSpan;
        var divider = RowsInTargetFormatToRowsInModelFormat(1);

        for (int rowInModelFormat = 0; rowInModelFormat < predictionsInModelFormat.Shape[0]; ++rowInModelFormat)
        {
            var rowInTargetFormat = rowInModelFormat / divider;
            //the first element of each row in target format is the index of this row (starting from 0)
            predictionsInTargetFormatSpan[rowInTargetFormat * predictionsInTargetFormat.Shape[1]] = rowInTargetFormat;
            int horizonId = RowInModelFormatToHorizonId(rowInModelFormat);
            int marketId = RowInModelFormatToMarketId(rowInModelFormat);
            //we load the row 'rowInModelFormat' in 'predictionsInTargetFormat' tensor
            for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : marketId); ++currentMarketId)
            {
                for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : horizonId); ++currentHorizonId)
                {
                    int colIndexInTargetFormat = 1 + Natixis70Utils.HorizonNames.Length * currentMarketId + currentHorizonId;
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
        Debug.Assert(predictionsInTargetFormat.Shape[1] == (1 + 39));
        return predictionsInTargetFormat;
    }
    public override bool FixErrors()
    {
        if (MergeHorizonAndMarketIdInSameFeature)
        {
            if (TryToPredictAllMarketsAtTheSameTime || TryToPredictAllHorizonAtTheSameTime)
            {
                return false;
            }
        }
        return true;
    }
    /// <summary>
    /// convert a dataset in LightGBM format (first column is the target 'y') to the challenge target format 
    /// </summary>
    /// name="dataframe_path">a dataset in LightGBM format
    /// <returns></returns>
    //public override CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(string dataframe_path)
    //{
    //    if (PredictionsInModelFormat_2_PredictionsInTargetFormat_Cache.TryGetValue(dataframe_path, out var res))
    //    {
    //        return res;
    //    }
    //    var predictionsInModelFormat = Dataframe.Load(dataframe_path, true, ',').Keep(new[] { "y" }).Tensor;
    //    var predictionInTargetFormat = PredictionsInModelFormat_2_PredictionsInTargetFormat(predictionsInModelFormat).DropColumns(new[] { 0 });
    //    Debug.Assert(predictionInTargetFormat.Shape.Length == 2);
    //    Debug.Assert(predictionInTargetFormat.Shape[1] == 39);
    //    PredictionsInModelFormat_2_PredictionsInTargetFormat_Cache.TryAdd(dataframe_path, predictionInTargetFormat);
    //    return predictionInTargetFormat;
    //}

    //public override (CpuTensor<float> trainPredictionsInTargetFormatWithoutIndex, CpuTensor<float> validationPredictionsInTargetFormatWithoutIndex, CpuTensor<float> testPredictionsInTargetFormatWithoutIndex) 
    //    LoadAllPredictionsInTargetFormatWithoutIndex()
    //{
    //    return LoadAllPredictionsInTargetFormatWithoutIndex(true, true, ',');
    //}
    public override List<string> CategoricalFeatures()
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
        return categoricalFeatures;
    }
    //public override IDataSet FullTraining()
    //{
    //    return NewDataSet(XTrainRawFile, YTrainRawFile);
    //}
    public override void SavePredictionsInTargetFormat(CpuTensor<float> predictionsInTargetFormat, string path)
    {
        var start = Stopwatch.StartNew();
        new Dataframe(predictionsInTargetFormat, Natixis70Utils.PredictionHeader.Split(','), "").Save(path);
        ISample.Log.Debug($"SavePredictionsInTargetFormat in {path} took {start.Elapsed.TotalSeconds}s");
    }
    public override IDataSet TestDataset()
    {
        return NewDataSet(XTestRawFile, null);
    }
    public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
    {
        using var fullTraining = NewDataSet(XTrainRawFile, YTrainRawFile);
        int rowsInTrainingSet = (int)(PercentageInTraining * fullTraining.Count + 0.1);
        rowsInTrainingSet -= rowsInTrainingSet % RowsInTargetFormatToRowsInModelFormat(1);
        return fullTraining.IntSplitIntoTrainingAndValidation(rowsInTrainingSet);
    }


    private int[] YShapeInModelFormat(int rowsInTargetFormat)
    {
        int yColCountInModelFormat = 1;
        if (TryToPredictAllHorizonAtTheSameTime)
        {
            yColCountInModelFormat *= Natixis70Utils.HorizonNames.Length;
        }
        if (TryToPredictAllMarketsAtTheSameTime)
        {
            yColCountInModelFormat *= Natixis70Utils.MarketNames.Length;
        }
        return new[] { RowsInTargetFormatToRowsInModelFormat(rowsInTargetFormat), yColCountInModelFormat };
    }
    private int[] YShapeInTargetFormat(int rowsInModelFormat)
    {
        var divider = RowsInTargetFormatToRowsInModelFormat(1);
        Debug.Assert(rowsInModelFormat%divider == 0);
        var rowsInTargetFormat = rowsInModelFormat / divider;
        return new[] { rowsInTargetFormat, 1 + Natixis70Utils.MarketNames.Length * Natixis70Utils.HorizonNames.Length };
    }
    private int RowsInTargetFormatToRowsInModelFormat(int rowsInTargetFormat)
    {
        int rowsInModelFormat = rowsInTargetFormat;
        if (!TryToPredictAllHorizonAtTheSameTime)
        {
            rowsInModelFormat *= Natixis70Utils.HorizonNames.Length;
        }
        if (!TryToPredictAllMarketsAtTheSameTime)
        {
            rowsInModelFormat *= Natixis70Utils.MarketNames.Length;
        }
        return rowsInModelFormat;
    }
    private List<string> FeatureNames()
    {
        var featureNames = new List<string>();
        for (int i = 0; i < Natixis70Utils.EmbeddingDimension; ++i)
        {
            featureNames.Add("embed_" + i);
        }
        featureNames.AddRange(CategoricalFeatures());
        return featureNames;
    }
    private int[] XShapeInModelFormat(int rowsInTargetFormat)
    {
        int xColCountInModelFormat = Natixis70Utils.EmbeddingDimension;

        if (MergeHorizonAndMarketIdInSameFeature)
        {
            Debug.Assert(!TryToPredictAllMarketsAtTheSameTime);
            Debug.Assert(!TryToPredictAllHorizonAtTheSameTime);
            xColCountInModelFormat += 1; //we'll have one single feature for both market to predict and horizon
        }
        else
        {
            if (!TryToPredictAllHorizonAtTheSameTime)
            {
                xColCountInModelFormat += 1; //we'll have one more feature : the horizon to predict (1d / 1w / 2w)
            }
            if (!TryToPredictAllMarketsAtTheSameTime)
            {
                xColCountInModelFormat += 1; //we'll have one more feature : the market to predict (VIX, EURUSD, etc...)
            }
        }
        return new[] { RowsInTargetFormatToRowsInModelFormat(rowsInTargetFormat), xColCountInModelFormat };
    }
    private IDataSet NewDataSet([JetBrains.Annotations.NotNull] string xFileInTargetFormat, [CanBeNull] string yFileInTargetFormatIfAny)
    {
        return new InMemoryDataSet(
            Load_XInModelFormat(xFileInTargetFormat),
            string.IsNullOrEmpty(yFileInTargetFormatIfAny) ? null : Load_YInModelFormat(yFileInTargetFormatIfAny),
            Natixis70Utils.NAME,
            Objective_enum.Regression,
            null,
            new[] { "NONE" },
            FeatureNames().ToArray(),
            CategoricalFeatures().ToArray(),
            false);
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
        return rowInModelFormat % Natixis70Utils.HorizonNames.Length;
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
            rowInModelFormat /= Natixis70Utils.HorizonNames.Length;
        }

        return rowInModelFormat % Natixis70Utils.MarketNames.Length;
    }
    /// <summary>
    /// path to the test dataset in LightGBM compatible format
    /// </summary>
    /// <returns></returns>
    private CpuTensor<float> Load_XInModelFormat(string xFileInTargetFormat)
    {
        var key = xFileInTargetFormat + "_" + ComputeHash();
        if (CacheDataset.TryGetValue(key, out var existingValue))
        {
            return existingValue;
        }

        //We load 'xFileInTargetFormat'
        var xInTargetFormatDataframe = Dataframe.Load(xFileInTargetFormat, true, ',');
        var xInTargetFormat = xInTargetFormatDataframe.Tensor;
        Debug.Assert(xInTargetFormat.Shape[1] == Natixis70Utils.EmbeddingDimension);
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
                xInModelFormatSpan[xInModelFormatSpanIndex++] = marketId * Natixis70Utils.HorizonNames.Length + horizonId;
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

        if (CacheDataset.TryAdd(key, xInModelFormat))
        {
            return xInModelFormat;
        }
        xInModelFormat.Dispose();
        return CacheDataset[key];
    }
    /// <summary>
    /// Load the content of the file 'yFileInTargetFormat' in a CpuTensor (in model format) and return it
    /// </summary>
    /// <param name="yFileInTargetFormat"></param>
    /// <returns></returns>
    private CpuTensor<float> Load_YInModelFormat(string yFileInTargetFormat)
    {
        var key = yFileInTargetFormat + "_" + ComputeHash();
        if (CacheDataset.TryGetValue(key, out var existingValue))
        {
            return existingValue;
        }

        Debug.Assert(File.Exists(yFileInTargetFormat));
        var yInTargetFormatDataframe = Dataframe.Load(yFileInTargetFormat, true, ',');
        var yInTargetFormat = yInTargetFormatDataframe.Tensor;
        var yInModelFormat = PredictionsInTargetFormat_2_PredictionsInModelFormat(yInTargetFormat);
        yInTargetFormat.Dispose();

        if (CacheDataset.TryAdd(key, yInModelFormat))
        {
            return yInModelFormat;
        }
        yInModelFormat.Dispose();
        return CacheDataset[key];
    }
    private static string XTrainRawFile => Path.Combine(Natixis70Utils.DataDirectory, "x_train_ACFqOMF.csv");
    // ReSharper disable once MemberCanBePrivate.Global
    private static string XTestRawFile => Path.Combine(Natixis70Utils.DataDirectory, "x_test_pf4T2aK.csv");
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
    private static string YTrainRawFile => Path.Combine(Natixis70Utils.WorkingDirectory, "Data", "y_train_HNMbC27.csv");
}
