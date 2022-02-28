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
    private static DoubleAccumulator[] Y_RAW_statistics { get; }
    private static DoubleAccumulator[] Y_RAW_abs_statistics { get; }
    private static readonly ConcurrentDictionary<string, CpuTensor<float>> PredictionsInModelFormat_2_PredictionsInTargetFormat_Cache = new();
    #endregion

    #region constructors
    static Natixis70DatasetSample()
    {
        var cpuTensor = Dataframe.Load(YTrainRawFile, true, ',').Drop(new[] { "" }).Tensor;
        Y_RAW_statistics = ExtractColumnStatistic(cpuTensor, false);
        Y_RAW_abs_statistics = ExtractColumnStatistic(cpuTensor, true);
    }
    public Natixis70DatasetSample() : base(new HashSet<string>())
    {
    }
    public static Natixis70DatasetSample ValueOfNatixis70DatasetSample(string workingDirectory, string modelName)
    {
        return (Natixis70DatasetSample)ISample.LoadConfigIntoSample(() => new Natixis70DatasetSample(), workingDirectory, modelName);
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
    
    public override bool PostBuild()
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
    /// <param name="dataframe_path">a dataset in LightGBM format</param>
    /// <returns></returns>
    public override CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(string dataframe_path)
    {
        if (PredictionsInModelFormat_2_PredictionsInTargetFormat_Cache.TryGetValue(dataframe_path, out var res))
        {
            return res;
        }
        var predictionsInModelFormat = Dataframe.Load(dataframe_path, true, ',').Keep(new[] { "y" }).Tensor;
        var predictionInTargetFormat = PredictionsInModelFormat_2_PredictionsInTargetFormat(predictionsInModelFormat).DropColumns(new[] { 0 });
        Debug.Assert(predictionInTargetFormat.Shape.Length == 2);
        Debug.Assert(predictionInTargetFormat.Shape[1] == 39);
        PredictionsInModelFormat_2_PredictionsInTargetFormat_Cache.TryAdd(dataframe_path, predictionInTargetFormat);
        return predictionInTargetFormat;
    }

    public override (CpuTensor<float> trainPredictions, CpuTensor<float> validationPredictions, CpuTensor<float> testPredictions) LoadAllPredictions()
    {
        return LoadAllPredictions(true, true, ',');
    }

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
    public override IDataSet FullTraining()
    {
        return NewDataSet(XTrainRawFile, YTrainRawFile);
    }
    public override void ComputeAndSavePredictions(CpuTensor<float> predictionsInModelFormat, string path)
    {
        var predictionsInTargetFormat = PredictionsInModelFormat_2_PredictionsInTargetFormat(predictionsInModelFormat);
        new Dataframe(predictionsInTargetFormat, Natixis70Utils.PredictionHeader.Split(','), "").Save(path);
    }

    protected override CpuTensor<float> UnnormalizeYIfNeeded(CpuTensor<float> y)
    {
        if (Normalization == normalize_enum.NONE)
        {
            return y; //no need to unnormalize y
        }

        var ySpan = y.AsReadonlyFloatCpuContent;
        var Yunnormalized = new CpuTensor<float>(y.Shape);
        var YunnormalizedSpan = Yunnormalized.AsFloatCpuSpan;

        int index = 0;
        for (int row = 0; row < Yunnormalized.Shape[0]; ++row)
        {
            var horizonId = RowToHorizonId(row);
            var marketId = RowToMarketId(row);
            //we load the row 'row' in 'y' tensor
            for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : marketId); ++currentMarketId)
            {
                for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : horizonId); ++currentHorizonId)
                {
                    int rawColIndex = 1 + Natixis70Utils.HorizonNames.Length * currentMarketId + currentHorizonId;
                    var yValue = ySpan[index];
                    if (Normalization == normalize_enum.MINUS_MEAN_DIVIDE_BY_VOL)
                    {
                        var colStatistics = Y_RAW_statistics[rawColIndex - 1];
                        var yUnnormalizedValue = (float)(yValue * colStatistics.Volatility + colStatistics.Average);
                        YunnormalizedSpan[index] = yUnnormalizedValue;
                    }
                    else if (Normalization == normalize_enum.DIVIDE_BY_ABS_MEAN)
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
    protected override IDataSet TestDataset()
    {
        return NewDataSet(XTestRawFile, null);
    }
    protected override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
    {
        using var fullTraining = NewDataSet(XTrainRawFile, YTrainRawFile);
        int rowsInTrainingSet = (int)(PercentageInTraining * fullTraining.Count + 0.1);
        rowsInTrainingSet -= rowsInTrainingSet % RawCountToCount(1);
        return fullTraining.IntSplitIntoTrainingAndValidation(rowsInTrainingSet);
    }

    private int[] Y_Shape(int yRawCount)
    {
        int yColCount = 1;
        if (TryToPredictAllHorizonAtTheSameTime)
        {
            yColCount *= Natixis70Utils.HorizonNames.Length;
        }
        if (TryToPredictAllMarketsAtTheSameTime)
        {
            yColCount *= Natixis70Utils.MarketNames.Length;
        }
        return new[] { RawCountToCount(yRawCount), yColCount };
    }
    private int[] YRaw_Shape(int yCount)
    {
        return new[] { CountToRawCount(yCount), 1 + Natixis70Utils.MarketNames.Length * Natixis70Utils.HorizonNames.Length };
    }
    private CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(CpuTensor<float> predictionsInModelFormat)
    {
        if (predictionsInModelFormat.Shape[1] == 39)
        {
            return CpuTensor<float>.AddIndexInFirstColumn(predictionsInModelFormat, 0);
        }

        var predictionsInModelFormatSpan = predictionsInModelFormat.AsReadonlyFloatCpuContent;
        var predictionsInModelFormatSpanIndex = 0;

        // the predictions in the expected format for the Natixis70 Challenge
        var predictionsInTargetFormat = new CpuTensor<float>(YRaw_Shape(predictionsInModelFormat.Shape[0]));
        var predictionsInTargetFormatSpan = predictionsInTargetFormat.AsFloatCpuSpan;
        var divider = RawCountToCount(1);

        for (int row = 0; row < predictionsInModelFormat.Shape[0]; ++row)
        {
            var rawRow = row / divider;
            predictionsInTargetFormatSpan[rawRow * predictionsInTargetFormat.Shape[1]] = rawRow;
            int horizonId = RowToHorizonId(row);
            int marketId = RowToMarketId(row);
            //we load the row 'row' in 'yRaw' tensor
            for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : marketId); ++currentMarketId)
            {
                for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : horizonId); ++currentHorizonId)
                {
                    int rawColIndex = 1 + Natixis70Utils.HorizonNames.Length * currentMarketId + currentHorizonId;
                    var yValue = predictionsInModelFormatSpan[predictionsInModelFormatSpanIndex++];
                    if (Math.Abs(yValue) < 1e-4)
                    {
                        yValue = 0;
                    }
                    predictionsInTargetFormatSpan[rawRow * predictionsInTargetFormat.Shape[1] + rawColIndex] = yValue;
                }
            }
        }
        Debug.Assert(predictionsInTargetFormat.Shape.Length == 2);
        Debug.Assert(predictionsInTargetFormat.Shape[1] == (1 + 39));
        return predictionsInTargetFormat;
    }
    private int CountToRawCount(int count)
    {
        int rawCount = count;
        if (!TryToPredictAllHorizonAtTheSameTime)
        {
            Debug.Assert(count % Natixis70Utils.HorizonNames.Length == 0);
            rawCount /= Natixis70Utils.HorizonNames.Length;
        }
        if (!TryToPredictAllMarketsAtTheSameTime)
        {
            Debug.Assert(count % Natixis70Utils.MarketNames.Length == 0);
            rawCount /= Natixis70Utils.MarketNames.Length;
        }
        return rawCount;
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
    private int[] X_Shape(int xRowCount)
    {
        int xColCount = Natixis70Utils.EmbeddingDimension;

        if (MergeHorizonAndMarketIdInSameFeature)
        {
            Debug.Assert(!TryToPredictAllMarketsAtTheSameTime);
            Debug.Assert(!TryToPredictAllHorizonAtTheSameTime);
            xColCount += 1; //we'll have one single feature for both market to predict and horizon
        }
        else
        {
            if (!TryToPredictAllHorizonAtTheSameTime)
            {
                xColCount += 1; //we'll have one more feature : the horizon to predict (1d / 1w / 2w)
            }
            if (!TryToPredictAllMarketsAtTheSameTime)
            {
                xColCount += 1; //we'll have one more feature : the market to predict (VIX, EURUSD, etc...)
            }
        }
        return new[] { RawCountToCount(xRowCount), xColCount };
    }
    private IDataSet NewDataSet([JetBrains.Annotations.NotNull] string xRawFile, [CanBeNull] string yRawFileIfAny)
    {
        return new InMemoryDataSet(
            Load_X(xRawFile),
            string.IsNullOrEmpty(yRawFileIfAny) ? null : Load_Y(yRawFileIfAny),
            Natixis70Utils.NAME,
            Objective_enum.Regression,
            null,
            new[] { "NONE" },
            FeatureNames().ToArray(),
            CategoricalFeatures().ToArray(),
            false);
    }
    /// <summary>
    /// return the horizonId associated with row 'row', or -1 if the row is associated with all horizon ids
    /// </summary>
    /// <param name="row"></param>
    /// <returns></returns>
    private int RowToHorizonId(int row)
    {
        if (TryToPredictAllHorizonAtTheSameTime)
        {
            return -1;
        }
        return row % Natixis70Utils.HorizonNames.Length;
    }
    /// <summary>
    /// return the marketId associated with row 'row', or -1 if the row is associated with all market ids
    /// </summary>
    /// <param name="row"></param>
    /// <returns></returns>
    private int RowToMarketId(int row)
    {
        if (TryToPredictAllMarketsAtTheSameTime)
        {
            return -1;
        }
        if (!TryToPredictAllHorizonAtTheSameTime)
        {
            row /= Natixis70Utils.HorizonNames.Length;
        }

        return row % Natixis70Utils.MarketNames.Length;
    }
    /// <summary>
    /// path to the test dataset in LightGBM compatible format
    /// </summary>
    /// <returns></returns>
    private CpuTensor<float> Load_X(string xRawFile)
    {
        var key = xRawFile + "_" + ComputeHash();
        if (CacheDataset.TryGetValue(key, out var existingValue))
        {
            return existingValue;
        }

        //We load 'xRaw'
        var xRawDataframe = Dataframe.Load(xRawFile, true, ',');
        var xRaw = xRawDataframe.Tensor;
        Debug.Assert(xRaw.Shape[1] == Natixis70Utils.EmbeddingDimension);
        int count = xRaw.Shape[0];
        var xRawSpan = xRaw.AsReadonlyFloatCpuContent;
        var x = new CpuTensor<float>(X_Shape(count));
        var xSpan = x.AsFloatCpuSpan;
        int xSpanIndex = 0;
        var divider = RawCountToCount(1);

        for (int row = 0; row < x.Shape[0]; row++)
        {
            int rawRow = row / divider;
            int horizonId = RowToHorizonId(row);
            int marketId = RowToMarketId(row);

            //we load the row 'row' in 'x' tensor
            for (int col = 0; col < xRaw.Shape[1]; ++col)
            {
                xSpan[xSpanIndex++] = xRawSpan[rawRow * xRaw.Shape[1] + col];
            }

            if (MergeHorizonAndMarketIdInSameFeature)
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

        if (CacheDataset.TryAdd(key, x))
        {
            return x;
        }
        x.Dispose();
        return CacheDataset[key];
    }
    /// <summary>
    /// Load the content of the file  'yRawFileIfAny' in a CpuTensor and return it
    /// </summary>
    /// <param name="yRawFileIfAny"></param>
    /// <returns></returns>
    private CpuTensor<float> Load_Y(string yRawFileIfAny)
    {
        var key = yRawFileIfAny + "_" + ComputeHash();
        if (CacheDataset.TryGetValue(key, out var existingValue))
        {
            return existingValue;
        }

        Debug.Assert(File.Exists(yRawFileIfAny));
        var yRawDataframe = Dataframe.Load(yRawFileIfAny, true, ',');
        var yRaw = yRawDataframe.Tensor;
        var yRawSpan = yRaw.AsReadonlyFloatCpuContent;
        var y = new CpuTensor<float>(Y_Shape(yRaw.Shape[0]));
        var ySpan = y.AsFloatCpuSpan;
        int ySpanIndex = 0;
        var divider = RawCountToCount(1);

        for (int row = 0; row < y.Shape[0]; row++)
        {
            int rawRow = row / divider;
            int horizonId = RowToHorizonId(row);
            int marketId = RowToMarketId(row);

            //we load the row 'row' in 'y' tensor
            for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : marketId); ++currentMarketId)
            {
                for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : horizonId); ++currentHorizonId)
                {
                    int rawColIndex = 1 + Natixis70Utils.HorizonNames.Length * currentMarketId + currentHorizonId;
                    var yRawValue = yRawSpan[rawRow * yRaw.Shape[1] + rawColIndex];
                    if (Normalization == normalize_enum.MINUS_MEAN_DIVIDE_BY_VOL)
                    {
                        var colStatistics = Y_RAW_statistics[rawColIndex - 1];
                        yRawValue = (float)((yRawValue - colStatistics.Average) / colStatistics.Volatility);
                    }
                    else if (Normalization == normalize_enum.DIVIDE_BY_ABS_MEAN)
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

        if (CacheDataset.TryAdd(key, y))
        {
            return y;
        }
        y.Dispose();
        return CacheDataset[key];
    }
    private int RawCountToCount(int rawCount)
    {
        int count = rawCount;
        if (!TryToPredictAllHorizonAtTheSameTime)
        {
            count *= Natixis70Utils.HorizonNames.Length;
        }
        if (!TryToPredictAllMarketsAtTheSameTime)
        {
            count *= Natixis70Utils.MarketNames.Length;
        }
        return count;
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
