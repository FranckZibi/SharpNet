using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Models;

namespace SharpNet.Datasets.Natixis70;

public class Natixis70DatasetHyperParameters : AbstractSample
{
    #region private fields
    private static readonly ConcurrentDictionary<string, CpuTensor<float>> cacheDataset = new();
    #endregion

    #region constructors
    public Natixis70DatasetHyperParameters() : base(new HashSet<string>())
    {
    }
    public static Natixis70DatasetHyperParameters ValueOf(string workingDirectory, string modelName)
    {
        return (Natixis70DatasetHyperParameters)ISample.LoadConfigIntoSample(() => new Natixis70DatasetHyperParameters(), workingDirectory, modelName);
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
    // ReSharper disable once MemberCanBePrivate.Global
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

    /// <summary>
    /// path to the test dataset in LightGBM compatible format
    /// </summary>
    /// <returns></returns>
    public string XTestDatasetPath()
    {
        using var test = NewDataSet(Natixis70Utils.XTestRawFile, null);
        return AbstractModel.DatasetPath(test, false, Natixis70Utils.NatixisDatasetDirectory);
    }
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
    public List<string> CategoricalFeatures()
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
    public List<string> TargetFeatures()
    {
        return new List<string>{"y"};
    }
    public int RawCountToCount(int rawCount)
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
    /// <summary>
    /// convert a dataset in LightGBM format (first column is the target 'y') to the challenge target format 
    /// </summary>
    /// <param name="lightGBM_dataset_path">a dataset in LightGBM format</param>
    /// <returns></returns>
    public CpuTensor<float> LightGBM_2_ExpectedPredictionFormat(string lightGBM_dataset_path)
    {
        var y_lightGBM = Dataframe.Load(lightGBM_dataset_path, true, ',').Keep(new[] { "y" }).Tensor;
        var y_target = LightGBM_2_ExpectedPredictionFormat(y_lightGBM).DropColumns(new[] { 0 });
        Debug.Assert(y_target.Shape.Length == 2);
        Debug.Assert(y_target.Shape[1] == 39);
        return y_target;
    }
    public int[] X_Shape(int xRowCount)
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
    public InMemoryDataSet NewDataSet([NotNull] string xRawFile, [CanBeNull] string yRawFileIfAny)
    {
        return new InMemoryDataSet(
            Load_X(xRawFile),
            string.IsNullOrEmpty(yRawFileIfAny) ? null : Load_Y(yRawFileIfAny),
            "Natixis70",
            Objective_enum.Regression,
            null,
            new[] { "NONE" },
            ComputeFeatureNames(),
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



    public CpuTensor<float> UnnormalizeYIfNeeded(CpuTensor<float> y)
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
                        var colStatistics = Natixis70Utils.Y_RAW_statistics[rawColIndex - 1];
                        var yUnnormalizedValue = (float)(yValue * colStatistics.Volatility + colStatistics.Average);
                        YunnormalizedSpan[index] = yUnnormalizedValue;
                    }
                    else if (Normalization == normalize_enum.DIVIDE_BY_ABS_MEAN)
                    {
                        var absColStatistics = Natixis70Utils.Y_RAW_abs_statistics[rawColIndex - 1];
                        var yUnnormalizedValue = (float)(yValue * absColStatistics.Average);
                        YunnormalizedSpan[index] = yUnnormalizedValue;
                    }

                    ++index;
                }
            }
        }
        return Yunnormalized;
    }

    /// <summary>
    /// Load the content of the file  'xRawFile' in a CpuTensor and return it
    /// </summary>
    /// <param name="xRawFile"></param>
    /// <returns></returns>
    private CpuTensor<float> Load_X(string xRawFile)
    {
        var key = xRawFile + "_" + ComputeHash();
        if (cacheDataset.TryGetValue(key, out var existingValue))
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

        if (cacheDataset.TryAdd(key, x))
        {
            return x;
        }
        x.Dispose();
        return cacheDataset[key];
    }
    /// <summary>
    /// Load the content of the file  'yRawFileIfAny' in a CpuTensor and return it
    /// </summary>
    /// <param name="yRawFileIfAny"></param>
    /// <returns></returns>
    private CpuTensor<float> Load_Y(string yRawFileIfAny)
    {
        var key = yRawFileIfAny + "_" + ComputeHash();
        if (cacheDataset.TryGetValue(key, out var existingValue))
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
                        var colStatistics = Natixis70Utils.Y_RAW_statistics[rawColIndex - 1];
                        yRawValue = (float)((yRawValue - colStatistics.Average) / colStatistics.Volatility);
                    }
                    else if (Normalization == normalize_enum.DIVIDE_BY_ABS_MEAN)
                    {
                        var absColStatistics = Natixis70Utils.Y_RAW_abs_statistics[rawColIndex - 1];
                        yRawValue = (float)(yRawValue / absColStatistics.Average);
                    }
                    ySpan[ySpanIndex++] = yRawValue;
                }
            }
        }
        Debug.Assert(ySpanIndex == y.Count);
        yRaw.Dispose();

        if (cacheDataset.TryAdd(key, y))
        {
            return y;
        }
        y.Dispose();
        return cacheDataset[key];
    }
    private string[] ComputeFeatureNames()
    {
        var featureNames = new List<string>();
        for (int i = 0; i < Natixis70Utils.EmbeddingDimension; ++i)
        {
            featureNames.Add("embed_" + i);
        }
        featureNames.AddRange(CategoricalFeatures());
        return featureNames.ToArray();
    }

    public void SavePredictions(CpuTensor<float> y_lightGBM, string path)
    {
        var y_target = LightGBM_2_ExpectedPredictionFormat(y_lightGBM);
        new Dataframe(y_target, Natixis70Utils.PredictionHeader.Split(','), "").Save(path);
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
    private CpuTensor<float> LightGBM_2_ExpectedPredictionFormat(CpuTensor<float> y_lightGBM)
    {
        var sb = new StringBuilder();
        sb.Append(Natixis70Utils.PredictionHeader + Environment.NewLine);

        var y_lightGBMSpan = y_lightGBM.AsReadonlyFloatCpuContent;
        var y_lightGBMSpanIndex = 0;

        // the predictions in the expected format for the Natixis70 Challenge
        var y_target = new CpuTensor<float>(YRaw_Shape(y_lightGBM.Shape[0]));
        var y_targetSpan = y_target.AsFloatCpuSpan;
        var divider = RawCountToCount(1);

        for (int row = 0; row < y_lightGBM.Shape[0]; ++row)
        {
            var rawRow = row / divider;
            y_targetSpan[rawRow * y_target.Shape[1]] = rawRow;
            int horizonId = RowToHorizonId(row);
            int marketId = RowToMarketId(row);
            //we load the row 'row' in 'yRaw' tensor
            for (int currentMarketId = (marketId < 0 ? 0 : marketId); currentMarketId <= (marketId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : marketId); ++currentMarketId)
            {
                for (int currentHorizonId = (horizonId < 0 ? 0 : horizonId); currentHorizonId <= (horizonId < 0 ? (Natixis70Utils.HorizonNames.Length - 1) : horizonId); ++currentHorizonId)
                {
                    int rawColIndex = 1 + Natixis70Utils.HorizonNames.Length * currentMarketId + currentHorizonId;
                    var yValue = y_lightGBMSpan[y_lightGBMSpanIndex++];
                    if (Math.Abs(yValue) < 1e-4)
                    {
                        yValue = 0;
                    }
                    y_targetSpan[rawRow * y_target.Shape[1] + rawColIndex] = yValue;
                }
            }
        }
        Debug.Assert(y_target.Shape.Length == 2);
        Debug.Assert(y_target.Shape[1] == (1 + 39));
        return y_target;
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

}
