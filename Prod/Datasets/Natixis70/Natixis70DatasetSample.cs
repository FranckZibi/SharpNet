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
    #endregion

    private CpuTensor<float> PredictionsInTargetFormat_2_PredictionsInModelFormat(CpuTensor<float> predictionsInTargetFormat)
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

    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat_with_IdColumns)
    {
        if (predictionsInModelFormat_with_IdColumns == null)
        {
            return null;
        }
        if (predictionsInModelFormat_with_IdColumns.Shape[1] == 39)
        {
            var cpuTensor = CpuTensor<float>.AddIndexInFirstColumn(predictionsInModelFormat_with_IdColumns.FloatCpuTensor(), 0);
            return DataFrame.New(cpuTensor, PredictionInTargetFormatHeader());
        }

        var predictionsInModelFormatSpan = predictionsInModelFormat_with_IdColumns.FloatCpuTensor().AsReadonlyFloatCpuContent;
        var predictionsInModelFormatSpanIndex = 0;

        // the predictions in the target format for the Natixis70 Challenge
        var predictionsInTargetFormat = new CpuTensor<float>(YShapeInTargetFormat(predictionsInModelFormat_with_IdColumns.Shape[0]));
        var predictionsInTargetFormatSpan = predictionsInTargetFormat.AsFloatCpuSpan;
        var divider = RowsInTargetFormatToRowsInModelFormat(1);

        for (int rowInModelFormat = 0; rowInModelFormat < predictionsInModelFormat_with_IdColumns.Shape[0]; ++rowInModelFormat)
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
        return DataFrame.New(predictionsInTargetFormat, PredictionInTargetFormatHeader());
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

    public override string[] IdColumns => new []{""};
    public override string[] TargetLabels => Natixis70Utils.PredictionHeader.Trim(',').Split(',');

    /// <summary>
    /// true if the test dataset must also have associated labels (so that we can compute a score for it)
    /// </summary>
    public static bool TestDatasetMustHaveLabels = false;

    
    public override DataSet TestDataset()
    {
        if (TestDatasetMustHaveLabels)
        {
            using var trainingAndValidationAndTestDataset = NewDataSet(XTrainRawFile, YTrainRawFile);
            var percentageInTraining = PercentageInTraining;
            if (PercentageInTraining >= 1.0)
            {
                percentageInTraining = 0.8;
            }
            int rowsForTrainingAndValidation = (int)(percentageInTraining * trainingAndValidationAndTestDataset.Count + 0.1);
            rowsForTrainingAndValidation -= rowsForTrainingAndValidation % DatasetRowsInModelFormatMustBeMultipleOf();
            return trainingAndValidationAndTestDataset.IntSplitIntoTrainingAndValidation(rowsForTrainingAndValidation).Test;
        }
        else
        {
            return NewDataSet(XTestRawFile, null);
        }
    }

    public override int DatasetRowsInModelFormatMustBeMultipleOf()
    {
        return RowsInTargetFormatToRowsInModelFormat(1);
    }

    public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
    {
        var percentageInTraining = PercentageInTraining;
        if (TestDatasetMustHaveLabels)
        {
            using var trainingAndValidationAndTestDataset = NewDataSet(XTrainRawFile, YTrainRawFile);
            if (PercentageInTraining >= 1.0)
            {
                percentageInTraining = 0.8;
            }
            int rowsForTrainingAndValidation = (int)(percentageInTraining * trainingAndValidationAndTestDataset.Count + 0.1);
            rowsForTrainingAndValidation -= rowsForTrainingAndValidation % DatasetRowsInModelFormatMustBeMultipleOf();
            var trainingAndValidationDataset = trainingAndValidationAndTestDataset.IntSplitIntoTrainingAndValidation(rowsForTrainingAndValidation).Training;
            if (PercentageInTraining >= 1.0)
            {
                return new TrainingAndTestDataset(trainingAndValidationDataset, null, trainingAndValidationAndTestDataset.Name);
            }
            int rowsForTraining = (int)(percentageInTraining * trainingAndValidationDataset.Count + 0.1);
            rowsForTraining -= rowsForTraining % DatasetRowsInModelFormatMustBeMultipleOf();
            return trainingAndValidationDataset.IntSplitIntoTrainingAndValidation(rowsForTraining);
        }
        else
        {
            using var trainingAndValidationDataset = NewDataSet(XTrainRawFile, YTrainRawFile);
            int rowsForTraining = (int)(percentageInTraining * trainingAndValidationDataset.Count + 0.1);
            rowsForTraining -= rowsForTraining % DatasetRowsInModelFormatMustBeMultipleOf();
            return trainingAndValidationDataset.IntSplitIntoTrainingAndValidation(rowsForTraining);
        }
    }

    protected override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        return EvaluationMetricEnum.Rmse;
    }

    public override Objective_enum GetObjective() => Objective_enum.Regression;


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
        featureNames.AddRange(CategoricalFeatures);
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
    private DataSet NewDataSet([JetBrains.Annotations.NotNull] string xFileInTargetFormat, [CanBeNull] string yFileInTargetFormatIfAny)
    {
        return new InMemoryDataSet(
            Load_XInModelFormat(xFileInTargetFormat),
            string.IsNullOrEmpty(yFileInTargetFormatIfAny) ? null : Load_YInModelFormat(yFileInTargetFormatIfAny),
            Natixis70Utils.NAME,
            GetObjective(),
            null,
            columnNames: FeatureNames().ToArray(),
            categoricalFeatures: CategoricalFeatures,
            useBackgroundThreadToLoadNextMiniBatch: false,
            separator: GetSeparator());
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
        var xInTargetFormatDataFrame = DataFrame.read_float_csv(xFileInTargetFormat);
        var xInTargetFormat = xInTargetFormatDataFrame.FloatCpuTensor();
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
        var yInTargetFormatDataFrame = DataFrame.read_float_csv(yFileInTargetFormat);
        var yInTargetFormat = yInTargetFormatDataFrame.FloatCpuTensor();
        var yInModelFormat = PredictionsInTargetFormat_2_PredictionsInModelFormat(yInTargetFormat);
        yInTargetFormat.Dispose();

        if (CacheDataset.TryAdd(key, yInModelFormat))
        {
            return yInModelFormat;
        }
        yInModelFormat.Dispose();
        return CacheDataset[key];
    }

    private const string FILE_SUFFIX = "";
    //private const string FILE_SUFFIX = "_small";

    private static string XTrainRawFile => Path.Combine(Natixis70Utils.DataDirectory, "x_train_ACFqOMF"+ FILE_SUFFIX+".csv");
    private static string XTestRawFile => Path.Combine(Natixis70Utils.DataDirectory, "x_test_pf4T2aK" + FILE_SUFFIX + ".csv");
    private static string YTrainRawFile => Path.Combine(Natixis70Utils.WorkingDirectory, "Data", "y_train_HNMbC27" + FILE_SUFFIX + ".csv");
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
