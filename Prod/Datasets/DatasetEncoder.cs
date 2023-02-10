﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using JetBrains.Annotations;
using log4net;

namespace SharpNet.Datasets;

/// <summary>
/// encode a dataset (that may contains categorical features) into a full numerical dataset.
/// each categorical feature will be encoded using the corresponding index of this categorical feature (starting from 0)
/// a negative value for a categorical feature means that the categorical feature is empty
/// </summary>
public class DatasetEncoder
{
    #region Private Fields
    private readonly bool _standardizeDoubleValues;
    private readonly bool _allDataFrameAreAlreadyNormalized;

    /// <summary>
    /// list of all categorical features in the dataset
    /// </summary>
    private string[] CategoricalFeatures { get; }
    /// <summary>
    /// name of the target features (to predict)
    /// (usually a single feature)
    /// </summary>
    [NotNull] private string[] TargetLabels { get; }
    /// <summary>
    /// the list of features that will be used to uniquely identify a row
    /// (usually a single feature)
    /// </summary>
    [NotNull] private string[] IdColumns { get; }

    private Objective_enum DatasetObjective { get; }

    [NotNull] private readonly Dictionary<string, ColumnStatistics> _columnStats = new();
    [NotNull] private static readonly ILog Log = LogManager.GetLogger(typeof(DatasetEncoder));

    /// <summary>
    /// number of time the method 'Fit' has been called
    /// </summary>
    private int _fitCallCount = 0;
    /// <summary>
    /// number of time the method 'Transform' has been called
    /// </summary>
    private int _transformCallCount = 0;
    /// <summary>
    /// number of time the method 'Inverse_Transform' has been called
    /// </summary>
    private int _inverseTransformCallCount = 0;

    #endregion
    /// <summary>
    /// 
    /// </summary>
    /// <param name="datasetSample"></param>
    /// <param name="standardizeDoubleValues"></param>
    /// <param name="allDataFrameAreAlreadyNormalized">
    /// true if we are sure that all fields have already been normalized (with all special characters removed)
    /// </param>
    public DatasetEncoder(AbstractDatasetSample datasetSample, bool standardizeDoubleValues, bool allDataFrameAreAlreadyNormalized)
    {
        CategoricalFeatures = datasetSample.CategoricalFeatures.ToArray();
        TargetLabels = datasetSample.TargetLabels.ToArray();
        IdColumns = datasetSample.IdColumns.ToArray();
        DatasetObjective = datasetSample.GetObjective();
        _standardizeDoubleValues = standardizeDoubleValues;
        _allDataFrameAreAlreadyNormalized = allDataFrameAreAlreadyNormalized;
    }
    public ColumnStatistics this[string featureName] => _columnStats[featureName];

    /// <summary>
    /// load the dataset 'xyDataset' and encode all categorical features into numerical values
    /// (so that it can be processed by LightGBM)
    /// this dataset contains both features (the 'x') and target columns (the 'y')
    /// </summary>
    /// <param name="xyDataset_df"></param>
    /// <param name="datasetSample"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    // ReSharper disable once UnusedMember.Global
    public DataFrameDataSet Transform_XYDataset(DataFrame xyDataset_df, AbstractDatasetSample datasetSample)
    {
        var xTrainEncoded = Transform(xyDataset_df.Drop(TargetLabels));
        var yTrain = xyDataset_df[TargetLabels];
        return NewDataFrameDataSet(xTrainEncoded, yTrain, datasetSample);
    }

    /// <summary>
    /// load the dataset 'xDataset' and encode all categorical features into numerical values
    /// (so that it can be processed by LightGBM)
    /// if 'yDataset' is not empty, it means that this second dataset contains the target 'y'
    /// </summary>
    public DataFrameDataSet Transform_X_and_Y_Dataset([NotNull] DataFrame x_df, [CanBeNull] DataFrame y_df, AbstractDatasetSample datasetSample)
    {
        return NewDataFrameDataSet(Transform(x_df), y_df, datasetSample);
    }

    private static DataFrameDataSet NewDataFrameDataSet(DataFrame xTrainEncoded, DataFrame yTrainEncoded, AbstractDatasetSample datasetSample)
    {
        return new DataFrameDataSet(
            datasetSample,
            xTrainEncoded,
            yTrainEncoded,
            false);
    }

    public DataFrame Fit_Transform(DataFrame string_df)
    {
        Fit(string_df);
        return Transform(string_df);
    }

    //public IList<float> GetDistinctCategoricalPercentage(string categoricalColumn)
    //{
    //    if (!IsCategoricalColumn(categoricalColumn))
    //    {
    //        throw new Exception($"Invalid column {categoricalColumn}: not a categorical column");
    //    }
    //    if (!_featureStats.ContainsKey(categoricalColumn))
    //    {
    //        return null;
    //    }
    //    return _featureStats[categoricalColumn].GetDistinctCategoricalPercentage();
    //}
    public void Fit(DataFrame df)
    {
        if (df == null)
        {
            return;
        }

        if (_transformCallCount != 0)
        {
            throw new ArgumentException($"can't call method {nameof(Fit)} because the method {nameof(Transform)} has already been called");
        }
        ++_fitCallCount;
        //we read the header
        foreach (var c in df.Columns)
        {
            if (!_columnStats.ContainsKey(c))
            {
                _columnStats[c] = new ColumnStatistics(IsCategoricalColumn(c), TargetLabels.Contains(c), IdColumns.Contains(c), _standardizeDoubleValues, _allDataFrameAreAlreadyNormalized);
            }
        }
    
        Parallel.For(0, df.Columns.Length, columnIdxIdDataFrame => FitColumn(df, columnIdxIdDataFrame));
    }


    public void FitMissingCategoricalColumns(params DataFrame[] dfs)
    {
        foreach (var df in dfs)
        {
            if (df == null)
            {
                continue;
            }

            for (var index = 0; index < df.Columns.Length; index++)
            {
                var c = df.Columns[index];
                if (!_columnStats.ContainsKey(c) && IsCategoricalColumn(c))
                {
                    _columnStats[c] = new ColumnStatistics(IsCategoricalColumn(c), TargetLabels.Contains(c), IdColumns.Contains(c), _standardizeDoubleValues, _allDataFrameAreAlreadyNormalized);
                    FitColumn(df, index);
                }
            }
        }
    }


    private void FitColumn(DataFrame df, int columnIdxIdDataFrame)
    {
        var rows = df.Shape[0];
        var columnStats = _columnStats[df.Columns[columnIdxIdDataFrame]];
        var tensorType = df.ColumnsDesc[columnIdxIdDataFrame].Item2;
        int tensorCols = df.EmbeddedTensors[tensorType].Shape[1];
        var columnIdxInTensor = df.ColumnsDesc[columnIdxIdDataFrame].Item3;
        Debug.Assert(columnIdxInTensor < tensorCols);
        switch (tensorType)
        {
            case DataFrame.STRING_TYPE_IDX:
                var strReadonlyContent = df.StringTensorEvenIfView.ReadonlyContent;
                for (var rowIndex = 0; rowIndex < rows; rowIndex++)
                {
                    columnStats.Fit(strReadonlyContent[columnIdxInTensor + rowIndex * tensorCols]);
                }
                break;
            case DataFrame.FLOAT_TYPE_IDX:
                var floatReadonlyContent = df.FloatTensorEvenIfView.ReadonlyContent;
                for (var rowIndex = 0; rowIndex < rows; rowIndex++)
                {
                    columnStats.Fit(floatReadonlyContent[columnIdxInTensor + rowIndex * tensorCols]);
                }
                break;
            case DataFrame.INT_TYPE_IDX:
                var intReadonlyContent = df.IntTensorEvenIfView.ReadonlyContent;
                for (var rowIndex = 0; rowIndex < rows; rowIndex++)
                {
                    columnStats.Fit(intReadonlyContent[columnIdxInTensor + rowIndex * tensorCols]);
                }
                break;
            default:
                throw new NotImplementedException($"Tensor Type {tensorType} not supported");
        }

    }
    // ReSharper disable once MemberCanBePrivate.Global
    public DataFrame Transform(DataFrame df)
    {
        if (_fitCallCount == 0)
        {
            throw new ArgumentException($"can't call method {nameof(Transform)} because the method {nameof(Fit)} has never been called");
        }
        ++_transformCallCount;
        //Debug.Assert(df.IsStringDataFrame);
        foreach (var c in df.Columns)
        {
            if (!_columnStats.ContainsKey(c))
            {
                throw new ArgumentException($"unknown column to transform : {c}, not among {string.Join(' ', df.Columns)}");
            }
        }

        var rows = df.Shape[0];
        var columns = df.Shape[1];
        var content = new float[rows * columns];
        
        void TransformColumn(int columnIdxIdDataFrame)
        {
            var columnStats = _columnStats[df.Columns[columnIdxIdDataFrame]];
            var tensorType = df.ColumnsDesc[columnIdxIdDataFrame].Item2;
            int tensorCols = df.EmbeddedTensors[tensorType].Shape[1];
            var columnIdxInTensor = df.ColumnsDesc[columnIdxIdDataFrame].Item3;
            Debug.Assert(columnIdxInTensor < tensorCols);
            switch (tensorType)
            {
                case DataFrame.STRING_TYPE_IDX:
                    var strReadonlyContent = df.StringTensorEvenIfView.ReadonlyContent;
                    for (var rowIndex = 0; rowIndex < rows; rowIndex++)
                    {
                        var contentIndexInDataFrame = columnIdxIdDataFrame + rowIndex * df.ColumnsDesc.Count;
                        var contentIndexInTensor = columnIdxInTensor + rowIndex * tensorCols;
                        content[contentIndexInDataFrame] = (float)columnStats.Transform(strReadonlyContent[contentIndexInTensor]);
                    }
                    break;
                case DataFrame.FLOAT_TYPE_IDX:
                    var floatReadonlyContent = df.FloatTensorEvenIfView.ReadonlyContent;
                    for (var rowIndex = 0; rowIndex < rows; rowIndex++)
                    {
                        var contentIndexInDataFrame = columnIdxIdDataFrame + rowIndex * df.Columns.Length;
                        var contentIndexInTensor = columnIdxInTensor + rowIndex * tensorCols;
                        content[contentIndexInDataFrame] = (float)columnStats.Transform(floatReadonlyContent[contentIndexInTensor]);
                    }
                    break;
                case DataFrame.INT_TYPE_IDX:
                    var intReadonlyContent = df.IntTensorEvenIfView.ReadonlyContent;
                    for (var rowIndex = 0; rowIndex < rows; rowIndex++)
                    {
                        var contentIndexInDataFrame = columnIdxIdDataFrame + rowIndex * df.Columns.Length;
                        var contentIndexInTensor = columnIdxInTensor + rowIndex * tensorCols;
                        content[contentIndexInDataFrame] = (float)columnStats.Transform(intReadonlyContent[contentIndexInTensor]);
                    }
                    break;
                default:
                    throw new NotImplementedException($"Tensor Type {tensorType} not supported");
            }

        }
        Parallel.For(0, df.Columns.Length, TransformColumn);

        return DataFrame.New(content, df.Columns);
    }

    /// <summary>
    /// Decode the dataframe 'df' (replacing numerical values by their categorical values) and return the content of the decoded DataFrame
    /// </summary>
    /// <param name="float_df"></param>
    /// <param name="missingNumberValue"></param>
    /// <returns></returns>
    public DataFrame Inverse_Transform(DataFrame float_df, string missingNumberValue = "")
    {
        ++_inverseTransformCallCount;
        Debug.Assert(float_df.IsFloatDataFrame);
        //we display a warning for column names without known encoding
        var unknownFeatureName = Utils.Without(float_df.Columns, _columnStats.Keys);
        unknownFeatureName.Remove("0");
        if (unknownFeatureName.Count != 0)
        {
            Log.Warn($"{unknownFeatureName.Count} unknown feature name(s): '{string.Join(' ', unknownFeatureName)}' (not in '{string.Join(' ', _columnStats.Keys)}')");
        }

        var encodedContent = float_df.FloatCpuTensor().ReadonlyContent;
        var decodedContent = new string[encodedContent.Length];
        for (int idx = 0; idx< encodedContent.Length;++idx)
        {
            var encodedValue = encodedContent[idx];
            int col = idx % float_df.Columns.Length;
            var decodedValueAsString = _columnStats.TryGetValue(float_df.Columns[col], out var featureStat)
                ? featureStat.Inverse_Transform(encodedValue, missingNumberValue)
                : encodedValue.ToString(CultureInfo.InvariantCulture);
            decodedContent[idx] = decodedValueAsString;
        }
        return DataFrame.New(decodedContent, float_df.Columns);
    }

    public DataFrame Inverse_Transform_float(DataFrame float_df)
    {
        ++_inverseTransformCallCount;
        Debug.Assert(float_df.IsFloatDataFrame);
        //we display a warning for column names without known encoding
        var unknownFeatureName = Utils.Without(float_df.Columns, _columnStats.Keys);
        unknownFeatureName.Remove("0");
        if (unknownFeatureName.Count != 0)
        {
            Log.Warn($"{unknownFeatureName.Count} unknown feature name(s): '{string.Join(' ', unknownFeatureName)}' (not in '{string.Join(' ', _columnStats.Keys)}')");
        }

        var encodedContent = float_df.FloatCpuTensor().ReadonlyContent;
        var decodedContent = new float[encodedContent.Length];
        for (int idx = 0; idx < encodedContent.Length; ++idx)
        {
            var encodedValue = encodedContent[idx];
            int col = idx % float_df.ColumnsDesc.Count;
            var decodedValueAsString = _columnStats.TryGetValue(float_df.Columns[col], out var featureStat)
                ? featureStat.Inverse_Transform_float(encodedValue)
                : encodedValue;
            decodedContent[idx] = decodedValueAsString;
        }
        return DataFrame.New(decodedContent, float_df.Columns);
    }


    /// <summary>
    /// return the distinct values of a 'categorical feature' or of a 'categorical target label '
    /// </summary>
    /// <param name="categoricalColumn"></param>
    /// <returns></returns>
    private IList<string> GetDistinctCategoricalValues(string categoricalColumn)
    {
        if (!IsCategoricalColumn(categoricalColumn))
        {
            throw new Exception($"Invalid column {categoricalColumn}: not a categorical column");
        }
        if (!_columnStats.ContainsKey(categoricalColumn))
        {
            return null;
        }
        return _columnStats[categoricalColumn].GetDistinctCategoricalValues();
    }
    private bool IsCategoricalColumn(string columnName)
    {
        if (CategoricalFeatures.Contains(columnName) || IdColumns.Contains(columnName))
        {
            return true;
        }
        if (DatasetObjective == Objective_enum.Classification && TargetLabels.Contains(columnName))
        {
            return true;
        }
        return false;
    }
}
