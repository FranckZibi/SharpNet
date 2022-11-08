using System;
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
    private readonly AbstractDatasetSample _datasetSample;
    private readonly bool _standardizeDoubleValues;

    /// <summary>
    /// list of all categorical features in the dataset
    /// </summary>
    private string[] CategoricalFeatures => _datasetSample.CategoricalFeatures;
    /// <summary>
    /// name of the target features (to predict)
    /// (usually a single feature)
    /// </summary>
    [NotNull] private string[] Targets => _datasetSample.TargetLabels;
    /// <summary>
    /// the list of features that will be used to uniquely identify a row
    /// (usually a single feature)
    /// </summary>
    [NotNull] private string[] IdColumns => _datasetSample.IdColumns;
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

    public DatasetEncoder(AbstractDatasetSample datasetSample, bool standardizeDoubleValues)
    {
        _datasetSample = datasetSample;
        _standardizeDoubleValues = standardizeDoubleValues;
    }
    public ColumnStatistics this[string featureName] => _columnStats[featureName];

    /// <summary>
    /// load the dataset 'xyDataset' and encode all categorical features into numerical values
    /// (so that it can be processed by LightGBM)
    /// this dataset contains both features (the 'x') and target columns (the 'y')
    /// </summary>
    /// <param name="xyDataset_string_df"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    // ReSharper disable once UnusedMember.Global
    public InMemoryDataSetV2 Transform_XYDataset(DataFrame xyDataset_string_df)
    {
        Debug.Assert(xyDataset_string_df.IsStringDataFrame);
        var xyTrainEncoded = Transform(xyDataset_string_df);
        var xTrainEncoded = xyTrainEncoded.Drop(Targets);
        var yTrainEncoded = xyTrainEncoded[Targets];
        return NewInMemoryDataSetV2(xTrainEncoded, yTrainEncoded, _datasetSample);
    }

    /// <summary>
    /// load the dataset 'xDataset' and encode all categorical features into numerical values
    /// (so that it can be processed by LightGBM)
    /// if 'yDataset' is not empty, it means that this second dataset contains the target 'y'
    /// </summary>
    public InMemoryDataSetV2 Transform_X_and_Y_Dataset([NotNull] DataFrame x_string_df, [CanBeNull] DataFrame y_string_df)
    {
        Debug.Assert(x_string_df.IsStringDataFrame);
        Debug.Assert(y_string_df == null || y_string_df.IsStringDataFrame);
        var xTrainEncoded = Transform(x_string_df);
        //we transform the y file if any
        var yTrainEncoded = (y_string_df != null)? Transform(y_string_df):null;
        return NewInMemoryDataSetV2(xTrainEncoded, yTrainEncoded, _datasetSample);
    }

    public static InMemoryDataSetV2 NewInMemoryDataSetV2(DataFrame xTrainEncoded, DataFrame yTrainEncoded, AbstractDatasetSample datasetSample)
    {
        return new InMemoryDataSetV2(
            datasetSample,
            xTrainEncoded,
            yTrainEncoded,
            false);
    }

    public string[] TargetLabelDistinctValues
    {
        get{
            if (_datasetSample.GetObjective() == Objective_enum.Regression)
            {
                return new string[0];
            }
            if (_datasetSample.TargetLabels.Length != 1)
            {
                throw new NotImplementedException($"invalid number of target labels {_datasetSample.TargetLabels.Length}, only 1 is supported");
            }
            var targetLabel = _datasetSample.TargetLabels[0];
            return GetDistinctCategoricalValues(targetLabel).ToArray();
        }
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
    public void Fit(DataFrame string_df)
    {
        if (_transformCallCount != 0)
        {
            throw new ArgumentException($"can't call method {nameof(Fit)} because the method {nameof(Transform)} has already been called");
        }

        ++_fitCallCount;
        Debug.Assert(string_df.IsStringDataFrame);
        //we read the header
        foreach (var c in string_df.Columns)
        {
            if (!_columnStats.ContainsKey(c))
            {
                _columnStats[c] = new ColumnStatistics(IsCategoricalColumn(c), Targets.Contains(c), IdColumns.Contains(c), _standardizeDoubleValues);
            }
        }

        var rows = string_df.Shape[0];
        void FitColumn(int columnIdx)
        {
            var readonlyContent = string_df.StringCpuTensor().ReadonlyContent;
            var columnStats = _columnStats[string_df.Columns[columnIdx]];
            for (var rowIndex = 0; rowIndex < rows; rowIndex++)
            {
                columnStats.Fit(readonlyContent[columnIdx + rowIndex * string_df.Columns.Length]);
            }
        }
        Parallel.For(0, string_df.Columns.Length, FitColumn);
    }

    // ReSharper disable once MemberCanBePrivate.Global
    public DataFrame Transform(DataFrame string_df)
    {
        if (_fitCallCount == 0)
        {
            throw new ArgumentException($"can't call method {nameof(Transform)} because the method {nameof(Fit)} has never been called");
        }
        ++_transformCallCount;
        Debug.Assert(string_df.IsStringDataFrame);
        foreach (var c in string_df.Columns)
        {
            if (!_columnStats.ContainsKey(c))
            {
                throw new ArgumentException($"unknown column to transform : {c}, not among {string.Join(' ', string_df.Columns)}");
            }
        }

        var rows = string_df.Shape[0];
        var columns = string_df.Shape[1];
        var content = new float[rows * columns];
        
        void TransformColumn(int columnIdx)
        {
            var readonlyContent = string_df.StringCpuTensor().ReadonlyContent;
            var columnStats = _columnStats[string_df.Columns[columnIdx]];
            for (var rowIndex = 0; rowIndex < rows; rowIndex++)
            {
                var contentIndex = columnIdx + rowIndex * string_df.Columns.Length;
                content[contentIndex] = (float)columnStats.Transform(readonlyContent[contentIndex]);
            }
        }
        Parallel.For(0, string_df.Columns.Length, TransformColumn);

        return DataFrame.New(content, string_df.Columns);
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
        if (_datasetSample.GetObjective() == Objective_enum.Classification && Targets.Contains(columnName))
        {
            return true;
        }
        return false;
    }
}
