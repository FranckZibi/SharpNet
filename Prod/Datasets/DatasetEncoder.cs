using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using JetBrains.Annotations;
using log4net;
using SharpNet.CPU;
using SharpNet.HyperParameters;

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
    #endregion

    public DatasetEncoder(AbstractDatasetSample datasetSample)
    {
        _datasetSample = datasetSample;
    }
    public ColumnStatistics this[string featureName] => _columnStats[featureName];

    /// <summary>
    /// load the dataset 'xyDataset' and encode all categorical features into numerical values
    /// (so that it can be processed by LightGBM)
    /// this dataset contains both features (the 'x') and target columns (the 'y')
    /// </summary>
    /// <param name="xyDataset"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    // ReSharper disable once UnusedMember.Global
    public InMemoryDataSet NumericalEncoding(string xyDataset)
    {
        var xyTrainEncoded = NumericalEncoding(DataFrame.read_string_csv(xyDataset));
        var xTrainEncoded = xyTrainEncoded.Drop(Targets);
        var yTrainEncoded = xyTrainEncoded[Targets];
        return NewInMemoryDataSet(xTrainEncoded, yTrainEncoded?.FloatCpuTensor(), _datasetSample);
    }
    /// <summary>
    /// load the dataset 'xDataset' and encode all categorical features into numerical values
    /// (so that it can be processed by LightGBM)
    /// if 'yDataset' is not empty, it means that this second dataset contains the target 'y'
    /// </summary>
    public InMemoryDataSet NumericalEncoding([NotNull] string xDataset, [CanBeNull] string yDataset)
    {
        var xTrainEncoded = NumericalEncoding(DataFrame.read_string_csv(xDataset));

        //we load the y file if any
        DataFrame yTrainEncoded = null;
        if (!string.IsNullOrEmpty(yDataset))
        {
            yTrainEncoded = NumericalEncoding(DataFrame.read_string_csv(yDataset));
        }
        return NewInMemoryDataSet(xTrainEncoded, yTrainEncoded?.FloatCpuTensor(), _datasetSample);
    }

    public static InMemoryDataSet NewInMemoryDataSet(DataFrame xTrainEncoded, CpuTensor<float> yTrainEncodedTensor, AbstractDatasetSample datasetSample)
    {
        return new InMemoryDataSet(
            xTrainEncoded.FloatCpuTensor(),
            yTrainEncodedTensor,
            datasetSample.Name,
            datasetSample.GetObjective(),
            null,
            xTrainEncoded.Columns,
            Utils.Intersect(datasetSample.CategoricalFeatures, xTrainEncoded.Columns).ToArray(), //we only take Categorical Features that actually appear in the training DataSet
            Utils.Intersect(datasetSample.IdColumns, xTrainEncoded.Columns).ToArray(), //we only take Id Columns Features that actually appear in the training DataSet
            datasetSample.TargetLabels,
            false,
            datasetSample.GetSeparator());
    }

    /// <summary>
    /// for classification problems, the number of distinct class to identify
    /// 1 if it is a regression problem
    /// </summary>
    /// <returns></returns>
    public int NumClasses()
    {
        if (_datasetSample.IsRegressionProblem)
        {
            return 1;
        }
        if (_datasetSample.TargetLabels.Length != 1)
        {
            throw new NotImplementedException($"invalid number of target labels {_datasetSample.TargetLabels.Length}, only 1 is supported");
        }
        var targetLabel = _datasetSample.TargetLabels[0];
        var allValues = GetDistinctCategoricalValues(targetLabel);
        if (allValues == null || allValues.Count == 0)
        {
            return 1;
        }
        return allValues.Count;
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
    public DataFrame NumericalEncoding(DataFrame df)
    {
        //we read the header
        foreach (var c in df.Columns)
        {
            if (!_columnStats.ContainsKey(c))
            {
                _columnStats[c] = new ColumnStatistics(IsCategoricalColumn(c), Targets.Contains(c), IdColumns.Contains(c));
            }
        }

        var rows = df.Shape[0];
        var columns = df.Shape[1];
        var content = new float[rows*columns];
        int contentIndex = 0;
        var readonlyContent = df.StringCpuTensor().ReadonlyContent;

        for (var rowIndex = 0; rowIndex < rows; rowIndex++)
        {
            foreach (var c in df.Columns)
            {
                content[contentIndex] = (float)_columnStats[c].NumericalEncoding(readonlyContent[contentIndex]);
                ++contentIndex;
            }
        }
        return DataFrame.New(content, df.Columns);
    }


    /// <summary>
    /// Decode the dataframe 'df' (replacing numerical values by their categorical values) and return the content of the decoded DataFrame
    /// </summary>
    /// <param name="df"></param>
    /// <param name="missingNumberValue"></param>
    /// <returns></returns>
    public DataFrame NumericalDecoding(DataFrame df, string missingNumberValue = "")
    {
        //we display a warning for column names without known encoding
        var unknownFeatureName = Utils.Without(df.Columns, _columnStats.Keys);
        if (unknownFeatureName.Count != 0)
        {
            Log.Warn($"{unknownFeatureName.Count} unknown feature name(s): '{string.Join(' ', unknownFeatureName)}' (not in '{string.Join(' ', _columnStats.Keys)}')");
        }

        var encodedContent = df.FloatCpuTensor().ReadonlyContent;
        var decodedContent = new string[encodedContent.Length];
        for (int idx = 0; idx< encodedContent.Length;++idx)
        {
            var encodedValue = encodedContent[idx];
            int col = idx % df.Columns.Length;
            var decodedValueAsString = _columnStats.TryGetValue(df.Columns[col], out var featureStat)
                ? featureStat.NumericalDecoding(encodedValue, missingNumberValue)
                : encodedValue.ToString(CultureInfo.InvariantCulture);
            decodedContent[idx] = decodedValueAsString;
        }
        return DataFrame.New(decodedContent, df.Columns);
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
        if (_datasetSample.IsClassificationProblem && Targets.Contains(columnName))
        {
            return true;
        }
        return false;
    }
}
