using System;
using System.Collections.Generic;
using System.Text;
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
    private readonly AbstractDatasetSample _datasetSample;

    /// <summary>
    /// list of all categorical features in the dataset
    /// </summary>
    private List<string> CategoricalFeatures => _datasetSample.CategoricalFeatures();
    /// <summary>
    /// name of the target features (to predict)
    /// (usually a single feature)
    /// </summary>
    [NotNull] private List<string> Targets => _datasetSample.TargetLabels();
    /// <summary>
    /// the list of features that will be used to uniquely identify a row
    /// (usually a single feature)
    /// </summary>
    [NotNull] private List<string> IdColumns => _datasetSample.IdColumns();
    [NotNull] private readonly Dictionary<string, FeatureStats> _featureStats = new();
    [NotNull] private static readonly ILog Log = LogManager.GetLogger(typeof(DatasetEncoder));

    public DatasetEncoder(AbstractDatasetSample datasetSample)
    {
        _datasetSample = datasetSample;
    }

    public FeatureStats this[string featureName]
    {
        get
        {
            return _featureStats[featureName];
        }
    }

    /// <summary>
    /// load the dataset 'xyDataset' and encode all categorical features into numerical values
    /// (so that it can be processed by LightGBM)
    /// this dataset contains both features (the 'x') and target columns (the 'y')
    /// </summary>
    /// <param name="xyDataset"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    // ReSharper disable once UnusedMember.Global
    public IDataSet NumericalEncoding(string xyDataset)
    {
        var rows = Utils.ReadCsvWithCache(xyDataset);
        var df = NumericalEncoding(rows);
        var (xTrainEncoded, yTrainEncoded) = df.Split(Targets);

        var x = xTrainEncoded.FloatCpuTensor();
        var y = yTrainEncoded.FloatCpuTensor();
        return new InMemoryDataSet(
            x,
            y,
            _datasetSample.Name,
            _datasetSample.GetObjective(),
            null,
            xTrainEncoded.ColumnNames,
            CategoricalFeatures.ToArray(),
            IdColumns.ToArray(),
            Targets.ToArray(),
            false,
            _datasetSample.GetSeparator());
    }

    
    /// <summary>
    /// load the dataset 'xDataset' and encode all categorical features into numerical values
    /// (so that it can be processed by LightGBM)
    /// if 'yDataset' is not empty, it means that this second dataset contains the target 'y'
    /// </summary>
    public IDataSet NumericalEncoding([NotNull] string xDataset, [CanBeNull] string yDataset)
    {
        var xRows = Utils.ReadCsvWithCache(xDataset);
        var xEncoding = NumericalEncoding(xRows);

        //we load the y file if any
        DataFrameT<float> yEncoding = null;
        if (!string.IsNullOrEmpty(yDataset))
        {
            var yRows = Utils.ReadCsvWithCache(yDataset);
            yEncoding = NumericalEncoding(yRows);
        }

        return new InMemoryDataSet(
            xEncoding.FloatCpuTensor(),
            yEncoding?.Tensor,
            _datasetSample.Name,
            _datasetSample.GetObjective(),
            null,
            xEncoding.ColumnNames,
            CategoricalFeatures.ToArray(),
            IdColumns.ToArray(),
            Targets.ToArray(),
            false,
            _datasetSample.GetSeparator());
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
        if (_datasetSample.TargetLabels().Count != 1)
        {
            throw new NotImplementedException($"invalid number of target labels {_datasetSample.TargetLabels().Count}, only 1 is supported");
        }
        var targetLabel = _datasetSample.TargetLabels()[0];
        var allValues = GetDistinctCategoricalValues(targetLabel);
        if (allValues == null || allValues.Count == 0)
        {
            return 1;
        }
        return allValues.Count;
    }

    /// <summary>
    /// return the distinct values of a 'categorical feature' or of a 'categorical target label '
    /// </summary>
    /// <param name="categoricalColumn"></param>
    /// <returns></returns>
    public IList<string> GetDistinctCategoricalValues(string categoricalColumn)
    {
        if (!IsCategoricalColumn(categoricalColumn))
        {
            throw new Exception($"Invalid column {categoricalColumn}: not a categorical column");
        }
        if (!_featureStats.ContainsKey(categoricalColumn))
        {
            return null;
        }
        return _featureStats[categoricalColumn].GetDistinctCategoricalValues();
    }

    public IList<int> GetDistinctCategoricalCount(string categoricalColumn)
    {
        if (!IsCategoricalColumn(categoricalColumn))
        {
            throw new Exception($"Invalid column {categoricalColumn}: not a categorical column");
        }
        if (!_featureStats.ContainsKey(categoricalColumn))
        {
            return null;
        }
        return _featureStats[categoricalColumn].GetDistinctCategoricalCount();
    }

    private bool IsCategoricalColumn(string columnName)
    {
        if (CategoricalFeatures.Contains(columnName)||IdColumns.Contains(columnName))
        {
            return true;
        }

        if (_datasetSample.IsClassificationProblem && Targets.Contains(columnName))
        {
            return true;
        }

        return false;
    }
    public DataFrameT<float> NumericalEncoding(IReadOnlyList<string[]> rows)
    {
        //the 1st row contains the header
        if (rows.Count < 2)
        {
            var errorMsg = $"fail to encode dataset '{_datasetSample.Name}', too few rows {rows.Count}";
            Log.Error(errorMsg);
            throw new Exception(errorMsg);
        }
        //we read the header
        var headerRow = rows[0];
        foreach (var featureName in headerRow)
        {
            if (!_featureStats.ContainsKey(featureName))
            {
                _featureStats[featureName] = new FeatureStats(IsCategoricalColumn(featureName), Targets.Contains(featureName), IdColumns.Contains(featureName));
            }
        }

        var content = new float[(rows.Count - 1) * headerRow.Length];
        int contentIndex = 0;

        //we skip the first row (== header row)
        for (var rowIndex = 1; rowIndex < rows.Count; rowIndex++)
        {
            var row = rows[rowIndex];
            if (row.Length != headerRow.Length)
            {
                Log.Warn($"invalid number of elements for row at index#{rowIndex}: found {row.Length} elements, expecting {headerRow.Length} (ignoring)");
            }
            for (int i = 0; i < headerRow.Length; ++i)
            {
                content[contentIndex++] = (float)_featureStats[headerRow[i]].NumericalEncoding(i<row.Length ? row[i] : "");
            }
        }
        var cpuTensor = new CpuTensor<float>(new [] { rows.Count - 1, headerRow.Length }, content);
        var df = DataFrame.New(cpuTensor, headerRow);
        return (DataFrameT<float>)df;
    }


    /// <summary>
    /// Decode the dataframe 'df' (replacing numerical values by their categorical values) and return the content of the decoded DataFrame
    /// </summary>
    /// <param name="df"></param>
    /// <param name="separator"></param>
    /// <param name="missingNumberValue"></param>
    /// <returns></returns>
    public string NumericalDecoding(DataFrame df, char separator, string missingNumberValue = "")
    {
        var featureNames = df.ColumnNames;

        //we ensure that categorical feature names are valid
        foreach (var featureName in featureNames)
        {
            if (!_featureStats.ContainsKey(featureName))
            {
                var errorMsg = $"Invalid feature name '{featureName}' : must be in {string.Join(' ', _featureStats.Keys)}";
                Log.Error(errorMsg);
                throw new Exception(errorMsg);
            }
        }

        var sb = new StringBuilder();
        sb.Append(string.Join(separator, featureNames) + Environment.NewLine);
        int nbRows = df.Shape[0];
        int nbColumns = df.Shape[1];
        var memoryContent = df.FloatCpuTensor().Content;
        for (int rowIndex = 0; rowIndex < nbRows; rowIndex++)
        {
            var rowContent = memoryContent.Slice(rowIndex * nbColumns, nbColumns).Span;
            for (int colIndex = 0; colIndex < nbColumns; ++colIndex)
            {
                var e = rowContent[colIndex];
                var featureStat = _featureStats[featureNames[colIndex]];
                var eAsString = featureStat.NumericalDecoding(e, missingNumberValue);
                if (colIndex != 0)
                {
                    sb.Append(separator);
                }
                sb.Append(eAsString);
            }
            if (rowIndex != nbRows - 1)
            {
                sb.Append(Environment.NewLine);
            }
        }
        return sb.ToString();
    }
}
