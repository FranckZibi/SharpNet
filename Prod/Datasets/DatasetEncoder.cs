using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using JetBrains.Annotations;
using log4net;
using SharpNet.CPU;

namespace SharpNet.Datasets;

/// <summary>
/// encode a dataset (that may contains categorical features) into a full numerical dataset.
/// each categorical feature will be encoded using the corresponding index of this categorical feature (starting from 0)
/// a negative value for a categorical feature means that the categorical feature is empty
/// </summary>
public class DatasetEncoder
{
    /// <summary>
    /// list of all categorical features in the dataset
    /// </summary>
    private readonly List<string> _categoricalFeatures;
    /// <summary>
    /// name of the target features (to predict)
    /// (usually a single feature)
    /// </summary>
    [NotNull] private readonly List<string> _targetFeatures;
    /// <summary>
    /// the list of features that will be used to uniquely identify a row
    /// (usually a single feature)
    /// </summary>
    [NotNull] private readonly List<string> _idFeatures;
    [NotNull] private readonly Dictionary<string, FeatureStats> _featureStats = new();
    [NotNull] private static readonly ILog Log = LogManager.GetLogger(typeof(DatasetEncoder));


    public DatasetEncoder(List<string> categoricalFeatures, [NotNull] List<string> idFeatures, [NotNull] List<string> targetFeatures)
    {
        _categoricalFeatures = categoricalFeatures;
        _idFeatures = idFeatures;
        _targetFeatures = targetFeatures;
    }

    public FeatureStats this[string featureName]
    {
        get
        {
            return _featureStats[featureName];
        }
    }
    
    /// <summary>
    /// true if the problem is a regression problem
    /// false if it is a categorization problem
    /// </summary>
    public bool IsRegressionProblem
    {
        get
        {
            foreach (var targetFeature in _targetFeatures)
            {
                if (_categoricalFeatures.Contains(targetFeature))
                {
                    return false;
                }
            }
            return true;
        }
    }

    /// <summary>
    /// load the dataset 'csvDataset' adn encode all categorical features into numerical values
    /// (so that it can be processed by LightGBM)
    /// </summary>
    /// <param name="csvDataset"></param>
    /// <returns></returns>
    /// <exception cref="Exception"></exception>
    // ReSharper disable once UnusedMember.Global
    public Dataframe NumericalEncoding(string csvDataset)
    {
        var rows = Utils.ReadCsv(csvDataset).ToList();
        return NumericalEncoding(rows, csvDataset);
    }

    public Dataframe NumericalEncoding(List<string[]> rows, string datasetName)
    {
        if (rows.Count < 2)
        {
            var errorMsg = $"fail to encode dataset '{datasetName}', too few rows {rows.Count}";
            Log.Error(errorMsg);
            throw new Exception(errorMsg);
        }
        //we read the header
        var headerRow = rows[0];
        foreach (var featureName in headerRow)
        {
            if (!_featureStats.ContainsKey(featureName))
            {
                _featureStats[featureName] = new FeatureStats(_categoricalFeatures.Contains(featureName), _targetFeatures.Contains(featureName), _idFeatures.Contains(featureName));
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
        var df = new Dataframe(cpuTensor, headerRow, datasetName);
        return df;
    }

    /// <summary>
    /// Decode the dataframe 'df' (replacing numerical values by their categorical values) and return the content of the decoded Dataframe
    /// </summary>
    /// <param name="df"></param>
    /// <param name="separator"></param>
    /// <param name="missingNumberValue"></param>
    /// <returns></returns>
    public string NumericalDecoding(Dataframe df, char separator, string missingNumberValue = "")
    {
        var featureNames = df.FeatureNames;

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
        int nbRows = df.Tensor.Shape[0];
        int nbColumns = df.Tensor.Shape[1];
        for (int rowIndex = 0; rowIndex < nbRows; rowIndex++)
        {
            var rowContent = df.Tensor.Content.Slice(rowIndex * nbColumns, nbColumns).Span;
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
