using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using SharpNet.CPU;

namespace SharpNet.Datasets;

/// <summary>
/// a DataFrame with all columns with elements of the same type (T)
/// </summary>
/// <typeparam name="T"></typeparam>
public class DataFrameT<T> : DataFrame
{
    private readonly Func<T, string> _toString;

    #region public fields
    public CpuTensor<T> Tensor { get; }
    #endregion


    #region Constructors
    public DataFrameT(CpuTensor<T> tensor, IList<string> featureNames, IList<string> categoricalFeatures, Func<T, string> toString) : base(featureNames, categoricalFeatures, Enumerable.Repeat(typeof(T), tensor.Shape[1]).ToList())
    {
        _toString = toString;
        Debug.Assert(tensor.Shape.Length == 2); //DataFrame works only with matrices
        if (tensor.Shape[1] != featureNames.Count)
        {
            var errorMsg = $"Invalid tensor columns count {tensor.Shape[1]}: should be {featureNames.Count} / {string.Join(' ', featureNames)} ";
            throw new Exception(errorMsg);
        }

        Tensor = tensor;
    }

    public override int[] Shape => Tensor.Shape;

    public static DataFrameT<T> Load(string path, bool hasHeader, Func<string, T> parser, IList<string> categoricalFeatures, Func<T, string> toString)
    {
        var content = new List<List<T>>();
        var featureNames = new List<string>();

        foreach (var lineContent in Utils.ReadCsv(path))
        {
            if (hasHeader && featureNames.Count == 0)
            {
                featureNames = lineContent.ToList();
                continue;
            }
            content.Add(lineContent.Select(parser).ToList());
        }

        if (!hasHeader)
        {
            featureNames = Enumerable.Range(0, content[0].Count).Select(t => t.ToString()).ToList();
        }

        var data = new T[content.Count * featureNames.Count];
        int idx = 0;
        foreach (var t in content)
        foreach (var d in t)
        {
            data[idx++] = d;
        }
        var tensor = new CpuTensor<T>(new[] { content.Count, featureNames.Count }, data);

        if (categoricalFeatures == null)
        {
            //TODO : enhance rule for detecting categorical features
            if (typeof(T) == typeof(float) || typeof(T) == typeof(double))
            {
                categoricalFeatures = Array.Empty<string>();
            }
            else
            {
                categoricalFeatures = featureNames.ToArray();
            }
        }

        return new DataFrameT<T>(tensor, featureNames, categoricalFeatures, toString);
    }
    #endregion



    public override (DataFrame first, DataFrame second) Split(IList<string> featuresForSecondDataFrame)
    {
        var first = Drop(featuresForSecondDataFrame);
        var second = Keep(featuresForSecondDataFrame);
        return (first, second);
    }

    public override DataFrameT<T> Drop(IList<string> featuresToDrop)
    {
        var newData = Tensor.DropColumns(FeatureNamesToIndexes(featuresToDrop));
        var newFeatures = FeatureNames.ToList();
        newFeatures.RemoveAll(featuresToDrop.Contains);
        var newCategoricalFeatures = CategoricalFeatures.ToList();
        newCategoricalFeatures.RemoveAll(featuresToDrop.Contains);
        return new DataFrameT<T>(newData, newFeatures.ToArray(), newCategoricalFeatures, _toString);
    }

    public override DataFrameT<T> Keep(IList<string> featuresToKeep)
    {
        var newData = Tensor.KeepColumns(FeatureNamesToIndexes(featuresToKeep));
        var newFeatures = FeatureNames.ToList();
        newFeatures.RemoveAll(f => !featuresToKeep.Contains(f));
        var newCategoricalFeatures = CategoricalFeatures.ToList();
        newCategoricalFeatures.RemoveAll(f => !featuresToKeep.Contains(f));

        return new DataFrameT<T>(newData, newFeatures, newCategoricalFeatures, _toString);
    }
    public override void to_csv(string path, string sep = ",", bool addHeader = false, int? index = null)
    {
        var sb = new StringBuilder();
        if (addHeader)
        {
            sb.Append(string.Join(sep, FeatureNames));
        }
        var dataAsSpan = Tensor.SpanContent;
        int currentIndex = index ?? -1;
        for (int i = 0; i < dataAsSpan.Length; ++i)
        {
            if (i % Tensor.Shape[1] == 0)
            {
                if (i != 0 || addHeader)
                {
                    sb.Append(Environment.NewLine);
                }
                if (index.HasValue)
                {
                    sb.Append(currentIndex+sep);
                    ++currentIndex;
                }
            }
            else
            {
                sb.Append(sep);
            }
            sb.Append(_toString(dataAsSpan[i]));
        }
        System.IO.File.WriteAllText(path, sb.ToString());
    }

}