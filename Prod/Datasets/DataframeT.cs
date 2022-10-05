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
    public DataFrameT(CpuTensor<T> tensor, IEnumerable<string> featureNames, IEnumerable<string> categoricalFeatures, Func<T, string> toString) : base(featureNames, categoricalFeatures, Enumerable.Repeat(typeof(T), tensor.Shape[1]))
    {
        _toString = toString;
        Debug.Assert(tensor.Shape.Length == 2); //DataFrame works only with matrices
        Tensor = tensor;
    }

    public override int[] Shape => Tensor.Shape;

    public static DataFrameT<T> Load(string path, bool hasHeader, Func<string, T> parser, IEnumerable<string> categoricalFeatures, Func<T, string> toString)
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

    public DataFrameT<T> Drop(IList<string> featuresToDrop)
    {
        var indexes = new HashSet<int>(FeatureNameToIndexes(featuresToDrop));
            
        var newData = Tensor.DropColumns(indexes);

        var newFeatures = FeatureNames.ToList();
        newFeatures.RemoveAll(featuresToDrop.Contains);
        var newCategoricalFeatures = CategoricalFeatures.ToList();
        newCategoricalFeatures.RemoveAll(featuresToDrop.Contains);


        return new DataFrameT<T>(newData, newFeatures.ToArray(), newCategoricalFeatures, _toString);
    }
    public DataFrameT<T> Keep(IList<string> featuresToKeep)
    {
        var newData = Tensor.KeepColumns(FeatureNameToIndexes(featuresToKeep));

        var newFeatures = FeatureNames.ToList();
        newFeatures.RemoveAll(f => !featuresToKeep.Contains(f));
        var newCategoricalFeatures = CategoricalFeatures.ToList();
        newCategoricalFeatures.RemoveAll(f => !featuresToKeep.Contains(f));

        return new DataFrameT<T>(newData, newFeatures, newCategoricalFeatures, _toString);
    }
    public override void Save(string path, int? index = null)
    {
        const char separator = ',';
        var sb = new StringBuilder();
        sb.Append(string.Join(separator, FeatureNames));
        var dataAsSpan = Tensor.SpanContent;
        int currentIndex = index ?? -1;
        for (int i = 0; i < dataAsSpan.Length; ++i)
        {
            if (i % Tensor.Shape[1] == 0)
            {
                sb.Append(Environment.NewLine);
                if (index.HasValue)
                {
                    sb.Append(currentIndex.ToString()+separator);
                    ++currentIndex;
                }
            }
            else
            {
                sb.Append(separator);
            }
            sb.Append(_toString(dataAsSpan[i]));
        }
        System.IO.File.WriteAllText(path, sb.ToString());
    }

}