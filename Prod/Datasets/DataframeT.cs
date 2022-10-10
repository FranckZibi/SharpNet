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
    public DataFrameT(CpuTensor<T> tensor, IList<string> columnsNames, Func<T, string> toString) : base(columnsNames, Enumerable.Repeat(typeof(T), tensor.Shape[1]).ToList())
    {
        _toString = toString;
        Debug.Assert(tensor.Shape.Length == 2); //DataFrame works only with matrices
        if (tensor.Shape[1] != columnsNames.Count)
        {
            var errorMsg = $"Invalid tensor columns count {tensor.Shape[1]}: should be {columnsNames.Count} / {string.Join(' ', columnsNames)} ";
            throw new Exception(errorMsg);
        }

        Tensor = tensor;
    }

    public override int[] Shape => Tensor.Shape;

    public static DataFrameT<T> Load(string path, bool hasHeader, Func<string, T> parser, Func<T, string> toString)
    {
        var content = new List<List<T>>();
        var columnsNames = new List<string>();

        foreach (var lineContent in Utils.ReadCsv(path))
        {
            if (hasHeader && columnsNames.Count == 0)
            {
                columnsNames = lineContent.ToList();
                continue;
            }
            content.Add(lineContent.Select(parser).ToList());
        }

        if (!hasHeader)
        {
            columnsNames = Enumerable.Range(0, content[0].Count).Select(t => t.ToString()).ToList();
        }

        var data = new T[content.Count * columnsNames.Count];
        int idx = 0;
        foreach (var t in content)
        foreach (var d in t)
        {
            data[idx++] = d;
        }
        var tensor = new CpuTensor<T>(new[] { content.Count, columnsNames.Count }, data);
        return new DataFrameT<T>(tensor, columnsNames, toString);
    }
    #endregion



    public override (DataFrame first, DataFrame second) Split(IList<string> columnsForSecondDataFrame)
    {
        var first = Drop(columnsForSecondDataFrame);
        var second = Keep(columnsForSecondDataFrame);
        return (first, second);
    }

    public override DataFrameT<T> Drop(IList<string> columnsToDrop)
    {
        var newData = Tensor.DropColumns(ColumnNamesToIndexes(columnsToDrop));
        var newColumnNames = ColumnNames.ToList();
        newColumnNames.RemoveAll(columnsToDrop.Contains);
        return new DataFrameT<T>(newData, newColumnNames.ToArray(), _toString);
    }

    public override DataFrameT<T> Keep(IList<string> columnsToKeep)
    {
        var newData = Tensor.KeepColumns(ColumnNamesToIndexes(columnsToKeep));
        var newColumnNames = ColumnNames.ToList();
        newColumnNames.RemoveAll(f => !columnsToKeep.Contains(f));
        return new DataFrameT<T>(newData, newColumnNames, _toString);
    }
    public override void to_csv(string path, string sep = ",", bool addHeader = false, int? index = null)
    {
        var sb = new StringBuilder();
        if (addHeader)
        {
            sb.Append(string.Join(sep, ColumnNames));
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
