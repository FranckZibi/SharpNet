﻿using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.HyperParameters;
using SharpNet.TextPreprocessing;

namespace SharpNet.Datasets;

public sealed class DataFrame
{
    #region private fields
    /// <summary>
    /// each Tuple element:
    ///     Item1: columnName
    ///     Item2: tensor index
    ///     Item3; column index in tensor
    /// 
    /// </summary>
    private readonly List<Tuple<string, int, int>> _columns;
    private readonly CpuTensor<float> _floatTensor;
    private readonly CpuTensor<string> _stringTensor;
    private readonly CpuTensor<int> _intTensor;
    private static readonly Type[] TensorIndexToType;
    private static readonly object[] TensorIndexToNonNullDefaultValue;
    private static readonly Dictionary<Type, int> TypeToTensorIndex = new();
    public const int FLOAT_TYPE_IDX = 0;
    public const int STRING_TYPE_IDX = 1;
    private const int INT_TYPE_IDX = 2;
    //protected static readonly ILog Log = LogManager.GetLogger(typeof(DataFrame));

    #endregion

    #region public fields and properties
    public string[] Columns { get; }
    public int[] Shape => new[] { Rows, _columns.Count };

    /// <summary>
    /// total number of cells in the DataFrame (including null/empty/nan cells
    /// </summary>
    public int CellCount => Utils.Product(Shape);

    public List<Tuple<string, int, int>> ColumnsDesc => _columns;


    #endregion


    public override string ToString()
    {
        return Tensor.ShapeToString(Shape);
    }

    public DataFrame ResizeWithNewNumberOfRows(int newRows)
    {
        return new DataFrame(_columns,
            ResizeWithNewNumberOfRows(_floatTensor, newRows),
            ResizeWithNewNumberOfRows(_stringTensor, newRows),
            ResizeWithNewNumberOfRows(_intTensor, newRows)
        );
    }


    public static DataFrame Average(params DataFrame[] dfs)
    {
        //TODO: ensure that each DataFrame has same chape and same format for columns
        var cpuTensor = new CpuTensor<float>(dfs[0].FloatTensor.Shape);
        var content = cpuTensor.SpanContent;
        foreach (var df in dfs)
        {
            var dfContent = df.FloatTensor.ReadonlyContent;
            for (int i = 0; i < content.Length; i++)
            {
                content[i] += (dfContent[i]) / dfs.Length;
            }
        }

        return new DataFrame(
            dfs[0]._columns,
            cpuTensor,
            dfs[0].StringTensor,
            dfs[0].IntTensor);
    }

    private  static CpuTensor<T> ResizeWithNewNumberOfRows<T>(CpuTensor<T> a, int newRows)
    {
        if (a == null)
        {
            return null;
        }
        Debug.Assert(a.Shape.Length == 2);
        if (newRows < a.Shape[0])
        {
            throw new ArgumentException($"invalid row {newRows} value, must be greater than {a.Shape[0]}");
        }
        var newTensor = new CpuTensor<T>(new [] { newRows , a.Shape[1]});
        a.CopyTo(newTensor.RowSlice(0, a.Shape[0]));
        return newTensor;
    }

    #region constructors
    static DataFrame()
    {
        TypeToTensorIndex[typeof(float)] = FLOAT_TYPE_IDX;
        TypeToTensorIndex[typeof(string)] = STRING_TYPE_IDX;
        TypeToTensorIndex[typeof(int)] = INT_TYPE_IDX;
        TensorIndexToType = TypeToTensorIndex.OrderBy(t => t.Value).Select(t => t.Key).ToArray();
        TensorIndexToNonNullDefaultValue = new object[TensorIndexToType.Length];
        TensorIndexToNonNullDefaultValue[FLOAT_TYPE_IDX] = 0.0f;
        TensorIndexToNonNullDefaultValue[STRING_TYPE_IDX] = "";
        TensorIndexToNonNullDefaultValue[INT_TYPE_IDX] = 0;
    }
    public DataFrame(List<Tuple<string, int, int>> columns,
        CpuTensor<float> floatTensor,
        CpuTensor<string> stringTensor,
        CpuTensor<int> intTensor)
    {
        Columns = columns.Select(t => t.Item1).ToArray();
        _columns = columns;
        _floatTensor = floatTensor;
        _stringTensor = stringTensor;
        _intTensor = intTensor;
    }
    public static DataFrame New(CpuTensor<float> floatTensor, CpuTensor<string> stringTensor, CpuTensor<int> intTensor, IList<string> columnsNames)
    {
        List<Tuple<string, int, int>> columns = new();
        if (floatTensor != null)
        {
            for (var i = 0; i < floatTensor.Shape[1]; i++)
            {
                columns.Add(Tuple.Create(columnsNames[columns.Count], FLOAT_TYPE_IDX, i));
            }
        }
        if (stringTensor != null)
        {
            for (var i = 0; i < stringTensor.Shape[1]; i++)
            {
                columns.Add(Tuple.Create(columnsNames[columns.Count], STRING_TYPE_IDX, i));
            }
        }
        if (intTensor != null)
        {
            for (var i = 0; i < intTensor.Shape[1]; i++)
            {
                columns.Add(Tuple.Create(columnsNames[columns.Count], INT_TYPE_IDX, i));
            }
        }
        Debug.Assert(columns.Count == columnsNames.Count);
        return new DataFrame(columns, floatTensor, stringTensor, intTensor);
    }
    public static DataFrame New(float[] content, IList<string> columnsNames)
    {
        var tensor = CpuTensor<float>.New(content, columnsNames.Count);
        return New(tensor, columnsNames);
    }
    public static DataFrame New(CpuTensor<float> tensor, IList<string> columnsNames)
    {
        return New(tensor, null, null, columnsNames);
    }
    public static DataFrame New(CpuTensor<float> tensor)
    {
        var columnsNames = Enumerable.Range(0, tensor.Shape[1]).Select(i => i.ToString()).ToArray();
        return New(tensor, columnsNames);
    }

    public static DataFrame New(string[] content, IList<string> columnsNames)
    {
        var tensor = CpuTensor<string>.New(content, columnsNames.Count);
        return new DataFrame(columnsNames.Select((n, i) => Tuple.Create(n, STRING_TYPE_IDX, i)).ToList(), null, tensor, null);
    }
    // ReSharper disable once UnusedMember.Global
    public static DataFrame New(int[] content, IList<string> columnsNames)
    {
        var tensor = CpuTensor<int>.New(content, columnsNames.Count);
        return new DataFrame(columnsNames.Select((n, i) => Tuple.Create(n, INT_TYPE_IDX, i)).ToList(), null, null, tensor);
    }
    #endregion

    public CpuTensor<float> FloatCpuTensor()
    {
        if (!IsFloatDataFrame)
        {
            throw new ArgumentException($"{this} is not a Float DataFrame");
        }
        return _floatTensor;
    }
    public CpuTensor<float> FloatTensor =>_floatTensor;
    public CpuTensor<string> StringTensor =>_stringTensor;
    public CpuTensor<int> IntTensor =>_intTensor;
    public CpuTensor<string> StringCpuTensor()
    {
        if (!IsStringDataFrame)
        {
            throw new ArgumentException($"{this} is not a String DataFrame");
        }
        return _stringTensor;
    }
    // ReSharper disable once UnusedMember.Global
    public CpuTensor<int> IntCpuTensor()
    {
        if (!IsIntDataFrame)
        {
            throw new ArgumentException($"{this} is not a Int DataFrame");
        }
        return _intTensor;
    }
    public string[] StringColumnContent(string columnName)
    {
        var colDesc= GetColumnDesc(columnName);
        if (colDesc.Item2 != STRING_TYPE_IDX)
        {
            throw new ArgumentException($"column {columnName} is not a string column");
        }
        var res = new string[Shape[0]];
        var content = _stringTensor.ReadonlyContent;
        for (int row = 0; row < res.Length; row++)
        {
            res[row] = content[colDesc.Item3 + row * _stringTensor.Shape[1]];
        }
        return res;
    }
    public void UpdateColumnInPlace(string columnName, Func<string, string> update) => UpdateColumnInPlace(columnName, StringTensor, update);
    public void UpdateColumnInPlace(string columnName, Func<float, float> update) => UpdateColumnInPlace(columnName, FloatTensor, update);
    public void UpdateColumnInPlace(string columnName, Func<int,int> update) => UpdateColumnInPlace(columnName, IntTensor, update);
    private void UpdateColumnInPlace<T>(string columnName, CpuTensor<T> tensor, Func<T, T> update)
    {
        var colDesc = GetColumnDesc(columnName);
        if (typeof(T) != TensorIndexToType[colDesc.Item2])
        {
            throw new ArgumentException($"column {columnName} is not a {typeof(T)} column but a {TensorIndexToType[colDesc.Item2]} column");
        }
        var content = tensor.SpanContent;
        for (int idx = colDesc.Item3; idx < content.Length; idx += tensor.Shape[1])
        {
            content[idx] = update(content[idx]);
        }
    }

    /// <summary>
    /// rename column 'originalColumnName' to 'newColumnName'
    /// </summary>
    /// <param name="originalColumnName"></param>
    /// <param name="newColumnName"></param>
    /// <exception cref="ArgumentException"></exception>
    public DataFrame RenameInPlace(string originalColumnName, string newColumnName)
    {
        if (!Columns.Contains(originalColumnName) || Columns.Contains(newColumnName))
        {
            throw new ArgumentException($"can't rename column {originalColumnName} to {newColumnName} because of existing columns {string.Join(' ', Columns)}");
        }
        var idxColumn = Array.IndexOf(Columns, originalColumnName);
        Columns[idxColumn] = newColumnName;
        for (var i = 0; i < _columns.Count; i++)
        {
            if (_columns[i].Item1 == originalColumnName)
            {
                _columns[i] = Tuple.Create(newColumnName, _columns[i].Item2, _columns[i].Item3);
            }
        }
        return this;
    }
    public static DataFrame read_float_csv(string path, bool hasHeader = true, bool isNormalized = false)
    {
        return read_csv(path, hasHeader, _ => typeof(float), isNormalized);
    }
    public static DataFrame read_string_csv(string path, bool hasHeader = true, bool isNormalized = false)
    {
        return read_csv(path, hasHeader, _ => typeof(string), isNormalized);
    }
    // ReSharper disable once UnusedMember.Global
    public static DataFrame read_int_csv(string path, bool hasHeader = true)
    {
        return read_csv(path, hasHeader, _ => typeof(int));
    }
    /// <summary>
    /// drop the columns 'columnsToDrop', and return a DataFrame without those columns
    /// throws an exception if a column name is invalid
    /// </summary>
    /// <param name="columnsToDrop"></param>
    /// <returns>a DataFrame without the specified columns</returns>
    public DataFrame Drop(params string[] columnsToDrop)
    {
        if (columnsToDrop == null || columnsToDrop.Length == 0)
        {
            return this;
        }
        AssertValidColumns(columnsToDrop);
        var columnsToKeep = Utils.Without(Columns, columnsToDrop).ToArray();
        return this[columnsToKeep];
    }
    /// <summary>
    /// drop the columns 'columnsToDrop', and return a DataFrame without those columns
    /// ignore invalid columns from the input list
    /// </summary>
    /// <param name="columnsToDrop">the column names to drop if they exist in the DataFrame (invalid columns will be ignored)</param>
    /// <returns>a DataFrame without the specified columns</returns>
    // ReSharper disable once UnusedMember.Global
    public DataFrame DropIgnoreErrors(params string[] columnsToDrop)
    {
        return Drop(Utils.Intersect(Columns, columnsToDrop).ToArray());
    }
    public DataFrame this[params string[] columnsToKeep]
    {
        get
        {
            List<Tuple<string, int, int>> newColumns = new();
            Dictionary<int, List<int>> tensorIndexToColumnsToKeep = new();
            foreach (var c in _columns)
            {
                if (columnsToKeep.Contains(c.Item1))
                {
                    var tensorIndex = c.Item2;
                    var oldIndex = c.Item3;
                    if (!tensorIndexToColumnsToKeep.ContainsKey(tensorIndex))
                    {
                        tensorIndexToColumnsToKeep[tensorIndex] = new List<int>();
                    }
                    var newIdx = tensorIndexToColumnsToKeep[tensorIndex].Count;
                    newColumns.Add(Tuple.Create(c.Item1, tensorIndex, newIdx));
                    tensorIndexToColumnsToKeep[c.Item2].Add(oldIndex);
                }
            }
            return new DataFrame(newColumns,
                tensorIndexToColumnsToKeep.ContainsKey(FLOAT_TYPE_IDX)? _floatTensor.KeepColumns(tensorIndexToColumnsToKeep[FLOAT_TYPE_IDX]) : null,
                tensorIndexToColumnsToKeep.ContainsKey(STRING_TYPE_IDX)? _stringTensor.KeepColumns(tensorIndexToColumnsToKeep[STRING_TYPE_IDX]) : null,
                tensorIndexToColumnsToKeep.ContainsKey(INT_TYPE_IDX)? _intTensor.KeepColumns(tensorIndexToColumnsToKeep[INT_TYPE_IDX]) : null);
        }


    }


    // ReSharper disable once UnusedMember.Global
    public static DataFrame MergeVertically(DataFrame top, DataFrame bottom)
    {
        if (top == null)
        {
            return bottom;
        }
        if (bottom == null)
        {
            return top;
        }
        Debug.Assert(top.Columns.SequenceEqual(bottom.Columns));
        return new DataFrame(
            top._columns.ToList(),
            CpuTensor<float>.MergeVertically(top._floatTensor, bottom._floatTensor),
            CpuTensor<string>.MergeVertically(top._stringTensor, bottom._stringTensor),
            CpuTensor<int>.MergeVertically(top._intTensor, bottom._intTensor)
        );
    }
    public static DataFrame MergeHorizontally(DataFrame left, DataFrame right)
    {
        if (left == null)
        {
            return right;
        }
        if (right == null)
        {
            return left;
        }
        var leftRows = left.EmbeddedTensors.Select(t => t?.Shape[1]??0).ToArray();
        var newColumns = left._columns.ToList();
        foreach (var c in right._columns)
        {
            var newTuple = Tuple.Create(c.Item1, c.Item2, leftRows[c.Item2]+c.Item3);
            newColumns.Add(newTuple);
        }
        return new DataFrame(
            newColumns,
            CpuTensor<float>.MergeHorizontally(left._floatTensor, right._floatTensor),
            CpuTensor<string>.MergeHorizontally(left._stringTensor, right._stringTensor),
            CpuTensor<int>.MergeHorizontally(left._intTensor, right._intTensor)
        );
    }
    public DataFrame SumBy(params string[] idColumns)
    {
        Debug.Assert(idColumns.Length == 1);
        return SumOrAvgForColumns(idColumns[0], false);
    }
    public DataFrame AverageBy(params string[] idColumns)
    {
        Debug.Assert(idColumns.Length == 1);
        return SumOrAvgForColumns(idColumns[0], true);
    }
    //public List<int> ColumnNamesToIndexes(IEnumerable<string> columnNames)
    //{
    //    var indexes = new List<int>();
    //    foreach (var f in columnNames)
    //    {
    //        int idx = Array.IndexOf(Columns, f);
    //        if (idx < 0)
    //        {
    //            throw new Exception($"Invalid {nameof(Columns)} name {f}");
    //        }
    //        indexes.Add(idx);
    //    }

    //    return indexes;
    //}

    /// <summary>
    /// encode the string column 'columnToEncode' using Tf*Idf with 'embeddingDim' words and return a new DataFrame with this encoding
    /// </summary>
    /// <param name="columnToEncode"></param>
    /// <param name="embeddingDim">the number of dimension for the encoding.
    /// Only the top frequent 'embeddingDim' words will be considered for the encoding.
    /// The other will be discarded</param>
    /// <param name="keepEncodedColumnName">
    /// Each new feature will have in its name the associated word for the TfIdf encoding</param>
    /// <param name="reduceEmbeddingDimIfNeeded"></param>
    /// <param name="norm"></param>
    /// <param name="scikitLearnCompatibilityMode"></param>
    /// <returns></returns>
    public DataFrame TfIdfEncode(string columnToEncode, int embeddingDim, bool keepEncodedColumnName = false, bool reduceEmbeddingDimIfNeeded = false, TfIdfEncoding.TfIdfEncoding_norm norm = TfIdfEncoding.TfIdfEncoding_norm.L2, bool scikitLearnCompatibilityMode = false)
    {
        return TfIdfEncoding.Encode(new[] { this }, columnToEncode, embeddingDim, keepEncodedColumnName, reduceEmbeddingDimIfNeeded, norm, scikitLearnCompatibilityMode)[0];
    }
    
    public static void TfIdfEncode(string[] csvFiles, bool hasHeader, bool isNormalized, string columnToEncode, int embeddingDim,
        bool keepEncodedColumnName = false, bool reduceEmbeddingDimIfNeeded = false, TfIdfEncoding.TfIdfEncoding_norm norm = TfIdfEncoding.TfIdfEncoding_norm.L2, bool scikitLearnCompatibilityMode = false)
    {
        string directory = Path.GetDirectoryName(csvFiles[0]) ?? "";
        
        var dfs = new List<DataFrame>();
        List<string> columnContent = new();
        foreach (var path in csvFiles)
        {
            ISample.Log.Info($"loading CSV file {path}");
            var df = DataFrame.read_string_csv(path, hasHeader, isNormalized);
            ISample.Log.Info($"extracting content of column {columnToEncode} in CSV file {path}");
            columnContent.AddRange(df.StringColumnContent(columnToEncode));
            dfs.Add(df);
        }

        columnContent.Sort();

        var df_ColumnToEncode = DataFrame.New(columnContent.ToArray(), new[] { columnToEncode });

        ISample.Log.Info($"Encoding column {columnToEncode}");
        var encoded_column_df = df_ColumnToEncode
            .TfIdfEncode(columnToEncode, embeddingDim, true, reduceEmbeddingDimIfNeeded, norm, scikitLearnCompatibilityMode)
            .AverageBy(columnToEncode);

        var encoded_column_df_path = Path.Combine(directory, "tfidf_for_"+columnToEncode + ".csv");
        ISample.Log.Info($"Encoded column file {encoded_column_df_path}");
        encoded_column_df.to_csv(encoded_column_df_path, ',', hasHeader);

        for (var index = 0; index < dfs.Count; index++)
        {
            var targetPath = Path.Combine(directory, Path.GetFileNameWithoutExtension(csvFiles[index])+"_with_tfidf_for_"+ columnToEncode+".csv");
            ISample.Log.Info($"Creating encoded DataFrame for {csvFiles[index]} and saving it to {targetPath}");
            var df = dfs[index];
            var df2 = df.LeftJoinWithoutDuplicates(encoded_column_df, new []{columnToEncode});
            if (!keepEncodedColumnName)
            {
                df2 = df2.Drop(columnToEncode);
            }
            df2.to_csv(targetPath, ',', hasHeader);
        }
        ISample.Log.Info($"All CSV files have been encoded");
    }




    //public DataFrame Normalize()
    //{
    //    return new DataFrame(_columns,
    //        _floatTensor?.Normalize().normalizedTensor,
    //        _stringTensor,
    //        _intTensor);
    //}

    //public DataFrame DeduceRowMean()
    //{
    //    return new DataFrame(_columns,
    //        _floatTensor?.DeduceRowMean(),
    //        _stringTensor,
    //        _intTensor);
    //}

    /// <summary>
    /// join the 2 DataFrame ('this' and 'right_df' using the key 'joinKey' to join.
    /// this key is considered as a unique key in the 'right_df' DataFrame
    /// (if the 'joinKey' key is duplicated in the 'right_df' DataFrame, then only the 1st occurrence of the key will be considered
    /// this : a DataFrame of shape (leftRows, leftColumns) containing a key 'joinKey'
    /// </summary>
    /// <param name="right">a DataFrame of shape (rightRows, rightColumns) containing a unique key 'joinKey'</param>
    /// <param name="leftJoinKey">the key to make the join for the left DataFrame</param>
    /// <param name="rightJoinKeys">the key to make the join for the right DataFrame.
    /// if missing, we'll use the same keys as for the left DataFrame</param>
    /// <returns>a DataFrame of shape (leftRows, leftColumns+rightColumns-1)</returns>
    /// <exception cref="ArgumentException"></exception>
    public DataFrame LeftJoinWithoutDuplicates(DataFrame right, string[] leftJoinKey, string[] rightJoinKeys = null)
    {
        var left = this;
        rightJoinKeys = rightJoinKeys ?? leftJoinKey;
        if (rightJoinKeys.Length != leftJoinKey.Length)
        {
            throw new ArgumentException($"both keys must have same length: {string.Join(",", leftJoinKey)} vs {string.Join(",", rightJoinKeys)}");
        }
        var leftRows = left.Shape[0];
        var leftSrcToTargetIndexes = new List<IList<int>>();
        var rightSrcToTargetIndexes = new List<IList<int>>();
        while (leftSrcToTargetIndexes.Count < EmbeddedTensors.Length)
        {
            leftSrcToTargetIndexes.Add(new List<int>());
            rightSrcToTargetIndexes.Add(new List<int>());
        }

        var targetCountByTensorType = new int[EmbeddedTensors.Length];
        List<Tuple<string, int, int>> targetColumnDesc = new ();
        foreach(var leftColDesc in left._columns)
        {
            int tensorType = leftColDesc.Item2;
            ++targetCountByTensorType[tensorType];
            targetColumnDesc.Add(leftColDesc);
            leftSrcToTargetIndexes[tensorType].Add(leftColDesc.Item3);
        }
        foreach (var rightColDesc in right._columns)
        {
            int tensorType = rightColDesc.Item2;
            if (rightJoinKeys.Contains(rightColDesc.Item1))
            {
                rightSrcToTargetIndexes[tensorType].Add(-1);
                continue;
            }
            int targetIndexForTensorType = targetCountByTensorType[tensorType];
            ++targetCountByTensorType[tensorType];
            rightSrcToTargetIndexes[tensorType].Add(targetIndexForTensorType);
            targetColumnDesc.Add(Tuple.Create(rightColDesc.Item1, tensorType, targetIndexForTensorType));
        }

        var targetRows = leftRows;
        var targetFloatTensor = targetCountByTensorType[FLOAT_TYPE_IDX] == 0 ? null : new CpuTensor<float>(new[] { targetRows, targetCountByTensorType[FLOAT_TYPE_IDX] });
        var targetStringTensor = targetCountByTensorType[STRING_TYPE_IDX] == 0 ? null : new CpuTensor<string>(new[] { targetRows, targetCountByTensorType[STRING_TYPE_IDX] });
        var targetIntTensor = targetCountByTensorType[INT_TYPE_IDX] == 0 ? null : new CpuTensor<int>(new[] { targetRows, targetCountByTensorType[INT_TYPE_IDX] });

             var target = new DataFrame(
            targetColumnDesc,
            targetFloatTensor,
            targetStringTensor,
            targetIntTensor);

        var rightKeys = right.ExtractKeyToRows(right.ToIndexesForEachTensorType(rightJoinKeys));
        left.CopyTo(leftSrcToTargetIndexes, target);
        var leftJoinKeyIndexes = left.ToIndexesForEachTensorType(leftJoinKey);
        //for (int leftRow = 0; leftRow < leftRows; ++leftRow)
        void ProcessLeftRow(int leftRow)
        {
            var targetRow = leftRow;
            var leftKey = left.ExtractKey(leftRow, leftJoinKeyIndexes);
            if (rightKeys.TryGetValue(leftKey, out var rightRowsMatchingKey))
            {
                var rightRow = rightRowsMatchingKey[0]; //we only take the first one
                right.CopyToSingleRow(rightRow, targetRow, rightSrcToTargetIndexes, target);
            }
        }

        Parallel.For(0, leftRows, ProcessLeftRow);


        return target;
    }


    private int[] TensorIndexToColumnCount()
    {
        var embeddedTensors = EmbeddedTensors;
        var tensorIndexToColumnCount = new int[embeddedTensors.Length];
        for (var tensorIndex = 0; tensorIndex < embeddedTensors.Length; tensorIndex++)
        {
            var t = embeddedTensors[tensorIndex];
            if (t != null)
            {
                tensorIndexToColumnCount[tensorIndex] += t.Shape[1];
            }
        }
        return tensorIndexToColumnCount;
    }


    private DataFrame SumOrAvgForColumns(string keyColumn, bool IsAverage)
    {
        var rows = Shape[0];
        var idxKeyColumn = Array.IndexOf(Columns, keyColumn);

        var newColumnsOldIndex = new List<Tuple<string, int, int>>();
        newColumnsOldIndex.Add(_columns[idxKeyColumn]);
        for (var i = 0; i < _columns.Count; i++)
        {
            if (i != idxKeyColumn && _columns[i].Item2 != STRING_TYPE_IDX)
            {
                newColumnsOldIndex.Add(_columns[i]);
            }
        }
        var countByNewTensor = new int[TensorIndexToType.Length];
        var newColumnsNewIndex = new List<Tuple<string, int, int>>();
        for (var i = 0; i < newColumnsOldIndex.Count; i++)
        {
            var c = newColumnsOldIndex[i];
            newColumnsNewIndex.Add(Tuple.Create(c.Item1, (i == 0) ? c.Item2 : FLOAT_TYPE_IDX, countByNewTensor[c.Item2]));
            ++countByNewTensor[c.Item2];
        }

        


        Dictionary<object, float[]> keyToSumArray = new();
        Dictionary<object, int> keyToCount = new();
        List<object> sortedKeys = new();

        var floatContent = _floatTensor == null ? new float[0] : _floatTensor.ReadonlyContent;
        var intContent = _intTensor == null ? new int[0] : _intTensor.ReadonlyContent;
        var embeddedTensors = EmbeddedTensors;

        for (int row = 0; row < rows; ++row)
        {
            object key = ExtractValue(row, _columns[idxKeyColumn]);
            if (key == null)
            {
                key = TensorIndexToNonNullDefaultValue[_columns[idxKeyColumn].Item2];
            }
            if (keyToSumArray.TryGetValue(key, out var sumArray))
            {
                ++keyToCount[key];
            }
            else
            {
                sortedKeys.Add(key);
                sumArray = new float[newColumnsNewIndex.Count - 1];
                keyToSumArray[key] = sumArray;
                keyToCount[key] = 1;
            }

            int dataIdx = 0;
            for (var i = 1; i < newColumnsNewIndex.Count; i++)
            {
                var c = newColumnsOldIndex[i];
                var idx = c.Item3 + row * embeddedTensors[c.Item2].Shape[1];
                switch (c.Item2)
                {
                    case FLOAT_TYPE_IDX: sumArray[dataIdx++] += floatContent[idx]; break;
                    case INT_TYPE_IDX: sumArray[dataIdx++] += intContent[idx]; break;
                    default: throw new ArgumentException($"can't convert {c.Item2} type to float");
                }
            }
        }

        var targetRows = keyToSumArray.Count;
        var targetFloatContent = new float[targetRows * countByNewTensor[FLOAT_TYPE_IDX]];
        var targetStringContent = new string[targetRows * countByNewTensor[STRING_TYPE_IDX]];
        var targetIntContent = new int[targetRows * countByNewTensor[INT_TYPE_IDX]];
        int idxTargetFloatContent = 0;
        var idxKeyColumnDesc = newColumnsNewIndex[0];
        for (var row = 0; row < sortedKeys.Count; row++)
        {
            var key = sortedKeys[row];
            var idx = idxKeyColumnDesc.Item3 + row * countByNewTensor[idxKeyColumnDesc.Item2];
            switch (idxKeyColumnDesc.Item2)
            {
                case FLOAT_TYPE_IDX: targetFloatContent[idx] = (float)key; break;
                case STRING_TYPE_IDX: targetStringContent[idx] = (string)key; break;
                default: targetIntContent[idx] = (int)key; break;
            }

            if (idxKeyColumnDesc.Item2 == FLOAT_TYPE_IDX)
            {
                ++idxTargetFloatContent;
            }
            var keyCount = keyToCount[key];
            foreach (var e in keyToSumArray[key])
            {
                targetFloatContent[idxTargetFloatContent++] = IsAverage ? (e / keyCount) : e;
            }
        }
        return new DataFrame(newColumnsNewIndex,
            CpuTensor<float>.New(targetFloatContent, countByNewTensor[FLOAT_TYPE_IDX]),
            CpuTensor<string>.New(targetStringContent, countByNewTensor[STRING_TYPE_IDX]),
            CpuTensor<int>.New(targetIntContent, countByNewTensor[INT_TYPE_IDX]));
    }

    // ReSharper disable once UnusedMember.Global
    public DataFrame Head(int n)
    {
        n = Math.Min(n, Shape[0]);
        return RowSlice(0, n, false);
    }
    // ReSharper disable once UnusedMember.Global
    public DataFrame Tail(int n)
    {
        n = Math.Min(n, Shape[0]);
        return RowSlice(n - Shape[0], n, false);
    }

    public DataFrame Clone()
    {
        return new DataFrame(_columns.ToList(),
            (CpuTensor<float>)_floatTensor?.Clone(),
            (CpuTensor<string>)_stringTensor?.Clone(),
            (CpuTensor<int>)_intTensor?.Clone());
    }

    public void Add(DataFrame other)
    {
        AssertSameShapeAndColumns(this, other);
        _floatTensor?.AddTensor(1, other._floatTensor, 1);
        other._intTensor?.CopyTo(_intTensor); //!D todo: do same thing for _intTensor
        other._stringTensor?.CopyTo(_stringTensor);
    }

    public void Mult(float multiplier)
    {
        _floatTensor?.Update_Multiplying_By_Alpha(multiplier);
        if (_intTensor != null)
        {
            //!D TODO: do the same thing for integers
            throw new NotImplementedException();
        }
    }


    public DataFrame sort_values(string columnName, bool ascending = true)
    {
        var colDesc = GetColumnDesc(columnName);

        int[] orderedRows = null;
        if (colDesc.Item2 == FLOAT_TYPE_IDX)
        {
            var floatSpan = _floatTensor.ReadonlyContent;
            List<Tuple<int, float>> columnContent = new();
            for (int row = 0; row < Shape[0]; ++row)
            {
                columnContent.Add(Tuple.Create(row, floatSpan[colDesc.Item3 + row * _floatTensor.Shape[1]]));
            }
            orderedRows = columnContent.OrderBy(t => t.Item2).Select(t => t.Item1).ToArray();
        }
        else if (colDesc.Item2 == INT_TYPE_IDX)
        {
            var intSpan = _intTensor.ReadonlyContent;
            List<Tuple<int, int>> columnContent = new();
            for (int row = 0; row < Shape[0]; ++row)
            {
                columnContent.Add(Tuple.Create(row, intSpan[colDesc.Item3 + row * _intTensor.Shape[1]]));
            }
            orderedRows = columnContent.OrderBy(t => t.Item2).Select(t => t.Item1).ToArray();
        }
        else if (colDesc.Item2 == STRING_TYPE_IDX)
        {
            var stringSpan = _stringTensor.ReadonlyContent;
            List<Tuple<int, string>> columnContent = new();
            for (int row = 0; row < Shape[0]; ++row)
            {
                columnContent.Add(Tuple.Create(row, stringSpan[colDesc.Item3 + row * _stringTensor.Shape[1]]));
            }
            orderedRows = columnContent.OrderBy(t => t.Item2).Select(t => t.Item1).ToArray();
        }
        else 
        {
            throw new ArgumentException($"invalid tensor type {colDesc.Item2}");
        }

        if (!ascending)
        {
            Array.Reverse(orderedRows);
        }

        return new DataFrame(
            _columns.ToList(),
            _floatTensor?.ApplyRowOrder(orderedRows),
            _stringTensor?.ApplyRowOrder(orderedRows),
            _intTensor?.ApplyRowOrder(orderedRows)
            );
    }


    //public void ClearContent()
    //{
    //    SetContent(_floatTensor, 0.0f);
    //    SetContent(_intTensor, 0);
    //    SetContent(_stringTensor, null);
    //}

    //private static void SetContent<T>(CpuTensor<T> t, T newValue)
    //{
    //    if (t == null)
    //    {
    //        return;
    //    }
    //    var content = t.Content.Span;
    //    for (int i = 0; i < content.Length; ++i)
    //    {
    //        content[i] = newValue;
    //    }
    //}

    public static bool SameShape(IList<DataFrame> df)
    {
        return df.All(t => t.Shape.SequenceEqual(df[0].Shape));
    }
    private static void AssertSameShapeAndColumns(DataFrame a, DataFrame b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
        {
            throw new ArgumentException($"not same shape between {a} and {b}");
        }
        if (!a._columns.SequenceEqual(b._columns))
        {
            throw new ArgumentException($"not same columns between {a} and {b}");
        }
    }


    public DataFrame RowSlice(int startRowIndex, int nbRows, bool inPlace)
    {
        Debug.Assert(startRowIndex >= 0);
        if (nbRows <= 0)
        {
            return null;
        }
        if (!inPlace)
        {
            return Clone().RowSlice(startRowIndex, nbRows, true);
        }
        if (startRowIndex == 0 && nbRows >= Shape[0])
        {
            return this;
        }
        return new DataFrame(_columns,
            (CpuTensor<float>)_floatTensor?.RowSlice(startRowIndex, nbRows),
            (CpuTensor<string>)_stringTensor?.RowSlice(startRowIndex, nbRows),
            (CpuTensor<int>)_intTensor?.RowSlice(startRowIndex, nbRows)
        );
    }

    /// <summary>
    /// build a DataFrame with all cells set to default values
    /// </summary>
    /// <param name="rows">number of rows in the DataFrame</param>
    /// <param name="columnsNames">column names of the DataFrame</param>
    /// <param name="columnNameToType">for each column name, the type associated with this column name</param>
    /// <returns></returns>
    public static DataFrame BuildEmptyDataframe(int rows, List<string> columnsNames, [CanBeNull] Func<string, Type> columnNameToType)
    {
        if (columnNameToType == null)
        {
            columnNameToType = _ => typeof(string);
        }
        List<Tuple<string, int, int>> newColumns = new();
        var columnIndexToTensorIndex = columnsNames.Select(i => TypeToTensorIndex[columnNameToType(i)]).ToArray();
        var tensorIndexToCount = columnIndexToTensorIndex.GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count());
        var countByTensorIndex = new int[TensorIndexToType.Length];
        for (int col = 0; col < columnsNames.Count; ++col)
        {
            var tensorIndex = columnIndexToTensorIndex[col];
            newColumns.Add(Tuple.Create(columnsNames[col], tensorIndex, countByTensorIndex[tensorIndex]));
            ++countByTensorIndex[tensorIndex];
        }

        var floatTensor = newColumns.Any(c => c.Item2 == FLOAT_TYPE_IDX) ? CpuTensor<float>.New(new float[rows * tensorIndexToCount[FLOAT_TYPE_IDX]], tensorIndexToCount[FLOAT_TYPE_IDX]) : null;
        var stringTensor = newColumns.Any(c => c.Item2 == STRING_TYPE_IDX) ? CpuTensor<string>.New(new string[rows * tensorIndexToCount[STRING_TYPE_IDX]], tensorIndexToCount[STRING_TYPE_IDX]) : null;
        var intTensor = newColumns.Any(c => c.Item2 == INT_TYPE_IDX) ? CpuTensor<int>.New(new int[rows * tensorIndexToCount[INT_TYPE_IDX]], tensorIndexToCount[INT_TYPE_IDX]) : null;
        return new DataFrame(newColumns, floatTensor, stringTensor, intTensor);
    }


    private static readonly object Lock_read_csv = new();

    /// <summary>
    /// read a csv that has been normalized:
    ///     we are sure that each field doesn't contain any special characters (new lines, comma, semi colon, separator, double quotes)
    ///     so we can simply load each file line and split it using the separator
    /// </summary>
    /// <param name="path">the path where the CSV file is located. If the path is invalid an exception will be thrown</param>
    /// <param name="sep">the separator used in the file (usually the comma)</param>
    /// <param name="hasHeader">true if the CSV file has a header</param>
    /// <param name="columnNameToType">for each column the type fo the associated column (among: float, string & int)
    /// if null we'll consider that each column is of string type
    /// </param>
    /// <returns></returns>
    /// <exception cref="ArgumentException">if the path is invalid</exception>
    public static DataFrame read_csv_normalized(string path, char sep = ',', bool hasHeader = true, [CanBeNull] Func<string, Type> columnNameToType = null)
    {
        if (!File.Exists(path))
        {
            throw new ArgumentException($"invalid CSV file {path}");
        }

        string[] allLinesReadAllLines;
        lock (Lock_read_csv)
        {
            allLinesReadAllLines = File.ReadAllLines(path);
        }
        var rows = hasHeader ? allLinesReadAllLines.Length - 1 : allLinesReadAllLines.Length;
        var columnsNames = hasHeader
            ? allLinesReadAllLines[0].Split(sep).ToList()
            : Enumerable.Range(0, allLinesReadAllLines[0].Split(sep).Length).Select(t => t.ToString()).ToList();

        var df = BuildEmptyDataframe(rows, columnsNames, columnNameToType);
        // this cache is used to avoid re creating the same string again and again
        var cacheString = new ConcurrentDictionary<int, string>();

        void ProcessRow(int row)
        {
            string line = allLinesReadAllLines[(hasHeader ? 1 : 0) + row];
            var lineSpan = line.AsSpan();
            var floatContent = df.FloatTensor == null ? null : df.FloatTensor.SpanContent;
            var stringContent = df.StringTensor == null ? null : df.StringTensor.SpanContent;
            var intContent = df.IntTensor == null ? null : df.IntTensor.SpanContent;
            int floatIdx = df.FloatTensor?.Idx(row) ?? -1;
            int stringIdx = df.StringTensor?.Idx(row) ?? -1;
            int intIdx = df.IntTensor?.Idx(row) ?? -1;

            // index in the current 'line' of the start of the next item to process
            int nextItemStart = 0;
            for (int col = 0; col < columnsNames.Count; ++col)
            {
                int nextItemEndExcluded = (nextItemStart >= line.Length) ?-1 : line.IndexOf(sep, nextItemStart);
                if (nextItemEndExcluded == -1)
                {
                    nextItemEndExcluded = line.Length;
                }
                int nextItemLength = nextItemEndExcluded - nextItemStart;
                switch (df.ColumnsDesc[col].Item2)
                {
                    case FLOAT_TYPE_IDX: 
                        floatContent[floatIdx++] = Utils.TryParseFloat(lineSpan, nextItemStart, nextItemLength);
                        break;
                    case STRING_TYPE_IDX:
                        stringContent[stringIdx++] = Utils.SubStringWithCache(lineSpan, nextItemStart, nextItemLength, cacheString);
                        break;
                    case INT_TYPE_IDX: 
                        intContent[intIdx++] = Utils.TryParseInt(lineSpan, nextItemStart, nextItemLength);
                        break;
                    default: throw new ArgumentException($"invalid Tensor Index {df.ColumnsDesc[col].Item2}");
                }
                nextItemStart = nextItemEndExcluded+1;
            }
        }
        Parallel.For(0, rows, ProcessRow);
        return df;
    }


    /// <summary>
    /// read a csv.
    /// it will detect automatically the separator
    /// </summary>
    /// <param name="path">the path where the CSV file is located. If the path is invalid an exception will be thrown</param>
    /// <param name="hasHeader">true if the CSV file has a header</param>
    /// <param name="columnNameToType">for each column the type fo the associated column (among: float, string & int)
    /// if null we'll consider that each column is of string type
    /// </param>
    /// <param name="isNormalized">
    /// true if the CSV file has been normalized (each field doesn't contain any special characters: new lines, comma, semi colon, separator, double quotes)
    /// false if some fields may contain special characters
    /// </param>
    /// <returns></returns>
    /// <exception cref="ArgumentException">if the path is invalid</exception>
    public static DataFrame read_csv(string path, bool hasHeader = true, [CanBeNull] Func<string, Type> columnNameToType = null, bool isNormalized = false)
    {
        if (isNormalized)
        {
            //we will use a faster method to load the file because we know we can split each line easily
            return read_csv_normalized(path, ',', hasHeader, columnNameToType);
        }
        if (!File.Exists(path))
        {
            throw new ArgumentException($"invalid CSV file {path}");
        }
        List<string[]> allLinesReadCsv;
        List<string> columnsNames;
        int rows;
        lock (Lock_read_csv)
        {
            allLinesReadCsv = Utils.ReadCsv(path).ToList();
            rows = hasHeader ? allLinesReadCsv.Count - 1 : allLinesReadCsv.Count;
            columnsNames = hasHeader
                ? allLinesReadCsv[0].ToList()
                : Enumerable.Range(0, allLinesReadCsv[0].Length).Select(t => t.ToString()).ToList();
        }

        var df = BuildEmptyDataframe(rows, columnsNames, columnNameToType);
        
        void ProcessRow(int row)
        {
            var lineContent = allLinesReadCsv[(hasHeader?1:0)+row];
            var floatContent = df.FloatTensor == null ? null : df.FloatTensor.SpanContent;
            var stringContent = df.StringTensor == null ? null : df.StringTensor.SpanContent;
            var intContent = df.IntTensor == null ? null : df.IntTensor.SpanContent;
            int floatIdx = df.FloatTensor?.Idx(row) ?? -1;
            int stringIdx = df.StringTensor?.Idx(row) ?? -1;
            int intIdx = df.IntTensor?.Idx(row) ?? -1;

            for (int col = 0; col < columnsNames.Count; ++col)
            {
                switch (df.ColumnsDesc[col].Item2)
                {
                    case FLOAT_TYPE_IDX:
                        if (!float.TryParse(lineContent[col], out var floatValue))
                        {
                            floatValue = float.NaN;
                        }
                        floatContent[floatIdx++] = floatValue; break;
                    case STRING_TYPE_IDX:  stringContent[stringIdx++] = lineContent[col]; break;
                    case INT_TYPE_IDX:  intContent[intIdx++] = int.Parse(lineContent[col]); break;
                    default: throw new ArgumentException($"invalid Tensor Index {df.ColumnsDesc[col].Item2}");
                }
            }
        }
        Parallel.For(0, rows, ProcessRow);
        return df;
    }

    public void to_csv([NotNull] string path, char sep = ',', bool addHeader = true, int? index = null)
    {
        int rows = Shape[0];
        int cols = Shape[1];
        //we'll process the file by chunk of 2000 rows
        var buffer = new StringBuilder[Math.Min(2000, rows + 1)];
        for (int i = 0; i < buffer.Length; ++i)
        {
            buffer[i] = new StringBuilder();
        }
        //we create the directory if missing
        var directory = Path.GetDirectoryName(path) ?? "";
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }
        File.WriteAllText(path + ".tmp", addHeader ? string.Join(sep, Columns) : "");
        var sbBuffer = new StringBuilder();

        // initialize the StringBuilder buffer at buffer[row - index_of_first_element_in_buffer]
        // with the row 'row' of the file
        void ProcessRow(int row, int row_in_first_index_of_buffer)
        {
            var sb = buffer[row - row_in_first_index_of_buffer];
            sb.Clear();
            var floatContent = _floatTensor == null ? null : _floatTensor.RowSpanSlice(row, 1);
            var stringContent = _stringTensor == null ? null : _stringTensor.RowSpanSlice(row, 1);
            var intContent = _intTensor == null ? null : _intTensor.RowSpanSlice(row, 1);

            for (int col = 0; col < cols; col++)
            {
                if (col == 0)
                {
                    if (index.HasValue)
                    {
                        sb.Append(index.Value + row);
                        sb.Append(sep);
                    }
                }
                else
                {
                    sb.Append(sep);
                }

                var c = _columns[col];
                var idx = c.Item3;
                switch (c.Item2)
                {
                    case FLOAT_TYPE_IDX:
                        if (!float.IsNaN(floatContent[idx]))
                        {
                            sb.Append(CultureInfo.InvariantCulture, $"{floatContent[idx]}");
                        }
                        break;
                    case STRING_TYPE_IDX:
                        var str = stringContent[idx];
                        if (!string.IsNullOrEmpty(str) && str.Contains(sep) && str[0] != '\"' && str.Last() != '\"')
                        {
                            str = "\"" + str + "\"";
                        }
                        sb.Append(str);
                        break;
                    case INT_TYPE_IDX:
                        sb.Append(intContent[idx]);
                        break;
                    default: throw new Exception($"invalid type index {c.Item2}");
                }
            }
        }

        for (int i = 0; i < rows; i += buffer.Length)
        {
            // we process the file by chunk of 'buffer.Length' rows
            var first_row_to_process = i;
            //the first row to process will be in the buffer at index 0
            int last_row_excluded = Math.Min(rows, first_row_to_process + buffer.Length);
            Parallel.For(first_row_to_process, last_row_excluded, row => ProcessRow(row, first_row_to_process));
            //for each processed row, we append the content to a StringBuilder buffer (sbBuffer)
            //when this StringBuilder buffer is more than 200MBytes, we write it to the file
            for (int row = first_row_to_process; row < last_row_excluded; ++row)
            {
                if (row != 0 || addHeader)
                {
                    sbBuffer.Append(Environment.NewLine);
                }
                sbBuffer.Append(buffer[row - first_row_to_process]);
                if (sbBuffer.Length >= 200_000_000 || row == last_row_excluded - 1)
                {
                    File.AppendAllText(path + ".tmp", sbBuffer.ToString());
                    sbBuffer.Clear();
                }
            }
        }
        Debug.Assert(sbBuffer.Length == 0);
        sbBuffer.Clear();
        foreach (var t in buffer)
        {
            t.Clear();
        }

        if (File.Exists(path))
        {
            File.Delete(path);
        }
        File.Move(path + ".tmp", path);
    }

    private static CpuTensor<T> ToCpuTensor<T>(List<T> content, Dictionary<int, int> tensorIndexToCount, int tensorIndex)
    {
        if (content == null || content.Count == 0 || !tensorIndexToCount.TryGetValue(tensorIndex, out var columnCount))
        {
            return null;
        }
        return CpuTensor<T>.New(content.ToArray(), columnCount);
    }
    public Tensor[] EmbeddedTensors => new Tensor[] { _floatTensor, _stringTensor, _intTensor };


    private Tuple<string, int, int> GetColumnDesc(string columnName)
    {
        return _columns.First(c => c.Item1 == columnName);
    }

    private List<List<int>> ToIndexesForEachTensorType(IEnumerable<string> columnNames)
    {
        var res = new List<List<int>>();
        while (res.Count < EmbeddedTensors.Length)
        {
            res.Add(new List<int>());
        }
        foreach (var c in columnNames)
        {
            var desc = GetColumnDesc(c);
            res[desc.Item2].Add(desc.Item3);
        }
        return res;
    }

    public void CopyTo(List<IList<int>> srcToTargetIndexes, DataFrame target)
    {
        Debug.Assert(srcToTargetIndexes.Count == EmbeddedTensors.Length);
        for (int row = 0; row < Shape[0]; ++row)
        {
            CopyTo(row, row, srcToTargetIndexes[FLOAT_TYPE_IDX], FloatTensor, target.FloatTensor);
            CopyTo(row, row, srcToTargetIndexes[STRING_TYPE_IDX], StringTensor, target.StringTensor);
            CopyTo(row, row, srcToTargetIndexes[INT_TYPE_IDX], IntTensor, target.IntTensor);
        }
    }

    public void CopyToSingleRow(int srcRow, int targetRow, List<IList<int>> srcToTargetIndexes, DataFrame target)
    {
        CopyTo(srcRow, targetRow, srcToTargetIndexes[FLOAT_TYPE_IDX], FloatTensor, target.FloatTensor);
        CopyTo(srcRow, targetRow, srcToTargetIndexes[STRING_TYPE_IDX], StringTensor, target.StringTensor);
        CopyTo(srcRow, targetRow, srcToTargetIndexes[INT_TYPE_IDX], IntTensor, target.IntTensor);

    }

    public static void CopyTo<T>(int srcRow, int targetRow, IList<int> srcToTargetIndexes, CpuTensor<T> srcTensor, CpuTensor<T> targetTensor)
    {
        if (srcTensor == null)
        {
            return;
        }
        var srcContent = srcTensor.RowSpanSlice(srcRow, 1);
        var targetContent = targetTensor.RowSpanSlice(targetRow, 1);
        for (var srcIndex = 0; srcIndex < srcToTargetIndexes.Count; srcIndex++)
        {
            var targetIndex = srcToTargetIndexes[srcIndex];
            if (targetIndex >= 0)
            {
                targetContent[targetIndex] = srcContent[srcIndex];
            }
        }
    }

    private IDictionary<string,List<int>> ExtractKeyToRows(List<List<int>> keyIndexes)
    {
        var keysToRow = new Dictionary<string, List<int>>();
        for (int row = 0; row < Shape[0]; ++row)
        {
            var key = ExtractKey(row, keyIndexes);
            if (!keysToRow.TryGetValue(key, out var val))
            {
                val = new List<int>();
                keysToRow[key] = val;
            }
            val.Add(row);
        }
        return keysToRow;
    }

    private string ExtractKey(int row, List<List<int>> keyIndexes)
    {
        Debug.Assert(keyIndexes.Count == EmbeddedTensors.Length);
        var keysAsString = new List<string>();
        foreach (var floatColIndex in keyIndexes[FLOAT_TYPE_IDX])
        {
            keysAsString.Add(Math.Round(ExtractFloatValue(row, floatColIndex), 8).ToString(CultureInfo.InvariantCulture));
        }
        foreach (var stringColIndex in keyIndexes[STRING_TYPE_IDX])
        {
            keysAsString.Add(ExtractStringValue(row, stringColIndex));
        }
        foreach (var intColIndex in keyIndexes[INT_TYPE_IDX])
        {
            keysAsString.Add(ExtractIntValue(row, intColIndex).ToString());
        }
        return string.Join(",", keysAsString);
    }

    private object ExtractValue(int row, Tuple<string, int, int> colDesc)
    {
        switch (colDesc.Item2)
        {
            case FLOAT_TYPE_IDX: return FloatTensor.RowSpanSlice(row,1)[colDesc.Item3];
            case STRING_TYPE_IDX: return StringTensor.RowSpanSlice(row, 1)[colDesc.Item3];
            default: return IntTensor.RowSpanSlice(row, 1)[colDesc.Item3];
        }
    }


    private float ExtractFloatValue(int row, int colIndex)
    {
        return FloatTensor.RowSpanSlice(row, 1)[colIndex];
    }
    private string ExtractStringValue(int row, int colIndex)
    {
        return StringTensor.RowSpanSlice(row, 1)[colIndex];
    }
    private int ExtractIntValue(int row, int colIndex)
    {
        return IntTensor.RowSpanSlice(row, 1)[colIndex];
    }

    public bool IsFloatDataFrame => _stringTensor == null && _intTensor == null;
    public bool IsStringDataFrame => _floatTensor == null && _intTensor == null;
    public bool IsIntDataFrame => _floatTensor == null && _stringTensor == null;
    /// <summary>
    /// ensure that all column name in 'columnNameToValidate' are valid and throws an exception if it is not the case
    /// </summary>
    /// <param name="columnNameToValidate"></param>
    /// <exception cref="Exception"></exception>
    private void AssertValidColumns(IEnumerable<string> columnNameToValidate)
    {
        var invalidColumns = Utils.Without(columnNameToValidate, Columns);
        if (invalidColumns.Count != 0)
        {
            throw new Exception($"found invalid column names {string.Join(',', invalidColumns)}");
        }
    }
    private int Rows
    {
        get
        {
            if (_stringTensor != null)
            {
                return _stringTensor.Shape[0];
            }
            if (_intTensor != null)
            {
                return _intTensor.Shape[0];
            }
            if (_floatTensor != null)
            {
                return _floatTensor.Shape[0];
            }
            return 0;
        }

    }
    //private Type[] Dtypes => _columns.Select(t => TensorIndexToType[t.Item2]).ToArray();
    public DataFrame ReduceFloatDimension(int totalReviewsEmbeddingDim)
    {
        if (_floatTensor.Shape[1] < totalReviewsEmbeddingDim)
        {
            throw new ArgumentException($"can't reduce dimension to {totalReviewsEmbeddingDim}, dimension is already {totalReviewsEmbeddingDim}");
        }

        var rows = _floatTensor.Shape[0];
        var oldCols = _floatTensor.Shape[1];
        var newCols = totalReviewsEmbeddingDim;
        var newContent = new float[rows * newCols];

        var srcContent = _floatTensor.AsReadonlyFloatCpuContent;
        for (int row = 0; row < rows; ++row)
        {
            for (int oldCol = 0; oldCol < oldCols; ++oldCol)
            {
                var newCol = oldCol%newCols;
                newContent[row*newCols+newCol] += srcContent[row*oldCols+oldCol];
            }
        }

        var newColumnDesc = _columns.Where(c => c.Item2 != FLOAT_TYPE_IDX || c.Item3 < newCols).ToList();
        var newFloatTensor = new CpuTensor<float>(new[] { rows, newCols }, newContent);
        return new DataFrame(newColumnDesc, newFloatTensor, _stringTensor, _intTensor);
    }


    public static void NormalizeAllCsvInDirectory(string directory, bool hasHeader = true, bool removeAccentedCharacters =false)
    {
        foreach (var file in Directory.GetFiles(directory, "*.csv"))
        {
            ISample.Log.Debug($"Normalizing file {file}");
            try
            {
                Normalize(file, hasHeader, removeAccentedCharacters);
            }
            catch (Exception e)
            {
                ISample.Log.Debug($"Fail to normalize file {file}: {e}");
            }
        }
    }
    public static void Normalize(string path, bool hasHeader = true, bool removeAccentedCharacters = false)
    {
        var fileName = Path.GetFileName(path);
        
        //We ensure that the file is encoded in UTF-8
        var encoding = Utils.GetEncoding(path);
        ISample.Log.Info($"Encoding of file '{fileName}' : '{encoding}'");
        if (!Equals(encoding, "UTF-8") && !Equals(encoding, "ASCII") && !string.IsNullOrEmpty(encoding))
        {
            ISample.Log.Error($"Encoding of file '{fileName}' seems to be '{encoding}'. Please encode the file in UTF-8.");
        }

        var string_df = read_string_csv(path, hasHeader, false);
        var tensor = string_df.StringTensor;
        int rows = tensor.Shape[0];
        int cols = tensor.Shape[1];

        void ProcessRow(int row)
        {
            var span = tensor.SpanContent;
            int startIdx = tensor.Idx(row);
            for (int col = 0; col < cols; ++col)
            {
                span[startIdx + col] = Utils.NormalizeCategoricalFeatureValue(span[startIdx + col]);
            }
        }
        Parallel.For(0, rows, ProcessRow);

        //we save a backup version of the original file
        var directory = Path.GetDirectoryName(path) ?? "";
        var backupDirectory = Path.Combine(directory, "backup_"+nameof(Normalize));
        if (!Directory.Exists(backupDirectory))
        {
            Directory.CreateDirectory(backupDirectory);
        }
        var backupPath = Path.Combine(backupDirectory, fileName);
        if (File.Exists(backupPath))
        {
            File.Delete(backupPath);
        }
        File.Move(path, backupPath);

        string_df.to_csv(path, ',', hasHeader);
        if (removeAccentedCharacters)
        {
            var newContent = Tokenizer.RemoveDiacritics(File.ReadAllText(path));
            File.WriteAllText(path, newContent);
        }
    }
}
