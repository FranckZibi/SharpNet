using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using SharpNet.CPU;
using SharpNet.Data;
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
    private const int FLOAT_TYPE_IDX = 0;
    private const int STRING_TYPE_IDX = 1;
    private const int INT_TYPE_IDX = 2;
    #endregion

    #region public fields and properties
    public string[] Columns { get; }
    public int[] Shape => new[] { Rows, _columns.Count };
    #endregion


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
    private DataFrame(List<Tuple<string, int, int>> columns,
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
    public static DataFrame New(string[] content, IList<string> columnsNames)
    {
        var tensor = CpuTensor<string>.New(content, columnsNames.Count);
        return new DataFrame(columnsNames.Select((n, i) => Tuple.Create(n, STRING_TYPE_IDX, i)).ToList(), null, tensor, null);
    }
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
    public CpuTensor<string> StringCpuTensor()
    {
        if (!IsStringDataFrame)
        {
            throw new ArgumentException($"{this} is not a String DataFrame");
        }
        return _stringTensor;
    }
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
        var colDesc= _columns.FirstOrDefault(c => c.Item1 == columnName);
        if (colDesc == null)
        {
            throw new ArgumentException($"invalid column name {columnName}");
        }
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
    public static DataFrame read_float_csv(string path, bool hasHeader = true)
    {
        return read_csv(path, hasHeader, _ => typeof(float));
    }
    public static DataFrame read_string_csv(string path, bool hasHeader = true)
    {
        return read_csv(path, hasHeader, _ => typeof(string));
    }
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
    public void to_csv(string path, char sep = ',', bool addHeader = true, int? index = null)
    {
        var sb = new StringBuilder();
        if (addHeader)
        {
            sb.Append(string.Join(sep, Columns));
        }
        var floatContent = _floatTensor == null ? new float[0] : _floatTensor.ReadonlyContent;
        var stringContent = _stringTensor == null ? new string[0] : _stringTensor.ReadonlyContent;
        var intContent = _intTensor == null ? new int[0] : _intTensor.ReadonlyContent;
        var embeddedTensors = EmbeddedTensors;
        int currentIndex = index ?? -1;
        int rows = Shape[0];
        int cols = Shape[1];
        for(int row=0;row<rows; row++)
        for(int col=0;col<cols; col++)
        {
            if (col == 0)
            {
                if (row != 0 || addHeader)
                {
                    sb.Append(Environment.NewLine);
                }
                if (index.HasValue)
                {
                    sb.Append(currentIndex + sep);
                    ++currentIndex;
                }
            }
            else
            {
                sb.Append(sep);
            }

            var c = _columns[col];
            var idx = c.Item3 + row * embeddedTensors[c.Item2].Shape[1];
            switch (c.Item2)
            {
                case FLOAT_TYPE_IDX: sb.Append(floatContent[idx].ToString(CultureInfo.InvariantCulture)); break;
                case STRING_TYPE_IDX: sb.Append(stringContent[idx]); break;
                case INT_TYPE_IDX: sb.Append(intContent[idx]);  break;
                default: throw new Exception($"invalid type index {c.Item2}");
            }
        }
        System.IO.File.WriteAllText(path, sb.ToString());
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
    public List<int> ColumnNamesToIndexes(IEnumerable<string> columnNames)
    {
        var indexes = new List<int>();
        foreach (var f in columnNames)
        {
            int idx = Array.IndexOf(Columns, f);
            if (idx < 0)
            {
                throw new Exception($"Invalid {nameof(Columns)} name {f}");
            }
            indexes.Add(idx);
        }

        return indexes;
    }
    public DataFrame TfIdfEncoding(string columnToEncode, int embeddingDim, bool keepEncodedColumnName = false, bool addTokenNameAsColumnNameSuffix = false)
    {
        var documents = StringColumnContent(columnToEncode);
        var df_encoded = TfIdf.ToTfIdf(documents, embeddingDim, columnToEncode, addTokenNameAsColumnNameSuffix);

        var result = this;
        if (!keepEncodedColumnName)
        {
            result = result.Drop(columnToEncode);
        }
        return MergeHorizontally(result, df_encoded);
    }
    /// <summary>
    /// join the 2 DataFrame ('this' and 'right_df' using the key 'joinKey' to join.
    /// this key is considered as a unique key in the 'right_df' DataFrame
    /// (if the 'joinKey' key is duplicated in the 'right_df' DataFrame, then only the 1st occurrence of the key will be considered
    /// this : a DataFrame of shape (leftRows, leftColumns) containing a key 'joinKey'
    /// </summary>
    /// <param name="rightDf">a DataFrame of shape (rightRows, rightColumns) containing a unique key 'joinKey'</param>
    /// <param name="joinKey">the key to make the left join</param>
    /// <returns>a DataFrame of shape (leftRows, leftColumns+rightColumns-1)</returns>
    /// <exception cref="ArgumentException"></exception>
    public DataFrame LeftJoinWithoutDuplicates(DataFrame rightDf, string joinKey)
    {
        int leftJoinKeyIndex = Array.IndexOf(Columns, joinKey);
        int rightJoinKeyIndex = Array.IndexOf(rightDf.Columns, joinKey);
        if (leftJoinKeyIndex == -1 || rightJoinKeyIndex == -1)
        {
            throw new ArgumentException($"invalid join Key name {joinKey}");
        }

        int leftRows = Shape[0];
        int leftColumns = Shape[1];
        var leftContent = FloatCpuTensor();

        int rightRows = rightDf.Shape[0];
        int rightColumns = rightDf.Shape[1];
        var rightContent = rightDf.FloatCpuTensor().ReadonlyContent;

        int newRows = leftRows;
        int newColumns = Shape[1] + rightDf.Shape[1] - 1;
        var newContent = new float[newRows * newColumns];

        var rightJoinKeyValueToFirstRightRow = new Dictionary<float, int>();
        for (int rightRow = 0; rightRow < rightRows; ++rightRow)
        {
            var rightJoinKeyValue = rightContent[rightJoinKeyIndex + rightRow * rightColumns];
            if (!rightJoinKeyValueToFirstRightRow.ContainsKey(rightJoinKeyValue))
            {
                rightJoinKeyValueToFirstRightRow[rightJoinKeyValue] = rightRow;
            }
        }

        for (int newRow = 0; newRow < newRows; ++newRow)
        {
            int nextIdxLeft = newRow * leftColumns;
            int nextIdxNew = newRow * newColumns;
            for (int leftColumn = 0; leftColumn < leftColumns; ++leftColumn)
            {
                newContent[nextIdxNew++] = leftContent[nextIdxLeft++];
            }
            var leftIdxValue = leftContent[leftJoinKeyIndex + newRow * leftColumns];
            if (rightJoinKeyValueToFirstRightRow.TryGetValue(leftIdxValue, out var rightRow))
            {
                for (int rightColumn = 0; rightColumn < rightColumns; ++rightColumn)
                {
                    if (rightColumn != rightJoinKeyIndex)
                    {
                        newContent[nextIdxNew++] = rightContent[rightColumn + rightRow * rightColumns];
                    }
                }
            }
        }
        var newColumnName = Utils.Join(Columns, Utils.Without(rightDf.Columns, new[] { joinKey }));
        return New(newContent, newColumnName);
    }


    private static DataFrame read_csv(string path, bool hasHeader = true, Func<int, Type> columnIndexToType = null)
    {
        List<string> columnsNames = null;
        int[] columnIndexToTensorIndex = null;
        Dictionary<int, int> tensorIndexToCount = new();
        List<Tuple<string, int, int>> newColumns = new();
        if (columnIndexToType == null)
        {
            columnIndexToType = _ => typeof(string);
        }
        List<float> floats = new();
        List<string> strings = new();
        List<int> ints = new();

        foreach (var lineContent in Utils.ReadCsv(path))
        {
            if (columnsNames == null)
            {
                if (hasHeader)
                {
                    columnsNames = lineContent.ToList();
                    continue;
                }
                columnsNames = Enumerable.Range(0, lineContent.Length).Select(t => t.ToString()).ToList();
            }

            if (columnIndexToTensorIndex == null)
            {
                columnIndexToTensorIndex = Enumerable.Range(0, columnsNames.Count).Select(i => TypeToTensorIndex[columnIndexToType(i)]).ToArray();
                tensorIndexToCount = columnIndexToTensorIndex.GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count());
                var countByTensorIndex = new int[TensorIndexToType.Length];
                for (int col = 0; col < columnsNames.Count; ++col)
                {
                    var tensorIndex = columnIndexToTensorIndex[col];
                    newColumns.Add(Tuple.Create(columnsNames[col], tensorIndex, countByTensorIndex[tensorIndex]));
                    ++countByTensorIndex[tensorIndex];
                }
            }


            for (int col = 0; col < lineContent.Length; ++col)
            {
                var item = lineContent[col];
                switch (columnIndexToTensorIndex[col])
                {
                    case FLOAT_TYPE_IDX: floats.Add(float.Parse(item)); break;
                    case STRING_TYPE_IDX: strings.Add(item); break;
                    case INT_TYPE_IDX: ints.Add(int.Parse(item)); break;
                    default: throw new ArgumentException($"invalid Tensor Index {columnIndexToTensorIndex[col]}");
                }
            }
        }
        return new DataFrame(
            newColumns,
            ToCpuTensor(floats, tensorIndexToCount, FLOAT_TYPE_IDX),
            ToCpuTensor(strings, tensorIndexToCount, STRING_TYPE_IDX),
            ToCpuTensor(ints, tensorIndexToCount, INT_TYPE_IDX)
            );
    }
    private static CpuTensor<T> ToCpuTensor<T>(List<T> content, Dictionary<int, int> tensorIndexToCount, int tensorIndex)
    {
        if (content == null || content.Count == 0 || !tensorIndexToCount.TryGetValue(tensorIndex, out var columnCount))
        {
            return null;
        }
        return CpuTensor<T>.New(content.ToArray(), columnCount);
    }
    private Tensor[] EmbeddedTensors => new Tensor[] { _floatTensor, _stringTensor, _intTensor };
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
        var stringContent = _stringTensor == null ? new string[0] : _stringTensor.ReadonlyContent;
        var intContent = _intTensor == null ? new int[0] : _intTensor.ReadonlyContent;
        var embeddedTensors = EmbeddedTensors;

        for (int row = 0; row < rows; ++row)
        {
            object key = ExtractValue(row, _columns[idxKeyColumn], floatContent, stringContent, intContent);
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
                    case INT_TYPE_IDX:  sumArray[dataIdx++] += intContent[idx]; break;
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
    private object ExtractValue(int row, Tuple<string, int, int> colDesc, ReadOnlySpan<float> floatContent, ReadOnlySpan<string> stringContent, ReadOnlySpan<int> intContent)
    {
        switch (colDesc.Item2)
        {
            case FLOAT_TYPE_IDX: return floatContent[colDesc.Item3 + row * _floatTensor.Shape[1]];
            case STRING_TYPE_IDX: return stringContent[colDesc.Item3 + row * _stringTensor.Shape[1]];
            default: return intContent[colDesc.Item3 + row * _intTensor.Shape[1]];
        }
    }
    private bool IsFloatDataFrame => _floatTensor != null && _stringTensor == null && _intTensor == null;
    private bool IsStringDataFrame => _floatTensor == null && _stringTensor != null && _intTensor == null;
    private bool IsIntDataFrame => _floatTensor == null && _stringTensor == null && _intTensor != null;
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
}
