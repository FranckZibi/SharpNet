using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
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
                case STRING_TYPE_IDX:
                    var str = stringContent[idx];
                    if (!string.IsNullOrEmpty(str) && str.Contains(sep) && str[0]!= '\"' && str.Last() != '\"')
                    {
                        str = "\""+str+"\"";
                    }
                    sb.Append(str); break;
                case INT_TYPE_IDX: sb.Append(intContent[idx]);  break;
                default: throw new Exception($"invalid type index {c.Item2}");
            }
        }
        File.WriteAllText(path, sb.ToString());
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
    /// <param name="joinKey">the key to make the left join</param>
    /// <returns>a DataFrame of shape (leftRows, leftColumns+rightColumns-1)</returns>
    /// <exception cref="ArgumentException"></exception>
    public DataFrame LeftJoinWithoutDuplicates(DataFrame right, string joinKey)
    {
        var left = this;
        int leftJoinKeyIndex = Array.IndexOf(Columns, joinKey);
        int rightJoinKeyIndex = Array.IndexOf(right.Columns, joinKey);
        if (leftJoinKeyIndex == -1 || rightJoinKeyIndex == -1)
        {
            throw new ArgumentException($"invalid join Key name {joinKey}");
        }

        var joinKeyTensorIndex = left._columns[leftJoinKeyIndex].Item2;
        if (joinKeyTensorIndex != right._columns[rightJoinKeyIndex].Item2)
        {
            throw new ArgumentException($"incoherent join Key type {joinKeyTensorIndex} vs {right._columns[rightJoinKeyIndex].Item2}");
        }


        var leftTensorIndexToColumnCount = left.TensorIndexToColumnCount();
        var rightTensorIndexToColumnCount = right.TensorIndexToColumnCount();
        var newTensorIndexToColumnCount = new int[leftTensorIndexToColumnCount.Length];
        for (var tensorIndex = 0; tensorIndex < leftTensorIndexToColumnCount.Length; tensorIndex++)
        {
            newTensorIndexToColumnCount[tensorIndex] = leftTensorIndexToColumnCount[tensorIndex] + rightTensorIndexToColumnCount[tensorIndex];
        }
        --newTensorIndexToColumnCount[joinKeyTensorIndex];


        var leftFloatContent = left._floatTensor == null ? new float[0] : left._floatTensor.ReadonlyContent;
        var leftStringContent = left._stringTensor == null ? new string[0] : left._stringTensor.ReadonlyContent;
        var leftIntContent = left._intTensor == null ? new int[0] : left._intTensor.ReadonlyContent;

        var rightFloatContent = right._floatTensor == null ? new float[0] : right._floatTensor.ReadonlyContent;
        var rightStringContent = right._stringTensor == null ? new string[0] : right._stringTensor.ReadonlyContent;
        var rightIntContent = right._intTensor == null ? new int[0] : right._intTensor.ReadonlyContent;

        int newRows = left.Shape[0];
        var newFloatContent = new float[newRows * newTensorIndexToColumnCount[FLOAT_TYPE_IDX]];
        var newStringContent = new string[newRows * newTensorIndexToColumnCount[STRING_TYPE_IDX]];
        var newIntContent = new int[newRows * newTensorIndexToColumnCount[INT_TYPE_IDX]];


        var rightJoinKeyValueToFirstRightRow = new Dictionary<object, int>();
        for (int rightRow = 0; rightRow < right.Shape[0]; ++rightRow)
        {
            var rightJoinKeyValue = right.ExtractValue(rightRow, right._columns[rightJoinKeyIndex], rightFloatContent, rightStringContent,  rightIntContent)??"";
            if (!rightJoinKeyValueToFirstRightRow.ContainsKey(rightJoinKeyValue))
            {
                rightJoinKeyValueToFirstRightRow[rightJoinKeyValue] = rightRow;
            }
        }


        int leftIdxFloat = 0;
        int leftIdxString = 0;
        int leftIdxInt = 0;
        var rightJoinKeyDesc = right._columns[rightJoinKeyIndex];
        for (int newRow = 0; newRow < newRows; ++newRow)
        {
            int newIdxFloat = newRow * newTensorIndexToColumnCount[FLOAT_TYPE_IDX];
            for (int i = 0; i < leftTensorIndexToColumnCount[FLOAT_TYPE_IDX]; ++i)
            {
                newFloatContent[newIdxFloat++] = leftFloatContent[leftIdxFloat++];
            }
            int newIdxString = newRow * newTensorIndexToColumnCount[STRING_TYPE_IDX];
            for (int i = 0; i < leftTensorIndexToColumnCount[STRING_TYPE_IDX]; ++i)
            {
                newStringContent[newIdxString++] = leftStringContent[leftIdxString++];
            }
            int newIdxInt = newRow * newTensorIndexToColumnCount[INT_TYPE_IDX];
            for (int i = 0; i < leftTensorIndexToColumnCount[INT_TYPE_IDX]; ++i)
            {
                newIntContent[newIdxInt++] = leftIntContent[leftIdxInt++];
            }

            var leftIdxValue = ExtractValue(newRow, left._columns[leftJoinKeyIndex], leftFloatContent, leftStringContent, leftIntContent) ?? "";
            if (rightJoinKeyValueToFirstRightRow.TryGetValue(leftIdxValue, out var rightRow))
            {
                int rightIdxFloat = rightRow * rightTensorIndexToColumnCount[FLOAT_TYPE_IDX];
                for (int i = 0; i < rightTensorIndexToColumnCount[FLOAT_TYPE_IDX]; ++i)
                {
                    if (rightJoinKeyDesc.Item2 == FLOAT_TYPE_IDX && rightJoinKeyDesc.Item3 == i)
                    {
                        continue; //We do not add again the joinKey;
                    }
                    newFloatContent[newIdxFloat++] = rightFloatContent[rightIdxFloat+i];
                }
                int rightIdxString = rightRow * rightTensorIndexToColumnCount[STRING_TYPE_IDX];
                for (int i = 0; i < rightTensorIndexToColumnCount[STRING_TYPE_IDX]; ++i)
                {
                    if (rightJoinKeyDesc.Item2 == STRING_TYPE_IDX && rightJoinKeyDesc.Item3 == i)
                    {
                        continue; //We do not add again the joinKey;
                    }
                    newStringContent[newIdxString++] = rightStringContent[rightIdxString+i];
                }
                int rightIdxInt = rightRow * rightTensorIndexToColumnCount[INT_TYPE_IDX];
                for (int i = 0; i < rightTensorIndexToColumnCount[INT_TYPE_IDX]; ++i)
                {
                    if (rightJoinKeyDesc.Item2 == INT_TYPE_IDX && rightJoinKeyDesc.Item3 == i)
                    {
                        continue; //We do not add again the joinKey;
                    }
                    newIntContent[newIdxInt++] = rightIntContent[rightIdxInt+i];
                }
            }
        }
        var newColumnsDesc = left._columns.ToList();
        var tensorIndexToRightColumnIndex = (int[])leftTensorIndexToColumnCount.Clone();
        foreach (var rightColumnDesc in right._columns)
        {
            if (rightColumnDesc.Item1 == joinKey)
            {
                continue;
            }
            var newTuple = Tuple.Create(rightColumnDesc.Item1, rightColumnDesc.Item2, tensorIndexToRightColumnIndex[rightColumnDesc.Item2]++);
            newColumnsDesc.Add(newTuple);
        }

        return new DataFrame(
            newColumnsDesc,
            CpuTensor<float>.New(newFloatContent, newTensorIndexToColumnCount[FLOAT_TYPE_IDX]), 
            CpuTensor<string>.New(newStringContent, newTensorIndexToColumnCount[STRING_TYPE_IDX]), 
            CpuTensor<int>.New(newIntContent, newTensorIndexToColumnCount[INT_TYPE_IDX]));
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
        var colDesc = _columns.FirstOrDefault(c => c.Item1 == columnName);
        if (colDesc == null)
        {
            throw new ArgumentException($"invalid column name {columnName}");
        }

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


    public static DataFrame read_csv(string path, bool hasHeader = true, Func<string, Type> columnNameToType = null)
    {
        if (!File.Exists(path))
        {
            throw new ArgumentException($"invalid CSV file {path}");
        }
        List<string> columnsNames = null;
        int[] columnIndexToTensorIndex = null;
        Dictionary<int, int> tensorIndexToCount = new();
        List<Tuple<string, int, int>> newColumns = new();
        if (columnNameToType == null)
        {
            columnNameToType = _ => typeof(string);
        }
        List<float> floats = new();
        List<string> strings = new();
        List<int> ints = new();


        List<string[]> allLines = null;
        lock (DataSet.Lock_to_csv)
        {
            allLines = Utils.ReadCsv(path).ToList();
        }
        foreach (var lineContent in allLines)
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
                columnIndexToTensorIndex = columnsNames.Select(i => TypeToTensorIndex[columnNameToType(i)]).ToArray();
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
    private object ExtractValue(int row, Tuple<string, int, int> colDesc, ReadOnlySpan<float> floatContent, ReadOnlySpan<string> stringContent, ReadOnlySpan<int> intContent)
    {
        switch (colDesc.Item2)
        {
            case FLOAT_TYPE_IDX: return floatContent[colDesc.Item3 + row * _floatTensor.Shape[1]];
            case STRING_TYPE_IDX: return stringContent[colDesc.Item3 + row * _stringTensor.Shape[1]];
            default: return intContent[colDesc.Item3 + row * _intTensor.Shape[1]];
        }
    }
    public bool IsFloatDataFrame => _floatTensor != null && _stringTensor == null && _intTensor == null;
    public bool IsStringDataFrame => _floatTensor == null && _stringTensor != null && _intTensor == null;
    public bool IsIntDataFrame => _floatTensor == null && _stringTensor == null && _intTensor != null;
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
