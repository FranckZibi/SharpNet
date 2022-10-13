using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public abstract class DataFrame
    {
        #region private and protected fields
        #endregion

        #region public fields
        public string[] ColumnNames { get; }
        private Type[] Dtypes { get; }
        public abstract int[] Shape { get; }
        public CpuTensor<float> FloatCpuTensor() => ((DataFrameT<float>)this).Tensor;

        #endregion

        #region Constructors

        protected DataFrame(IList<string> columnsNames, IList<Type> dtypes)
        {
            ColumnNames = columnsNames.ToArray();
            Dtypes = dtypes.ToArray();
            if (Dtypes.Length != columnsNames.Count())
            {
                var errorMsg = $"invalid Dtypes length {Dtypes.Length}, should be {ColumnNames.Length}";
                throw new Exception(errorMsg);

            }
        }
        #endregion


        public override string ToString()
        {
            return "(" + string.Join(", ", Shape) + ")";
        }

        public static string Float2String(float f) => f.ToString(CultureInfo.InvariantCulture);

        public abstract DataFrame Drop(IList<string> columnsToDrop);
        public abstract DataFrame Keep(IList<string> columnsToKeep);
        public abstract (DataFrame first, DataFrame second) Split(IList<string> columnsForSecondDataFrame);

        public abstract void to_csv(string path, string sep = ",", bool addHeader = false, int? index = null);

        public List<int> ColumnNamesToIndexes(IEnumerable<string> columnNames)
        {
            var indexes = new List<int>();
            foreach (var f in columnNames)
            {
                int idx = Array.IndexOf(ColumnNames, f);
                if (idx < 0)
                {
                    throw new Exception($"Invalid {nameof(ColumnNames)} name {f}");
                }
                indexes.Add(idx);
            }

            return indexes;
        }

        public static DataFrame New(CpuTensor<float> tensor, IList<string> columnsNames)
        {
            return new DataFrameT<float>(tensor, columnsNames, Float2String);
        }

        public static DataFrameT<float> LoadFloatDataFrame(string path, bool hasHeader)
        {
            return DataFrameT<float>.Load(path, hasHeader, float.Parse, Float2String);
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
            var mergedTensor = CpuTensor<float>.MergeHorizontally(left.FloatCpuTensor(), right.FloatCpuTensor());
            var mergedColumnNames = Utils.Join(left.ColumnNames, right.ColumnNames);
            return New(mergedTensor, mergedColumnNames);
        }

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
            Debug.Assert(top.ColumnNames.SequenceEqual(bottom.ColumnNames));
            var mergedTensor = CpuTensor<float>.MergeVertically(top.FloatCpuTensor(), bottom.FloatCpuTensor());
            return New(mergedTensor, top.ColumnNames);
        }

    }
}
