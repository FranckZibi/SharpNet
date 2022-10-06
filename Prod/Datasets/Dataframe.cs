using System;
using System.Collections.Generic;
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
        public string[] FeatureNames { get; }
        public string[] CategoricalFeatures{ get; }
        private Type[] Dtypes { get; }
        public abstract int[] Shape { get; }
        public CpuTensor<float> FloatCpuTensor() => ((DataFrameT<float>)this).Tensor;

        #endregion

        #region Constructors

        protected DataFrame(IList<string> featureNames, IList<string> categoricalFeatures, IList<Type> dtypes)
        {
            FeatureNames = featureNames.ToArray();
            CategoricalFeatures = categoricalFeatures.ToArray();
            Dtypes = dtypes.ToArray();

            foreach (var f in CategoricalFeatures)
            {
                if (!FeatureNames.Contains(f))
                {
                    var errorMsg = $"invalid categorical feature {f}, not among {string.Join(' ', FeatureNames)}";
                    throw new Exception(errorMsg);
                }
            }
            if (Dtypes.Length != featureNames.Count())
            {
                var errorMsg = $"invalid Dtypes length {Dtypes.Length}, should be {FeatureNames.Length}";
                throw new Exception(errorMsg);

            }

        }
        #endregion

        public static string Float2String(float f) => f.ToString(CultureInfo.InvariantCulture);

        public abstract DataFrame Drop(IList<string> featuresToDrop);
        public abstract DataFrame Keep(IList<string> featuresToKeep);
        public abstract (DataFrame first, DataFrame second) Split(IList<string> featuresForSecondDataFrame);

        public static bool SameShape(IList<DataFrame> tensors)
        {
            return tensors.All(t => t.Shape.SequenceEqual(tensors[0].Shape));
        }
        public abstract void to_csv(string path, string sep = ",", bool addHeader = false, int? index = null);

        protected List<int> FeatureNamesToIndexes(IEnumerable<string> featureNames)
        {
            var indexes = new List<int>();
            foreach (var f in featureNames)
            {
                int idx = Array.IndexOf(FeatureNames, f);
                if (idx < 0)
                {
                    throw new Exception($"Invalid feature name {f}");
                }
                indexes.Add(idx);
            }

            return indexes;
        }

        public static DataFrame New(CpuTensor<float> tensor, IList<string> featureNames, IList<string> categoricalFeatures)
        {
            return new DataFrameT<float>(tensor, featureNames, categoricalFeatures, Float2String);
        }

        public static DataFrameT<float> LoadFloatDataFrame(string path, bool hasHeader)
        {
            return DataFrameT<float>.Load(path, hasHeader, float.Parse, null, Float2String);
        }

        public static DataFrame MergeHorizontally(DataFrame left, DataFrame right)
        {
            var mergedTensor = CpuTensor<float>.MergeHorizontally(left.FloatCpuTensor(), right.FloatCpuTensor());
            var mergedFeatureNames = Utils.Join(left.FeatureNames, right.FeatureNames);
            var mergedCategoricalFeatures = Utils.Join(left.CategoricalFeatures, right.CategoricalFeatures);
            return New(mergedTensor, mergedFeatureNames, mergedCategoricalFeatures);
        }
    }
}
