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

        protected DataFrame(IEnumerable<string> featureNames, IEnumerable<string> categoricalFeatures, IEnumerable<Type> dtypes)
        {
            FeatureNames = featureNames.ToArray();
            CategoricalFeatures = categoricalFeatures.ToArray();
            Dtypes = dtypes.ToArray();
        }
        #endregion

        public static string Float2String(float f) => f.ToString(CultureInfo.InvariantCulture);


        public static bool SameShape(IList<DataFrame> tensors)
        {
            return tensors.All(t => t.Shape.SequenceEqual(tensors[0].Shape));
        }
        public abstract void Save(string path, int? index = null);

        protected List<int> FeatureNameToIndexes(IEnumerable<string> featureNames)
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

        public static DataFrameT<float> New(CpuTensor<float> tensor, IEnumerable<string> featureNames, IEnumerable<string> categoricalFeatures)
        {
            return new DataFrameT<float>(tensor, featureNames, categoricalFeatures, Float2String);
        }

    }
}
