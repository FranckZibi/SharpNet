using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using SharpNet.CPU;

namespace SharpNet.LightGBM
{
    public class Dataframe
    {
        public CpuTensor<float> Tensor { get; }
        public string[] FeatureNames { get; }
        public string Name { get; }


        public int[] Shape => Tensor.Shape;

        public Dataframe(CpuTensor<float> tensor, IEnumerable<string> featureNames, string name)
        {
            Debug.Assert(tensor.Shape.Length == 2); //Dataframe works only with matrices
            Tensor = tensor;
            FeatureNames = featureNames.ToArray();
            Name = name;
        }



        public List<int> FeatureNameToIndexes(IEnumerable<string> featureNames)
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

        public Dataframe Drop(IList<string> featuresToDrop)
        {
            var indexes = new HashSet<int>(FeatureNameToIndexes(featuresToDrop));
            
            var newData = Tensor.DropColumns(indexes);

            var newFeatures = FeatureNames.ToList();
            newFeatures.RemoveAll(featuresToDrop.Contains);
            
            return new Dataframe(newData, newFeatures.ToArray(), Name+"_drop_"+string.Join("_", featuresToDrop));
        }


        public Dataframe Keep(IList<string> featuresToKeep)
        {
            var newData = Tensor.KeepColumns(FeatureNameToIndexes(featuresToKeep));

            var newFeatures = FeatureNames.ToList();
            newFeatures.RemoveAll(f => !featuresToKeep.Contains(f));

            return new Dataframe(newData, newFeatures, Name + "_keep_" + string.Join("_", featuresToKeep));
        }



        public void Save(string path)
        {
            const char separator = ',';
            var sb = new StringBuilder();
            sb.Append(string.Join(separator, FeatureNames));
            var dataAsSpan = Tensor.SpanContent;
            for (int i = 0; i < dataAsSpan.Length; ++i)
            {
                if (i % Tensor.Shape[1] == 0)
                {
                    sb.Append(Environment.NewLine);
                }
                else
                {
                    sb.Append(separator);
                }
                sb.Append(dataAsSpan[i].ToString(CultureInfo.InvariantCulture));
            }
            System.IO.File.WriteAllText(path, sb.ToString());
        }

        public static Dataframe Load(string path, bool hasHeader, char separator)
        {
            var content=  new List<List<float>>();
            var featureNames = new List<string>();

            foreach (var l in System.IO.File.ReadAllLines(path))
            {
                var lineContent = l.Split(separator);
                if (hasHeader && featureNames.Count == 0)
                {
                    featureNames = lineContent.ToList();
                    continue;
                }
                content.Add(lineContent.Select(float.Parse).ToList());
            }

            if (!hasHeader)
            {
                featureNames = Enumerable.Range(0, content[0].Count).Select(t => t.ToString()).ToList();
            }

            var data = new float[content.Count * featureNames.Count];
            int idx = 0;
            foreach(var t in content)
            foreach (var d in t)
            {
                data[idx++] = d;
            }
            var tensor = new CpuTensor<float>(new [] {content.Count, featureNames.Count}, data);

            return new Dataframe(tensor, featureNames, path);
        }

    }
}
