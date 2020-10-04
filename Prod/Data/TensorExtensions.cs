using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using SharpNet.CPU;

namespace SharpNet.Data
{
    public static class TensorExtensions
    {
        public static string ToNumpy(this Tensor t, int maxLength = 2000)
        {
            var sb = new StringBuilder();
            int idx = 0;
            var tContent = t.ContentAsFloatArray();

            NumpyArrayHelper(t, tContent, 0, ref idx, sb);
            var res = sb.ToString();
            if (res.Length > maxLength)
            {
                res = res.Substring(0, maxLength / 2) + " .../... " + res.Substring(res.Length - maxLength / 2);
            }

            return res; 
        }

        public static Tensor FromNumpyArray(string s)
        {
            //we extract the shape of the numpy array
            int currentDepth = -1;
            var shape = new List<int>();
            foreach (var c in s)
            {
                if (c == '[')
                {
                    ++currentDepth;
                    if (currentDepth > shape.Count - 1)
                    {
                        shape.Add(1);
                    }
                    shape[currentDepth] = 1;
                }
                else if (c == ']')
                {
                    --currentDepth;
                }
                else if ((c == ',') && (currentDepth >= 0))
                {
                    ++shape[currentDepth];
                }
            }

            bool isInteger = s.Contains("numpy.int");
            if (s.Contains("numpy.double"))
            {
                throw new NotImplementedException();
            }

            s = s.ToLowerInvariant().Replace("numpy.array", "").Replace("numpy.float", "").Replace("numpy.int32", "")
                .Replace("numpy.double", "").Replace("(", " ").Replace(")", " ").Replace("[", " ").Replace("]", " ")
                .Replace("array", " ")
                .Trim(',', ' ');

            if (isInteger)
            {
                return new CpuTensor<int>(shape.ToArray(), s.Split(',').Select(int.Parse).ToArray());
            }
            //float
            return new CpuTensor<float>(shape.ToArray(), s.Split(',').Select(float.Parse).ToArray());
        }

        private static void NumpyArrayHelper(Tensor t, ReadOnlySpan<float> tContent, int currentDepth, ref int idx, StringBuilder sb)
        {
            //if (currentDepth == 0)
            //{
            //    sb.Append("numpy.array(");
            //}
            sb.Append("[");
            for (int indexInCurrentDepth = 0; indexInCurrentDepth < t.Shape[currentDepth]; ++indexInCurrentDepth)
            {
                if (currentDepth == t.Shape.Length - 1)
                {
                    sb.Append(tContent[idx++].ToString("G9", CultureInfo.InvariantCulture));
                }
                else
                {
                    NumpyArrayHelper(t, tContent, currentDepth + 1, ref idx, sb);
                }
                if (indexInCurrentDepth != t.Shape[currentDepth] - 1)
                {
                    sb.Append(",");
                }
            }
            sb.Append("]");
            //if (currentDepth == 0)
            //{
            //    sb.Append(", numpy.float)");
            //}
        }
        public static bool Equals(this Tensor a, Tensor b, double epsilon, string id, ref string errors)
        {
            if (a == b)
            {
                return true;
            }
            if (b == null)
            {
                errors += id + ":a: b is null" + Environment.NewLine;
                return false;
            }
            if (a == null)
            {
                errors += id + ":b: a is null" + Environment.NewLine;
                return false;
            }
            id += ":a";
            if (!a.SameShape(b))
            {
                errors += id + ":Shape: " + string.Join(",", a.Shape) + " != " + string.Join(",", b.Shape) + Environment.NewLine;
                return false;
            }
            if (a.GetType() != b.GetType())
            {
                errors += id + ":Type: " + a.GetType() + " != " + b.GetType() + Environment.NewLine;
                return false;
            }
            var contentA = a.ContentAsFloatArray();
            var contentB = b.ContentAsFloatArray();
            int nbFoundDifferences = 0;
            for (int i = 0; i < a.Count; ++i)
            {
                if (!Utils.Equals(contentA[i], contentB[i], epsilon, id + "[" + i + "]", ref errors))
                {
                    ++nbFoundDifferences;
                    if (nbFoundDifferences >= 10)
                    {
                        return false;
                    }
                }
            }
            if (nbFoundDifferences != 0)
            {
                return false;
            }
            return true;
        }
    }
}
