using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNet.CPU;

namespace SharpNet.Data
{
    public static class TensorExtensions
    {
        public static string ToNumpy(this Tensor t)
        {
            var sb = new StringBuilder();
            int idx = 0;
            var tContent = t.ContentAsDoubleArray();

            NumpyArrayHelper(t, tContent, 0, ref idx, sb);
            return sb.ToString();
        }

        public static Tensor FromNumpyArray(string s, string description)
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
            bool isDouble = s.Contains("numpy.double");

            s = s.ToLowerInvariant().Replace("numpy.array", "").Replace("numpy.float", "").Replace("numpy.int32", "")
                .Replace("numpy.double", "").Replace("(", " ").Replace(")", " ").Replace("[", " ").Replace("]", " ")
                .Replace("array", " ")
                .Trim(',', ' ');

            if (isInteger)
            {
                return new CpuTensor<int>(shape.ToArray(), s.Split(',').Select(int.Parse).ToArray(), description);
            }
            if (isDouble)
            {
                return new CpuTensor<double>(shape.ToArray(), s.Split(',').Select(double.Parse).ToArray(), description);
            }
            //float
            return new CpuTensor<float>(shape.ToArray(), s.Split(',').Select(float.Parse).ToArray(), description);
        }

        private static void NumpyArrayHelper(Tensor t, double[] tContent, int currentDepth, ref int idx, StringBuilder sb)
        {
            if (currentDepth == 0)
            {
                sb.Append("numpy.array(");
            }
            sb.Append("[");
            for (int indexInCurrentDepth = 0; indexInCurrentDepth < t.Shape[currentDepth]; ++indexInCurrentDepth)
            {
                if (currentDepth == t.Shape.Length - 1)
                {
                    sb.Append(tContent[idx++]);
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
            if (currentDepth == 0)
            {
                sb.Append(t.UseSinglePrecision ? ", numpy.float)" : ", numpy.double)");
            }
        }
        public static bool Equals(this Tensor a, Tensor b, double epsilon, string id, ref string errors)
        {
            if (a == b)
            {
                return true;
            }
            if (b == null)
            {
                errors += id + ":" + a.Description + ": b is null" + Environment.NewLine;
                return false;
            }
            if (a == null)
            {
                errors += id + ":" + b.Description + ": a is null" + Environment.NewLine;
                return false;
            }
            if (!Equals(a.Description, b.Description))
            {
                errors += id + ":Description: " + a.Description + " != " + b.Description + Environment.NewLine;
                return false;
            }
            id += ":" + a.Description;
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
            var contentA = a.ContentAsDoubleArray();
            var contentB = b.ContentAsDoubleArray();
            int nbFoundDifferences = 0;
            for (int i = 0; i < contentA.Length; ++i)
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

        /*
        public static bool HasNan(this Tensor t)
        {
            if (t.UseDoublePrecision)
            {
                var content = t.ToCpu<double>().AsDoubleCpuContent;
                if (content.Any(double.IsNaN))
                {
                    return true;
                }
                return false;
            }
            else
            {
                var content = t.ToCpu<float>().AsFloatCpuContent;
                if (content.Any(float.IsNaN))
                {
                    return true;
                }
                return false;
            }
        }*/
    }
}
