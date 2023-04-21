using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Text;
using SharpNet.CPU;

namespace SharpNet.Data
{
    public static class TensorExtensions
    {
        public static string ToShapeAndNumpy(this Tensor t, int maxLength = 2000)
        {
            return Tensor.ShapeToString(t.Shape) + Environment.NewLine + t.ToNumpy(maxLength);
        }


        public static CpuTensor<float> DeduceRowMean(this CpuTensor<float> tensor)
        {
            Debug.Assert(tensor.Shape.Length == 2);
            var rows = tensor.Shape[0];
            var columns = tensor.Shape[1];
            var resultTensor = (CpuTensor<float>)tensor.Clone();
            var resultContent = resultTensor.Content.ToArray();

            for (int row = 0; row < rows; row++)
            {
                float sumForRow = 0.0f;
                for (int column = 0; column < columns; column++)
                {
                    int idx = column + row * columns;
                    sumForRow += resultContent[idx];
                }
                float meanForRow = sumForRow/rows;
                for (int column = 0; column < columns; column++)
                {
                    int idx = column + row * columns;
                    resultContent[idx] -= meanForRow;
                }
            }

            return resultTensor;
        }
        
        public static (CpuTensor<float> normalizedTensor, CpuTensor<float> mean, CpuTensor<float> variance) Normalize(this CpuTensor<float> toNormalize)
        {
            Debug.Assert(toNormalize.Shape.Length == 2);
            var rows = toNormalize.Shape[0];
            var columns = toNormalize.Shape[1];
            var mean = CpuTensor<float>.New(new float[columns], columns);
            var variance = CpuTensor<float>.New(new float[columns], columns);
            toNormalize.Compute_Column_Mean_Variance(mean, variance);
            var normalizedContent = toNormalize.Content.ToArray();
            var meanContent = mean.ReadonlyContent;
            var varianceContent = variance.ReadonlyContent;
            for (int row = 0; row < rows; row++)
            for (int column = 0; column < columns; column++)
            {
                var meanValue = meanContent[column];
                var varianceValue = varianceContent[column];
                int idx = column + row * columns;
                normalizedContent[idx] = varianceValue < 1e-5
                    ? 0
                    : (normalizedContent[idx] - meanValue) / MathF.Sqrt(varianceValue);
            }
            return (new CpuTensor<float>(toNormalize.Shape, normalizedContent), mean, variance);
        }



        // ReSharper disable once UnusedMember.Global
        public static string ToCsv(this CpuTensor<float> t, char separator, bool prefixWithRowIndex = false)
        {
            var sb = new StringBuilder();
            var tSpan = t.AsReadonlyFloatCpuSpan;
            int index = 0;
            if (t.Shape.Length != 2)
            {
                throw new ArgumentException($"can only create csv from matrix Tensor, not {t.Shape.Length} dimension tensor");
            }
            for (int row = 0; row < t.Shape[0]; ++row)
            {
                if (prefixWithRowIndex)
                {
                    sb.Append(row.ToString()+separator);
                }
                for (int col = 0; col < t.Shape[1]; ++col)
                {
                    if (col != 0)
                    {
                        sb.Append(separator);
                    }
                    sb.Append(tSpan[index++].ToString(CultureInfo.InvariantCulture));
                }
                if (row != t.Shape[0] - 1)
                {
                    sb.Append(Environment.NewLine);
                }
            }
            return sb.ToString();
        }

     
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
     
        public static double ComputeAccuracy(CpuTensor<float> y_true, CpuTensor<float> y_pred)
        {
            using var buffer = new CpuTensor<float>(new[] { y_true.Shape[0] });
            return y_true.ComputeAccuracy(y_pred, buffer);
        }


        public static bool SameFloatContent(Tensor a, Tensor b, double epsilon)
        {
            return SameFloatContent(a, b, epsilon, out _);
        }
        public static bool SameFloatContent(Tensor a, Tensor b, double epsilon, out string difference)
        {
            difference = "";
            if (!SameShapeContent(a, b, out difference))
            {
                return false;
            }
            if (a == null || b == null)
            {
                return true;
            }
            return Utils.SameContent(a.ContentAsFloatArray(), b.ContentAsFloatArray(), epsilon, out difference);
        }

        private static bool SameShapeContent(Tensor a, Tensor b, out string difference)
        {
            difference = "";
            if (a == null || b == null)
            {
                if (a == null && b == null)
                {
                    return true;
                }
                difference = $"only one of the 2 tensors is null";
                return false;
            }
            if (!a.SameShape(b))
            {
                difference = $"different shapes between first ({Tensor.ShapeToString(a.Shape)}) and second ({Tensor.ShapeToString(b.Shape)}) tensor";
                return false;
            }
            return true;
        }

        public static bool SameIntContent(CpuTensor<int> a, CpuTensor<int> b, out string difference)
        {
            difference = "";
            if (!SameShapeContent(a, b, out difference))
            {
                return false;
            }
            if (a == null || b == null)
            {
                return true;
            }
            var aContent = a.Content.ToArray();
            var bContent = b.Content.ToArray();
            for (int i = 0; i < aContent.Length; ++i)
            {
                if (aContent[i] != bContent[i])
                {
                    difference = $"different content at index {i} between first ({aContent[i]}) and second ({bContent[i]}) tensor";
                    return false;
                }
            }
            return true;
        }
        public static bool SameStringContent(CpuTensor<string> a, CpuTensor<string> b, out string difference)
        {
            difference = "";
            if (!SameShapeContent(a, b, out difference))
            {
                return false;
            }
            if (a == null || b == null)
            {
                return true;
            }
            var aContent = a.Content.ToArray();
            var bContent = b.Content.ToArray();
            for (int i = 0; i < aContent.Length; ++i)
            {
                if (aContent[i] != bContent[i])
                {
                    difference = $"different content at index {i} between first ({aContent[i]}) and second ({bContent[i]}) tensor";
                    return false;
                }
            }
            return true;
        }
    }
}
