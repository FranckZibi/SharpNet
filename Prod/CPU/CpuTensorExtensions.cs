using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.Data;

namespace SharpNet.CPU
{
    public static class CpuTensorExtensions
    {
        public static void Softmax<T>(this CpuTensor<T> x, Tensor y) where T : struct
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> {x, y}));
            var batchSize = x.Shape[0];
            var xContent = x.AsFloatCpuContent;
            var yContent = y.AsFloatCpuContent;
            for (int row = 0; row < batchSize; ++row)
            {
                int start = row * x.MultDim0;
                int end = start + x.MultDim0;
                var rowMax = float.MinValue;
                for (int i = start; i < end; ++i)
                {
                    rowMax = Math.Max(rowMax, xContent[i]);
                }
                var rowExpSum = 0f;
                for (int i = start; i < end; ++i)
                {
                    var tmp = (float)Math.Exp(xContent[i] - rowMax);
                    rowExpSum += tmp;
                    yContent[i] = tmp;
                }
                for (int i = start; i < end; ++i)
                {
                    yContent[i] /= rowExpSum;
                }
            }
        }
        public static void Tanh<T>(this CpuTensor<T> x, Tensor y) where T : struct
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> {x, y}));
            //x.AsFloatCpu.Map(v => (float) (1.7159 * Math.Tanh(0.66666667 * v)), y.AsFloatCpu);
            x.AsFloatCpu.Map(v=> (float)Math.Tanh(v), y.AsFloatCpu);
        }
        public static void Relu<T>(this CpuTensor<T> x, Tensor y) where T : struct
        {
            x.AsFloatCpu.Map(v => Math.Max(0, v), y.AsFloatCpu);
        }
        public static void Elu<T>(this CpuTensor<T> x, Tensor y, double alpha) where T : struct
        {
            x.AsFloatCpu.Map(v => (v >= 0) ? v : (float)(alpha * (Math.Exp(v) - 1)), y.AsFloatCpu);
        }
        public static void Sigmoid<T>(this CpuTensor<T> x, Tensor y) where T : struct
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> {x, y}));
            x.AsFloatCpu.Map(v => (float) (1 / (1 + Math.Exp(-v))), y.AsFloatCpu);
        }
        public static CpuTensor<float> ToSinglePrecision<T>(this CpuTensor<T> x) where T : struct
        {
            return x as CpuTensor<float>;
        }
        public static CpuTensor<TY> ToCategorical<TX, TY>(this CpuTensor<TX> y, TY one, out IDictionary<TX, int> categoryToIndex) where TX : struct where TY : struct
        {
            Debug.Assert(y.MultDim0 == 1);
            var batchSize = y.Shape[0];

            categoryToIndex = new Dictionary<TX, int>();
            var distinctCategories = new HashSet<TX>(y.Content).ToList();
            distinctCategories.Sort();
            for (int i = 0; i < distinctCategories.Count; ++i)
            {
                categoryToIndex[distinctCategories[i]] = i;
            }
            var newShape = new[] { batchSize, distinctCategories.Count };
            var newY = new CpuTensor<TY>(newShape, y.Description);
            for (int n = 0; n < batchSize; ++n)
            {
                newY.Set(n, categoryToIndex[y.Get(n, 0)], one);
            }
            return newY;
        }
    }
}
