using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.Data;

namespace SharpNet.CPU
{
    public static class CpuTensorActivationFunctions
    {
        #region Softmax
        public static void Softmax<T>(CpuTensor<T> X, Tensor Y) where T : struct
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> {X, Y}));
            var batchSize = X.Shape[0];
            var xContent = X.AsFloatCpuContent;
            var yContent = Y.AsFloatCpuContent;
            for (int row = 0; row < batchSize; ++row)
            {
                int start = row * X.MultDim0;
                int end = start + X.MultDim0;
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
        public static void SoftmaxGradient(Tensor y, Tensor dy, Tensor dx)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { y, dy, dx }));
            var yContent = y.AsFloatCpuContent;
            var dyContent = dy.AsFloatCpuContent;
            var dxContent = dx.AsFloatCpuContent;
            for (int i = 0; i < dx.Count; ++i)
            {
                var yi = yContent[i];
                var dyi = dyContent[i];
                dxContent[i] = (Math.Abs(dyi - 1.0) < 1e-6) ? (yi * (1 - yi)) : (-yi * dyi);
            }
        }
        #endregion

        #region Swish
        public static void Swish<T>(CpuTensor<T> X, Tensor Y) where T : struct
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { X, Y }));
            X.AsFloatCpu.Map(x => (float)(x / (1 + Math.Exp(-x))), Y.AsFloatCpu);
        }
        public static void SwishGradient(Tensor Y, Tensor dY, Tensor X, Tensor dX)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { Y, dY, dX }));
            dX.AsFloatCpu.BuildEntirelyFromInput(Y, dY, X, (y, dy, x) =>
            {
                // y = x * sigmoid(x)
                float sigmoid_x = (Math.Abs(x) < 0.0001f) ? 0.5f : y / x;
                return dy * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x));
            });
        }
        #endregion

        #region Tanh
        public static void Tanh<T>(CpuTensor<T> X, Tensor Y) where T : struct
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> {X, Y}));
            //X.AsFloatCpu.Map(x => (float) (1.7159 * Math.Tanh(0.66666667 * x)), Y.AsFloatCpu);
            X.AsFloatCpu.Map(x=> (float)Math.Tanh(x), Y.AsFloatCpu);
        }
        public static void TanhGradient(Tensor Y, Tensor dY, Tensor dX)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { Y, dY, dX }));
            dX.AsFloatCpu.BuildEntirelyFromInput(Y, dY, (y, dy) => dy * (1 - y*y));
        }
        #endregion

        #region Relu
        public static void Relu<T>(CpuTensor<T> X, Tensor Y) where T : struct
        {
            X.AsFloatCpu.Map(x => Math.Max(0, x), Y.AsFloatCpu);
        }
        public static void ReluGradient(Tensor dY, Tensor X, Tensor dX)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { dY, X, dX }));
            dX.AsFloatCpu.BuildEntirelyFromInput(dY, X, (dy, x) => (x >= 0f ? dy : 0f));
        }
        #endregion

        #region Elu
        public static void Elu<T>(CpuTensor<T> X, Tensor Y, double alpha) where T : struct
        {
            X.AsFloatCpu.Map(x => (x >= 0) ? x : (float)(alpha * (Math.Exp(x) - 1)), Y.AsFloatCpu);
        }
        public static void EluGradient(Tensor Y, Tensor dY, Tensor X, Tensor dX, float alpha)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { dY, X, dX }));
            dX.AsFloatCpu.BuildEntirelyFromInput(Y, dY, X, (y, dy, x) => (x >= 0.0 ? dy : dy * (y + alpha)));
        }
        #endregion

        #region Sigmoid
        public static void Sigmoid<T>(CpuTensor<T> X, Tensor Y) where T : struct
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> {X, Y}));
            X.AsFloatCpu.Map(x => (float) (1 / (1 + Math.Exp(-x))), Y.AsFloatCpu);
        }
        public static void SigmoidGradient(Tensor Y, Tensor dY, Tensor dX)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { Y, dY, dX }));
            dX.AsFloatCpu.BuildEntirelyFromInput(Y, dY, (y, dy) => dy * y * (1f - y));
        }
        #endregion
    }
}
