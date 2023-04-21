using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using SharpNet.Data;

namespace SharpNet.CPU
{
    public static unsafe class CpuTensorActivationFunctions
    {
        #region Softmax

        public static void SoftmaxLastDimension<T>(CpuTensor<T> X, Tensor Y)
        {
            Softmax(X.Reshape(-1, X.Shape[^1]).AsFloatCpu, Y.Reshape(-1, Y.Shape[^1]));
        }

        public static void Softmax<T>(CpuTensor<T> X, Tensor Y)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> {X, Y}));
            var batchSize = X.Shape[0];
            var xContent = X.AsReadonlyFloatCpuSpan;
            var yContent = Y.AsFloatCpuSpan;
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

        public static void SoftmaxWithHierarchy<T>(CpuTensor<T> X, Tensor Y, Tensor activationParameter)
        {
            X.CopyTo(Y.AsFloatCpu);
            var activationParameterPointer = (float*)activationParameter.Pointer;
            var yPointer = (float*) Y.Pointer;
            int colSize = Y.MultDim0;
            Parallel.For(0, Y.Shape[0], m =>{int idx = 0;SoftmaxWithHierarchy(activationParameterPointer, yPointer+m*colSize, colSize, &idx);});
        }

        private static void SoftmaxWithHierarchy(float* activationParameter, float* y, int endIndexExcluded, int *pNexIndexToCheck)
        {
            float param = activationParameter[*pNexIndexToCheck];
            y[*pNexIndexToCheck] = param;
            int subCategoriesCount = ExtractCount(param);
            *pNexIndexToCheck += 1;
            int[] indexesProba = new int[subCategoriesCount];
            float maxProba = -1e9f;
            bool probaFound = false;

            for(int subCategoriesFound = 0;subCategoriesFound < subCategoriesCount; ++subCategoriesFound)
            {
                float expectedProba = activationParameter[*pNexIndexToCheck];
                if (IsProba(expectedProba))
                {
                    maxProba = fmaxf(maxProba, y[*pNexIndexToCheck]);
                    indexesProba[subCategoriesFound] = *pNexIndexToCheck;
                    probaFound = true;
                    * pNexIndexToCheck += 1;
                    if (*pNexIndexToCheck < endIndexExcluded && IsCountAssociateWithAboveProba(activationParameter[*pNexIndexToCheck]))
                    {
                        SoftmaxWithHierarchy(activationParameter, y, endIndexExcluded, pNexIndexToCheck);
                    }
                }
                else
                {
                   SoftmaxWithHierarchy(activationParameter, y, endIndexExcluded,  pNexIndexToCheck);
                }
            }

            if (probaFound)
            {
                float sumExp = 0.0f;
                for (int i = 0; i < subCategoriesCount; ++i)
                {
                    int idx = indexesProba[i];
                    float tmp = expf(y[idx] - maxProba);
                    sumExp += tmp;
                    y[idx] = tmp;
                }
                for (int i = 0; i < subCategoriesCount; ++i)
                {
                    y[indexesProba[i]] /= sumExp;
                }
            }
        }

        public static void SoftmaxGradientLastDimension(Tensor y, Tensor dy, Tensor dx)
        {
            SoftmaxGradient(y.Reshape(-1, y.Shape[^1]), dy.Reshape(-1, dy.Shape[^1]), dx.Reshape(-1, dx.Shape[^1]));
        }

        public static void SoftmaxGradient(Tensor y, Tensor dy, Tensor dx)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { y, dy, dx }));
            var yContent = y.AsFloatCpuSpan;
            var dyContent = dy.AsFloatCpuSpan;
            var dxContent = dx.AsFloatCpuSpan;
            for (int i = 0; i < dx.Count; ++i)
            {
                var yi = yContent[i];
                var dyi = dyContent[i];
                dxContent[i] = (MathF.Abs(dyi - 1.0f) < 1e-6) ? (yi * (1 - yi)) : (-yi * dyi);
            }
        }
        public static void SoftmaxGradientWitHierarchy(Tensor y, Tensor dy, Tensor dx, Tensor activationParameter)
        {
            var activationParameterPointer = (float*)activationParameter.Pointer;
            var yPointer = (float*)y.Pointer;
            var dyPointer = (float*)dy.Pointer;
            var dxPointer = (float*)dx.Pointer;
            int colSize = dx.MultDim0;
            Parallel.For(0, dx.Shape[0], m => SoftmaxGradientWitHierarchy(activationParameterPointer, yPointer+m*colSize, dyPointer + m * colSize, dxPointer + m * colSize, colSize));
        }
        private static void SoftmaxGradientWitHierarchy(float* activationParameter, float* y, float* dy, float* dx, int endIndexExcluded)
        {
            for (int i = 0; i < endIndexExcluded; ++i)
            {
                float expectedProba = activationParameter[i];
                if (IsProba(expectedProba))
                {
                    float dyi = dy[i];
                    float yi = y[i];
                    dx[i] = (fabsf(dyi - 1.0f) < 1e-6) ? (yi * (1 - yi)) : (-yi * dyi);
                }
                else
                {
                    dx[i] = expectedProba;
                }
            }
        }

        #endregion

        #region Swish
        public static void Swish<T>(CpuTensor<T> X, Tensor Y)
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
                float sigmoid_x = (MathF.Abs(x) < 0.0001f) ? 0.5f : y / x;
                return dy * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x));
            });
        }
        #endregion

        #region Ln
        public static void Ln<T>(CpuTensor<T> X, Tensor Y)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { X, Y }));
            X.AsFloatCpu.Map(x => x <= 0 ? -100.0f : MathF.Log(x), Y.AsFloatCpu);
        }
        public static void LnGradient(Tensor dY, Tensor X, Tensor dX)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> {dY, X, dX}));
            dX.AsFloatCpu.BuildEntirelyFromInput(X, dY, (x, dy) => dy / x);
        }
        #endregion

        #region Tanh
        public static void Tanh<T>(CpuTensor<T> X, Tensor Y)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> {X, Y}));
            //X.AsFloatCpu.Map(x => (float) (1.7159 * Math.Tanh(0.66666667 * x)), Y.AsFloatCpu);
            X.AsFloatCpu.Map(MathF.Tanh, Y.AsFloatCpu);
        }
        public static void TanhGradient(Tensor Y, Tensor dY, Tensor dX)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { Y, dY, dX }));
            dX.AsFloatCpu.BuildEntirelyFromInput(Y, dY, (y, dy) => dy * (1 - y*y));
        }
        #endregion

        #region Relu
        public static void Relu<T>(CpuTensor<T> X, Tensor Y)
        {
            X.AsFloatCpu.Map(x => Math.Max(0, x), Y.AsFloatCpu);
        }
        public static void ReluGradient(Tensor Y, Tensor dY, Tensor dX)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { dY, Y, dX }));
            dX.AsFloatCpu.BuildEntirelyFromInput(dY, Y, (dy, y) => (y > 0f ? dy : 0f));
        }
        #endregion

        #region Leaky Relu
        public static void LeakyRelu<T>(CpuTensor<T> X, Tensor Y, double alpha)
        {
            X.AsFloatCpu.Map(x => (x>=0)?x:((float)alpha)*x, Y.AsFloatCpu);
        }
        public static void LeakyReluGradient(Tensor Y, Tensor dY, Tensor dX, double alpha)
        {
            Debug.Assert(alpha>=0);
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { dY, Y, dX }));
            dX.AsFloatCpu.BuildEntirelyFromInput(dY, Y, (dy, y) => (y >= 0f ? dy : ((float)alpha)*dy));
        }
        #endregion

        #region Elu
        public static void Elu<T>(CpuTensor<T> X, Tensor Y, double alpha)
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
        public static void Sigmoid<T>(CpuTensor<T> X, Tensor Y)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> {X, Y}));
            X.AsFloatCpu.Map(Utils.Sigmoid, Y.AsFloatCpu);
        }
        public static void SigmoidGradient(Tensor Y, Tensor dY, Tensor dX)
        {
            Debug.Assert(Tensor.AreCompatible(new List<Tensor> { Y, dY, dX }));
            dX.AsFloatCpu.BuildEntirelyFromInput(Y, dY, (y, dy) => dy * y * (1f - y));
        }
        #endregion


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsCountAssociateWithAboveProba(float f) { return f > 5.0f && ((int)(f + 0.1f)) % 10 == 1; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool IsProba(float f) { return fabsf(f) < 5.0f; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int ExtractCount(float f) { return (int)(f + 0.5f) / 10; }
        #region CUDA methods
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        // ReSharper disable once InconsistentNaming
        private static float fabsf(float f) { return MathF.Abs(f); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        // ReSharper disable once InconsistentNaming
        private static float expf(float f) { return MathF.Exp(f); }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        // ReSharper disable once InconsistentNaming
        private static float fmaxf(float a, float b) { return Math.Max(a, b); }
        #endregion
    }
}
