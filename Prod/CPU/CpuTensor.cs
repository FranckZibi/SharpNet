﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;

namespace SharpNet.CPU
{
    public class CpuTensor<T> : Tensor
    {
        #region fields
        public T[] Content { get; private set; }
        private HostPinnedMemory<T> _hostPinnedMemory;
        #endregion

        public CpuTensor(int[] shape, T[] data, int typeSize, string description) : base(shape, typeSize, false, description)
        {
            Content = data ?? new T[Count];
            CapacityInBytes = (ulong)(Content.Length * TypeSize);
        }

        public CpuTensor(int[] shape, T[] data, string description) : this(shape, data, Marshal.SizeOf(typeof(T)), description)
        {
        }
        public CpuTensor(int[] shape, string description) : this(shape, null, description)
        {
        }

        /// <summary>
        /// pointer to (pinned) host memory (in CPU)
        /// </summary>
        public IntPtr HostPointer
        {
            get
            {
                if (_hostPinnedMemory == null)
                {
                    _hostPinnedMemory = new HostPinnedMemory<T>(Content);
                }
                return _hostPinnedMemory.Pointer;
            }
        }


        /// <summary>
        /// resize the current Cpu tensor to a different shape (both bigger or smaller)
        /// </summary>
        /// <param name="newShape"></param>
        public override void Reshape(int[] newShape)
        {
            if (ReallyNeededMemoryInBytesForShape(newShape) <= CapacityInBytes)
            {
                //smaller shape
                Shape = newShape;
            }
            else
            {
                //bigger shape
                _hostPinnedMemory?.Dispose();
                _hostPinnedMemory = null;
                Content = new T[Utils.Product(newShape)];
                CapacityInBytes = (ulong)(Content.Length * TypeSize);
                Shape = newShape;
            }
            RecomputeMultDim();
        }

        public CpuTensor<T> WithNewShape(int[] newShape)
        {
            if (Utils.Product(newShape) != Count)
            {
                throw new ArgumentException("invalid shape " + string.Join(",", newShape) + " for " + this+" : must have the same size");
            }
            return new CpuTensor<T>(newShape, Content, Description);
        }


        public T this[int i]
        {
            get => Content[i];
            set => Content[i] = value;
        }
        public CpuTensor<T> From_HNC_to_NCH()
        {
            var transformedShape = new[] { Shape[1], Shape[2], Shape[0] };
            var result = new CpuTensor<T>(transformedShape, Description);
            for (int n = 0; n < transformedShape[0]; ++n)
            {
                for (int c = 0; c < transformedShape[1]; ++c)
                {
                    for (int h = 0; h < transformedShape[2]; ++h)
                    {
                        result.Set(n, c, h, Get(h, n, c));
                    }
                }
            }
            return result;
        }

        public CpuTensor<T> ChangeAxis(int[] newAxis)
        {
            Debug.Assert(newAxis.Length == Dimension);
            Debug.Assert(newAxis.Min() == 0);
            Debug.Assert(newAxis.Max() == Dimension-1);

            var transformedShape = new int[Dimension];
            for (int i = 0; i < Dimension; ++i)
            {
                transformedShape[newAxis[i]] = Shape[i];
            }

            var result = new CpuTensor<T>(transformedShape, Description);

            var newIndexes =  new int[Dimension];
            for (int n = 0; n < Shape[0]; ++n)
            {
                newIndexes[newAxis[0]] = n;
                for (int h = 0; h < Shape[1]; ++h)
                {
                    newIndexes[newAxis[1]] = h;
                    for (int w = 0; w < Shape[2]; ++w)
                    {
                        newIndexes[newAxis[2]] = w;
                        for (int c = 0; c < Shape[3]; ++c)
                        {
                            newIndexes[newAxis[3]] = c;
                            result.Set(newIndexes[0], newIndexes[1], newIndexes[2], newIndexes[3], Get(n, h, w, c));
                        }
                    }
                }
            }

            return result;
        }
     
        public override void From_NCH_to_NH(Tensor tensor_NH, int channel)
        {
            Debug.Assert(Shape[0] == tensor_NH.Shape[0]);  //N
            Debug.Assert(Shape[2] == tensor_NH.Shape[1]);  //H
            Debug.Assert(channel < Shape[1]);
            var cpuTensor_NH = tensor_NH as CpuTensor<T>;
            Debug.Assert(cpuTensor_NH != null);
            for (int n = 0; n < tensor_NH.Shape[0]; ++n)
            {
                for (int h = 0; h < tensor_NH.Shape[1]; ++h)
                {
                    // ReSharper disable once PossibleNullReferenceException
                    cpuTensor_NH.Set(n, h, Get(n, channel, h));
                }
            }
        }
        public T Get(int n, int c)
        {
            return this[Idx(n, c)];
        }
        public T Get(int n, int c, int h)
        {
            Debug.Assert(Dimension == 3);
            return this[Idx(n, c, h)];
        }
        public T Get(int n, int c, int h, int w)
        {
            Debug.Assert(Dimension == 4);
            return this[Idx(n, c, h, w)];
        }
        public void Set(int n, int c, T t)
        {
            Debug.Assert(Dimension == 2);
            this[Idx(n, c)] = t;
        }
        // ReSharper disable once MemberCanBeProtected.Global
        public void Set(int n, int c, int h, T t)
        {
            Debug.Assert(Dimension == 3);
            this[Idx(n, c, h)] = t;
        }
        public void Set(int n, int c, int h, int w, T t)
        {
            Debug.Assert(Dimension == 4);
            this[Idx(n, c, h, w)] = t;
        }
        public void Map(Func<T, T> func, CpuTensor<T> result)
        {
            Debug.Assert(SameShape(result));
            for (int i = 0; i < Count; ++i)
            {
                result[i] = func(this[i]);
            }
        }

        /// <summary>
        /// Transform the 'this' tensor into another tensor by transforming:
        ///   each element 'val' of the  'this' tensor at position (m,c,h,w)
        /// into
        ///   the value returned by the method func(m,c,val)
        /// </summary>
        /// <typeparam name="TY"></typeparam>
        /// <param name="func"></param>
        /// <returns></returns>
        public CpuTensor<TY> Select<TY>(Func<int,int, T, TY> func) where TY : struct
        {
            var result = new CpuTensor<TY>(Shape, Description);
            Debug.Assert(SameShape(result));
            for (int m = 0; m < Shape[0]; ++m)
            {
                for (int c = 0; c < Shape[1]; ++c)
                {
                    int startIdx = Idx(m, c);
                    for (int idx = startIdx; idx < (startIdx + MultDim1); ++idx)
                    {
                        result[idx] = func(m, c, Content[idx]);
                    }
                }
            }
            return result;
        }
        public override Tensor Clone(GPUWrapper notUsed)
        {
            return new CpuTensor<T>((int[])Shape.Clone(), (T[])Content.Clone(), Description);
        }

        #region Tensor implementation
        public override void UpdateSGDOptimizer(double learningRate, double momentum, bool usenesterov, Tensor dW, Tensor velocity)
        {
            var W = this;
            var wContent = W.AsCpu<float>().Content;
            var dWContent = dW.AsCpu<float>().Content;
            var velocityContent = velocity.AsCpu<float>().Content;
            var learningRateFloat = (float) learningRate;
            var momentumFloat = (float)momentum;
            for (int i = 0; i < W.Count; ++i)
            {
                velocityContent[i] = (momentumFloat * velocityContent[i]) - (dWContent[i] * learningRateFloat);
                if (usenesterov)
                {
                    wContent[i] += momentumFloat * velocityContent[i] - (dWContent[i] * learningRateFloat);
                }
                else
                {
                    wContent[i] += velocityContent[i];
                }
            }
        }
        public override void BatchNormalization(Tensor y, Tensor bnScale, Tensor bnBias, double exponentialAverageFactor, Tensor resultRunningMean, Tensor resultRunningVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor resultSaveMean, Tensor resultSaveVariance, bool isTraining)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor>{x,y,bnScale,bnBias,resultRunningMean,resultRunningVariance,resultSaveMean,resultSaveVariance}));
            Debug.Assert(x.SameShape(y));
            Debug.Assert(bnScale.SameShape(bnBias, resultRunningMean, resultRunningVariance, resultSaveMean, resultSaveVariance));
            bool is1C11Shape = bnBias.Count == bnBias.Shape[1];
            var meanDivider = Count / bnBias.Count;  // = batchSize if (1,C,H,W) , and = batchSize*H*W if (1,C,1,1)

            Compute_Column_Mean_Variance(resultSaveMean, resultSaveVariance);
            var batchSize = x.Shape[0];

            int idx = 0;
            var xContent = x.AsFloatCpuContent;
            var yContent = y.AsFloatCpuContent;
            var bnScaleContent = bnScale.AsFloatCpuContent;
            var bnBiasContent = bnBias.AsFloatCpuContent;
            var resultSaveMeanContent = resultSaveMean.AsFloatCpuContent;
            var resultSaveVarianceContent = resultSaveVariance.AsFloatCpuContent;
            var resultRunningMeanContent = resultRunningMean.AsFloatCpuContent;
            var resultRunningVarianceContent = resultRunningVariance.AsFloatCpuContent;
            if (isTraining)
            {
                for (int j = 0; j < resultRunningVarianceContent.Length; ++j)
                {
                    resultRunningMeanContent[j] = (float) (resultSaveMeanContent[j] * exponentialAverageFactor + resultRunningMeanContent[j] * (1 - exponentialAverageFactor));
                    resultRunningVarianceContent[j] = (float)(resultSaveVarianceContent[j] * exponentialAverageFactor + resultRunningVarianceContent[j] * (1 - exponentialAverageFactor));
                }
            }
            for (int j = 0; j < resultSaveVarianceContent.Length; ++j)
            {
                resultSaveVarianceContent[j] = (float) (1.0 / Math.Sqrt(((meanDivider - 1) * resultSaveVarianceContent[j]) / meanDivider + epsilon));
            }

            for (int n = 0; n < batchSize; ++n)
            {
                for (int j = 0; j < MultDim0; ++j)
                {
                    int scaleIndex = is1C11Shape ? (j / MultDim1) : j;
                    var xOriginal = xContent[idx];
                    var xTarget = isTraining
                        ? ((xOriginal - resultSaveMeanContent[scaleIndex]) * resultSaveVarianceContent[scaleIndex])
                        : (float)((xOriginal - resultRunningMeanContent[scaleIndex]) / Math.Sqrt(resultRunningVarianceContent[scaleIndex] + epsilon));
                    yContent[idx++] = bnScaleContent[scaleIndex] * xTarget + bnBiasContent[scaleIndex];
                }
            }
        }
        public override void BatchNormalizationBackward(Tensor dy, Tensor dx, Tensor bnScale, Tensor resultBnScaleDiff, Tensor resultBnBiasDiff, cudnnBatchNormMode_t mode, double epsilon, Tensor resultSaveMean, Tensor resultSaveVariance)
        {
            var x = this;
            var batchSize = x.Shape[0];
            Debug.Assert(AreCompatible(new List<Tensor> {x, dy, dx, bnScale, resultBnScaleDiff, resultBnBiasDiff, resultSaveMean, resultSaveVariance}));
            Debug.Assert(x.SameShape(dy, dx));
            Debug.Assert(bnScale.SameShape(resultBnScaleDiff, resultBnBiasDiff, resultSaveMean, resultSaveVariance));
            bool is1C11Shape = bnScale.Count == bnScale.Shape[1];
            var meanDivider = Count / bnScale.Count;  // = batchSize if (1,C,H,W) , and = batchSize*H*W if (1,C,1,1)
            resultBnScaleDiff.ZeroMemory();
            dx?.ZeroMemory();

            //we compute resultBnBiasDiff
            dy.AsCpu<float>().ComputeSumByColumn(resultBnBiasDiff);
            //we compute resultBnScaleDiff
            var xContent = x.AsFloatCpuContent;
            var dyContent = dy.AsFloatCpuContent;
            var dxContent = dx?.AsFloatCpuContent ?? new float[x.Count];
            var resultBnBiasDiffContent = resultBnBiasDiff.AsFloatCpuContent;
            var resultBnScaleDiffContent = resultBnScaleDiff.AsFloatCpuContent;
            var bnScaleContent = bnScale.AsFloatCpuContent;
            var resultSaveMeanContent = resultSaveMean.AsFloatCpuContent;
            var resultSaveVarianceContent = resultSaveVariance.AsFloatCpuContent;
            for (int j = 0; j < MultDim0; ++j)
            {
                int meanIndex = is1C11Shape ? (j / MultDim1) : j;
                double result = 0.0;
                for (int n = 0; n < batchSize; ++n)
                {

                    int idx = n * MultDim0 + j;
                    result += dyContent[idx] * (xContent[idx] - resultSaveMeanContent[meanIndex]);
                }
                resultBnScaleDiffContent[meanIndex] += (float) (result * resultSaveVarianceContent[meanIndex]);
            }
            //we compute dx
            for (int i = 0; i < batchSize; ++i)
            {
                for (int j = 0; j < MultDim0; ++j)
                {
                    int meanIndex = is1C11Shape ? (j / MultDim1) : j;
                    int idx = i * MultDim0 + j;
                    double result = meanDivider * dyContent[idx] - resultBnBiasDiffContent[meanIndex] - resultBnScaleDiffContent[meanIndex] * resultSaveVarianceContent[meanIndex] * (xContent[idx] - resultSaveMeanContent[meanIndex]);
                    dxContent[idx] += (float) ((bnScaleContent[meanIndex] * resultSaveVarianceContent[meanIndex] * result) / meanDivider);
                }
            }
        }
        public override void DropoutForward(Tensor y, double dropProbability, bool isTraining, Random dropoutRandom, Tensor dropoutMaskBuffer)
        {
            var x = this;
            Debug.Assert(dropoutMaskBuffer != null);
            if (!isTraining)
            {
                x.CopyTo(y);
                return;
            }
       
            var dropProbabilityFloat = (float)dropProbability;
            Utils.Randomize(dropoutMaskBuffer.AsFloatCpuContent, dropoutRandom, 0.0, 1.0);
            y.AsCpu<float>().BuildEntirelyFromInput(x, dropoutMaskBuffer, (prevLayer, prob) => prob < dropProbability ? 0f : prevLayer / (1 - dropProbabilityFloat));
        }
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropProbability, Tensor usedDropoutMask)
        {
            var _dropProbabilityFloat = (float)dropProbability;
            dx.AsCpu<float>().BuildEntirelyFromInput(dy, usedDropoutMask, (dOutput, prob) => prob < _dropProbabilityFloat ? 0f : dOutput / (1 - _dropProbabilityFloat));
        }
        //this = dy

        public override void ActivationForward(cudnnActivationMode_t activationType, Tensor y)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> {x, y}));
            switch (activationType)
            {
                case cudnnActivationMode_t.CUDNN_ACTIVATION_RELU:
                    CpuTensorActivationFunctions.Relu(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_ELU:
                    CpuTensorActivationFunctions.Elu(x, y, 1.0);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_TANH:
                    CpuTensorActivationFunctions.Tanh(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID:
                    CpuTensorActivationFunctions.Sigmoid(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX:
                    CpuTensorActivationFunctions.Softmax(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH:
                    CpuTensorActivationFunctions.Swish(x, y);
                    return;
                default:
                    throw new ArgumentException("invalid activation mode " + activationType);
            }
        }
        public override void ActivationBackward(Tensor dy, Tensor x, cudnnActivationMode_t activationType, Tensor dx)
        {
            var y = this;
            Debug.Assert(AreCompatible(new List<Tensor> { y, dy, x, dx }));
            switch (activationType)
            {
                case cudnnActivationMode_t.CUDNN_ACTIVATION_RELU:
                    CpuTensorActivationFunctions.ReluGradient(dy, x, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_ELU:
                    CpuTensorActivationFunctions.EluGradient(y, dy, x, dx, 1f);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_TANH:
                    CpuTensorActivationFunctions.TanhGradient(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID:
                    CpuTensorActivationFunctions.SigmoidGradient(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX:
                    CpuTensorActivationFunctions.SoftmaxGradient(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH:
                    CpuTensorActivationFunctions.SwishGradient(y, dy, x, dx);
                    return;
                default:
                    throw new ArgumentException("invalid activation mode " + activationType);
            }
        }

        public override void Update_Adding_Alpha_X(float alpha, Tensor x)
        {
            var y = this;
            Debug.Assert(AreCompatible(new List<Tensor> {y, x}));
            Debug.Assert(x.Count == y.Count);
            MKL_BLAS.cblas_saxpy(x.Count, alpha, x.AsFloatCpuContent, 1, y.AsFloatCpuContent, 1);
        }

        // compute: this = alpha * x + beta * this
        public override void AddTensor(float alpha, Tensor x, float beta)
        {
            // this = beta * this
            Update_Multiplying_By_Alpha(beta);
            // this = alpha * x + beta * this
            Update_Adding_Alpha_X(alpha, x);
        }

        public override void MultiplyTensor(Tensor a, Tensor x)
        {
            Debug.Assert(this.SameShape(a));
            Debug.Assert(a.Count >= x.Count);
            Debug.Assert(Count % x.Count == 0);

            var aFloat = a.AsFloatCpuContent;
            var xFloat = x.AsFloatCpuContent;
            var thisFloat = AsFloatCpuContent;
            if (a.Count == x.Count)
            {
                for (int i = 0; i < x.Count; ++i)
                {
                    thisFloat[i] = aFloat[i] * xFloat[i];
                }
            }
            else
            {
                Debug.Assert(x.Shape[0]*x.Shape[1] == x.Count);
                int indexInX = 0;
                int indexInThis = 0;
                int toAddInThis = a.Count / x.Count;
                while(indexInThis<Count)
                {
                    int endIndexInThis = indexInThis + toAddInThis;
                    var alphaFromX = xFloat[indexInX++];
                    while (indexInThis< endIndexInThis)
                    {
                        thisFloat[indexInThis] = aFloat[indexInThis] * alphaFromX;
                        indexInThis++;
                    }
                }
            }
        }

        public override void MultiplyEachRowIntoSingleValue(Tensor a, Tensor b)
        {
            Debug.Assert(a.SameShape(b));
            int nbRows = Count;
            Debug.Assert(nbRows <= a.Count);
            Debug.Assert(a.Count % nbRows == 0);
            int nbColumns_in_a_and_b = b.Count / nbRows;
            var thisFloat = AsFloatCpuContent;
            var aFloat = a.AsFloatCpuContent;
            var bFloat = b.AsFloatCpuContent;
            int indexIn_a_or_b = 0;
            for (int row = 0; row < nbRows; ++row)
            {
                float rowSum = 0;
                for (int col = 0; col < nbColumns_in_a_and_b; ++col)
                {
                    rowSum += aFloat[indexIn_a_or_b] * bFloat[indexIn_a_or_b];
                    ++indexIn_a_or_b;
                }
                thisFloat[row] = rowSum;
            }
        }

        public override void ZeroPadding(Tensor src, int topPadding, int bottomPadding, int leftPadding, int rightPadding)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, src }));
            Debug.Assert(Dimension == 4);
            Debug.Assert(Dimension == src.Dimension);
            Debug.Assert(Shape[0] == src.Shape[0]); //same batch size
            Debug.Assert(Shape[1] == src.Shape[1]); //same number of channels
            Debug.Assert(Shape[2] == (topPadding + src.Shape[2] + bottomPadding)); //valid height for destination
            Debug.Assert(Shape[3] == (leftPadding + src.Shape[3] + rightPadding)); //valid width destination
            ZeroMemory();
            int h_src = src.Shape[2];
            int w_src = src.Shape[3];
            // copy the row 'srcRowId' from 'src' tensor (n, c, h_src, w_src) to dest tensor (n, c, h_dest, w_dest)
            // the number of distinct rows in 'src' tensor is : n*c*h_src
            void ApplyZeroPaddingForRowId(int srcRowId)
            {
                // 0 <= srcRowId < n*c*h_src
                int row_src = (srcRowId % h_src);
                int srcRowIndex = srcRowId * w_src;
                int destRowIndex = ((srcRowId / h_src) * Shape[2] + row_src + topPadding) * Shape[3] + leftPadding;
                src.CopyTo(srcRowIndex, this, destRowIndex, w_src);
            }
            System.Threading.Tasks.Parallel.For(0, src.Shape[0] * src.Shape[1] * src.Shape[2], ApplyZeroPaddingForRowId);
        }

        public override void AssertIsNotDisposed()
        {
            if (_disposed)
            {
                throw new Exception("Tensor is disposed " + this);
            }
        }

        /// <summary>
        /// Add tests
        /// </summary>
        /// <returns></returns>
        public override Tensor Transpose()
        {
            Debug.Assert(Dimension == 2);
            var output = new CpuTensor<T>(new[] { Shape[1], Shape[0] }, Description);
            for (int row = 0; row < Shape[0]; ++row)
            {
                for (int col = 0; col < Shape[1]; ++col)
                {
                    output.Set(col, row, Get(row, col));
                }
            }
            return output;
        }

        public override void Concatenate(Tensor a, Tensor b)
        {
            CheckConcatenate(a, b);
            void ConcatenateSingleRow(int m)
            {
                a.CopyTo(a.Idx(m), this, Idx(m), a.MultDim0);
                b.CopyTo(b.Idx(m), this, Idx(m) + a.MultDim0, b.MultDim0);
            }
            System.Threading.Tasks.Parallel.For(0, Shape[0], ConcatenateSingleRow);
        }
        public override void Split(Tensor a, Tensor b)
        {
            CheckConcatenate(a, b);
            void SplitSingleRow(int m)
            {
                CopyTo(Idx(m), a, a.Idx(m), a.MultDim0);
                CopyTo(Idx(m) + a.MultDim0, b, b.Idx(m), b.MultDim0);
            }
            System.Threading.Tasks.Parallel.For(0, Shape[0], SplitSingleRow);
        }
        public static CpuTensor<float> CreateOneHotTensor(Func<int,int> elementIdToCategoryIndex, int elementCount, int categoriesCount)
        {
            var Y = new CpuTensor<float>(new[] { elementCount, categoriesCount }, "YOneHot");
            for (int elementId = 0; elementId < elementCount; ++elementId)
            {
                var categoryIndex = elementIdToCategoryIndex(elementId);
                if (categoryIndex < 0)
                {
                    continue;
                }
                Y.Content[elementId * categoriesCount + categoryIndex] = 1f;
            }
            return Y;
        }


        // compute:     this = alpha * this
        public override void Update_Multiplying_By_Alpha(float alpha)
        {
            MKL_BLAS.cblas_sscal(Count, alpha, AsFloatCpuContent, 1);
        }
        #region pooling layers


        public override void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride)
        {
            var x = this;
#if DEBUG
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            int hOutput = y.Shape[2];
            int wOutput = y.Shape[3];
            int hInput = x.Shape[2];
            int wInput = x.Shape[3];
            Debug.Assert(x.Dimension == 4);
            Debug.Assert(y.Dimension == 4);
            Debug.Assert(x.Shape[0] == y.Shape[0]); //same batch size
            Debug.Assert(x.Shape[1] == y.Shape[1]); //same number of channels
            int hExpected = (hInput - poolingHeight) / poolingStride + 1;
            Debug.Assert(hOutput == hExpected);
            int wExpected = (wInput - poolingWidth) / poolingStride + 1;
            Debug.Assert(wOutput == wExpected);
#endif
            int batchSize = x.Shape[0];
            if (PoolingLayer.IsMaxPooling(poolingMode))
            {
                System.Threading.Tasks.Parallel.For(0, batchSize, elementIndex => MaxPoolingForSingleElement(y, poolingHeight, poolingWidth, poolingStride, elementIndex ));
            }
            else
            {
                System.Threading.Tasks.Parallel.For(0, batchSize, elementIndex => AvgPoolingForSingleElement(y, poolingHeight, poolingWidth, poolingStride, elementIndex));
            }
        }
        private void AvgPoolingForSingleElement(Tensor y, int poolingHeight, int poolingWidth, int poolingStride, int elementIndex)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            int hOutput = y.Shape[2];
            int wOutput = y.Shape[3];
            //the first (top left) point in 'y' is computed from a filter starting at (0,0)
            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    int col_filter_start = 0;
                    for (int colAfterPooling = 0; colAfterPooling < wOutput; ++colAfterPooling)
                    {
                        //we want to compute the point in y[n, channelId, row_output, col_output]
                        //it is computed by applying an avg filter located (for its top left) in (row_filter_start,col_filter_start) in the x 
                        float outputPointSum = 0f;
                        int count = 0;
                        for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingHeight); ++rowBeforePooling)
                        {
                            for (int colBeforePooling = col_filter_start; colBeforePooling < (col_filter_start + poolingWidth); ++colBeforePooling)
                            {
                                outputPointSum += x.AsFloatCpu.Get(elementIndex, c, rowBeforePooling, colBeforePooling);
                                ++count;
                            }
                        }
                        y.AsFloatCpu.Set(elementIndex, c, rowAfterPooling, colAfterPooling, outputPointSum / count);
                        col_filter_start += poolingStride;
                    }
                    row_filter_start += poolingStride;
                }
            }
        }
        private void MaxPoolingForSingleElement(Tensor y, int poolingHeight, int poolingWidth, int poolingStride, int elementIndex)
        {
            var x = this;
            int hOutput = y.Shape[2];
            int wOutput = y.Shape[3];
            //the first (top left) point in 'y' is computed from a filter starting at (0,0)
            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    int col_filter_start = 0;
                    for (int colAfterPooling = 0; colAfterPooling < wOutput; ++colAfterPooling)
                    {
                        //we want to compute the point in y[n, channelId, row_output, col_output]
                        //it is computed by applying a max filter located (for its top left) in (row_filter_start,col_filter_start) in the x 
                        float outputPointResult = float.MinValue;
                        for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingHeight); ++rowBeforePooling)
                        {
                            for (int colBeforePooling = col_filter_start; colBeforePooling < (col_filter_start + poolingWidth); ++colBeforePooling)
                            {
                                outputPointResult = Math.Max(outputPointResult, x.AsFloatCpu.Get(elementIndex, c, rowBeforePooling, colBeforePooling));
                            }
                        }
                        y.AsFloatCpu.Set(elementIndex, c, rowAfterPooling, colAfterPooling, outputPointResult);
                        col_filter_start += poolingStride;
                    }
                    row_filter_start += poolingStride;
                }
            }
        }
        public override void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode,
            int poolingHeight, int poolingWidth, int poolingStride)
        {
            int batchSize = x.Shape[0];
#if DEBUG
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, y, x, dx }));
            Debug.Assert(x.Dimension == 4);
            Debug.Assert(x.Shape[0] == dy.Shape[0]); //same batchSize
            Debug.Assert(x.Shape[1] == dy.Shape[1]); //same number of channels
            Debug.Assert(dx.SameShape(x));
            int hOutput = dy.Shape[2];
            int wOutput = dy.Shape[3];
            Debug.Assert(hOutput == ((x.Shape[2] - poolingHeight) / poolingStride + 1));
            Debug.Assert(wOutput == ((x.Shape[3] - poolingWidth) / poolingStride + 1));
#endif
            dx.ZeroMemory();
            if (PoolingLayer.IsMaxPooling(poolingMode))
            {
                System.Threading.Tasks.Parallel.For(0, batchSize, elementIndex => MaxPoolingGradientForSingleElement(x, dx, poolingHeight, poolingWidth, poolingStride, elementIndex));
            }
            else
            {
                System.Threading.Tasks.Parallel.For(0, batchSize, elementIndex => AvgPoolingGradientForSingleElement(x, dx, poolingHeight, poolingWidth, poolingStride, elementIndex));
            }
        }
        private void AvgPoolingGradientForSingleElement(Tensor x, Tensor dx, int poolingHeight, int poolingWidth, int poolingStride, int elementIndex)
        {
            var dy = this;
            int hOutput = dy.Shape[2];
            int wOutput = dy.Shape[3];
            double doubleMultiplier = 1.0 / (poolingHeight * poolingWidth);
            float floatMultiplier = (float)doubleMultiplier;

            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    int col_filter_start = 0;
                    for (int colAfterPooling = 0; colAfterPooling < wOutput; ++colAfterPooling)
                    {
                        for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingHeight); ++rowBeforePooling)
                        {
                            for (int colBeforePooling = col_filter_start; colBeforePooling < (col_filter_start + poolingWidth); ++colBeforePooling)
                            {
                                var pointGradient = dy.AsFloatCpu.Get(elementIndex, c, rowAfterPooling, colAfterPooling);
                                dx.AsFloatCpu.Set(elementIndex, c, rowBeforePooling, colBeforePooling, floatMultiplier * pointGradient);
                            }
                        }
                        col_filter_start += poolingStride;
                    }
                    row_filter_start += poolingStride;
                }
            }
        }
        //compute 'dx' from ('dy' and 'x')
        private void MaxPoolingGradientForSingleElement(Tensor x, Tensor dx, int poolingHeight, int poolingWidth, int poolingStride, int elementIndex)
        {
            var dy = this;
            int hOutput = dy.Shape[2];
            int wOutput = dy.Shape[3];

            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    int col_filter_start = 0;
                    for (int colAfterPooling = 0; colAfterPooling < wOutput; ++colAfterPooling)
                    {
                        //we retrieve the coordinate of the max value in 'x'
                        double outputPointResult = double.MinValue;
                        int maxRowBeforePooling = 0;
                        int maxColBeforePooling = 0;
                        for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingHeight); ++rowBeforePooling)
                        {
                            for (int colBeforePooling = col_filter_start; colBeforePooling < (col_filter_start + poolingWidth); ++colBeforePooling)
                            {
                                var currentPointValue = x.AsFloatCpu.Get(elementIndex, c, rowBeforePooling, colBeforePooling);
                                if (currentPointValue > outputPointResult)
                                {
                                    outputPointResult = currentPointValue;
                                    maxRowBeforePooling = rowBeforePooling;
                                    maxColBeforePooling = colBeforePooling;
                                }
                            }
                        }
                        var pointGradient = dy.AsFloatCpu.Get(elementIndex, c, rowAfterPooling, colAfterPooling);
                        dx.AsFloatCpu.Set(elementIndex, c, maxRowBeforePooling, maxColBeforePooling, pointGradient);
                        col_filter_start += poolingStride;
                    }
                    row_filter_start += poolingStride;
                }
            }
        }
#endregion
        public override void BroadcastAddVectorToOutput(Tensor y)
        {
            var bias = this;
            Debug.Assert(AreCompatible(new List<Tensor> {bias, y}));
            Debug.Assert(y.Dimension >= 2);
            Debug.Assert(y.MultDim0 == Count);
            var batchSize = y.Shape[0];

            var singleRowMatrixContent = bias.AsFloatCpuContent;
            var yContent = y.AsFloatCpuContent;
            for (int colIndex = 0; colIndex < Count; ++colIndex)
            {
                var valueToAddToColumn = singleRowMatrixContent[colIndex];
                for (int n = 0; n < batchSize; ++n)
                {
                    var idx = y.Idx(n, colIndex);
                    yContent[idx] += valueToAddToColumn;
                }
            }
        }

        #region Convolution
        public override void Convolution(Tensor convolution, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int stride, Tensor y, bool isDepthwiseConvolution)
        {
            var x = this;
            int inputChannels = x.Shape[1];
            int outputChannels = y.Shape[1];
            Debug.Assert(inputChannels == convolution.Shape[1]);
            if (isDepthwiseConvolution)
            {
                Debug.Assert(inputChannels == y.Shape[1]);
                Debug.Assert(outputChannels == convolution.Shape[1]);
                var depthMultiplier = convolution.Shape[0];
                if (depthMultiplier != 1)
                {
                    throw new NotImplementedException("only depthMultiplier=1 is supported");
                }
            }
            else
            {
                Debug.Assert(outputChannels == convolution.Shape[0]);
            }
            Debug.Assert(AreCompatible(new List<Tensor> {x, convolution, y}));
            int batchSize = x.Shape[0];
            int hInput = x.Shape[2];
            int wInput = x.Shape[3];
            int F = convolution.Shape[2];
            Debug.Assert(F == convolution.Shape[3]);
            int hOutput = y.Shape[2];
            int wOutput = y.Shape[3];
            Debug.Assert(batchSize == y.Shape[0]);

            //the first (top left) point in 'y' is computed from a filter starting at (-padding,-padding)
            void ComputeForBatch(int m)
            {
                var convolutionContentAsFloat = convolution.AsFloatCpuContent;
                var xContentAsFloat = x.AsFloatCpuContent;

                for (int outputChannelId = 0; outputChannelId < outputChannels; ++outputChannelId)
                {
                    int rowFilterStart = -paddingTop;
                    for (int rowOutput = 0; rowOutput < hOutput; ++rowOutput)
                    {
                        int colFilterStart = -paddingLeft;
                        var rowInputStart = Math.Max(0, rowFilterStart);
                        var rowInputEndExcluded = Math.Min(hInput, rowFilterStart + F);
                        for (int colOutput = 0; colOutput < wOutput; ++colOutput)
                        {
                            //we want to compute the point in y[m, filterId, row_output, col_output]
                            //it is computed by applying a filter located (for its top left) in (row_filter_start,col_filter_start) in the x 
                            double outputPointResult = 0.0;
                            var colInputStart = Math.Max(0, colFilterStart);
                            var colInputEndExcluded = Math.Min(wInput, colFilterStart + F);

                            int startInputChannelId = isDepthwiseConvolution ? outputChannelId : 0;
                            int endInputChannelId = isDepthwiseConvolution ? (outputChannelId+1) : inputChannels;
                            for (int inputChannelId = startInputChannelId; inputChannelId < endInputChannelId; ++inputChannelId)
                            {
                                var convolutionIdxForStartRow = convolution.Idx(isDepthwiseConvolution?0:outputChannelId, inputChannelId, rowInputStart - rowFilterStart, colInputStart - colFilterStart);
                                var xIdxForStartRow = x.Idx(m, inputChannelId, rowInputStart, colInputStart);
                                for (int rowInput = rowInputStart; rowInput < rowInputEndExcluded; ++rowInput)
                                {
                                    var convolutionIdx = convolutionIdxForStartRow;
                                    var xIdx = xIdxForStartRow;
                                    for (int colInput = colInputStart; colInput < colInputEndExcluded; ++colInput)
                                    {
                                        outputPointResult +=
                                            convolutionContentAsFloat[convolutionIdx] *
                                            xContentAsFloat[xIdx];
                                        ++convolutionIdx;
                                        ++xIdx;
                                    }
                                    convolutionIdxForStartRow += convolution.Shape[3];
                                    xIdxForStartRow += x.Shape[3];
                                }
                            }
                            y.AsFloatCpu.Set(m, outputChannelId, rowOutput, colOutput, (float) outputPointResult);
                            colFilterStart += stride;
                        }
                        rowFilterStart += stride;
                    }
                }
            }

            System.Threading.Tasks.Parallel.For(0, batchSize, ComputeForBatch);
        }


        public override void BroadcastConvolutionBiasToOutput(Tensor y)
        {
            var convolutionBias = this;
            Debug.Assert(AreCompatible(new List<Tensor> { convolutionBias, y }));
            Debug.Assert(y.Dimension >= 2);
            Debug.Assert(convolutionBias.Shape.SequenceEqual(new []{1, y.Shape[1], 1, 1}));
            var batchSize = y.Shape[0];
            var yContent = y.AsFloatCpuContent;
            for (int n = 0; n < batchSize; ++n)
            {
                int startIndex = n * y.MultDim0;
                for (int filterId = 0; filterId < y.Shape[1]; ++filterId, startIndex += y.MultDim1)
                {
                    var toAdd = convolutionBias.AsFloatCpuContent[filterId];
                    for (int i = startIndex; i < (startIndex + y.MultDim1); ++i)
                    {
                        yContent[i] += toAdd;
                    }
                }
            }
        }

        public override void ConvolutionGradient(Tensor convolution, Tensor dy, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, int stride, Tensor dx, Tensor convGradient, bool isDepthwiseConvolution)
        {
            var x = this;
            int inputChannels = x.Shape[1];
            int outputChannels = dy.Shape[1];
            Debug.Assert(inputChannels == convolution.Shape[1]);
            if (isDepthwiseConvolution)
            {
                Debug.Assert(inputChannels == dy.Shape[1]);
                Debug.Assert(outputChannels == convolution.Shape[1]);
                var depthMultiplier = convolution.Shape[0];
                if (depthMultiplier != 1)
                {
                    throw new NotImplementedException("only depthMultiplier=1 is supported");
                }
            }
            else
            {
                Debug.Assert(outputChannels == convolution.Shape[0]);
            }
            Debug.Assert(AreCompatible(new List<Tensor> { x, convolution, dy, dx, convGradient }));
            int batchSize = x.Shape[0];
            Debug.Assert(batchSize == dy.Shape[0]);
            int hInput = x.Shape[2];
            int wInput = x.Shape[3];
            int F = convolution.Shape[2];
            Debug.Assert(F == convolution.Shape[3]);
            int hOutput = dy.Shape[2];
            Debug.Assert(hOutput == ((hInput - F + paddingTop+paddingBottom) / stride + 1));
            int wOutput = dy.Shape[3];
            Debug.Assert(wOutput == ((wInput - F + paddingLeft+paddingRight) / stride + 1));
            dx?.ZeroMemory();
            convGradient.ZeroMemory();

            //the first (top left) point in 'y' is computed from a filter starting at (-padding,-padding)

            void ComputeForBatch(int m)
            {
                //every thread needs to update 'convolutionGradient'
                //to be thread safe, each thread will update a local object 'convolutionGradientContent' and at the end
                //will update the object 'convolutionGradient' with a local
                float[] convolutionGradientForLocalThreadFloat = new float[convGradient.Count];
                for (int outputChannelId = 0; outputChannelId < outputChannels; ++outputChannelId)
                {
                    int rowFilterStart = -paddingTop;
                    for (int rowOutput = 0; rowOutput < hOutput; ++rowOutput)
                    {
                        int colFilterStart = -paddingLeft;
                        var rowInputStart = Math.Max(0, rowFilterStart);
                        var rowInputEndExcluded = Math.Min(hInput, rowFilterStart + F);
                        for (int colOutput = 0; colOutput < wOutput; ++colOutput)
                        {
                            //we want to compute the point in y[m, filterId, rowOutput, colOutput]
                            //it is computed by applying a filter located (for its top left) in (row_filter_start,col_filter_start) in the x 
                            // and centered at this particular location
                            var chainGradientFloat = dy.AsFloatCpu.Get(m, outputChannelId, rowOutput, colOutput);
                            var colInputStart = Math.Max(0, colFilterStart);
                            var colInputEndExcluded = Math.Min(wInput, colFilterStart + F);
                            int startInputChannelId = isDepthwiseConvolution ? outputChannelId : 0;
                            int endInputChannelId = isDepthwiseConvolution ? (outputChannelId + 1) : inputChannels;
                            for (int inputChannelId = startInputChannelId; inputChannelId < endInputChannelId; ++inputChannelId)
                            {
                                int convIdxStartRow = convGradient.Idx(isDepthwiseConvolution ? 0 : outputChannelId, inputChannelId, rowInputStart - rowFilterStart, colInputStart - colFilterStart);
                                int APrevLayerIdxStartRow = x.Idx(m, inputChannelId, rowInputStart, colInputStart);

                                for (int rowInput = rowInputStart; rowInput < rowInputEndExcluded; ++rowInput)
                                {
                                    var convIdx = convIdxStartRow;
                                    var APrevLayerIdx = APrevLayerIdxStartRow;
                                    for (int colInput = colInputStart; colInput < colInputEndExcluded; ++colInput)
                                    {
                                        convolutionGradientForLocalThreadFloat[convIdx] += x.AsFloatCpuContent[APrevLayerIdx] * chainGradientFloat;
                                        if (dx != null)
                                        {
                                            dx.AsFloatCpuContent[APrevLayerIdx] += convolution.AsFloatCpuContent[convIdx] * chainGradientFloat;
                                        }
                                        ++convIdx;
                                        ++APrevLayerIdx;
                                    }
                                    convIdxStartRow += convGradient.Shape[3];
                                    APrevLayerIdxStartRow += wInput;
                                }
                            }
                            colFilterStart += stride;
                        }
                        rowFilterStart += stride;
                    }
                }
                lock (convGradient)
                {
                    for (int i = 0; i < convGradient.Count; ++i)
                    {
                        convGradient.AsFloatCpuContent[i] += convolutionGradientForLocalThreadFloat[i];
                    }
                }
            }

            System.Threading.Tasks.Parallel.For(0, batchSize, ComputeForBatch);
        }
        public override void ConvolutionBackwardBias(Tensor bias)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, bias }));
            Debug.Assert(bias.Dimension == 4);
            Debug.Assert(bias.Shape[1] == bias.Count);
            Debug.Assert(dy.Shape[1] == bias.Shape[1]); // number of distinct filters
            Debug.Assert(dy.Dimension == 4);

            bias.ZeroMemory();
            var batchSize = dy.Shape[0];
            for (int n = 0; n < batchSize; ++n)
            {
                for (int filterId = 0; filterId < dy.Shape[1]; ++filterId)
                {
                    int startIndex = n * dy.MultDim0 + filterId * dy.MultDim1;
                    var endIndex = startIndex + dy.MultDim1;
                    var convolutionBackwardBiasContent = bias.AsFloatCpuContent;
                    var dyContent = dy.AsFloatCpuContent;
                    for (int i = startIndex; i < endIndex; ++i)
                    {
                        convolutionBackwardBiasContent[filterId] += dyContent[i];
                    }
                }
            }
        }
        #endregion

        public override void Compute_BiasGradient_from_dy(Tensor biasGradient)
        {
            ComputeSumByColumn(biasGradient);
        }
        //this = Weights or Bias
        public override void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon, Tensor dW, Tensor adam_vW, Tensor adam_sW, int timestep)
        {
            var beta1_power = Math.Pow(beta1, timestep);
            var beta2_power = Math.Pow(beta2, timestep);
            var multiplicative_factor = learningRate * (Math.Sqrt(1.0 - beta2_power) / (1.0 - beta1_power));

            var W = this;
            adam_vW.AsCpu<float>().Update(dW, (adam_vw, dw) => (float) (beta1 * adam_vw + (1 - beta1) * dw));
            adam_sW.AsCpu<float>().Update(dW, (adam_sw, dw) => (float) (beta2 * adam_sw + (1 - beta2) * dw * dw));
            W.AsCpu<float>().Update(adam_vW, adam_sW, (w, adam_vw, adam_sw) => (float) (w - multiplicative_factor * (adam_vw / (Math.Sqrt(adam_sw) + epsilon))));
        }
        //this = yExpected
        public override double ComputeLoss(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer)
        {
            var yExpected = this;
            Debug.Assert(yPredicted != null);
            Debug.Assert(!yPredicted.UseGPU);
            Debug.Assert(yPredicted.SameShape(yExpected));
            var batchSize = yExpected.Shape[0];
            var categoryCount = yExpected.Shape[1];
            double cost;
            switch (lossFunction)
            {
                case NetworkConfig.LossFunctionEnum.BinaryCrossentropy:
                    cost = (-1.0 / (batchSize * categoryCount)) * yPredicted.AsCpu<float>().Merge(yExpected.AsCpu<float>(), (prediction, expected) => (float)(expected * Math.Log(prediction) + (1 - expected) * Math.Log(1 - prediction)), "BinaryCrossentropy").NaNSum();
                    break;
                case NetworkConfig.LossFunctionEnum.CategoricalCrossentropy:
                    cost = (-1.0 / (batchSize)) * yPredicted.AsCpu<float>().Merge(yExpected.AsCpu<float>(), (prediction, expected) => (float)(expected * Math.Log(prediction)), "CategoricalCrossentropy").NaNSum();
                    break;
                default:
                    throw new NotImplementedException("don't know how to calculate cost for " + lossFunction);
            }

            //!D TODO using CUDA
            /*
            //L2 regularization
            if (Layers.Skip(1).Any(l => l.UseL2Regularization))
            {
                double L2RegularizationCost = 0.0;
                foreach (var l in Layers.Skip(1))
                {
                    if (l.UseL2Regularization && l is DenseLayer)
                    {
                        DenseLayer fclayer = l as DenseLayer;
                        L2RegularizationCost += (1.0 / batchSize) * (fclayer.lambdaL2Regularization / 2) * fclayer.Weights.SumSquare();
                    }
                }
                cost += L2RegularizationCost;
            }
            */

            return cost;
        }

        public override double ComputeLossFromCategoryIndexes(Tensor yPredictedTensor, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer)
        {
            var categoryIndexes = AsCpu<int>().Content;
            Debug.Assert(yPredictedTensor != null);
            Debug.Assert(!yPredictedTensor.UseGPU);
            var batchSize = yPredictedTensor.Shape[0];
            Debug.Assert(categoryIndexes.Length == batchSize);
            var categoryCount = yPredictedTensor.Shape[1];
            var yPredicted = yPredictedTensor.AsCpu<float>().Content;

            switch (lossFunction)
            {
                case NetworkConfig.LossFunctionEnum.BinaryCrossentropy:
                    double binaryCrossentropyLoss= 0.0;
                    for (int i = 0; i < batchSize; ++i)
                    {
                        int categoryIndex = categoryIndexes[i]; /* the expected category index for element at index 'i' */
                        int startIndex = i * categoryCount;
                        for (int category = 0; category < categoryCount; ++category)
                        {
                            float predicted = yPredicted[startIndex + category];
                            float error = (category == categoryIndex) ? predicted : (1.0f - predicted);
                            if (error > 0)
                            {
                                binaryCrossentropyLoss -= Math.Log(error);
                            }
                        }
                    }
                    return binaryCrossentropyLoss / (batchSize * categoryCount);
                case NetworkConfig.LossFunctionEnum.CategoricalCrossentropy:
                    //cost = (-1.0 / (batchSize)) * yPredicted.AsCpu<float>().Merge(categoryIndexes.AsCpu<int>(), (prediction, expected) => (float)(expected * Math.Log(prediction)), "CategoricalCrossentropy").NaNSum();
                    double categoricalCrossentropyLoss = 0.0;
                    for (int i=0;i< batchSize ;++i)
                    {
                        int categoryIndex = categoryIndexes[i]; /* the expected category index for element at index 'i' */
                        int startIndex = i * categoryCount;
                        float predictedForExpectedCategory = yPredicted[startIndex + categoryIndex];
                        if (predictedForExpectedCategory > 0)
                        {
                            categoricalCrossentropyLoss -= Math.Log(predictedForExpectedCategory);
                        }
                    }
                    return categoricalCrossentropyLoss / (batchSize);
                default:
                    throw new NotImplementedException("don't know how to calculate cost for " + lossFunction);
            }
        }

        public override void RandomMatrixNormalDistribution(Random rand, double mean, double stdDev)
        {
            Utils.RandomizeNormalDistribution(AsFloatCpuContent, rand, mean, stdDev);
        }
        public override void NewSameValueTensor(double sameValue)
        {
            var array = AsFloatCpuContent;
            var sameValueAsFloat = (float)sameValue;
            for (int i = 0; i < array.Length; ++i)
            {
                array[i] = sameValueAsFloat;
            }
        }
        public override float[] ContentAsFloatArray()
        {
            return AsFloatCpuContent;
        }
        //this method is only called for display / logging testing
        //this = yExpectedOneHot
        public override double ComputeAccuracy(Tensor yPredicted, Tensor notUsedBuffer)
        {
            var yExpectedOneHot = this;
            Debug.Assert(AreCompatible(new List<Tensor> { yExpectedOneHot, yPredicted }));
            Debug.Assert(yExpectedOneHot.SameShape(yPredicted));
            Debug.Assert(!yExpectedOneHot.UseGPU);
            int batchSize = yExpectedOneHot.Shape[0];
            int result = 0;

            var yExpectedOneHotCpu = yExpectedOneHot.AsCpu<float>();
            var yPredictedCpu = yPredicted.AsCpu<float>();
            for (int m = 0; m < batchSize; ++m)
            {
                result += ComputeSingleAccuracyCount(yExpectedOneHotCpu, yPredictedCpu, m, out _);
            }
            return ((double)result)/Shape[0];
        }


        //this method is only called for display / logging testing
        //this = category indexes
        public override double ComputeAccuracyFromCategoryIndexes(Tensor yPredicted, Tensor notUsedBuffer)
        {
            var categoryIndexes = AsCpu<int>().Content;
            int batchSize = yPredicted.Shape[0];
            Debug.Assert(batchSize == categoryIndexes.Length);
            Debug.Assert(!yPredicted.UseGPU);
            int result = 0;

            var yPredictedCpu = yPredicted.AsCpu<float>();
            for (int m = 0; m < batchSize; ++m)
            {
                result += ComputeSingleAccuracyCountFromCategoryIndexes(categoryIndexes, yPredictedCpu, m, out _);
            }
            return ((double)result) / Shape[0];
        }

        protected override IntPtr DevicePointer => throw new NotImplementedException("not available for CPU");


        /// <summary>
        /// compute the prediction embedded in the tensor (in each line the index with max value)
        /// </summary>
        /// <returns>array with prediction (=category) of each element</returns>
        public int[] ComputePrediction()
        {
            int batchSize = Shape[0];
            int[] categories = new int[batchSize];
            var yPredictedCpu = AsCpu<float>();
            for (int m = 0; m < batchSize; ++m)
            {
                ComputeSingleAccuracyCount(yPredictedCpu, yPredictedCpu, m, out categories[m]);
            }
            return categories;
        }

        public override void CopyTo(Tensor b)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, b }));
            Debug.Assert(Count == b.Count);
            MKL_BLAS.cblas_scopy(AsFloatCpuContent.Length, AsFloatCpuContent, 1, b.AsFloatCpuContent, 1);
        }
        public override void CopyTo(int startElement, Tensor other, int bStartElement, int elementCount)
        {
            Array.Copy(AsFloatCpuContent, startElement, other.AsFloatCpuContent, bStartElement, elementCount);
        }
        public override Tensor ExtractSubTensor(int startRowIndex, int nbRows)
        {
            Debug.Assert(Shape.Length >= 2);
            Debug.Assert(startRowIndex >= 0);
            Debug.Assert(startRowIndex < Shape[0]);
            Debug.Assert(startRowIndex + nbRows - 1 < Shape[0]);
            var extractedShape = (int[])Shape.Clone();
            extractedShape[0] = nbRows; //news number of rows
            var extractedCount = nbRows * MultDim0;
            var extractedContent = new T[extractedCount];
            int rowLengthInBytes = MultDim0 * TypeSize;
            Buffer.BlockCopy(Content, startRowIndex * rowLengthInBytes, extractedContent, 0, nbRows * rowLengthInBytes);
            return new CpuTensor<T>(extractedShape, extractedContent, Description);
        }
        public override void ZeroMemory()
        {
            Array.Clear(Content, 0, Content.Length);
        }
        public override void Dot(Tensor a, bool transposeA, Tensor b, bool transposeB, float alpha, float beta)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, a, b }));
            Debug.Assert(a.Dimension >= 2);
            Debug.Assert(b.Dimension >= 2);
            Debug.Assert(Dimension >= 2);
            BlasServices.DotMkl(a.AsFloatCpuContent, a.Shape[0], a.MultDim0, transposeA, b.AsFloatCpuContent,
                b.Shape[0], b.MultDim0, transposeB, AsFloatCpuContent, alpha, beta);
            //MathServices.DotOpenblas(a.Content, a.Height, a.Width, b.Content, b.Height, b.Width, y.Content);
            //var tmpTranspose = new double[b.Count];
            //MathServices.DotCSharp(a.Content, a.Height, a.Width, b.Content, b.Height, b.Width, tmpTranspose, y.Content);
        }

        #endregion

        #region Dispose pattern
        public override void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _hostPinnedMemory?.Dispose();
            _hostPinnedMemory = null;
            Content = null;
        }
        #endregion

        /// <summary>
        /// Compute the mean and volatility of each channel of the tensor
        /// </summary>
        /// <param name="toFloat">Function to convert 'T' type to double</param>
        /// <returns>A list of Tuple (one Tuple per channel)
        /// In each channel Tuple: Tuple.Item1: mean of the channel / Tuple.Item2: vol of the channel</returns>
        // ReSharper disable once UnusedMember.Global
        public List<Tuple<float, float>> ComputeMeanAndVolatilityOfEachChannel(Func<T, float> toFloat)
        {
            return Enumerable.Range(0, Shape[1]).Select(c => ComputeMeanAndVolatilityOfChannel(c, toFloat)).ToList();
        }

        /// <summary>
        /// Computes the mean and volatility of the selected channel in the 'this' tensor
        /// </summary>
        /// <param name="c">The channel to compute in the tensor</param>
        /// <param name="toFloat">Function to convert 'T' type to double</param>
        /// <returns>Tuple.Item1: mean of the channel / Tuple.Item2: vol of the channel</returns>
        private Tuple<float, float> ComputeMeanAndVolatilityOfChannel(int c, Func<T, float> toFloat)
        {
            double sum = 0f;
            double sumSquare = 0.0;
            int count = 0;
            for (int m = 0; m < Shape[0]; ++m)
            {
                int startIdx = Idx(m, c, 0, 0);
                for (int idx = startIdx; idx < (startIdx + MultDim1); ++idx)
                {
                    var val = toFloat(Content[idx]);
                    sum += val;
                    sumSquare += val * val;
                    ++count;
                }
            }
            if (count == 0)
            {
                return Tuple.Create(0f, 0f);
            }
            var mean = (sum / count);
            var variance = (sumSquare / count) - mean * mean;
            var volatility = Math.Sqrt(Math.Max(0, variance));
            return Tuple.Create((float)mean, (float)volatility);
        }
        private CpuTensor<T> Merge(CpuTensor<T> b, Func<T, T, T> func, string description)
        {
            Debug.Assert(Dimension == b.Dimension);
            Debug.Assert(Count == b.Count);
            var content = new T[Count];
            for (int i = 0; i < Count; ++i)
            {
                content[i] = func(this[i], b[i]);
            }
            return new CpuTensor<T>(Shape, content, description);
        }
        private double NaNSum()
        {
            return AsFloatCpuContent.Select(x => float.IsNaN(x) ? 0 : x).Sum();
        }
        private void Update(Tensor a, Tensor b, Func<T, T, T, T> funcInput)
        {
            Debug.Assert(AreCompatible(new List<Tensor> {this, a, b}));
            Debug.Assert(SameShape(a, b));
            var aCpu = a.AsCpu<T>();
            var bCpu = b.AsCpu<T>();
            for (int i = 0; i < Count; ++i)
            {
                this[i] = funcInput(this[i], aCpu[i], bCpu[i]);
            }
        }
        private void Update(Tensor b, Func<T, T, T> funcInput)
        {
            Debug.Assert(AreCompatible(new List<Tensor> {this, b}));
            Debug.Assert(SameShape(b));
            var bCpu = b.AsCpu<T>();
            for (int i = 0; i < Count; ++i)
            {
                this[i] = funcInput(this[i], bCpu[i]);
            }
        }
        public void BuildEntirelyFromInput(Tensor a, Tensor b, Func<T, T, T> funcInput)
        {
            Debug.Assert(AreCompatible(new List<Tensor> {this, a, b}));
            Debug.Assert(SameShape(a, b));
            var aCpu = a.AsCpu<T>().Content;
            var bCpu = b.AsCpu<T>().Content;
            for (int i = 0; i < a.Count; ++i)
            {
                this[i] = funcInput(aCpu[i], bCpu[i]);
            }
        }
        public void BuildEntirelyFromInput(Tensor a, Tensor b, Tensor c, Func<T, T, T, T> funcInput)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, a, b, c }));
            Debug.Assert(SameShape(a, b));
            var aCpu = a.AsCpu<T>().Content;
            var bCpu = b.AsCpu<T>().Content;
            var cCpu = c.AsCpu<T>().Content;
            for (int i = 0; i < a.Count; ++i)
            {
                this[i] = funcInput(aCpu[i], bCpu[i], cCpu[i]);
            }
        }
        private void ComputeSumByColumn(Tensor sumByColumn)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, sumByColumn }));
            Debug.Assert(Dimension >= 2);
            var batchSize = Shape[0];
            bool is1C11Shape = sumByColumn.Count == sumByColumn.Shape[1];

            sumByColumn.ZeroMemory();
            var content = AsFloatCpuContent;
            var columnSumContent = sumByColumn.AsFloatCpuContent;
            for (int n = 0; n < batchSize; ++n)
            {
                int start = MultDim0 * n;
                for (int i = 0; i < MultDim0; ++i)
                {
                    int sumByColumnIndex = is1C11Shape ? (i / MultDim1) : i;
                    columnSumContent[sumByColumnIndex] += content[start + i];
                }
            }
        }
        private void Compute_Column_Mean_Variance(Tensor mean, Tensor variance)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, mean, variance }));
            var batchSize = Shape[0];
            Debug.Assert(mean.SameShape(variance));
            //true if we have a (1,C,1,1) shape for scale and bias
            //false is we have a (1,C,H,W) shape for scale and bias
            bool is1C11Shape = mean.Count == mean.Shape[1];

            mean.ZeroMemory();
            variance.ZeroMemory();
            var content = AsFloatCpuContent;
            //we'll store in meanContent Sum(X) and in varianceContent Sum(X^2)
            var meanContent = mean.AsFloatCpuContent;
            var varianceContent = variance.AsFloatCpuContent;
            for (int n = 0; n < batchSize; ++n)
            {
                int start = MultDim0 * n;
                for (int i = 0; i < MultDim0; ++i)
                {
                    var d = content[start + i];
                    int scaleIndex = is1C11Shape ? (i / MultDim1) : i;
                    meanContent[scaleIndex] += d;
                    varianceContent[scaleIndex] += d * d;
                }
            }
            var meanDivider = Count / mean.Count;  // = batchSize if (1,C,H,W) , and = batchSize*H*W if (1,C,1,1)
            for (int i = 0; i < varianceContent.Length; ++i)
            {
                meanContent[i] /= meanDivider;
                //Variance(X) = E(X^2) - E(X) ^2
                //varianceContent[i] = varianceContent[i]/meanDivider - meanContent[i] * meanContent[i];
                varianceContent[i] = (meanDivider <= 1) ? 1f : (varianceContent[i] - meanDivider * meanContent[i] * meanContent[i]) / (meanDivider - 1);
            }
        }
        private static int ComputeSingleAccuracyCount(CpuTensor<float> yExpectedOneHot, CpuTensor<float> yPredicted, int m, out int maxIndexPredicted)
        {
            Debug.Assert(yExpectedOneHot.SameShape(yPredicted));
            Debug.Assert(yExpectedOneHot.Dimension == 2);
            maxIndexPredicted = 0;
            var categoryCount = yExpectedOneHot.Shape[1];
            if (categoryCount == 1)
            {
                var error = Math.Abs(yExpectedOneHot.Get(m, 0) - yPredicted.Get(m, 0));
                return (error < 0.5) ? 1 : 0;
            }
            int maxIndexExpected = 0;
            for (int j = 1; j < categoryCount; ++j)
            {
                if (yPredicted.Get(m, j) > yPredicted.Get(m, maxIndexPredicted))
                {
                    maxIndexPredicted = j;
                }
                if (yExpectedOneHot.Get(m, j) > yExpectedOneHot.Get(m, maxIndexExpected))
                {
                    maxIndexExpected = j;
                }
            }
            if (maxIndexExpected == maxIndexPredicted)
            {
                return 1;
            }
            return 0;
        }
        private static int ComputeSingleAccuracyCountFromCategoryIndexes(int[] categoryIndexes, CpuTensor<float> yPredicted, int m, out int maxIndexPredicted)
        {
            Debug.Assert(categoryIndexes.Length == yPredicted.Shape[0]);
            Debug.Assert(yPredicted.Dimension == 2);
            maxIndexPredicted = 0;
            var categoryCount = yPredicted.Shape[1];
            int categoryIndex = categoryIndexes[m]; /* the expected category index for element at index 'i' */
            for (int j = 1; j < categoryCount; ++j)
            {
                if (yPredicted.Get(m, j) > yPredicted.Get(m, maxIndexPredicted))
                {
                    maxIndexPredicted = j;
                }
            }
            return categoryIndex == maxIndexPredicted ? 1 : 0;
        }

    }
}
