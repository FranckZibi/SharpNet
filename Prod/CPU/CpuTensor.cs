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
    public class CpuTensor<T> : Tensor where T : struct
    {
        #region fields
        public T[] Content { get; private set; }
        private HostPinnedMemory<T> _hostPinnedMemory;
        #endregion

        public CpuTensor(int[] shape, T[] data, string description) : base(shape, Marshal.SizeOf(typeof(T)), false, description)
        {
            Content = data ?? new T[Count];
            CapacityInBytes = (ulong)(Content.Length * TypeSize);
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
        public override void ConvolutionBackwardBias(Tensor convolutionBackwardBias)
        {
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> {dy, convolutionBackwardBias}));
            Debug.Assert(convolutionBackwardBias.Dimension == 4);
            Debug.Assert(convolutionBackwardBias.Shape[1] == convolutionBackwardBias.Count);
            Debug.Assert(dy.Shape[1] == convolutionBackwardBias.Shape[1]); // number of distinct filters
            Debug.Assert(dy.Dimension == 4);

            convolutionBackwardBias.ZeroMemory();
            var batchSize = dy.Shape[0];
            for (int n = 0; n < batchSize; ++n)
            {
                for (int filterId = 0; filterId < dy.Shape[1]; ++filterId)
                {
                    int startIndex = n * dy.MultDim0 + filterId * dy.MultDim1;
                    var endIndex = startIndex + dy.MultDim1;
                    var convolutionBackwardBiasContent = convolutionBackwardBias.AsFloatCpuContent;
                    var dyContent = dy.AsFloatCpuContent;
                    for (int i = startIndex; i < endIndex; ++i)
                    {
                        convolutionBackwardBiasContent[filterId] += dyContent[i];
                    }
                }
            }
        }
        public override void ActivationForward(cudnnActivationMode_t activationType, Tensor y)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> {x, y}));
            switch (activationType)
            {
                case cudnnActivationMode_t.CUDNN_ACTIVATION_RELU:
                    x.Relu(y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_ELU:
                    x.Elu(y, 1.0);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_TANH:
                    x.Tanh(y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID:
                    x.Sigmoid(y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX:
                    x.Softmax(y);
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
                    ReluGradient(dy, x, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_ELU:
                    EluGradient(y, dy, x, dx, 1f);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID:
                    SigmoidGradient(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX:
                    SoftmaxGradient(y, dy, dx);
                    return;
                default:
                    throw new ArgumentException("invalid activation mode " + activationType);
            }
        }
        // compute:  this += alpha * x
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

        public override void AssertIsNotDisposed()
        {
            if (_disposed)
            {
                throw new Exception("Tensor is disposed " + this);
            }
        }

        /// <summary>
        /// TODO : add tests
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
        public static CpuTensor<float> CreateOneHotTensor(Func<int,int> elementIdToCategoryId, int elementCount, int categoriesCount)
        {
            var Y = new CpuTensor<float>(new[] { elementCount, categoriesCount }, "YOneHot");
            for (int elementId = 0; elementId < elementCount; ++elementId)
            {
                var category = elementIdToCategoryId(elementId);
                if (category < 0)
                {
                    continue;
                }
                Y.Content[elementId * categoriesCount + category] = 1f;
            }
            return Y;
        }


        // compute:     this = alpha * this
        public override void Update_Multiplying_By_Alpha(float alpha)
        {
            MKL_BLAS.cblas_sscal(Count, alpha, AsFloatCpuContent, 1);
        }
        #region pooling layers
        public override void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingSize, int poolingStride)
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
            int hExpected = (hInput - poolingSize) / poolingStride + 1;
            Debug.Assert(hOutput == hExpected);
            int wExpected = (wInput - poolingSize) / poolingStride + 1;
            Debug.Assert(wOutput == wExpected);
#endif
            int batchSize = x.Shape[0];
            if (PoolingLayer.IsMaxPooling(poolingMode))
            {
                System.Threading.Tasks.Parallel.For(0, batchSize, elementIndex => MaxPoolingForSingleElement(y, poolingSize, poolingStride, elementIndex ));
            }
            else
            {
                System.Threading.Tasks.Parallel.For(0, batchSize, elementIndex => AvgPoolingForSingleElement(y, poolingSize, poolingStride, elementIndex));
            }
        }
        private void AvgPoolingForSingleElement(Tensor y, int poolingSize, int poolingStride, int elementIndex)
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
                        for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingSize); ++rowBeforePooling)
                        {
                            for (int colBeforePooling = col_filter_start; colBeforePooling < (col_filter_start + poolingSize); ++colBeforePooling)
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
        private void MaxPoolingForSingleElement(Tensor y, int poolingSize, int poolingStride, int elementIndex)
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
                        for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingSize); ++rowBeforePooling)
                        {
                            for (int colBeforePooling = col_filter_start; colBeforePooling < (col_filter_start + poolingSize); ++colBeforePooling)
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
        public override void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode, int poolingSize, int poolingStride)
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
            Debug.Assert(hOutput == ((x.Shape[2] - poolingSize) / poolingStride + 1));
            Debug.Assert(wOutput == ((x.Shape[3] - poolingSize) / poolingStride + 1));
#endif
            dx.ZeroMemory();
            if (PoolingLayer.IsMaxPooling(poolingMode))
            {
                System.Threading.Tasks.Parallel.For(0, batchSize, elementIndex => MaxPoolingGradientForSingleElement(x, dx, poolingSize, poolingStride, elementIndex));
            }
            else
            {
                System.Threading.Tasks.Parallel.For(0, batchSize, elementIndex => AvgPoolingGradientForSingleElement(x, dx, poolingSize, poolingStride, elementIndex));
            }
        }
        private void AvgPoolingGradientForSingleElement(Tensor x, Tensor dx, int poolingSize, int poolingStride, int elementIndex)
        {
            var dy = this;
            int hOutput = dy.Shape[2];
            int wOutput = dy.Shape[3];
            double doubleMultiplier = 1.0 / (poolingSize * poolingSize);
            float floatMultiplier = (float)doubleMultiplier;

            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    int col_filter_start = 0;
                    for (int colAfterPooling = 0; colAfterPooling < wOutput; ++colAfterPooling)
                    {
                        for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingSize); ++rowBeforePooling)
                        {
                            for (int colBeforePooling = col_filter_start; colBeforePooling < (col_filter_start + poolingSize); ++colBeforePooling)
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
        private void MaxPoolingGradientForSingleElement(Tensor x, Tensor dx, int poolingSize, int poolingStride, int elementIndex)
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
                        for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingSize); ++rowBeforePooling)
                        {
                            for (int colBeforePooling = col_filter_start; colBeforePooling < (col_filter_start + poolingSize); ++colBeforePooling)
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
        // x = (batchSize, ChannelsDepth, H_beforePooling, W_beforePooling)
        public override void BroadcastConvolutionBiasToOutput(Tensor y)
        {
            var convolutionBias = this;
            Debug.Assert(AreCompatible(new List<Tensor> {convolutionBias, y}));
            Debug.Assert(y.Dimension >= 2);
            Debug.Assert(convolutionBias.Dimension == 4);
            Debug.Assert(y.Shape[1] == convolutionBias.Shape[1]); // number of distinct filters

            var batchSize = y.Shape[0];
            for (int n = 0; n < batchSize; ++n)
            {
                int startIndex = n * y.MultDim0;
                for (int filterId = 0; filterId < y.Shape[1]; ++filterId, startIndex += y.MultDim1)
                {
                    var yContent = y.AsFloatCpuContent;
                    var toAdd = convolutionBias.AsFloatCpuContent[filterId];
                    for (int i = startIndex; i < (startIndex + y.MultDim1); ++i)
                    {
                        yContent[i] += toAdd;
                    }
                }
            }
        }
        //Compute:      y = x (conv) Convolution  (with padding / stride)
        public override void Convolution(Tensor convolution, int padding, int stride, Tensor y)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> {x, convolution, y}));
            int batchSize = x.Shape[0];
            int channelCount = x.Shape[1];
            int hInput = x.Shape[2];
            int wInput = x.Shape[3];
            int fitlerCount = convolution.Shape[0];
            int F = convolution.Shape[2];
            int hOutput = y.Shape[2];
            int wOutput = y.Shape[3];
            Debug.Assert(F == convolution.Shape[3]);
            Debug.Assert(channelCount == convolution.Shape[1]);
            Debug.Assert(fitlerCount == y.Shape[1]);
            Debug.Assert(batchSize == y.Shape[0]);
            Debug.Assert(hOutput == ((hInput - F + 2 * padding) / stride + 1));
            Debug.Assert(wOutput == ((x.Shape[3] - F + 2 * padding) / stride + 1));

            //the first (top left) point in 'y' is computed from a filter starting at (-padding,-padding)
            void ComputeForBatch(int m)
            {
                var convolutionContentAsFloat = convolution.AsFloatCpuContent;
                var xContentAsFloat = x.AsFloatCpuContent;

                for (int filterId = 0; filterId < fitlerCount; ++filterId)
                {
                    int rowFilterStart = -padding;
                    for (int rowOutput = 0; rowOutput < hOutput; ++rowOutput)
                    {
                        int colFilterStart = -padding;
                        var rowInputStart = Math.Max(0, rowFilterStart);
                        var rowInputEndExcluded = Math.Min(hInput, rowFilterStart + F);
                        for (int colOutput = 0; colOutput < wOutput; ++colOutput)
                        {
                            //we want to compute the point in y[m, filterId, row_output, col_output]
                            //it is computed by applying a filter located (for its top left) in (row_filter_start,col_filter_start) in the x 
                            double outputPointResult = 0.0;
                            var colInputStart = Math.Max(0, colFilterStart);
                            var colInputEndExcluded = Math.Min(wInput, colFilterStart + F);
                            for (int channelId = 0; channelId < channelCount; ++channelId)
                            {
                                var convolutionIdxForStartRow = convolution.Idx(filterId, channelId,
                                    rowInputStart - rowFilterStart, colInputStart - colFilterStart);
                                var xIdxForStartRow = x.Idx(m, channelId, rowInputStart, colInputStart);
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
                            y.AsFloatCpu.Set(m, filterId, rowOutput, colOutput, (float) outputPointResult);
                            colFilterStart += stride;
                        }
                        rowFilterStart += stride;
                    }
                }
            }

            System.Threading.Tasks.Parallel.For(0, batchSize, ComputeForBatch);
        }
        public override void ConvolutionGradient(Tensor conv, Tensor dy, int padding, int stride, Tensor dx, Tensor convGradient)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> {x, conv, dy, dx, convGradient}));
            int batchSize = x.Shape[0];
            int channelCount = x.Shape[1];
            int hInput = x.Shape[2];
            int wInput = x.Shape[3];
            int fitlerCount = conv.Shape[0];
            int F = conv.Shape[2];
            Debug.Assert(F == conv.Shape[3]);
            Debug.Assert(channelCount == conv.Shape[1]);
            Debug.Assert(fitlerCount == dy.Shape[1]);
            Debug.Assert(batchSize == dy.Shape[0]);
            int hOutput = dy.Shape[2];
            Debug.Assert(hOutput == ((hInput - F + 2 * padding) / stride + 1));
            int wOutput = dy.Shape[3];
            Debug.Assert(wOutput == ((wInput - F + 2 * padding) / stride + 1));
            dx?.ZeroMemory();
            convGradient.ZeroMemory();

            //the first (top left) point in 'y' is computed from a filter starting at (-padding,-padding)

            void ComputeForBatch(int m)
            {
                //every thread needs to update 'convolutionGradient'
                //to be thread safe, each thread will update a local object 'convolutionGradientContent' and at the end
                //will update the object 'convolutionGradient' with a local
                float[] convolutionGradientForLocalThreadFloat = new float[convGradient.Count];
                for (int filterId = 0; filterId < fitlerCount; ++filterId)
                {
                    int rowFilterStart = -padding;
                    for (int rowOutput = 0; rowOutput < hOutput; ++rowOutput)
                    {
                        int colFilterStart = -padding;
                        var rowInputStart = Math.Max(0, rowFilterStart);
                        var rowInputEndExcluded = Math.Min(hInput, rowFilterStart + F);
                        for (int colOutput = 0; colOutput < wOutput; ++colOutput)
                        {
                            //we want to compute the point in y[m, filterId, rowOutput, colOutput]
                            //it is computed by applying a filter located (for its top left) in (row_filter_start,col_filter_start) in the x 
                            // and centered at this particular location
                            var chainGradientFloat = dy.AsFloatCpu.Get(m, filterId, rowOutput, colOutput);
                            var colInputStart = Math.Max(0, colFilterStart);
                            var colInputEndExcluded = Math.Min(wInput, colFilterStart + F);
                            for (int channelId = 0; channelId < channelCount; ++channelId)
                            {
                                int convIdxStartRow = convGradient.Idx(filterId, channelId,
                                    rowInputStart - rowFilterStart, colInputStart - colFilterStart);
                                int APrevLayerIdxStartRow = x.Idx(m, channelId, rowInputStart, colInputStart);
                                for (int rowInput = rowInputStart; rowInput < rowInputEndExcluded; ++rowInput)
                                {
                                    var convIdx = convIdxStartRow;
                                    var APrevLayerIdx = APrevLayerIdxStartRow;
                                    for (int colInput = colInputStart; colInput < colInputEndExcluded; ++colInput)
                                    {
                                        convolutionGradientForLocalThreadFloat[convIdx] +=
                                            x.AsFloatCpuContent[APrevLayerIdx] * chainGradientFloat;
                                        if (dx != null)
                                        {
                                            dx.AsFloatCpuContent[APrevLayerIdx] +=
                                                conv.AsFloatCpuContent[convIdx] * chainGradientFloat;
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
                result += ComputeSingleAccuracyCount(yExpectedOneHotCpu, yPredictedCpu, m);
            }
            return ((double)result)/Shape[0];
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
        /// <param name="toDouble">Function to convert 'T' type to double</param>
        /// <returns>A list of Tuple (one Tuple per channel)
        /// In each channel Tuple: Tuple.Item1: mean of the channel / Tuple.Item2: vol of the channel</returns>
        // ReSharper disable once UnusedMember.Global
        public List<Tuple<double, double>> ComputeMeanAndVolatilityOfEachChannel(Func<T, double> toDouble)
        {
            return Enumerable.Range(0, Shape[1]).Select(c => ComputeMeanAndVolatilityOfChannel(c, toDouble)).ToList();
        }

        /// <summary>
        /// Computes the mean and volatility of the selected channel in the 'this' tensor
        /// </summary>
        /// <param name="c">The channel to compute in the tensor</param>
        /// <param name="toDouble">Function to convert 'T' type to double</param>
        /// <returns>Tuple.Item1: mean of the channel / Tuple.Item2: vol of the channel</returns>
        private Tuple<double, double> ComputeMeanAndVolatilityOfChannel(int c, Func<T, double> toDouble)
        {
            double sum = 0.0;
            double sumSquare = 0.0;
            int count = 0;
            for (int m = 0; m < Shape[0]; ++m)
            {
                int startIdx = Idx(m, c, 0, 0);
                for (int idx = startIdx; idx < (startIdx + MultDim1); ++idx)
                {
                    var val = toDouble(Content[idx]);
                    sum += val;
                    sumSquare += val * val;
                    ++count;
                }
            }
            if (count == 0)
            {
                return Tuple.Create(0.0, 0.0);
            }
            var mean = (sum / count);
            var variance = (sumSquare / count) - mean * mean;
            var volatility = Math.Sqrt(Math.Max(0, variance));
            return Tuple.Create(mean, volatility);
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
        private void BuildEntirelyFromInput(Tensor a, Tensor b, Func<T, T, T> funcInput)
        {
            Debug.Assert(AreCompatible(new List<Tensor> {this, a, b}));
            Debug.Assert(SameShape(a, b));
            var aCpu = a.AsCpu<T>();
            var bCpu = b.AsCpu<T>();
            for (int i = 0; i < a.Count; ++i)
            {
                this[i] = funcInput(aCpu[i], bCpu[i]);
            }
        }
        private void BuildEntirelyFromInput(Tensor a, Tensor b, Tensor c, Func<T, T, T, T> funcInput)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, a, b, c }));
            Debug.Assert(SameShape(a, b));
            var aCpu = a.AsCpu<T>();
            var bCpu = b.AsCpu<T>();
            var cCpu = c.AsCpu<T>();
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
        private static void ReluGradient(Tensor dy, Tensor x, Tensor dx)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { dy, x, dx }));
            dx.AsFloatCpu.BuildEntirelyFromInput(dy, x, (da, z) => (z >= 0f ? da : 0f));
        }
        private static void EluGradient(Tensor y, Tensor dy, Tensor x, Tensor dx, float alpha)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { dy, x, dx }));
            dx.AsFloatCpu.BuildEntirelyFromInput(y, dy, x, (a, da, z) => (z >= 0.0 ? da : da * (a + alpha)));
        }
        private static void SigmoidGradient(Tensor y, Tensor dy, Tensor dx)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { y, dy, dx }));
            dx.AsFloatCpu.BuildEntirelyFromInput(y, dy, (a, da) => da * a * (1f - a));
        }
        private static void SoftmaxGradient(Tensor y, Tensor dy, Tensor dx)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { y, dy, dx }));
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
        private static int ComputeSingleAccuracyCount(CpuTensor<float> yExpectedOneHot, CpuTensor<float> yPredicted, int m)
        {
            Debug.Assert(yExpectedOneHot.SameShape(yPredicted));
            Debug.Assert(yExpectedOneHot.Dimension == 2);
            var categoryCount = yExpectedOneHot.Shape[1];
            if (categoryCount == 1)
            {
                var error = Math.Abs(yExpectedOneHot.Get(m, 0) - yPredicted.Get(m, 0));
                return (error < 0.5) ? 1 : 0;
            }
            int maxIndexPredicted = 0;
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
    }
}
