using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.HyperParameters;
using SharpNet.Layers;
using SharpNet.MathTools;
using static SharpNet.GPU.GPUWrapper;

namespace SharpNet.CPU
{
    public unsafe class CpuTensor<T> : Tensor
    {
        #region fields
        public Memory<T> Content { get; private set; }
        /// <summary>
        /// used only if the tensor is NOT the owner of the memory
        /// </summary>
        private readonly IntPtr _ptrToOwnerPinnedMemory;
        /// <summary>
        /// used only if the tensor is the owner of the memory
        /// </summary>
        private HostPinnedMemory<T> _hostPinnedMemory;
        #endregion

        #region constructors
        public CpuTensor(int[] shape, T[] data, int typeSize) : base(shape, typeSize, false)
        {
            Content = data ?? new T[Count];
            CapacityInBytes = (ulong)(Content.Length * TypeSize);
            _ptrToOwnerPinnedMemory = IntPtr.Zero;
        }
        public CpuTensor(int[] shape, T[] data = null) : this(shape, data, typeof(T).IsValueType ?Marshal.SizeOf(typeof(T)) : IntPtr.Size)
        {
        }
        private CpuTensor(int[] shape, CpuTensor<T> memoryOwner, int startIndex) : base(shape, memoryOwner.TypeSize, false)
        {
            Content = memoryOwner.Content.Slice(startIndex, Utils.Product(shape));
            CapacityInBytes = (ulong)(Content.Length * TypeSize);
            _ptrToOwnerPinnedMemory = memoryOwner.Pointer + TypeSize * startIndex;
        }
        public static CpuTensor<T> New(T[] content, int columns)
        {
            if (content == null || content.Length == 0)
            {
                return null;
            }
            Debug.Assert(content.Length % columns == 0);
            int rows = content.Length / columns;
            return new CpuTensor<T>(new[] { rows, columns }, content);
        }
        #endregion

        /// <summary>
        /// pointer to (pinned) host memory (in CPU)
        /// </summary>
        public override IntPtr Pointer
        {
            get
            {
                if (!IsOwnerOfMemory)
                {
                    //the memory owner has its memory already pinned
                    Debug.Assert(_ptrToOwnerPinnedMemory != IntPtr.Zero);
                    return _ptrToOwnerPinnedMemory;
                }
                Debug.Assert(_ptrToOwnerPinnedMemory == IntPtr.Zero);
                if (_hostPinnedMemory == null)
                {
                    _hostPinnedMemory = new HostPinnedMemory<T>(Content);
                }
                return _hostPinnedMemory.Pointer;
            }
        }

        /// <summary>
        /// true if the tensor memory is currently pinned
        /// </summary>
        private bool HasPinnedMemory => !IsOwnerOfMemory || _hostPinnedMemory != null;

        public override void WordEmbeddingForwardPropagation(Tensor x, Tensor wordEmbedding, int xIndexInLastDimensionToUse, int yIndexInLastDimensionToUse, int copyCountBeforeIndex, int copyCountAfterIndex)
        {
            var y = this;
            Debug.Assert(wordEmbedding.Shape.Length == 2);
            Debug.Assert(x.Shape[0] == y.Shape[0]); //same batchSize
            Debug.Assert(x.Shape[1] == y.Shape[1]); //same timeSteps
            Debug.Assert(x.Shape.Length == 3);
            Debug.Assert(y.Shape.Length == 3);

            Debug.Assert(xIndexInLastDimensionToUse>=0);
            Debug.Assert(yIndexInLastDimensionToUse>=0);
            var timeSteps = x.Shape[1];
            var embeddingDim = wordEmbedding.Shape[1];

            void ProcessBatch(int batchIndex)
            {
                var xSpan = x.AsReadonlyFloatCpuSpan;
                var ySpan = y.AsFloatCpuSpan;
                var wordEmbeddingSpan = wordEmbedding.AsReadonlyFloatCpuSpan;

                for (int timeStep = 0; timeStep < timeSteps; ++timeStep)
                {
                    int xTimeStepIndex = x.Idx(batchIndex, timeStep, xIndexInLastDimensionToUse);
                    int yTimeStepIndex = y.Idx(batchIndex, timeStep, yIndexInLastDimensionToUse);


                    //for the current timeStep, we copy the elements from 'x' to 'y' before 'indexInLastDimensionToUse'
                    //int xElementsBeforeEmbeddingIndex = indexInLastDimensionToUse;
                    if (copyCountBeforeIndex > 0)
                    {
                        //we copy 'xElementsBeforeEmbeddingIndex' elements before index 'indexInLastDimensionToUse'
                        xSpan.Slice(xTimeStepIndex- copyCountBeforeIndex, copyCountBeforeIndex).CopyTo(ySpan.Slice(yTimeStepIndex- copyCountBeforeIndex, copyCountBeforeIndex));
                    }

                    int wordIndex = (int)(xSpan[xTimeStepIndex] + 0.1);
                    wordEmbeddingSpan.Slice(wordIndex*embeddingDim, embeddingDim).CopyTo(ySpan.Slice(yTimeStepIndex, embeddingDim));

                    //for the current timeStep, we copy the elements from 'x' to 'y' after 'indexInLastDimensionToUse'
                    //int xElementsAfterEmbeddingIndex = inputSize - indexInLastDimensionToUse - 1;
                    if (copyCountAfterIndex > 0)
                    {
                        //we copy the 'xElementsAfterEmbeddingIndex' elements after index 'indexInLastDimensionToUse'
                        xSpan.Slice(xTimeStepIndex+ 1, copyCountAfterIndex).CopyTo(ySpan.Slice(yTimeStepIndex+ embeddingDim, copyCountAfterIndex));
                    }
                }
            }
            Parallel.For(0, x.Shape[0], ProcessBatch);
        }

        public override void WordEmbeddingBackwardPropagation(/*in*/ Tensor x, /*out*/ Tensor dx, /*in*/ Tensor dy, int dxIndexInLastDimensionToUse, int dyIndexInLastDimensionToUse, int copyCountBeforeIndex, int copyCountAfterIndex)
        {
            var dW = this;

            Debug.Assert(dW.Shape.Length == 2);
            Debug.Assert(x.Shape.Length == 3);
            Debug.Assert(dy.Shape.Length == 3);
            Debug.Assert(x.Shape[0] == dy.Shape[0]); //same batchSize
            Debug.Assert(x.Shape[1] == dy.Shape[1]); //same timeSteps
            Debug.Assert(dxIndexInLastDimensionToUse >= 0);
            Debug.Assert(dyIndexInLastDimensionToUse >= 0);

            var xCpu = (CpuTensor<float>)x;
            var dyCpu = (CpuTensor<float>)dy;

            dW.ZeroMemory();
            var batchSize = dy.Shape[0];
            var timeSteps = x.Shape[1];
            var embeddingDim = dW.Shape[1];

            var xSpan = x.AsReadonlyFloatCpuSpan;
            var dxSpan = dx.AsFloatCpuSpan;
            var dWSpan = dW.AsFloatCpuSpan;
            var dySpan = dy.AsReadonlyFloatCpuSpan;

            for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
            {
                for (int timeStep = 0; timeStep < timeSteps; ++timeStep)
                {
                    //we initialize 'dw' for the current batchIndex & timeStep
                    int wordIndex = (int)(xSpan[xCpu.Idx(batchIndex, timeStep, dxIndexInLastDimensionToUse)] + 0.1);
                    int indexInDw = dW.Idx(wordIndex, 0);
                    int indexIndY = dyCpu.Idx(batchIndex, timeStep, dyIndexInLastDimensionToUse);
                    for (int embeddingId = 0; embeddingId < embeddingDim; ++embeddingId)
                    {
                        dWSpan[indexInDw] += dySpan[indexIndY];
                        ++indexInDw;
                        ++indexIndY;
                    }


                    int dyTimeStepIndex = dy.Idx(batchIndex, timeStep, dyIndexInLastDimensionToUse);
                    int dxTimeStepIndex = dx.Idx(batchIndex, timeStep, dxIndexInLastDimensionToUse);

                    //we initialize 'dx' for the current batchIndex & timeStep
                    //for the current timeStep, we copy the elements from 'dy' to 'dx' before 'indexInLastDimensionToUse'
                    //int dyElementsBeforeEmbeddingIndex = prevXIndexInLastDimensionToUse==-1?xIndexInLastDimensionToUse:(xIndexInLastDimensionToUse-prevXIndexInLastDimensionToUse-1);
                    if (copyCountBeforeIndex > 0)
                    {
                        //we copy 'xElementsBeforeEmbeddingIndex' elements before index 'xIndexInLastDimensionToUse'
                        dySpan.Slice(dyTimeStepIndex- copyCountBeforeIndex, copyCountBeforeIndex).CopyTo(dxSpan.Slice(dxTimeStepIndex- copyCountBeforeIndex, copyCountBeforeIndex));
                    }
                    dxSpan[dxTimeStepIndex] = 0;
                    //for the current timeStep, we copy the elements from 'dy' to 'dx' after 'xIndexInLastDimensionToUse'
                    //int dyElementsAfterEmbeddingIndex = nextXIndexInLastDimensionToUse==-1 ? (inputSize - xIndexInLastDimensionToUse - 1):(nextXIndexInLastDimensionToUse-xIndexInLastDimensionToUse-1);
                    if (copyCountAfterIndex > 0)
                    {
                        //we copy the 'xElementsAfterEmbeddingIndex' elements after index 'indexInLastDimensionToUse'
                        dySpan.Slice(dyTimeStepIndex + embeddingDim, copyCountAfterIndex).CopyTo(dxSpan.Slice(dxTimeStepIndex + 1, copyCountAfterIndex));
                    }
                }
            }
        }

        /// <summary>
        /// resize the current Cpu tensor to a different shape (both bigger or smaller)
        /// </summary>
        /// <param name="newShape"></param>
        public override void ReshapeInPlace(params int[] newShape)
        {
            newShape = FillMinusOneIfAny(Shape, newShape);
            if (SameShape(newShape))
            {
                return;
            }
            else if (HasEnoughCapacityForTensor(newShape))
            {
                //smaller shape
                Shape = newShape;
            }
            else
            {
                //bigger shape
                if (!IsOwnerOfMemory)
                {
                    throw new ArgumentException("must be memory owner to increase memory associated with the 'this' Tensor");
                }
                _hostPinnedMemory?.Dispose();
                _hostPinnedMemory = null;
                Content = new T[Utils.Product(newShape)];
                CapacityInBytes = (ulong)(Content.Length * TypeSize);
                Shape = newShape;
            }
            RecomputeMultDim();
        }
        public override Tensor Reshape(params int[] newShape)
        {
            AssertIsNotDisposed();
            newShape = FillMinusOneIfAny(Shape, newShape);
            if (SameShape(newShape))
            {
                return this;
            }
            if (ReallyNeededMemoryInBytesForShape(newShape) <= CapacityInBytes)
            {
                return new CpuTensor<T>(newShape, this, 0);
            }
            //bigger shape : we do not have enough space to store it
            throw new ArgumentException("CapacityInBytes: " + CapacityInBytes + " but need memory  " + ReallyNeededMemoryInBytesForShape(newShape) + " for " + this);
        }


        public T this[int i]
        {
            get => ReadonlyContent[i];
            set => SpanContent[i] = value;
        }

        public override void Switch_First_2_axis(Tensor target)
        {
            Debug.Assert(target.Count == Count);
            Debug.Assert(Shape.Length >= 2);
            int aLength = Shape[0];
            int bLength = Shape[1];
            int cLength = MultDim1;
            int multDim0 = bLength * cLength;
            var srcContent = AsReadonlyFloatCpuSpan;
            var targetContent = target.AsFloatCpuSpan;

            for (int idx_src = 0; idx_src < Count; ++idx_src)
            {
                int a_src = idx_src / multDim0;
                int tmp = idx_src % multDim0;
                int b_src = tmp / cLength;
                int c_src = tmp % cLength;
                int idx_target = b_src * aLength * cLength + a_src * cLength + c_src;
                targetContent[idx_target] = srcContent[idx_src];
            }

            var targetShape = (int[]) Shape.Clone();
            targetShape[0] = bLength;
            targetShape[1] = aLength;
            target.ReshapeInPlace(targetShape);
        }

        public override void SwitchSecondAndThirdDimension(Tensor target)
        {
            Debug.Assert(Shape.Length == 3 || (Shape.Length==4&&Shape[3] == 1));
            var srcContent = AsReadonlyFloatCpuSpan;
            var targetContent = target.AsFloatCpuSpan;
            for (int n = 0; n < Shape[0]; ++n)
            {
                for (int c = 0; c < Shape[1]; ++c)
                {
                    for (int h = 0; h < Shape[2]; ++h)
                    {
                        targetContent[target.Idx(n, h, c)] = srcContent[Idx(n, c, h)];
                    }
                }
            }
        }

        public override void TransposeSecondAndThirdDimension(Tensor target)
        {
            Debug.Assert(Shape.Length >= 3);
            var targetShape = (int[])Shape.Clone();
            (targetShape[1], targetShape[2]) = (targetShape[2], targetShape[1]);
            target.ReshapeInPlace(targetShape);
            var src = Reshape(Shape[0], Shape[1], Shape[2], -1);
            var srcSpan = src.AsFloatCpuSpan;
            var targetSpan = target.AsFloatCpuSpan;

            int A = Shape[0];
            int B = Shape[1];
            int C = Shape[2];
            int D = Count/(A*B*C);
            for (int a=0;a<A;++a)
            for(int b=0;b<B;++b)
            for(int c=0;c<C;++c)
            for (int d = 0; d < D; ++d)
            {
                targetSpan[target.Idx(a, c, b, d)] = srcSpan[src.Idx(a, b, c, d)];
            }
        }
        

        public override Tensor ChangeAxis(int[] targetAxisToSrcAxis)
        {
            Debug.Assert(targetAxisToSrcAxis.Length == Dimension);
            Debug.Assert(targetAxisToSrcAxis.Min() == 0);
            Debug.Assert(targetAxisToSrcAxis.Max() == Dimension-1);

            var srcAxisToTargetAxis = new int[Dimension];
            for (int targetAxis = 0; targetAxis < Dimension; ++targetAxis)
            {
                srcAxisToTargetAxis[targetAxisToSrcAxis[targetAxis]] = targetAxis;
            }

            var targetShape = new int[Dimension];
            for (int targetAxis = 0; targetAxis < Dimension; ++targetAxis)
            {
                targetShape[targetAxis] = Shape[targetAxisToSrcAxis[targetAxis]];
            }

            var result = new CpuTensor<T>(targetShape);

            var idxInSrcAxis = new int[Dimension];
            var srcMultDim = new int[Dimension];
            var idxInTargetAxis =  new int[Dimension];
            var targetMultDim = new int[Dimension];
            srcMultDim[^1] = 1;
            targetMultDim[^1] = 1;
            for (int dim = Dimension - 2; dim >= 0; --dim)
            {
                srcMultDim[dim] = Shape[dim + 1] * srcMultDim[dim + 1];
                targetMultDim[dim] = targetShape[dim + 1] * targetMultDim[dim + 1];
            }

            void ProcessDimension(int axisSrc)
            {
                for (int idxInAxis = 0; idxInAxis < Shape[axisSrc]; ++idxInAxis)
                {
                    idxInTargetAxis[srcAxisToTargetAxis[axisSrc]] = idxInAxis;
                    idxInSrcAxis[axisSrc] = idxInAxis;
                    if (axisSrc == Dimension - 1)
                    {
                        int targetIdx = 0;
                        int srcIdx = 0;
                        for (int axis = 0; axis < Dimension; ++axis)
                        {
                            srcIdx += idxInSrcAxis[axis] * srcMultDim[axis];
                            targetIdx += idxInTargetAxis[srcAxisToTargetAxis[axis]] * targetMultDim[srcAxisToTargetAxis[axis]];
                        }
                        result[targetIdx] = this[srcIdx];
                    }
                    else
                    {
                        ProcessDimension(axisSrc + 1);
                    }
                }
            }

            ProcessDimension(0);
            return result;
        }

        public override bool IsOwnerOfMemory => _ptrToOwnerPinnedMemory == IntPtr.Zero;
        public ReadOnlySpan<T> ReadonlyContent => Content.Slice(0, Count).Span;
        public Span<T> SpanContent => Content.Slice(0, Count).Span;
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
            Debug.Assert(Count == result.Count);
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
            var result = new CpuTensor<TY>(Shape);
            Debug.Assert(SameShape(result));
            var content = ReadonlyContent;
            for (int m = 0; m < Shape[0]; ++m)
            {
                for (int c = 0; c < Shape[1]; ++c)
                {
                    int startIdx = Idx(m, c);
                    for (int idx = startIdx; idx < (startIdx + MultDim1); ++idx)
                    {
                        result[idx] = func(m, c, content[idx]);
                    }
                }
            }
            return result;
        }

        public CpuTensor<TY> Select<TY>(Func<T, TY> func) where TY : struct
        {
            var result = new CpuTensor<TY>(Shape);
            Debug.Assert(SameShape(result));
            var content = ReadonlyContent;
            var resultSpan = result.SpanContent;
            for (int i = 0; i < Count; ++i)
            {
                resultSpan[i] = func(content[i]);
            }
            return result;
        }

        #region Tensor implementation
        public override void UpdateSGDOptimizer(double learningRate, double momentum, bool usenesterov, Tensor dW, Tensor velocity)
        {
            var W = this;
            var wContent = W.AsFloatCpuSpan;
            var dWContent = dW.AsFloatCpuSpan;
            var velocityContent = velocity.AsFloatCpuSpan;
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
        public override void BatchNormalization(Tensor y, Tensor scale, Tensor bias, double exponentialAverageSmoothingFactor, Tensor runningInputMean, Tensor runningInputVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer, bool isTraining)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor>{x,y,scale,bias,runningInputMean,runningInputVariance,meanBuffer,invertOfUnbiasedVolatilityBuffer}));
            Debug.Assert(x.SameShape(y));
            Debug.Assert(scale.SameShape(bias, runningInputMean, runningInputVariance, meanBuffer, invertOfUnbiasedVolatilityBuffer));
            bool is1C11Shape = bias.Count == bias.Shape[1];
            var meanDivider = Count / bias.Count;  // = batchSize if (1,C,H,W) , and = batchSize*H*W if (1,C,1,1)

            
            var batchSize = x.Shape[0];

            var xContent = x.AsFloatCpuSpan;
            var yContent = y.AsFloatCpuSpan;
            var scaleContent = scale.AsFloatCpuSpan;
            var biasContent = bias.AsFloatCpuSpan;

            // 'meanBuffer' & 'invertOfUnbiasedVolatilityBuffer' will only be used when isTraining = true
            var meanContent = isTraining?meanBuffer.AsFloatCpuSpan:null;
            var invertOfUnbiasedVolatility = isTraining ? invertOfUnbiasedVolatilityBuffer.AsFloatCpuSpan:null;

            var runningInputMeanContent = runningInputMean.AsFloatCpuSpan;
            var runningInputVarianceContent = runningInputVariance.AsFloatCpuSpan;


            if (isTraining)
            {
                //'invertOfUnbiasedVolatilityBuffer' will temporary store the variance of the input 
                Compute_Column_Mean_Variance(meanBuffer, invertOfUnbiasedVolatilityBuffer);
                var variance = invertOfUnbiasedVolatilityBuffer.AsFloatCpuSpan;

                //we need to update 'runningInputMean' and 'runningInputVariance'
                for (int j = 0; j < runningInputVariance.Count; ++j)
                {
                    runningInputMeanContent[j] = (float) (meanContent[j] * exponentialAverageSmoothingFactor + runningInputMeanContent[j] * (1 - exponentialAverageSmoothingFactor));
                    runningInputVarianceContent[j] = (float)(variance[j] * exponentialAverageSmoothingFactor + runningInputVarianceContent[j] * (1 - exponentialAverageSmoothingFactor));
                }

                //we update 'invertOfUnbiasedVolatilityBuffer' so that it stores the invert of the unbiased volatility of the input
                for (int j = 0; j < invertOfUnbiasedVolatilityBuffer.Count; ++j)
                {
                    invertOfUnbiasedVolatility[j] = (float)(1.0 / Math.Sqrt(((meanDivider - 1) * variance[j]) / meanDivider + epsilon));
                }
            }

            int idx = 0;
            for (int n = 0; n < batchSize; ++n)
            {
                for (int j = 0; j < MultDim0; ++j)
                {
                    int scaleIndex = is1C11Shape ? (j / MultDim1) : j;
                    var xTarget = isTraining
                        ? ((xContent[idx] - meanContent[scaleIndex]) * invertOfUnbiasedVolatility[scaleIndex])
                        : (float)((xContent[idx] - runningInputMeanContent[scaleIndex]) / Math.Sqrt(runningInputVarianceContent[scaleIndex] + epsilon));
                    yContent[idx++] = scaleContent[scaleIndex] * xTarget + biasContent[scaleIndex];
                }
            }
        }
        public override void BatchNormalizationBackward(Tensor dy, Tensor dx, Tensor scale, Tensor scaleGradient, Tensor biasGradient, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer)
        {
            var x = this;
            var batchSize = x.Shape[0];
            Debug.Assert(AreCompatible(new List<Tensor> {x, dy, dx, scale, scaleGradient, biasGradient, meanBuffer, invertOfUnbiasedVolatilityBuffer}));
            Debug.Assert(x.SameShape(dy, dx));
            Debug.Assert(scale.SameShape(scaleGradient, biasGradient, meanBuffer, invertOfUnbiasedVolatilityBuffer));
            bool is1C11Shape = scale.Count == scale.Shape[1];
            var meanDivider = Count / scale.Count;  // = batchSize if (1,C,H,W) , and = batchSize*H*W if (1,C,1,1)
            scaleGradient.ZeroMemory();
            dx?.ZeroMemory();

            //we compute resultBnBiasDiff
            dy.AsFloatCpu.ComputeSumByColumn(biasGradient);
            //we compute resultBnScaleDiff
            var xContent = x.AsFloatCpuSpan;
            var dyContent = dy.AsFloatCpuSpan;
            Span<float> dxContent = null;
            if (dx != null)
            {
                dxContent = dx.AsFloatCpuSpan;
            }

            var biasGradientContent = biasGradient.AsFloatCpuSpan;
            var scaleGradientContent = scaleGradient.AsFloatCpuSpan;
            var scaleContent = scale.AsFloatCpuSpan;
            var meanBufferContent = meanBuffer.AsFloatCpuSpan;
            var invertOfUnbiasedVolatility = invertOfUnbiasedVolatilityBuffer.AsFloatCpuSpan;
            for (int j = 0; j < MultDim0; ++j)
            {
                int meanIndex = is1C11Shape ? (j / MultDim1) : j;
                double result = 0.0;
                for (int n = 0; n < batchSize; ++n)
                {

                    int idx = n * MultDim0 + j;
                    result += dyContent[idx] * (xContent[idx] - meanBufferContent[meanIndex]);
                }
                scaleGradientContent[meanIndex] += (float) (result * invertOfUnbiasedVolatility[meanIndex]);
            }
            //we compute dx
            for (int i = 0; i < batchSize; ++i)
            {
                for (int j = 0; j < MultDim0; ++j)
                {
                    int meanIndex = is1C11Shape ? (j / MultDim1) : j;
                    int idx = i * MultDim0 + j;
                    double result = meanDivider * dyContent[idx] - biasGradientContent[meanIndex] - scaleGradientContent[meanIndex] * invertOfUnbiasedVolatility[meanIndex] * (xContent[idx] - meanBufferContent[meanIndex]);
                    if (dxContent != null)
                    {
                        dxContent[idx] += (float) ((scaleContent[meanIndex] * invertOfUnbiasedVolatility[meanIndex] * result) /meanDivider);
                    }
                }
            }
        }


        public override void StandardizeInPlace(Tensor row_mean, Tensor row_variance, int axis, float epsilon)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { this, row_mean, row_variance}));
            Debug.Assert(row_mean.SameShape(row_variance));
            if (axis == 1)
            {
                //we'll standardize each row
                int rows = row_mean.Count;
                if (x.Count % rows != 0)
                {
                    throw new ArgumentException("The number of elements in the tensor must be a multiple of the number of rows");
                }
                int cols = x.Count / rows;
                void ProcessRow(int row)
                {
                    var xSpan = x.AsFloatCpuSpan;
                    var row_mean_value = row_mean.AsFloatCpuSpan[row];
                    var row_variance_value = row_variance.AsFloatCpuSpan[row];
                    int startIndex = row * cols;
                    int endIndex = startIndex + cols - 1;
                    for (int i = startIndex; i <= endIndex; ++i)
                    {
                        xSpan[i] = (xSpan[i] - row_mean_value) / MathF.Sqrt(row_variance_value + epsilon);
                    }
                }
                Parallel.For(0, rows, ProcessRow);
                return;
            }
            throw new NotSupportedException("Only axis=1 is supported");
        }

        public override void StandardizeRowsInPlaceBroadcastGammasBetas(Tensor row_mean, Tensor row_variance, float epsilon, Tensor col_gammas, Tensor col_betas)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { this, row_mean, row_variance }));
            Debug.Assert(row_mean.SameShape(row_variance));
            //we'll standardize each row
            int rows = row_mean.Count;
            if (x.Count % rows != 0)
            {
                throw new ArgumentException("The number of elements in the tensor must be a multiple of the number of rows");
            }
            int cols = x.Count / rows;
            void ProcessRow(int row)
            {
                var xSpan = x.AsFloatCpuSpan;
                var row_mean_value = row_mean.AsFloatCpuSpan[row];
                var row_variance_value = row_variance.AsFloatCpuSpan[row];
                var col_gammas_span = col_gammas.AsReadonlyFloatCpuSpan;
                var col_betas_span = col_betas.AsReadonlyFloatCpuSpan;

                int startIndex = row * cols;
                int endIndex = startIndex + cols - 1;
                int col = 0;
                for (int i = startIndex; i <= endIndex; ++i)
                {
                    xSpan[i] = (xSpan[i] - row_mean_value) / MathF.Sqrt(row_variance_value + epsilon);
                    xSpan[i] = col_gammas_span[col]* xSpan[i] + col_betas_span[col];
                    ++col;
                }
            }
            Parallel.For(0, rows, ProcessRow);
        }

        public override void numpy_sum(Tensor sum_result, int axis)
        {
            var a = this;
            Debug.Assert(AreCompatible(new List<Tensor> { a, sum_result}));
            sum_result.ZeroMemory();
            var sum_result_as_span = sum_result.AsFloatCpuSpan;
            var aSpan = a.AsReadonlyFloatCpuSpan;
            if (axis == 1)
            {
                int rows = sum_result.Count;
                if (a.Count % rows != 0)
                {
                    throw new ArgumentException("x.Count % rows != 0");
                }
                int cols = a.Count / rows;
                for (int row = 0; row < rows; ++row)
                {
                    var row_sum = 0.0f;
                    for (int col = 0; col < cols; ++col)
                    {
                        row_sum += aSpan[row * cols + col];
                    }
                    sum_result_as_span[row] = row_sum;
                }

                return;
            }
            if (axis == 0)
            {
                int cols = sum_result.Count;
                if (a.Count % cols != 0)
                {
                    throw new ArgumentException("x.Count % cols != 0");
                }
                int rows = a.Count / cols;
                for (int row = 0; row < rows; ++row)
                {
                    for (int col = 0; col < cols; ++col)
                    {
                        sum_result_as_span[col] += aSpan[row * cols + col];
                    }
                }
                return;
            }
            throw new ArgumentException("axis != 0 && axis != 1");
        }
        

        public override void Compute_Row_Mean_Variance(Tensor row_mean, Tensor row_variance, bool unbiasedVariance)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { this, row_mean, row_variance}));
            Debug.Assert(row_mean.SameShape(row_variance));
            int rows = row_mean.Count;
            if (x.Count % rows != 0)
            {
                throw new ArgumentException("x.Count % rows != 0");
            }
            int cols = x.Count / rows;

            void ProcessBlock(int rowId)
            {
                var xSpan = x.AsFloatCpuSpan;
                int startIndex = rowId * cols;
                int endIndex = startIndex + cols - 1;
                double sum = 0.0;
                double sumSquare = 0.0;
                for (int i = startIndex; i <= endIndex; ++i)
                {
                    double xValue = xSpan[i];
                    sum += xValue;
                    sumSquare += xValue * xValue;
                }
                var row_mean_value = (sum / cols);
                var divider = unbiasedVariance ? (cols - 1) : cols;
                var row_variance_value = Math.Abs(sumSquare - cols * row_mean_value * row_mean_value) / divider;
                row_mean.AsFloatCpuSpan[rowId] = (float)row_mean_value;
                row_variance.AsFloatCpuSpan[rowId] = (float)row_variance_value;
            }
            Parallel.For(0, rows, ProcessBlock);
        }

        public override void LayerNormalizationBackward(Tensor dy, Tensor dx, Tensor col_gammas, Tensor row_mean, Tensor row_variance, float epsilon, Tensor dmean, Tensor dvariance)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, dy, dx, col_gammas, row_mean, row_variance}));
            Debug.Assert(x.SameShape(dy, dx));
            Debug.Assert(row_mean.SameShape(row_variance));
            int rows = row_mean.Count;
            int cols= col_gammas.Count;
            if (x.Count != rows * cols)
            {
                throw new ArgumentException("x.Count != rows * cols");
            }

            void ComputeDxForRow(int row)
            {
                var gammaSpan = col_gammas.AsFloatCpuSpan;
                var mean_row = row_mean.AsFloatCpuSpan[row];
                var variance_row = row_variance.AsFloatCpuSpan[row];
                var volatility_row = MathF.Sqrt(variance_row + epsilon);
                var x_row = x.AsFloatCpu.SpanSlice(row * cols, cols);
                var dy_row = dy.AsFloatCpu.SpanSlice(row * cols, cols);
                var dvariance_row = 0f;
                var dmean_row = 0f;
                for (int col = 0; col < cols; ++col)
                {
                    var tmp0 = (dy_row[col] * gammaSpan[col]);
                    dvariance_row += tmp0 * (x_row[col]-mean_row);
                    dmean_row -= tmp0;
                }
                dvariance_row *= (-0.5f * MathF.Pow(variance_row + epsilon, -1.5f));
                dmean_row /= volatility_row;
                for (int col = 0; col < cols; ++col)
                {
                    dmean_row += dvariance_row*(x_row[col] -mean_row) * (-2f/cols);
                }
                var dx_row = dx.AsFloatCpu.SpanSlice(row * cols, cols);
                for (int col = 0; col < cols; ++col)
                {
                    dx_row[col] = (dy_row[col] * gammaSpan[col]) /volatility_row
                                + dvariance_row * (2f / cols) * (x_row[col] - mean_row)
                                + dmean_row / cols;
                }
            }
            Parallel.For(0, rows, ComputeDxForRow);
        }

        public override void DropoutForward(Tensor y, double dropoutRate, bool isTraining, Random dropoutRandom, Tensor dropoutReservedSpaceForTraining)
        {
            var x = this;
            if (!isTraining)
            {
                x.CopyTo(y);
                return;
            }
            Debug.Assert(!dropoutReservedSpaceForTraining.UseGPU);
            var dropoutRateFloat = (float)dropoutRate;
            Utils.UniformDistribution(dropoutReservedSpaceForTraining.AsFloatCpuSpan, dropoutRandom, 0.0, 1.0);
            y.AsFloatCpu.BuildEntirelyFromInput(x, dropoutReservedSpaceForTraining, (prevLayer, prob) => prob < dropoutRate ? 0f : prevLayer / (1 - dropoutRateFloat));
        }
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropoutRate, Tensor dropoutReserveSpace)
        {
            Debug.Assert(!dropoutReserveSpace.UseGPU);
            var dropoutRateFloat = (float)dropoutRate;
            dx.AsFloatCpu.BuildEntirelyFromInput(dy, dropoutReserveSpace, (dOutput, prob) => prob < dropoutRateFloat ? 0f : dOutput / (1 - dropoutRateFloat));
        }
        //this = dy

        public override void ActivationForward(cudnnActivationMode_t activationType, Tensor activationParameter, Tensor y)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> {x, y}));
            switch (activationType)
            {
                case cudnnActivationMode_t.CUDNN_ACTIVATION_RELU:
                    CpuTensorActivationFunctions.Relu(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU:
                    Debug.Assert(activationParameter.Dimension == 1);
                    Debug.Assert(activationParameter.Count == 1);
                    CpuTensorActivationFunctions.LeakyRelu(x, y, activationParameter.AsReadonlyFloatCpuSpan[0]);
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
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION:
                    CpuTensorActivationFunctions.SoftmaxLastDimension(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY:
                    Debug.Assert(activationParameter.Dimension == 1);
                    CpuTensorActivationFunctions.SoftmaxWithHierarchy(x, y, activationParameter);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH:
                    CpuTensorActivationFunctions.Swish(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LN:
                    CpuTensorActivationFunctions.Ln(x, y);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY:
                    x.CopyTo(y);
                    return;
                default:
                    throw new ArgumentException("invalid activation mode " + activationType);
            }
        }
        public override void ActivationBackward(cudnnActivationMode_t activationType, Tensor activationParameter, Tensor dy, Tensor x, Tensor y)
        {
            var dx = this;
            Debug.Assert(AreCompatible(new List<Tensor> { y, dy, x, dx }));
            switch (activationType)
            {
                case cudnnActivationMode_t.CUDNN_ACTIVATION_RELU:
                    CpuTensorActivationFunctions.ReluGradient(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LEAKY_RELU:
                    Debug.Assert(activationParameter.Dimension == 1);
                    Debug.Assert(activationParameter.Count == 1);
                    CpuTensorActivationFunctions.LeakyReluGradient(y, dy, dx, activationParameter.AsReadonlyFloatCpuSpan[0]);
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
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_LAST_DIMENSION:
                    CpuTensorActivationFunctions.SoftmaxGradientLastDimension(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX:
                    CpuTensorActivationFunctions.SoftmaxGradient(y, dy, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY:
                    Debug.Assert(activationParameter.Dimension == 1);
                    CpuTensorActivationFunctions.SoftmaxGradientWitHierarchy(y, dy, dx, activationParameter);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH:
                    CpuTensorActivationFunctions.SwishGradient(y, dy, x, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_LN:
                    CpuTensorActivationFunctions.LnGradient(dy, x, dx);
                    return;
                case cudnnActivationMode_t.CUDNN_ACTIVATION_IDENTITY:
                    dy.CopyTo(dx);
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
            MKL_BLAS.cblas_saxpy(x.Count, alpha, x.AsFloatPointer, 1, y.AsFloatPointer, 1);
        }

        // compute: this = alpha * x + beta * this
        public override void AddTensor(float alpha, Tensor x, float beta)
        {
            // this = beta * this
            Update_Multiplying_By_Alpha(beta);
            // this = alpha * x + beta * this
            Update_Adding_Alpha_X(alpha, x);
        }

        public override void LinearFunction(float slope, Tensor x, float intercept)
        {
            Debug.Assert(this.SameShape(x));
            var yAsSpan = AsFloatCpuSpan;
            var xAsSpan = x.AsReadonlyFloatCpuSpan;
            for (int i = 0; i < xAsSpan.Length; ++i)
            {
                yAsSpan[i] = slope * xAsSpan[i] + intercept;
            }
        }

        public override void MultiplyTensor(Tensor a, Tensor diagonalMatrix)
        {
            Debug.Assert(this.SameShape(a));
            Debug.Assert(a.Count >= diagonalMatrix.Count);
            Debug.Assert(Count % diagonalMatrix.Count == 0);

            var aFloat = a.AsFloatCpuSpan;
            var xFloat = diagonalMatrix.AsFloatCpuSpan;
            var thisFloat = AsFloatCpuSpan;
            if (a.Count == diagonalMatrix.Count)
            {
                for (int i = 0; i < diagonalMatrix.Count; ++i)
                {
                    thisFloat[i] = aFloat[i] * xFloat[i];
                }
            }
            else
            {
                Debug.Assert(diagonalMatrix.Shape[0]*diagonalMatrix.Shape[1] == diagonalMatrix.Count);
                int indexInX = 0;
                int indexInThis = 0;
                int toAddInThis = a.Count / diagonalMatrix.Count;
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

        public override void UpSampling2D(Tensor tensorBeforeUpSampling, int rowFactor, int colFactor, UpSampling2DLayer.InterpolationEnum interpolation)
        {
            Debug.Assert(rowFactor >= 1);
            Debug.Assert(colFactor >= 1);
            if (interpolation == UpSampling2DLayer.InterpolationEnum.Bilinear)
            {
                throw new NotImplementedException("only " + UpSampling2DLayer.InterpolationEnum.Nearest + " interpolation is supported (not " + interpolation + ")");
            }
            var beforeUpSampling = (CpuTensor<T>)tensorBeforeUpSampling;
            var afterUpSampling = this;
            Debug.Assert(rowFactor * beforeUpSampling.Shape[2] == afterUpSampling.Shape[2]);
            Debug.Assert(colFactor * beforeUpSampling.Shape[3] == afterUpSampling.Shape[3]);
            for (int m = 0; m < afterUpSampling.Shape[0]; ++m)
            for (int c = 0; c < afterUpSampling.Shape[1]; ++c)
            for (int row = 0; row < afterUpSampling.Shape[2]; ++row)
            for (int col = 0; col < afterUpSampling.Shape[3]; ++col)
            {
                afterUpSampling.Set(m, c, row, col, beforeUpSampling.Get(m, c, row / rowFactor, col / colFactor));
            }
        }
        public override void DownSampling2D(Tensor tensorBeforeDownSampling, int rowFactor, int colFactor)
        {
            var beforeDownSampling = (CpuTensor<float>)tensorBeforeDownSampling;
            var afterDownSampling = AsFloatCpu;
            afterDownSampling.ZeroMemory();
            Debug.Assert(rowFactor >= 1);
            Debug.Assert(colFactor >= 1);
            Debug.Assert(rowFactor * afterDownSampling.Shape[2] == beforeDownSampling.Shape[2]);
            Debug.Assert(colFactor * afterDownSampling.Shape[3] == beforeDownSampling.Shape[3]);
            for (int m = 0; m < beforeDownSampling.Shape[0]; ++m)
            for (int c = 0; c < beforeDownSampling.Shape[1]; ++c)
            for (int row = 0; row < beforeDownSampling.Shape[2]; ++row)
            for (int col = 0; col < beforeDownSampling.Shape[3]; ++col)
            {
                var toAdd = beforeDownSampling.Get(m, c, row , col );
                var prevValue = afterDownSampling.Get(m, c, row / rowFactor, col / colFactor);
                afterDownSampling.Set(m, c, row / rowFactor, col / colFactor, toAdd + prevValue);
            }
        }

        public override void MultiplyEachRowIntoSingleValue(Tensor a, Tensor b)
        {
            Debug.Assert(a.SameShape(b));
            int nbRows = Count;
            Debug.Assert(nbRows <= a.Count);
            Debug.Assert(a.Count % nbRows == 0);
            int nbColumns_in_a_and_b = b.Count / nbRows;
            var thisFloat = AsFloatCpuSpan;
            var aFloat = a.AsFloatCpuSpan;
            var bFloat = b.AsFloatCpuSpan;
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

        public override void Clip(float lower, float upper)
        {
            Debug.Assert(upper >= lower);
            var thisFloat = AsFloatCpuSpan;
            for (int i = 0; i < Count; ++i)
            {
                var curValue = thisFloat[i];
                thisFloat[i] = Math.Min(Math.Max(curValue, lower), upper);
            }
        }

        public override void ZeroPadding(Tensor unpaddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            //we are adding padding to 'unpaddedTensor' to initialize 'paddedTensor'
            var paddedTensor = this;
            paddedTensor.ZeroMemory();
            ZeroPadding_and_Unpadding(unpaddedTensor, paddingTop, paddingLeft, false);
        }
        public override void ZeroUnpadding(Tensor paddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            ((CpuTensor<T>)paddedTensor).ZeroPadding_and_Unpadding(this, paddingTop, paddingLeft, true);
        }

        private void ZeroPadding_and_Unpadding(Tensor unpaddedTensor, int paddingTop, int paddingLeft, bool isUnpadding)
        {
            var paddedTensor = this;
            Debug.Assert(AreCompatible(new List<Tensor> { paddedTensor, unpaddedTensor }));
            Debug.Assert(paddedTensor.Dimension == 4);
            Debug.Assert(paddedTensor.Dimension == unpaddedTensor.Dimension);
            Debug.Assert(paddedTensor.Shape[0] == unpaddedTensor.Shape[0]); //same batch size
            Debug.Assert(paddedTensor.Shape[1] == unpaddedTensor.Shape[1]); //same number of channels
            int h_src = unpaddedTensor.Shape[2];
            int w_src = unpaddedTensor.Shape[3];
            // copy the row 'srcRowId' from 'src' tensor (n, c, h_src, w_src) to dest tensor (n, c, h_dest, w_dest)
            // the number of distinct rows in 'src' tensor is : n*c*h_src

            void ApplyZeroPaddingForRowId(int srcRowId)
            {
                // 0 <= srcRowId < n*c*h_src
                int row_src = (srcRowId % h_src);
                int unpaddedRowIndex = srcRowId * w_src;
                int paddedRowIndex = ((srcRowId / h_src) * paddedTensor.Shape[2] + row_src + paddingTop) * paddedTensor.Shape[3] + paddingLeft;
                if (isUnpadding)
                {
                    paddedTensor.CopyTo(paddedRowIndex, unpaddedTensor, unpaddedRowIndex, w_src);
                }
                else
                {
                    unpaddedTensor.CopyTo(unpaddedRowIndex, paddedTensor, paddedRowIndex, w_src);
                }
            }
            Parallel.For(0, unpaddedTensor.Shape[0] * unpaddedTensor.Shape[1] * unpaddedTensor.Shape[2], ApplyZeroPaddingForRowId);
        }

        public override void AssertIsNotDisposed()
        {
            if (_disposed)
            {
                throw new Exception("Tensor is disposed " + this);
            }
        }
        public override void Concatenate(IList<Tensor> tensors)
        {
            CheckConcatenate(tensors);
            void ConcatenateSingleRow(int m)
            {
                int startIdx = Idx(m);
                foreach (var t in tensors)
                {
                    t.CopyTo(t.Idx(m), this, startIdx, t.MultDim0);
                    startIdx += t.MultDim0;
                }
            }
            Parallel.For(0, Shape[0], ConcatenateSingleRow);
        }
        public override void Split(IList<Tensor> tensors)
        {
            CheckConcatenate(tensors);
            void SplitSingleRow(int m)
            {
                int startIdx = Idx(m);
                foreach (var t in tensors)
                {
                    CopyTo(startIdx, t, t.Idx(m), t.MultDim0);
                    startIdx += t.MultDim0;
                }
            }
            Parallel.For(0, Shape[0], SplitSingleRow);
        }

        ///// <summary>
        ///// return a (square) diagonal matrix of length (rowCount, rowCount)
        ///// each element in the diagonal will be 1, all other will be 0
        ///// </summary>
        ///// <param name="rowCount">number of rows and columns of the diagonal matrix</param>
        ///// <returns></returns>
        //public static CpuTensor<float> NewFloatDiagonalMatrix(int rowCount)
        //{
        //    var data = new float[rowCount * rowCount];
        //    for (int row = 0; row < rowCount; ++row)
        //    {
        //        data[row * rowCount + row] = 1f;
        //    }
        //    return new CpuTensor<float>(new[] { rowCount, rowCount }, data);
        //}


        // compute:     this = alpha * this
        public override void Update_Multiplying_By_Alpha(float alpha)
        {
            MKL_BLAS.cblas_sscal(Count, alpha, AsFloatPointer, 1);
        }
        #region pooling layers

        public override void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride)
        {
            var x = this;
#if DEBUG
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            Debug.Assert(x.Shape[0] == y.Shape[0]); //same batch size
            Debug.Assert(x.Shape[1] == y.Shape[1]); //same number of channels
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 4);
            int hOutput = y.Shape[2];
            int hInput = x.Shape[2];
            int hExpected = (hInput - poolingHeight) / verticalStride + 1;
            Debug.Assert(hOutput == hExpected);
            int wOutput = y.Shape[3];
            int wInput = x.Shape[3];
            int wExpected = (wInput - poolingWidth) / horizontalStride + 1;
            Debug.Assert(wOutput == wExpected);
#endif
            int batchSize = x.Shape[0];
            if (PoolingLayer.IsMaxPooling(poolingMode))
            {
                Parallel.For(0, batchSize, elementIndex => x.MaxPoolingForSingleElement4D(y, poolingHeight, poolingWidth, verticalStride, horizontalStride, elementIndex ));
            }
            else
            {
                Parallel.For(0, batchSize, elementIndex => x.AvgPoolingForSingleElement4D(y, poolingHeight, poolingWidth, verticalStride, horizontalStride, elementIndex));
            }
        }
        private void AvgPoolingForSingleElement4D(Tensor y, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride, int elementIndex)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 4);
            int hOutput = y.Shape[2];
            int wOutput = y.Shape.Length>=4?y.Shape[3]:1;
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
                        col_filter_start += horizontalStride;
                    }
                    row_filter_start += verticalStride;
                }
            }
        }
        private void MaxPoolingForSingleElement4D(Tensor y, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride, int elementIndex)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 4);
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
                        col_filter_start += horizontalStride;
                    }
                    row_filter_start += verticalStride;
                }
            }
        }
        public override void PoolingGradient(Tensor yNotUsed, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride)
        {
            var dy = this;
            int batchSize = x.Shape[0];
#if DEBUG
            Debug.Assert(AreCompatible(new List<Tensor> { dy, x, dx }));
            Debug.Assert(x.Shape[0] == dy.Shape[0]); //same batchSize
            Debug.Assert(x.Shape[1] == dy.Shape[1]); //same number of channels
            Debug.Assert(dx.SameShape(x));
            Debug.Assert(x.Shape.Length == 4);
            Debug.Assert(dx.Shape.Length == 4);
            Debug.Assert(dy.Shape.Length == 4);
            int hOutput = dy.Shape[2];
            int wOutput = dy.Shape[3];
            Debug.Assert(hOutput == ((x.Shape[2] - poolingHeight) / verticalStride + 1));
            Debug.Assert(wOutput == ((x.Shape[3] - poolingWidth) / horizontalStride + 1));
#endif
            dx.ZeroMemory();
            if (PoolingLayer.IsMaxPooling(poolingMode))
            {
                Parallel.For(0, batchSize, elementIndex => dy.MaxPoolingGradientForSingleElement4D(x, dx, poolingHeight, poolingWidth, verticalStride, horizontalStride, elementIndex));
            }
            else
            {
                Parallel.For(0, batchSize, elementIndex => dy.AvgPoolingGradientForSingleElement4D(x, dx, poolingHeight, poolingWidth, verticalStride, horizontalStride, elementIndex));
            }
        }
        private void AvgPoolingGradientForSingleElement4D(Tensor x, Tensor dx, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride, int elementIndex)
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
                        col_filter_start += horizontalStride;
                    }
                    row_filter_start += verticalStride;
                }
            }
        }
        //compute 'dx' from ('dy' and 'x')
        private void MaxPoolingGradientForSingleElement4D(Tensor x, Tensor dx, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride, int elementIndex)
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
                        col_filter_start += horizontalStride;
                    }
                    row_filter_start += verticalStride;
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

            var singleRowMatrixContent = bias.AsFloatCpuSpan;
            var yContent = y.AsFloatCpuSpan;
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
        public override void Convolution(Tensor convolution, int paddingTop, int paddingBottom, int paddingLeft,
            int paddingRight, int stride, Tensor y, bool isDepthwiseConvolution,
            ConvolutionAlgoPreference forwardAlgoPreference, TensorMemoryPool memoryPool)
        {
            if (forwardAlgoPreference != ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM)
            {
                throw new NotImplementedException("only "+ ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM+" is available on CPU ("+ forwardAlgoPreference+ " is not supported)");
            }
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
            int kernelHeight = convolution.Shape[2];
            int kernelWidth = convolution.Shape[3];
            int hOutput = y.Shape[2];
            int wOutput = y.Shape[3];
            Debug.Assert(batchSize == y.Shape[0]);

            //the first (top left) point in 'y' is computed from a filter starting at (-padding,-padding)
            void ComputeForBatch(int m)
            {
                var convolutionContentAsFloat = convolution.AsFloatCpuSpan;
                var xContentAsFloat = x.AsFloatCpuSpan;

                for (int outputChannelId = 0; outputChannelId < outputChannels; ++outputChannelId)
                {
                    int rowFilterStart = -paddingTop;
                    for (int rowOutput = 0; rowOutput < hOutput; ++rowOutput)
                    {
                        int colFilterStart = -paddingLeft;
                        var rowInputStart = Math.Max(0, rowFilterStart);
                        var rowInputEndExcluded = Math.Min(hInput, rowFilterStart + kernelHeight);
                        for (int colOutput = 0; colOutput < wOutput; ++colOutput)
                        {
                            //we want to compute the point in y[m, filterId, row_output, col_output]
                            //it is computed by applying a filter located (for its top left) in (row_filter_start,col_filter_start) in the x 
                            double outputPointResult = 0.0;
                            var colInputStart = Math.Max(0, colFilterStart);
                            var colInputEndExcluded = Math.Min(wInput, colFilterStart + kernelWidth);

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
            Parallel.For(0, batchSize, ComputeForBatch);
        }


        public override void BroadcastConvolutionBiasToOutput(Tensor y)
        {
            var convolutionBias = this;
            Debug.Assert(AreCompatible(new List<Tensor> { convolutionBias, y }));
            Debug.Assert(y.Dimension >= 2);
            Debug.Assert(convolutionBias.Shape.SequenceEqual(new []{1, y.Shape[1], 1, 1}));
            var batchSize = y.Shape[0];
            var yContent = y.AsFloatCpuSpan;
            for (int n = 0; n < batchSize; ++n)
            {
                int startIndex = n * y.MultDim0;
                for (int filterId = 0; filterId < y.Shape[1]; ++filterId, startIndex += y.MultDim1)
                {
                    var toAdd = convolutionBias.AsFloatCpuSpan[filterId];
                    for (int i = startIndex; i < (startIndex + y.MultDim1); ++i)
                    {
                        yContent[i] += toAdd;
                    }
                }
            }
        }

        public override void ConvolutionGradient(Tensor convolution, Tensor dy, int paddingTop, int paddingBottom,
            int paddingLeft, int paddingRight, int stride, Tensor dx, Tensor convGradient, bool isDepthwiseConvolution,
            ConvolutionAlgoPreference backwardAlgoPreference, TensorMemoryPool memoryPool)
        {
            if (backwardAlgoPreference != ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM)
            {
                throw new NotImplementedException("only " + ConvolutionAlgoPreference.FASTEST_DETERMINIST_NO_TRANSFORM + " is available on CPU (" + backwardAlgoPreference + " is not supported)");
            }
            var x = this;
            int inputChannels = x.Shape[1];
            int outputChannels = dy.Shape[1];
            Debug.Assert(inputChannels == convolution.Shape[1]);
            Debug.Assert(dx == null ||dx.SameShape(x));
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
            int kernelHeight = convolution.Shape[2];
            int kernelWidth = convolution.Shape[3];
            int hOutput = dy.Shape[2];
            Debug.Assert(hOutput == ((hInput - kernelHeight + paddingTop+paddingBottom) / stride + 1));
            int wOutput = dy.Shape[3];
            Debug.Assert(wOutput == ((wInput - kernelWidth + paddingLeft+paddingRight) / stride + 1));
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
                        var rowInputEndExcluded = Math.Min(hInput, rowFilterStart + kernelHeight);
                        for (int colOutput = 0; colOutput < wOutput; ++colOutput)
                        {
                            //we want to compute the point in y[m, filterId, rowOutput, colOutput]
                            //it is computed by applying a filter located (for its top left) in (row_filter_start,col_filter_start) in the x 
                            // and centered at this particular location
                            var chainGradientFloat = dy.AsFloatCpu.Get(m, outputChannelId, rowOutput, colOutput);
                            var colInputStart = Math.Max(0, colFilterStart);
                            var colInputEndExcluded = Math.Min(wInput, colFilterStart + kernelWidth);
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
                                        convolutionGradientForLocalThreadFloat[convIdx] += x.AsFloatCpuSpan[APrevLayerIdx] * chainGradientFloat;
                                        if (dx != null)
                                        {
                                            dx.AsFloatCpuSpan[APrevLayerIdx] += convolution.AsFloatCpuSpan[convIdx] * chainGradientFloat;
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
                        convGradient.AsFloatCpuSpan[i] += convolutionGradientForLocalThreadFloat[i];
                    }
                }
            }
            Parallel.For(0, batchSize, ComputeForBatch);
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
                    var convolutionBackwardBiasContent = bias.AsFloatCpuSpan;
                    var dyContent = dy.AsFloatCpuSpan;
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

        /// <summary>
        /// Use Adam optimizer (see https://arxiv.org/pdf/1412.6980.pdf)
        /// this = Weights or Bias
        /// </summary>
        /// <param name="learningRate"></param>
        /// <param name="beta1"></param>
        /// <param name="beta2"></param>
        /// <param name="epsilon"></param>
        /// <param name="adamW_l2Regularization"></param>
        /// <param name="dW"></param>
        /// <param name="adam_vW">biased first moment estimate</param>
        /// <param name="adam_sW">biased second raw moment estimate</param>
        /// <param name="timeStep"></param>
        public override void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon,
            double adamW_l2Regularization, Tensor dW, Tensor adam_vW, Tensor adam_sW, int timeStep)
        {
            var beta1_power = Math.Pow(beta1, timeStep);
            var beta2_power = Math.Pow(beta2, timeStep);

            var W = this;
            //Update biased first moment estimate
            adam_vW.AsFloatCpu.Update(dW, (adam_vw, dw) => (float) (beta1 * adam_vw + (1 - beta1) * dw));
            //Update biased second raw moment estimate
            adam_sW.AsFloatCpu.Update(dW, (adam_sw, dw) => (float) (beta2 * adam_sw + (1 - beta2) * dw * dw));
            var multiplicative_factor = learningRate * (Math.Sqrt(1.0 - beta2_power) / (1.0 - beta1_power));
            //Update parameters
            W.AsFloatCpu.Update(adam_vW, adam_sW, (w, adam_vw, adam_sw) => (float)(w - ( multiplicative_factor * (adam_vw / (Math.Sqrt(adam_sw) + epsilon)) + adamW_l2Regularization*w )));
        }

        public override void NormalDistribution(Random rand, double mean, double stdDev)
        {
            Utils.NormalDistribution(AsFloatCpuSpan, rand, mean, stdDev);
        }


        public override void UniformDistribution(Random rand, double minValue, double maxValue)
        {
            Utils.UniformDistribution(AsFloatCpuSpan, rand, minValue, maxValue);
        }
        
        public override void SetValue(float sameValue)
        {
            var array = AsFloatCpuSpan;
            for (int i = 0; i < Count; ++i)
            {
                array[i] = sameValue;
            }
        }
        public override float[] ContentAsFloatArray()
        {
            return AsFloatCpuSpan.ToArray();
        }

        public override Half[] ContentAsHalfArray()
        {
            return AsHalfCpuSpan.ToArray();
        }


        public override Tensor Clone()
        {
            var cloned = new CpuTensor<T>(Shape);
            CopyTo(cloned);
            return cloned;
        }


        private static bool IsAccuratePredictionForCategoricalCrossentropyWithHierarchy(float* expected, float* predicted, int endIndexExcluded, int *pNexIndexToCheck, int subCategoriesCount, List<int> observedPrediction)
        {
            int subCategoriesFound = 0;
            int predictedSubCategoryId = -1;
            float bestPredictedSubCategoryProba = -1.0f;
            int expectedSubCategoryId = -1;
            float bestExpectedSubCategoryProba = -1.0f;
            bool isAccurate = true;
            bool previousIndexWasProba = false;

            while (subCategoriesFound < subCategoriesCount && (*pNexIndexToCheck < endIndexExcluded))
            {
                float expectedProba = expected[*pNexIndexToCheck];
                float predictedProba = predicted[*pNexIndexToCheck];
                if (fabsf(expectedProba) < 9.5f)
                {
                    previousIndexWasProba = true;
                    ++subCategoriesFound;
                    if (expectedProba > bestExpectedSubCategoryProba)
                    {
                        bestExpectedSubCategoryProba = expectedProba;
                        expectedSubCategoryId = subCategoriesFound-1;
                    }
                    if (predictedProba > bestPredictedSubCategoryProba)
                    {
                        bestPredictedSubCategoryProba = predictedProba;
                        predictedSubCategoryId = subCategoriesFound-1;
                    }
                    *pNexIndexToCheck += 1;
                }
                else
                {
                    int count = (int)(fabsf(expectedProba) + 0.5f) / 10;
                    if (expectedProba < 0)
                    {
                        //we need to skip 'count' indexes
                        *pNexIndexToCheck += count;
                    }
                    else
                    {
                        *pNexIndexToCheck += 1;
                        bool subCategoryIsAccurate = IsAccuratePredictionForCategoricalCrossentropyWithHierarchy(expected, predicted, endIndexExcluded, pNexIndexToCheck, count, observedPrediction);
                        isAccurate = subCategoryIsAccurate && isAccurate;
                    }
                    if (!previousIndexWasProba)
                    {
                        ++subCategoriesFound;
                    }
                    previousIndexWasProba = false;
                }
            }
            observedPrediction.Insert(0, predictedSubCategoryId);
            return (expectedSubCategoryId == predictedSubCategoryId) && isAccurate;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        // ReSharper disable once InconsistentNaming
        private static float fabsf(float f) {return Math.Abs(f);}

        #region Compute of Evaluation Metrics

        protected override double ComputePearsonCorrelation([NotNull] Tensor y_pred)
        {
            var y_true = this;
            Debug.Assert(AreCompatible(new List<Tensor> { y_true, y_pred }));
            Debug.Assert(y_true.Shape.Length == 2);
            Debug.Assert(y_true.Shape[1] == 1);
            Debug.Assert(y_true.SameShape(y_pred));
            Debug.Assert(!y_true.UseGPU);
            var y_true_span = y_true.AsReadonlyFloatCpuSpan;
            var y_pred_span = y_pred.AsReadonlyFloatCpuSpan;

            var lr = new LinearRegression();
            for (int i = 0; i < y_true_span.Length; i++)
            {
                lr.Add(y_true_span[i], y_pred_span[i]);
            }
            return lr.PearsonCorrelationCoefficient;  
        }

        protected override double ComputeSpearmanCorrelation([NotNull] Tensor y_pred)
        {
            static int[] CreateRankForSpearmanCorrelation(ReadOnlySpan<float> s)
            {
                var res = new int[s.Length];
                var sorted = new List<Tuple<float, int>>();
                for (int i = 0; i < s.Length; i++)
                {
                    sorted.Add(new Tuple<float, int>(s[i], i));
                }
                int currentRank = 0;
                foreach (var e in sorted.OrderByDescending(a => a.Item1))
                {
                    res[e.Item2] = currentRank++;
                }
                return res;
            }
            
            var y_true = this;
            Debug.Assert(AreCompatible(new List<Tensor> { y_true, y_pred }));
            Debug.Assert(y_true.Shape.Length == 2);
            Debug.Assert(y_true.Shape[1] == 1);
            Debug.Assert(y_true.SameShape(y_pred));
            Debug.Assert(!y_true.UseGPU);
            var y_true_rank = CreateRankForSpearmanCorrelation(y_true.AsReadonlyFloatCpuSpan);
            var y_pred_rank = CreateRankForSpearmanCorrelation(y_pred.AsReadonlyFloatCpuSpan);

            double sum_delta_rank_square = 0;
            for (int i = 0; i < y_true_rank.Length; i++)
            {
                double delta_rank = y_true_rank[i] - y_pred_rank[i];
                sum_delta_rank_square += delta_rank * delta_rank;
            }
            double n = y_true.Shape[0];
            var spearmanCorrelation  = 1.0 - (6 * sum_delta_rank_square) / (n * (n * n - 1));
            Debug.Assert(spearmanCorrelation < 1.001);
            Debug.Assert(spearmanCorrelation > -1.001);
            return spearmanCorrelation;
        }

        [SuppressMessage("ReSharper", "PossibleNullReferenceException")]
        protected override void ComputeAccuracyBuffer([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted)
        {
            var buffer = this;
            Debug.Assert(AreCompatible(new List<Tensor> { yExpected, yPredicted }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(!yExpected.UseGPU);
            Debug.Assert(buffer.Shape.Length == 1);
            Debug.Assert(buffer.Shape[0] == yPredicted.Shape[0]);
            int batchSize = yExpected.Shape[0];

            var bufferPointer = (float*)buffer.Pointer;
            var yExpectedCpu = yExpected.AsFloatCpu;
            var yPredictedCpu = yPredicted.AsFloatCpu;
            Parallel.For(0, batchSize, row => bufferPointer[row] = ComputeSingleAccuracy(yExpectedCpu, yPredictedCpu, row));
        }

        private static float ComputeSingleAccuracy(CpuTensor<float> yExpected, CpuTensor<float> yPredicted, int row)
        {
            Debug.Assert(yExpected.SameShape(yPredicted));
            int maxIndexPredicted = 0;
            var numClass = yExpected.Shape[1];
            if (numClass == 1)
            {
                var error = Math.Abs(yExpected.Get(row, 0) - yPredicted.Get(row, 0));
                return (error < 0.5) ? 1 : 0;
            }
            int maxIndexExpected = 0;
            for (int j = 1; j < numClass; ++j)
            {
                if (yPredicted.Get(row, j) > yPredicted.Get(row, maxIndexPredicted))
                {
                    maxIndexPredicted = j;
                }
                if (yExpected.Get(row, j) > yExpected.Get(row, maxIndexExpected))
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


        protected override void SparseCategoricalCrossentropyLossBuffer([NotNull] Tensor yExpectedSparse, [NotNull] Tensor yPredicted)
        {
            var buffer = this;
            (yExpectedSparse, yPredicted, _) = ReformatTo2DTensorsSparse(yExpectedSparse, yPredicted);
            //yExpectedSparse shape:    (batchSize*timeSteps, 1)
            //yPredicted shape:         (batchSize*timeSteps, numClass)

            var bufferSpan = buffer.AsFloatCpuSpan;
            var yExpectedSpan = yExpectedSparse.AsReadonlyFloatCpuSpan;
            var yPredictedSpan = yPredicted.AsReadonlyFloatCpuSpan;
            var numClass = yPredicted.MultDim0;
            for (int row = 0; row < yExpectedSpan.Length; ++row)
            {
                var yClass = Utils.NearestInt(yExpectedSpan[row]);
                Debug.Assert(yClass >= 0);
                Debug.Assert(yClass < numClass);
                float predicted = yPredictedSpan[row * numClass + yClass];
                bufferSpan[row]= -MathF.Log(predicted);
            }
        }

        protected override void CategoricalCrossentropyLossBuffer(Tensor yExpectedOneHot, Tensor yPredicted)
        {
            MergeInPlaceByRow(yExpectedOneHot.AsFloatCpu, yPredicted.AsFloatCpu, (expected, prediction) => -expected * MathF.Log(prediction), 1);
        }

        protected override void MaeLossBuffer(Tensor yExpected, Tensor yPredicted)
        {
            MergeInPlaceByRow(yExpected.AsFloatCpu, yPredicted.AsFloatCpu, (expected, prediction) => MathF.Abs(expected - prediction), yPredicted.MultDim0);
        }

        protected override void MseLossBuffer(Tensor yExpected, Tensor yPredicted)
        {
            MergeInPlaceByRow(yExpected.AsFloatCpu, yPredicted.AsFloatCpu, (expected, prediction) => MathF.Pow(expected - prediction, 2), yPredicted.MultDim0);
        }

        protected override void MeanSquaredLogErrorLossBuffer(Tensor yExpected, Tensor yPredicted)
        {
            MergeInPlaceByRow(yExpected.AsFloatCpu, yPredicted.AsFloatCpu, (expected, prediction) => MathF.Pow(MathF.Log(1 + expected) - MathF.Log(1 + prediction), 2f), yPredicted.MultDim0);
        }

        protected override void CategoricalCrossentropyWithHierarchyLossBuffer(Tensor yExpected, Tensor yPredicted)
        {
            var batchSize = yPredicted.Shape[0];
            var buffer = this;
            Parallel.For(0, batchSize, m => { buffer.AsFloatCpuSpan[m] = CategoricalCrossentropyWithHierarchyLossBuffer_Helper(yExpected.RowSlice(m, 1).AsReadonlyFloatCpuSpan, yPredicted.RowSlice(m, 1).AsReadonlyFloatCpuSpan); });
        }
        private static float CategoricalCrossentropyWithHierarchyLossBuffer_Helper(ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted)
        {
            Debug.Assert(expected.Length == predicted.Length);
            double loss = 0;
            for (int i = 0; i < expected.Length; ++i)
            {
                var expectedValue = expected[i];
                if (Math.Abs(expectedValue) < 9.5f)
                {
                    //expectedValue contains a proba between 0 and 1
                    Debug.Assert(expectedValue >= 0);
                    Debug.Assert(expectedValue <= 1.0);
                    Debug.Assert(predicted[i] >= 0.0);
                    Debug.Assert(predicted[i] <= 1.0);
                    if (expectedValue > 1e-6)
                    {
                        Debug.Assert(Math.Abs(expectedValue - 1.0) < 1e-6);
                        loss += expectedValue * Math.Log(Math.Max(1e-6, predicted[i]));
                    }
                }
                else
                {
                    //expectedValue contains a description : there is no associated loss
                    if (expectedValue < 0)
                    {
                        var count = (int)(Math.Abs(expectedValue) + 0.5) / 10;
                        //we need to skip 'count' indexes
                        i += count - 1; //-1 because the for(;;) loop will also increment 'i'
                    }
                }
            }
            return -(float)loss;
        }



        protected override void BinaryCrossentropyLossBuffer(Tensor yExpected, Tensor yPredicted)
        {
            MergeInPlaceByRow(yPredicted.AsFloatCpu, yExpected.AsFloatCpu, (y_pred, y_true) => BinaryCrossentropyLossBufferSingle(y_true, y_pred) , yPredicted.MultDim0);
        }

        private static float BinaryCrossentropyLossBufferSingle(float y_true, float y_pred)
        {
            const float epsilon = 1e-9f;
            if (y_true > (1-epsilon))
            {
                return -MathF.Log(Math.Max(epsilon, y_pred));
            }
            if (y_true < epsilon)
            {
                return -MathF.Log(Math.Max(epsilon, 1 - y_pred));
            }
            return -y_true * MathF.Log(Math.Max(epsilon, y_pred)) - (1 - y_true) * MathF.Log(Math.Max(epsilon, 1 - y_pred));
        }

        protected override void BCEContinuousYLossBuffer(Tensor yExpected, Tensor yPredicted)
        {
            MergeInPlaceByRow(yPredicted.AsFloatCpu, yExpected.AsFloatCpu, (prediction, expected) => -MathF.Log(1 - MathF.Abs(prediction - expected)), yPredicted.MultDim0);
        }

        protected override void BCEWithFocalLossLossBuffer(Tensor yExpected, Tensor yPredicted, float percentageInTrueClass, float gamma)
        {
            var bceWithFocalLossLossBuffer = this;
            // all tensors must be of shape (rows, 1)
            Debug.Assert(yExpected.Shape.Length == 2);
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(bceWithFocalLossLossBuffer.Count == yPredicted.Shape[0]);
            var y_true = yExpected.AsFloatCpuSpan;
            var y_pred = yPredicted.AsFloatCpuSpan;
            var loss = bceWithFocalLossLossBuffer.AsFloatCpuSpan;

            //loss = -POWER(ABS(y_true-y_pred)/MAX(y_true;1-y_true);gamma)*LN(1-ABS(y_true-y_pred))

            int idx = 0;
            var rows = yExpected.Shape[0];
            var numClass = yExpected.Shape[1];
            var imbalancedCoeffForTrueClass = 1 / (2 * percentageInTrueClass);
            var imbalancedCoeffForFalseClass = 1 / (2 * (1 - percentageInTrueClass));

            for (int row = 0; row < rows; ++row)
            {
                float rowLoss = 0;
                for (int col = 0; col < numClass; ++col)
                {
                    //the gradient value for standard binary cross entropy (without focal loss)
                    var absNonScaledGradient = MathF.Abs(y_pred[idx] - y_true[idx]);
                    var nonScaledLoss = -MathF.Log(Math.Max(1 - absNonScaledGradient, 1e-6f));
                    var focalLossCoeff = 1.0f;
                    if (gamma > 0)
                    {
                        //we need to adjust the loss value for focal loss
                        float maxValueForNonScaledGradient = Math.Max(y_true[idx], 1 - y_true[idx]);
                        focalLossCoeff = MathF.Pow(absNonScaledGradient / maxValueForNonScaledGradient, gamma);
                    }

                    // we take into account the imbalance between the true and false class
                    // if one class is over represented, we reduce the loss value for this class
                    float imbalancedCoeffForCurrentClass = imbalancedCoeffForFalseClass + y_true[idx] * (imbalancedCoeffForTrueClass - imbalancedCoeffForFalseClass);

                    rowLoss += focalLossCoeff * imbalancedCoeffForCurrentClass * nonScaledLoss;
                    ++idx;
                }
                loss[row] = rowLoss/numClass;
            }
        }
        

        protected override void ComputeSparseAccuracyBuffer(Tensor yExpectedSparse, Tensor yPredicted)
        {
            var buffer = this;
            (yExpectedSparse,yPredicted,_) = ReformatTo2DTensorsSparse(yExpectedSparse, yPredicted);
            //yExpectedSparse shape:    (batchSize*timeSteps, 1)
            //yPredicted shape:         (batchSize*timeSteps, numClass)
            Debug.Assert(buffer.Shape.Length == 1);
            Debug.Assert(yExpectedSparse.Count == buffer.Shape[0]);

            int rows = yPredicted.Shape[0];
            var bufferPointer = (float*)buffer.Pointer;
            var yExpectedSparseCpu = yExpectedSparse.AsFloatCpu;
            var yPredictedCpu = yPredicted.AsFloatCpu;
            Parallel.For(0, rows, row => bufferPointer[row] = ComputeSparseAccuracyBuffer_Helper(yExpectedSparseCpu, yPredictedCpu, row));
        }

        private static float ComputeSparseAccuracyBuffer_Helper(CpuTensor<float> yExpectedSparse, CpuTensor<float> yPredicted, int row)
        {
            int expectedClassIndex = Utils.NearestInt(yExpectedSparse.Get(row, 0));
            var yPredictedSpan = yPredicted.RowSpanSlice(row, 1);
            int predictedClassIndex = 0;
            for (int j = 1; j < yPredictedSpan.Length; ++j)
            {
                if (yPredictedSpan[j] > yPredictedSpan[predictedClassIndex])
                {
                    predictedClassIndex = j;
                }
            }
            return expectedClassIndex == predictedClassIndex ? 1 : 0;
        }

        public override void ComputeAUCBuffer(Tensor yExpected, Tensor yPredicted)
        {
            var buffer = this;
            Debug.Assert(buffer.Count == 1);
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(yExpected.Shape.Length == 2);
            Debug.Assert(yExpected.Shape[1] == 1);

            var yExpectedCpu = yExpected.AsFloatCpu.SpanContent;
            var yPredictedCpu = yPredicted.AsFloatCpu.SpanContent;

            int batchSize = yExpected.Shape[0];
            List<Tuple<float, float>> data = new(batchSize);
            for (var i = 0; i < batchSize; ++i)
            {
                data.Add(Tuple.Create(yExpectedCpu[i], yPredictedCpu[i]));
            }
            data.Sort((x,y)=>x.Item2.CompareTo(y.Item2));
            double falseCount = 0;
            double auc = 0;
            for (int i = 0; i < batchSize; ++i)
            {
                float y_i = data[i].Item1;
                falseCount += (1 - y_i);
                auc += y_i * falseCount;
            }

            if (falseCount == 0 || falseCount == batchSize)
            {
                auc = 0.0;
            }
            else
            {
                auc /= (falseCount * (batchSize - falseCount));
            }
            buffer.AsFloatCpu[0] = (float)auc;
        }

        public override void ComputeAveragePrecisionScoreBuffer(Tensor yExpected, Tensor yPredicted)
        {
            var buffer = this;
            Debug.Assert(buffer.Count == 1);
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(yExpected.Shape.Length == 2);
            Debug.Assert(yExpected.Shape[1] == 1);

            var y_true_cpu = yExpected.AsFloatCpu.SpanContent;
            var y_pred_cpu = yPredicted.AsFloatCpu.SpanContent;

            List<Tuple<float, float>> data = new(y_true_cpu.Length);
            float trueCount = 0;
            for (var i = 0; i < y_true_cpu.Length; ++i)
            {
                data.Add(Tuple.Create(y_true_cpu[i], y_pred_cpu[i]));
                trueCount += y_true_cpu[i];
            }
            data.Sort((a, b) => b.Item2.CompareTo(a.Item2));
            float cumSum = 0;
            float res = 0;
            float recall_i_minus_1 = 0f;
            for (int i = 0; i < y_true_cpu.Length; ++i)
            {
                cumSum+= data[i].Item1;
                float precision_i = cumSum / (i + 1);
                float recall_i = trueCount<=0?1:(cumSum / trueCount);
                res+= precision_i*(recall_i- recall_i_minus_1);
                recall_i_minus_1 = recall_i;
            }
            buffer.AsFloatCpu[0] = res;
        }


        [SuppressMessage("ReSharper", "PossibleNullReferenceException")]
        protected override void ComputeAccuracyCategoricalCrossentropyWithHierarchyBuffer(Tensor yExpected, Tensor yPredicted)
        {
            var buffer = this;
            Debug.Assert(AreCompatible(new List<Tensor> { yExpected, yPredicted }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(!yExpected.UseGPU);
            Debug.Assert(buffer.Shape.Length == 1);
            Debug.Assert(buffer.Shape[0] == yPredicted.Shape[0]);
            int batchSize = yExpected.Shape[0];

            var bufferPointer = (float*)buffer.Pointer;
            var expected = (float*)yExpected.Pointer;
            var predicted = (float*)yPredicted.Pointer;
            int nbCols = yExpected.Shape[1];
            Parallel.For(0, batchSize, i =>
            {
                int nexIndexToCheck = 0;
                bufferPointer[i] = IsAccuratePredictionForCategoricalCrossentropyWithHierarchy(expected + i * nbCols, predicted + i * nbCols, nbCols, &nexIndexToCheck, int.MaxValue, new List<int>()) ? 1f : 0f;
            });
        }

        protected override void MseOfLogLossBuffer(Tensor yExpected, Tensor yPredicted, float epsilon)
        {
            var buffer = this;
            int batchSize = yExpected.Shape[0];
            Debug.Assert(buffer.SameShape(new[] { batchSize }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            Parallel.For(0, batchSize, batchId => { MseOfLogLossHelper(batchId, buffer.AsFloatCpuSpan, yExpected.RowSlice(batchId, 1).AsReadonlyFloatCpuSpan, yPredicted.RowSlice(batchId, 1).AsReadonlyFloatCpuSpan, epsilon); });
        }

        private static int MaxIndex(ReadOnlySpan<float> a, int startIndex, int count)
        {
            int maxIndex = startIndex;
            for (int j = startIndex + 1; j < startIndex + count; ++j)
            {
                if (a[j] > a[maxIndex])
                {
                    maxIndex = j;
                }
            }
            return maxIndex;
        }
        public override (float f1, float precision, float recall) F1PrecisionRecallMicro(Tensor yExpected, Tensor yPredicted)
        {
            Debug.Assert(yExpected.SameShape(yPredicted));
            var y_true_span = yExpected.AsReadonlyFloatCpuSpan;
            var y_pred_span = yPredicted.AsReadonlyFloatCpuSpan;
            int true_count = 0;

            var rows = yExpected.Shape[0];
            var num_class = yExpected.Shape[1];
            if (num_class >= 2)
            {
                for (int row = 0; row<rows; ++row)
                {
                    int idxStart = row * num_class;
                    var idxTrue = MaxIndex(y_true_span, idxStart, num_class);
                    var idxPred = MaxIndex(y_pred_span, idxStart, num_class);
                    if (idxTrue == idxPred)
                    {
                        ++true_count;
                    }
                }
            }
            else
            {
                for (int row = 0; row <rows; ++row)
                {
                    var idxTrue = Utils.NearestInt(y_true_span[row]);
                    var idxPred = Utils.NearestInt(y_pred_span[row]);
                    if (idxTrue == idxPred)
                    {
                        ++true_count;
                    }
                }
            }

            var precisionMicro = true_count/(float)rows;
            var recallMicro = precisionMicro;
            var f1Micro = (2 * precisionMicro * recallMicro) / (precisionMicro + recallMicro);
            return (f1Micro, precisionMicro, recallMicro);
        }

        private static void MseOfLogLossHelper(int batchId, Span<float> mseLoss, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted, float epsilon)
        {
            Debug.Assert(expected.Length == predicted.Length);
            var loss = 0.0f;
            for (int i = 0; i < expected.Length; ++i)
            {
                var adjustedPredicted = Math.Max(epsilon, predicted[i]);
                var error = Math.Log(adjustedPredicted) - Math.Log(expected[i]);
                loss += (float)(error * error);
            }
            mseLoss[batchId] = loss / expected.Length;
        }

        public override void CosineSimilarityLossBuffer(Tensor yExpected, Tensor yPredicted, int timeSeriesLength)
        {
            var cosineSimilarityLoss = this;
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(cosineSimilarityLoss.Count == timeSeriesLength);
            Debug.Assert(yPredicted.Count%timeSeriesLength == 0);
            Parallel.For(0, timeSeriesLength, t => { CosineSimilarityLoss(t, cosineSimilarityLoss.AsFloatCpuSpan, yExpected.AsReadonlyFloatCpuSpan, yPredicted.AsReadonlyFloatCpuSpan, timeSeriesLength); });
        }
        private static void CosineSimilarityLoss(int day, Span<float> cosineSimilarityLoss, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted, int timeSeriesLength)
        {
            Debug.Assert(expected.Length == predicted.Length);
            Debug.Assert(cosineSimilarityLoss.Length == timeSeriesLength);
            var top = 0.0f;
            var expectedSquares = 0.0f;
            var predictedSquares = 0.0f;
            for (int t = day; t < expected.Length; t+= timeSeriesLength)
            {
                var pred = predicted[t];
                var exp = expected[t];
                top += pred * exp;
                expectedSquares += exp * exp;
                predictedSquares += pred * pred;
            }
            var l2_norm_expected = Math.Sqrt(expectedSquares);
            var l2_norm_predicted = Math.Sqrt(predictedSquares);
            cosineSimilarityLoss[day] = (float)(top / (l2_norm_expected * l2_norm_predicted));
        }

        public override void HuberLossBuffer(Tensor yExpected, Tensor yPredicted, float huberDelta)
        {
            var huberLoss = this;
            int batchSize = yExpected.Shape[0];
            Debug.Assert(huberLoss.SameShape(new[] { batchSize }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            Parallel.For(0, batchSize, batchId => { HuberLossHelper(batchId, huberLoss.AsFloatCpuSpan, yExpected.RowSlice(batchId, 1).AsReadonlyFloatCpuSpan, yPredicted.RowSlice(batchId, 1).AsReadonlyFloatCpuSpan, huberDelta); });
        }
        private static void HuberLossHelper(int batchId, Span<float> huberLoss, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted, float huberDelta)
        {
            Debug.Assert(expected.Length == predicted.Length);
            var loss = 0.0f;
            for (int i = 0; i < expected.Length; ++i)
            {
                var error = predicted[i] - expected[i];
                if (Math.Abs(error) <= huberDelta)
                {
                    loss += 0.5f * error * error;
                }
                else
                {
                    loss += huberDelta * Math.Abs(error) - 0.5f * huberDelta * huberDelta;
                }
            }
            huberLoss[batchId] = loss;
        }
        #endregion


        #region Compute of Gradients (for backward propagation)
        public override void CosineSimilarityGradient(Tensor yExpected, Tensor yPredicted, int timeSeriesLength)
        {
            var cosineSimilarityGradient = this;
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(cosineSimilarityGradient.Count == yExpected.Count);
            Parallel.For(0, timeSeriesLength, t => { CosineSimilarityGradient_Helper(t, cosineSimilarityGradient.AsFloatCpuSpan, yExpected.AsReadonlyFloatCpuSpan, yPredicted.AsReadonlyFloatCpuSpan, timeSeriesLength); });
        }
        private static void CosineSimilarityGradient_Helper(int day, Span<float> cosineSimilarityGradient, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted, int timeSeriesLength)
        {
            Debug.Assert(expected.Length == predicted.Length);
            Debug.Assert(cosineSimilarityGradient.Length == expected.Length);
            var top = 0.0f;
            var expectedSquares = 0.0f;
            var predictedSquares = 0.0f;
            for (int t = day; t < expected.Length; t += timeSeriesLength)
            {
                var pred = predicted[t];
                var exp = expected[t];
                top += pred * exp;
                expectedSquares += exp * exp;
                predictedSquares += pred * pred;
            }
            var l2_norm_expected = Math.Sqrt(expectedSquares);
            var l2_norm_predicted = Math.Sqrt(predictedSquares);
            var multiplier1 = 1.0f/(l2_norm_expected * l2_norm_predicted);
            var multiplier2 = (-top)/(l2_norm_predicted* l2_norm_predicted * l2_norm_predicted * l2_norm_expected);
            for (int t = day; t < expected.Length; t += timeSeriesLength)
            {
                cosineSimilarityGradient[t] = -(float) (multiplier1*expected[t] + multiplier2*predicted[t]);
            }
        }

        public override void HuberGradient(Tensor yExpected, Tensor yPredicted, float huberDelta)
        {
            var huberGradient = this;
            Debug.Assert(huberGradient.SameShape(yExpected));
            Debug.Assert(huberGradient.SameShape(yPredicted));
            Parallel.For(0, huberGradient.Shape[0], m => { HuberGradient_Helper(huberGradient.RowSlice(m, 1).AsFloatCpuSpan, yExpected.RowSlice(m, 1).AsReadonlyFloatCpuSpan, yPredicted.RowSlice(m, 1).AsReadonlyFloatCpuSpan, huberDelta); });
        }
        private static void HuberGradient_Helper(Span<float> gradient, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted, float huberDelta)
        {
            Debug.Assert(gradient.Length == expected.Length);
            Debug.Assert(gradient.Length == predicted.Length);
            for (int i = 0; i < gradient.Length; ++i)
            {
                var error = predicted[i] - expected[i];
                gradient[i] = Math.Max(Math.Min(error, huberDelta), -huberDelta);
                gradient[i] /= gradient.Length;
            }
        }


        public override void SparseCategoricalCrossentropyGradient(Tensor yExpectedSparse, Tensor yPredicted)
        {
            (yExpectedSparse, yPredicted, var sparseCategoricalCrossentropyGradient) = ReformatTo2DTensorsSparse(yExpectedSparse, yPredicted, this);
            //yExpected shape:  (batchSize*timeSteps, 1)
            //yPredicted shape: (batchSize*timeSteps, numClass)
            //sparseCategoricalCrossentropyGradient shape: (batchSize*timeSteps, numClass)
            int numClass = yPredicted.Shape[^1];
            yPredicted.CopyTo(sparseCategoricalCrossentropyGradient);
            var yExpectedSpan = yExpectedSparse.AsReadonlyFloatCpuSpan;
            var yGradient = sparseCategoricalCrossentropyGradient.AsFloatCpuSpan;
            for (int row = 0; row < yExpectedSpan.Length; ++row)
            {
                var yClass = Utils.NearestInt(yExpectedSpan[row]);
                Debug.Assert(yClass >= 0);
                Debug.Assert(yClass < numClass);
                yGradient[row * numClass + yClass] -= 1.0f;
            }
        }
        public override void MseGradient(Tensor yExpected, Tensor yPredicted)
        {
            var mseGradient = this;
            Debug.Assert(mseGradient.SameShape(yExpected));
            Debug.Assert(mseGradient.SameShape(yPredicted));
            Parallel.For(0, mseGradient.Shape[0], m => { MseGradient_Helper(mseGradient.RowSlice(m, 1).AsFloatCpuSpan, yExpected.RowSlice(m, 1).AsReadonlyFloatCpuSpan, yPredicted.RowSlice(m, 1).AsReadonlyFloatCpuSpan); });
        }

        private static void MseGradient_Helper(Span<float> gradient, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted)
        {
            Debug.Assert(gradient.Length == expected.Length);
            Debug.Assert(gradient.Length == predicted.Length);
            for (int i = 0; i < gradient.Length; ++i)
            {
                var error = predicted[i] - expected[i];
                gradient[i] = error / gradient.Length;
            }
        }

        public override void MaeGradient(Tensor yExpected, Tensor yPredicted)
        {
            var maeGradient = this;
            Debug.Assert(maeGradient.SameShape(yExpected));
            Debug.Assert(maeGradient.SameShape(yPredicted));
            Parallel.For(0, maeGradient.Shape[0], m => { MaeGradient_Helper(maeGradient.RowSlice(m, 1).AsFloatCpuSpan, yExpected.RowSlice(m, 1).AsReadonlyFloatCpuSpan, yPredicted.RowSlice(m, 1).AsReadonlyFloatCpuSpan); });
        }
        private static void MaeGradient_Helper(Span<float> gradient, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted)
        {
            Debug.Assert(gradient.Length == expected.Length);
            Debug.Assert(gradient.Length == predicted.Length);
            for (int i = 0; i < gradient.Length; ++i)
            {
                var error = predicted[i] - expected[i];
                gradient[i] = Math.Sign(error) / (float)gradient.Length;
            }
        }

        public override void MseOfLogGradient(Tensor yExpected, Tensor yPredicted, float epsilon)
        {
            var mseOfLogGradient = this;
            Debug.Assert(mseOfLogGradient.SameShape(yExpected));
            Debug.Assert(mseOfLogGradient.SameShape(yPredicted));
            Parallel.For(0, mseOfLogGradient.Shape[0], m => { MseOfLogGradient_Helper(mseOfLogGradient.RowSlice(m, 1).AsFloatCpuSpan, yExpected.RowSlice(m, 1).AsReadonlyFloatCpuSpan, yPredicted.RowSlice(m, 1).AsReadonlyFloatCpuSpan, epsilon); });
        }
        private static void MseOfLogGradient_Helper(Span<float> gradient, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted, float epsilon)
        {
            Debug.Assert(gradient.Length == expected.Length);
            Debug.Assert(gradient.Length == predicted.Length);
            for (int i = 0; i < gradient.Length; ++i)
            {
                var adjustedPredicted = Math.Max(epsilon, predicted[i]);
                var error = Math.Log(adjustedPredicted) - Math.Log(expected[i]);
                gradient[i] = (float)(2 * error / (adjustedPredicted * gradient.Length));
            }
        }

        protected override void BCEWithFocalLossGradient(Tensor yExpected, Tensor yPredicted, float percentageInTrueClass, float gamma)
        {
            var bceWithFocalLossGradient = this;
            Debug.Assert(yExpected.Shape.Length == 2);
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(yExpected.SameShape(bceWithFocalLossGradient));
            var y_true = yExpected.AsFloatCpuSpan;
            var y_pred = yPredicted.AsFloatCpuSpan;
            var gradients = bceWithFocalLossGradient.AsFloatCpuSpan;
            var rows = yExpected.Shape[0];
            var numClass = yExpected.Shape[1];
            int idx = 0;

            var imbalancedCoeffForTrueClass = 1 / (2*percentageInTrueClass);
            var imbalancedCoeffForFalseClass = 1 / (2*(1-percentageInTrueClass));

            for (int row = 0; row < rows; ++row)
            {
                for (int col = 0; col < numClass; ++col)
                {
                    //the gradient value for standard binary cross entropy (without focal loss)
                    var nonScaledGradient = y_pred[idx] - y_true[idx];
                    var focalLossCoeff = 1.0f;
                    if (gamma > 0)
                    {
                        //we need to adjust the gradient value for focal loss
                        float maxValueForNonScaledGradient = Math.Max(y_true[idx], 1 - y_true[idx]);
                        focalLossCoeff = (gamma + 1) * MathF.Pow(MathF.Abs(nonScaledGradient) / maxValueForNonScaledGradient, gamma);
                    }

                    // we take into account the imbalance between the true and false class
                    // if one class is over represented, we reduce the gradient value for this class
                    float imbalancedCoeffForCurrentClass = imbalancedCoeffForFalseClass +  y_true[idx] * (imbalancedCoeffForTrueClass - imbalancedCoeffForFalseClass);

                    gradients[idx++] = (focalLossCoeff * imbalancedCoeffForCurrentClass * nonScaledGradient) /numClass;
                }
            }
        }

        public override void CategoricalCrossentropyWithHierarchyGradient(Tensor yExpected, Tensor yPredicted)
        {
            var categoricalCrossentropyWithHierarchyGradient = this;
            Debug.Assert(categoricalCrossentropyWithHierarchyGradient.SameShape(yExpected));
            Debug.Assert(categoricalCrossentropyWithHierarchyGradient.SameShape(yPredicted));
            Debug.Assert(categoricalCrossentropyWithHierarchyGradient.Dimension == 2);
            categoricalCrossentropyWithHierarchyGradient.ZeroMemory();
            Parallel.For(0, categoricalCrossentropyWithHierarchyGradient.Shape[0], m => { CategoricalCrossentropyWithHierarchyGradient_Helper(categoricalCrossentropyWithHierarchyGradient.RowSlice(m, 1).AsFloatCpuSpan, yExpected.RowSlice(m, 1).AsReadonlyFloatCpuSpan, yPredicted.RowSlice(m, 1).AsReadonlyFloatCpuSpan); });
        }

        private static void CategoricalCrossentropyWithHierarchyGradient_Helper(Span<float> loss, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted)
        {
            Debug.Assert(loss.Length == expected.Length);
            Debug.Assert(loss.Length == predicted.Length);
            for (int i = 0; i < loss.Length; ++i)
            {
                var expectedValue = expected[i];
                if (Math.Abs(expectedValue) < 9.5f)
                {
                    //expectedValue contains a proba between 0 and 1
                    Debug.Assert(expectedValue >= 0);
                    Debug.Assert(expectedValue <= 1.0);
                    Debug.Assert(predicted[i] >= 0.0);
                    Debug.Assert(predicted[i] <= 1.0);
                    loss[i] = predicted[i] - expectedValue;
                }
                else
                {
                    //expectedValue contains a description : there is no associated loss
                    if (expectedValue < 0)
                    {
                        var count = (int)(Math.Abs(expectedValue) + 0.5) / 10;
                        //we need to skip 'count' indexes
                        i += count - 1; //-1 because the for(;;) loop will also increment 'i'
                    }
                }
            }
        }
        #endregion


        public override void CopyTo(Tensor b)
        {
            if (Count != b.Count)
            {
                throw new ArgumentException("can't copy "+this+" to "+b);
            }
            if (b.UseGPU)
            {
                //copy from CPU ('this' tensor) to GPU ('b' tensor)
                if (HasPinnedMemory)
                {
                    //the tensor memory is already pinned
                    b.AsGPU<T>().InitializeFromHostPinnedMemory(Pointer);
                }
                else
                {
                    b.AsGPU<T>().InitializeFromHostMemory(Content);
                }
            }
            else
            {
                //copy from CPU ('this' tensor) to CPU ('b' tensor)
                Content.Slice(0, Count).CopyTo( ((CpuTensor<T>)b).Content.Slice(0, Count));
            }
        }


        ///// <summary>
        ///// return a new Tensor keeping only columns at index 'columnIndexesToKeep'
        ///// </summary>
        ///// <param name="columnIndexesToRemove">the column indexes to remove</param>
        ///// <returns></returns>
        //public CpuTensor<T> DropColumns(IEnumerable<int> columnIndexesToRemove)
        //{
        //    if (Shape.Length != 2)
        //    {
        //        throw new Exception($"{nameof(DropColumns)} only works with matrix");
        //    }
        //    if (columnIndexesToRemove == null || !columnIndexesToRemove.Any())
        //    {
        //        return (CpuTensor<T>)Clone();
        //    }
        //    var columnIndexesToKeep = Enumerable.Range(0, Shape[1]).ToList();
        //    foreach (var col in columnIndexesToRemove)
        //    {
        //        columnIndexesToKeep.Remove(col);
        //    }
        //    return KeepColumns(columnIndexesToKeep);
        //}

        /// <summary>
        /// Return a new Tensor keeping only columns at index 'columnIndexesToKeep'
        /// Those columns will be in the same order as the one provided in 'columnIndexesToKeep'
        /// </summary>
        /// <param name="columnIndexesToKeep"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        public CpuTensor<T> KeepColumns(List<int> columnIndexesToKeep)
        {
            if (Shape.Length != 2)
            {
                throw new Exception($"{nameof(KeepColumns)} only works with matrix");
            }
            var srcContent = SpanContent;
            var srcCols = Shape[1];
            var targetShape = new[] { Shape[0], columnIndexesToKeep.Count };
            var targetContent = new T[targetShape[0] * targetShape[1]];
            int newIdx = 0;
            for (int row = 0; row < Shape[0]; ++row)
            {
                foreach (var srcCol in columnIndexesToKeep)
                {
                    targetContent[newIdx++] = srcContent[srcCol+row*srcCols];
                }
            }
            return new CpuTensor<T>(targetShape, targetContent);
        }

        public static CpuTensor<float> CreateOneHotTensor(Func<int, int> elementIdToCategoryIndex, int elementCount, int numClass)
        {
            var result = new CpuTensor<float>(new[] { elementCount, numClass });
            var yContent = result.SpanContent;
            for (int elementId = 0; elementId < elementCount; ++elementId)
            {
                var categoryIndex = elementIdToCategoryIndex(elementId);
                if (categoryIndex >= 0)
                {
                    yContent[elementId * numClass + categoryIndex] = 1f;
                }
            }
            return result;
        }



        /// <summary>
        /// transform a tensor with class indexes to a tensor with a probability distribution of each class
        /// </summary>
        /// <param name="argMax">for each row, the index of the class associated with the row</param>
        /// <param name="numClasses">total number of distinct classes</param>
        /// <returns>a new tensor of shape (rows, numClasses) with for each row a 1 at the idx of the associated class and 0 elsewhere </returns>
        public static CpuTensor<float> FromClassIndexToProba(CpuTensor<float> argMax, int numClasses)
        {
            var argMaxContent = argMax.ReadonlyContent;
            int rows = argMax.Shape[0];
            var content = new float[rows * numClasses];
            for (int row = 0; row < rows; ++row)
            {
                int idx = Utils.NearestInt(argMaxContent[row]);
                if (idx >= numClasses)
                {
                    throw new Exception($"invalid index {idx} at row {row}, must be less than {numClasses}");
                }
                content[row * numClasses + idx] = 1; //100% for being class 'idx', 0% for all other classes
            }
            return new CpuTensor<float>(new[] { rows, numClasses }, content);
        }
        
        public static CpuTensor<T> MergeHorizontally(params CpuTensor<T>[] tensors)
        {
            tensors = tensors.Where(t => t != null).ToArray();
            if (tensors.Length == 0)
            {
                return null;
            }
            if (tensors.Length == 1)
            {
                return tensors[0];
            }

            var newColumns = tensors.Select(t => t.Shape[1]).Sum();
            var result = new CpuTensor<T>(new[] { tensors[0].Shape[0], newColumns });
            int nextColumnToInsert = 0;
            foreach (var t in tensors)
            {
                result.InsertOtherAtColumnIndex(t, nextColumnToInsert);
                nextColumnToInsert += t.Shape[1];
            }
            return result;
        }

        public static CpuTensor<T> MergeVertically(CpuTensor<T> top, CpuTensor<T> bottom)
        {
            if (top == null)
            {
                return bottom;
            }
            if (bottom == null)
            {
                return top;
            }
            return InsertAtRowIndex(top, bottom, top.Shape[0]);
        }


        public static CpuTensor<T> InsertAtColumnIndex(CpuTensor<T> source, CpuTensor<T> toAddAtColumnIndex, int columnIndex)
        {
            Debug.Assert(source.Shape.Length == 2);
            Debug.Assert(toAddAtColumnIndex.Shape.Length == 2);
            //same number of rows
            Debug.Assert(source.Shape[0] == toAddAtColumnIndex.Shape[0]);
            Debug.Assert(columnIndex <= source.Shape[1]);
            var newShape = new[] { source.Shape[0], source.Shape[1] + toAddAtColumnIndex.Shape[1] };

            var sourceSpan = source.SpanContent;
            var toAddSpan = toAddAtColumnIndex.SpanContent;

            var newData = new T[newShape[0] * newShape[1]];
            int nextIndexInSourceSpan = 0;
            int nextIndexInToAddAtColumnIndex = 0;
            int nextIndexInNewData = 0;
            for (int row = 0; row < newShape[0]; ++row)
            {
                for (int col = 0; col < columnIndex; ++col)
                {
                    newData[nextIndexInNewData++] = sourceSpan[nextIndexInSourceSpan++];
                }
                for (int col = 0; col < toAddAtColumnIndex.Shape[1]; ++col)
                {
                    newData[nextIndexInNewData++] = toAddSpan[nextIndexInToAddAtColumnIndex++];
                }
                for (int col = columnIndex; col < source.Shape[1]; ++col)
                {
                    newData[nextIndexInNewData++] = sourceSpan[nextIndexInSourceSpan++];
                }
            }
            return new CpuTensor<T>(newShape, newData);
        }


        private void InsertOtherAtColumnIndex(CpuTensor<T> other, int columnIndex)
        {
            Debug.Assert(Shape.Length == 2);
            Debug.Assert(other.Shape.Length == 2);
            //same number of rows
            Debug.Assert(Shape[0] == other.Shape[0]);
            Debug.Assert(columnIndex <= Shape[1]);
            var otherSpan = other.ReadonlyContent;
            var content = SpanContent;
            int nextIndexInToAddAtColumnIndex = 0;
            for (int row = 0; row < Shape[0]; ++row)
            {
                int nextIndexInNewData = columnIndex+row * Shape[1];
                for (int col = 0; col < other.Shape[1]; ++col)
                {
                    content[nextIndexInNewData++] = otherSpan[nextIndexInToAddAtColumnIndex++];
                }
            }
        }



        public static CpuTensor<T> InsertAtRowIndex(CpuTensor<T> source, CpuTensor<T> toAddAtRowIndex, int rowIndex)
        {
            Debug.Assert(source.Shape.Length == 2);
            Debug.Assert(toAddAtRowIndex.Shape.Length == 2);
            //same number of rows
            var columns = source.Shape[1];
            Debug.Assert(columns == toAddAtRowIndex.Shape[1]);
            Debug.Assert(rowIndex <= source.Shape[0]);
            var newShape = new[] { source.Shape[0] + toAddAtRowIndex.Shape[0], columns };

            var newTensor = new CpuTensor<T>(newShape);
            source.CopyTo(0, newTensor, 0, rowIndex* columns);
            toAddAtRowIndex.CopyTo(0, newTensor, rowIndex * columns, toAddAtRowIndex.Count);
            if (rowIndex < source.Shape[0])
            {
                source.CopyTo(rowIndex * columns, newTensor, rowIndex * columns+ toAddAtRowIndex.Count, source.Count-rowIndex * columns);
            }
            return newTensor;
        }
        
        public static CpuTensor<float> NewCpuTensor(IList<float[]> rows)
        {
            var x = new CpuTensor<float>(new[] { rows.Count, rows[0].Length });
            var xSpan = x.AsFloatCpuSpan;
            int xSpanIndex = 0;
            foreach (var row in rows)
            {
                Debug.Assert(row.Length == x.Shape[1]);
                foreach (var t in row)
                {
                    xSpan[xSpanIndex++] = t;
                }
            }
            Debug.Assert(xSpanIndex == x.Count);
            return x;
        }

        /// <summary>
        /// copy the columns at indexes 'columnsToLoadFromSource' from 'source' tensor to 'this' tensor
        /// </summary>
        /// <param name="source"></param>
        /// <param name="columnsToLoadFromSource"></param>

        public void LoadColumnsFromSource(CpuTensor<T> source, IList<int> columnsToLoadFromSource)
        {
            Debug.Assert(SameShape(source));
            Debug.Assert(Shape.Length == 2);
            var sourceSpan = source.SpanContent;
            var thisSpan = SpanContent;
            for (int row = 0; row < Shape[0]; ++row)
            {
                int firstIndex = row * Shape[1];
                foreach (var columnIndex in columnsToLoadFromSource)
                {
                    thisSpan[firstIndex+ columnIndex] = sourceSpan[firstIndex + columnIndex];
                }
            }
        }

        public override void CopyTo(int startElement, Tensor other, int otherStartElement, int elementCount)
        {
            var src = Content.Slice(startElement, elementCount);
            var dest = ((CpuTensor<T>)other).Content.Slice(otherStartElement, elementCount);
            src.CopyTo(dest);
        }
       
        public override Tensor Slice(int startIndex, int[] sliceShape)
        {
            Debug.Assert(startIndex >= 0);
            return new CpuTensor<T>((int[])sliceShape.Clone(), this, startIndex);
        }


        public Span<T> RowSpanSlice(int startRowIndex, int nbRows)
        {
            Debug.Assert(Shape.Length >= 2);
            Debug.Assert(startRowIndex >= 0);
            Debug.Assert(startRowIndex < Shape[0]);
            Debug.Assert(startRowIndex + nbRows - 1 < Shape[0]);
            return SpanSlice(startRowIndex* MultDim0, MultDim0);
        }

        private Span<T> SpanSlice(int start, int length)
        {
            Debug.Assert(start >= 0);
            return Content.Slice(start, length).Span;
        }

        public T[] ColumnContent(int columnIndex)
        {
            Debug.Assert(Shape.Length == 2);
            var res = new T[Shape[0]];
            var content = ReadonlyContent;
            for (int row = 0; row < res.Length; row++)
            {
                res[row] = content[columnIndex + row * Shape[1]];
            }
            return res;
        }
        
        public override void YOLOV3Forward(Tensor x, int inputImageHeight, int inputImageWidth, int[] anchors)
        {
            Debug.Assert(anchors.Length %2 == 0);
            int nbAnchors = anchors.Length / 2;
            var y = AsFloatCpu;
            Debug.Assert(inputImageHeight % x.Shape[2] == 0);
            Debug.Assert(inputImageWidth % x.Shape[3] == 0);
            Debug.Assert(y.Shape[0] == x.Shape[0]);
            Debug.Assert(x.Shape[1] % nbAnchors == 0);
            Debug.Assert(nbAnchors * y.Shape[2] == x.Shape[1]);
            Debug.Assert(y.Shape[1] == nbAnchors * x.Shape[2] * x.Shape[3]);

            var xContent = x.AsFloatCpuSpan;
            var yContent = y.SpanContent;

            //2 for box centers + 2 for box size + 1 for box confidence + N for categories (N == 80 for COCO)
            int predictionLength = x.Shape[1] / nbAnchors;
            int categories = predictionLength - 5;
            int rowStride = inputImageHeight / x.Shape[2];
            int colStride = inputImageWidth / x.Shape[3];

            var yNextIndex = 0;
            for (int n = 0; n < x.Shape[0]; ++n)
            for (int h = 0; h < x.Shape[2]; ++h)
            for (int w = 0; w < x.Shape[3]; ++w)
            {
                for (int boxId = 0; boxId < anchors.Length/2; ++boxId)
                {
                    //box center
                    var xNextIndex = x.Idx(n, boxId* predictionLength, h, w);
                    yContent[yNextIndex++] = (w + Utils.Sigmoid(xContent[xNextIndex])) * colStride;
                    xNextIndex += x.MultDim1;
                    yContent[yNextIndex++] = (h + Utils.Sigmoid(xContent[xNextIndex])) * rowStride;
                    xNextIndex += x.MultDim1;

                    //box size
                    var anchorWidth = anchors[2 * boxId];
                    yContent[yNextIndex++] = (float) (anchorWidth * Math.Exp(xContent[xNextIndex]));
                    xNextIndex += x.MultDim1;
                    var anchorHeight = anchors[2 * boxId+1];
                    yContent[yNextIndex++] = (float)(anchorHeight * Math.Exp(xContent[xNextIndex]));
                    xNextIndex += x.MultDim1;

                    //box confidence
                    yContent[yNextIndex++] = Utils.Sigmoid(xContent[xNextIndex]);
                    xNextIndex += x.MultDim1;

                    //categories
                    for (int i = 0; i < categories; ++i)
                    {
                        yContent[yNextIndex++] = Utils.Sigmoid(xContent[xNextIndex]);
                        xNextIndex += x.MultDim1;
                    }
                }
            }
        }

        public override void UpdateWithPositionalEncoding_AttnIsAllYouNeed(int n)
        {
            Debug.Assert(Shape.Length == 3);
            int batchSize = Shape[0];
            int timeSteps = Shape[1];
            int embeddingDim = Shape[2];
            var spanContent = AsFloatCpuSpan;
            int idx = 0;
            for (int batch = 0; batch < batchSize; ++batch)
            {
                for (int k = 0; k < timeSteps; ++k)
                {
                    for (int col = 0; col < embeddingDim; ++col)
                    {
                        int i = col / 2;
                        float value = col % 2 == 0 
                            ? MathF.Sin(k / MathF.Pow(n, (2f * i) / embeddingDim)) 
                            : MathF.Cos(k / MathF.Pow(n, (2f * i) / embeddingDim));
                        spanContent[idx++] += value;
                    }
                }
            }
        }

        public override void ZeroMemory()
        {
            SpanContent.Clear();
        }
        public override void Dot(Tensor a, bool transposeA, Tensor b, bool transposeB, float alpha, float beta)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, a, b }));
            Debug.Assert(a.Dimension >= 2);
            Debug.Assert(b.Dimension >= 2);
            Debug.Assert(Dimension >= 2);

            BlasServices.DotMkl(a.AsFloatPointer, a.Shape[0], a.MultDim0, transposeA, b.AsFloatPointer, b.Shape[0], b.MultDim0, transposeB, AsFloatPointer, alpha, beta);
            //Utils.DotOpenblas(a.Content, a.Height, a.Width, b.Content, b.Height, b.Width, y.Content);
            //var tmpTranspose = new double[b.Count];
            //Utils.DotCSharp(a.Content, a.Height, a.Width, b.Content, b.Height, b.Width, tmpTranspose, y.Content);
        }
        
        public override void BatchMatrixMultiplication(Tensor a_3D, bool transposeA, Tensor b_3D, bool transposeB, float alpha, float beta)
        {
            var c_3D = this;
            Debug.Assert(a_3D.Shape.Length == 3);
            Debug.Assert(b_3D.Shape.Length == 3);
            Debug.Assert(c_3D.Shape.Length == 3);
            Debug.Assert(a_3D.Shape[0] == b_3D.Shape[0]);
            Debug.Assert(a_3D.Shape[0] == c_3D.Shape[0]);
            int nbMatrices = a_3D.Shape[0];

            var aShape = a_3D.Shape.Skip(1).ToArray();
            var bShape = b_3D.Shape.Skip(1).ToArray();
            var cShape = c_3D.Shape.Skip(1).ToArray();

            for (int i = 0; i < nbMatrices; ++i)
            {
                var a = a_3D.GetSubTensor(i, aShape);
                var b = b_3D.GetSubTensor(i, bShape);
                var c = c_3D.GetSubTensor(i, cShape);
                c.Dot(a, transposeA, b, transposeB, alpha, beta);
            }
        }

        public override void SetToZeroAllElementsBelowMainDiagonal()
        {
            var spanContent = AsFloatCpuSpan;
            for (int row = 0; row < Shape[0]; ++row)
                for (int col = 0; col < Math.Min(Shape[1], row); ++col)
                {
                    spanContent[row*Shape[1]+col] = 0;
                }
        }

        public override void SetAllElementsAboveMainDiagonal(float valueForElementsAboveMainDiagonal)
        {
            Debug.Assert(Shape.Length == 2 || Shape.Length == 3);
            (int matrices_count,int rows_by_matrix,int cols_by_matrix) = Shape.Length == 3
                ? (Shape[0], Shape[1], Shape[2])
                : (1, Shape[0], Shape[1]);
            for (int matrixId = 0; matrixId < matrices_count; ++matrixId)
            {
                var spanContent = Shape.Length == 3 ?ElementSlice(matrixId).AsFloatCpuSpan : AsFloatCpuSpan;
                for (int row = 0; row < rows_by_matrix; ++row)
                    for (int col = row + 1; col < cols_by_matrix; ++col)
                    {
                        spanContent[row * cols_by_matrix + col] = valueForElementsAboveMainDiagonal;
                    }
            }
        }

        public override void SetIdentityMatrix()
        {
            Debug.Assert(Shape.Length == 2);
            Debug.Assert(Shape[0] == Shape[1]);
            ZeroMemory();
            var spanContent = AsFloatCpuSpan;
            for (int row = 0; row < Shape[0]; ++row)
            {
                spanContent[row * Shape[1] + row] = 1f;
            }
        }

        public override void Orthogonal(Random rand)
        {
            NormalDistribution(rand, 0, 1);
            Q_Factorization();
        }
        public override int QRFactorization_FloatBufferLength()
        {
            return 1;
        }
        public override void QRFactorization(Tensor Q, Tensor R, Tensor buffer)
        {
            Q_Factorization(Q);
            R.Dot(Q, true, this, false, 1.0f, 0.0f);
        }
        /// <summary>
        /// Make input 'this' an orthogonal matrix using Gram–Schmidt process
        /// this: A matrix with shape: (rows, cols)
        /// See: https://en.wikipedia.org/wiki/QR_decomposition
        /// </summary>
        /// <param name="Q">An orthogonal Matrix of shape (rows, col)</param>
        public void Q_Factorization(Tensor Q = null)
        {
            //Debug.Assert(matrix.Length == rows * cols);

            //var aSpan = AsFloatCpuSpan;
            int rows = Shape[0];
            int cols = MultDim0;

            int rowsTransposed = cols;
            int colsTransposed = rows;
            var aTransposed = new CpuTensor<float>(new[] { rowsTransposed, colsTransposed });

            Transpose(aTransposed);
            var aTransposedSpan = aTransposed.AsFloatCpuSpan;

            //We compute the U matrix as described in: https://en.wikipedia.org/wiki/QR_decomposition 
            var U = new Span<float>(new float[aTransposedSpan.Length]);
            aTransposedSpan.CopyTo(U);
            for (int row = 1; row < rowsTransposed; ++row)
            {
                //we compute row 'row' of 'U' matrix
                var aRow = aTransposedSpan.Slice(colsTransposed * row, colsTransposed);
                var uRow = U.Slice(colsTransposed * row, colsTransposed);
                for (int subRow = 0; subRow < row; ++subRow)
                {
                    var uSubRow = U.Slice(colsTransposed * subRow, colsTransposed);
                    float multiplier = InnerProduct(uSubRow, aRow) / InnerProduct(uSubRow, uSubRow);
                    for (int col = 0; col < uSubRow.Length; ++col)
                    {
                        uRow[col] -= multiplier * uSubRow[col];
                    }
                }
            }

            //We compute the Q (= rectangularMatrix) matrix:
            //  it is an orthogonal matrix that we can compute from the U matrix
            //  (by normalizing each row of the U matrix)
            U.CopyTo(aTransposedSpan);
            for (int row = 0; row < rowsTransposed; ++row)
            {
                var aTransposedRow = aTransposedSpan.Slice(colsTransposed * row, colsTransposed);
                float normalizer = (float)Math.Sqrt(InnerProduct(aTransposedRow, aTransposedRow));
                for (int col = 0; col < aTransposedRow.Length; ++col)
                {
                    aTransposedRow[col] /= normalizer;
                }
            }

            aTransposed.Transpose(Q??this);
        }

        public override void Transpose(Tensor transposed)
        {
            Debug.Assert(Dimension == 2);
            if (transposed.CapacityInBytes < ReallyNeededMemoryInBytesForShape(Shape))
            {
                throw new ArgumentException("Can't transpose to tensor: not enough capacity");
            }
            transposed.ReshapeInPlace(new[] { Shape[1], Shape[0] });
            Debug.Assert(transposed.Dimension == Dimension);
            Debug.Assert(transposed.Shape[0] == Shape[1]);
            Debug.Assert(transposed.Shape[1] == Shape[0]);

            var inputSpan = AsReadonlyFloatCpuSpan;
            var outputSpan = transposed.AsFloatCpuSpan;
            for (int row = 0; row < Shape[0]; ++row)
            for (int col = 0; col < Shape[1]; ++col)
            {
                var srcIndex = row * Shape[1] + col;
                var srcValue = inputSpan[srcIndex];
                var targetIndex = col * Shape[0] + row;
                outputSpan[targetIndex] = srcValue;
            }
        }

        private static float InnerProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
        {
            Debug.Assert(a.Length == b.Length);
            float result = 0;
            for (int i = 0; i < a.Length; ++i)
            {
                result += a[i] * b[i];
            }

            return result;
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
            var content = ReadonlyContent;
            for (int m = 0; m < Shape[0]; ++m)
            {
                int startIdx = Idx(m, c, 0, 0);
                for (int idx = startIdx; idx < (startIdx + MultDim1); ++idx)
                {
                    var val = toFloat(content[idx]);
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
        private void MergeInPlaceByRow(CpuTensor<float> a, CpuTensor<float> b, Func<float, float, float> func, float rowDivider)
        {
            var buffer = this;
            Debug.Assert(a.Dimension == b.Dimension);
            Debug.Assert(a.Count == b.Count);
            Debug.Assert(buffer.Count == a.Shape[0]);
            var content = (float*)buffer.Pointer;
            var aSpan = a.ReadonlyContent;
            var bSpan = b.ReadonlyContent;

            int idx = 0;
            for (int row = 0; row < buffer.Count; ++row)
            {
                var rowResult = 0f;
                for (int col = 0; col <a.MultDim0; ++col)
                {
                    rowResult += func(aSpan[idx], bSpan[idx]);
                    ++idx;
                }
                content[row] = rowResult/ rowDivider;
            }
        }
        private void Update(Tensor a, Tensor b, Func<T, T, T, T> funcInput)
        {
            Debug.Assert(AreCompatible(new List<Tensor> {this, a, b}));
            Debug.Assert(SameShape(a, b));
            var aSpan = a.AsCpu<T>().ReadonlyContent;
            var bSpan = b.AsCpu<T>().ReadonlyContent;
            var thisSpan = SpanContent;
            for (int i = 0; i < Count; ++i)
            {
                thisSpan[i] = funcInput(thisSpan[i], aSpan[i], bSpan[i]);
            }
        }
        private void Update(Tensor b, Func<T, T, T> funcInput)
        {
            Debug.Assert(AreCompatible(new List<Tensor> {this, b}));
            Debug.Assert(SameShape(b));
            var bSpan = b.AsCpu<T>().ReadonlyContent;
            var thisSpan = SpanContent;
            for (int i = 0; i < Count; ++i)
            {
                thisSpan[i] = funcInput(thisSpan[i], bSpan[i]);
            }
        }


        /// <summary>
        /// update the entire tensor in place
        /// </summary>
        /// <param name="update"></param>
        public void UpdateInPlace(Func<T, T> update)
        {
            var content = SpanContent;
            for (int i = 0; i < Count; ++i)
            {
                content[i] = update(content[i]);
            }
        }


        /// <summary>
        /// for each column index in 'columnIndexToUpdate' ,
        /// update the column values by applying 'update' function
        /// </summary>
        /// <param name="update"></param>
        /// <param name="columnIndexToUpdate"></param>
        public void UpdateInPlace(Func<T, T> update, params int[] columnIndexToUpdate)
        {
            if (columnIndexToUpdate.Length == 0)
            {
                return; //nothing to do
            }
            Debug.Assert(Shape.Length == 2);
            var rows = Shape[0];
            var cols = Shape[1];
            var content = SpanContent;
            for(int row=0;row<rows;++row)
            {
                foreach (var col in columnIndexToUpdate)
                {
                    var idx = row * cols + col;
                    content[idx] = update(content[idx]);
                }
            }
        }

        public void BuildEntirelyFromInput(Tensor a, Tensor b, Func<T, T, T> funcInput)
        {
            Debug.Assert(AreCompatible(new List<Tensor> {this, a, b}));
            Debug.Assert(SameShape(a, b));
            var aSpan = a.AsCpu<T>().ReadonlyContent;
            var bSpan = b.AsCpu<T>().ReadonlyContent;
            var thisSpan = SpanContent;
            for (int i = 0; i < a.Count; ++i)
            {
                thisSpan[i] = funcInput(aSpan[i], bSpan[i]);
            }
        }
        public void BuildEntirelyFromInput(Tensor a, Tensor b, Tensor c, Func<T, T, T, T> funcInput)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, a, b, c }));
            Debug.Assert(SameShape(a, b));
            var aSpan = a.AsCpu<T>().ReadonlyContent;
            var bSpan = b.AsCpu<T>().ReadonlyContent;
            var cSpan = c.AsCpu<T>().ReadonlyContent;
            var thisSpan = SpanContent;
            for (int i = 0; i < a.Count; ++i)
            {
                thisSpan[i] = funcInput(aSpan[i], bSpan[i], cSpan[i]);
            }
        }
        private void ComputeSumByColumn(Tensor sumByColumn)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, sumByColumn }));
            Debug.Assert(Dimension >= 2);
            var batchSize = Shape[0];
            bool is1C11Shape = sumByColumn.Count == sumByColumn.Shape[1];

            sumByColumn.ZeroMemory();
            var content = AsFloatCpuSpan;
            var columnSumContent = sumByColumn.AsFloatCpuSpan;
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
        public void Compute_Column_Mean_Variance(Tensor mean, Tensor variance)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, mean, variance }));
            var batchSize = Shape[0];
            Debug.Assert(mean.SameShape(variance));
            //true if we have a (1,C,1,1) shape for scale and bias
            //false is we have a (1,C,H,W) shape for scale and bias
            bool is1C11Shape = mean.Count == mean.Shape[1];

            mean.ZeroMemory();
            variance.ZeroMemory();
            var content = AsFloatCpuSpan;
            //we'll store in meanContent Sum(X) and in varianceContent Sum(X^2)
            var meanContent = mean.AsFloatCpuSpan;
            var varianceContent = variance.AsFloatCpuSpan;
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
            for (int i = 0; i < variance.Count; ++i)
            {
                meanContent[i] /= meanDivider;
                //Variance(X) = E(X^2) - E(X) ^2
                //varianceContent[i] = varianceContent[i]/meanDivider - meanContent[i] * meanContent[i];
                varianceContent[i] = (meanDivider <= 1) ? 1f : (varianceContent[i] - meanDivider * meanContent[i] * meanContent[i]) / (meanDivider - 1);
            }
        }



        // ReSharper disable once UnusedMember.Global
        public void Save(string filePath, Func<int, bool> shouldSaveRow, bool addColumnWithRowIndex, string header = null)
        {
            var sb = new StringBuilder();
            if (!string.IsNullOrEmpty(header))
            {
                sb.Append(header + Environment.NewLine);
            }

            int rowIndex = 0;
            for (int row = 0; row < Shape[0]; ++row)
            {
                if (!shouldSaveRow(row))
                {
                    continue;
                }

                if (addColumnWithRowIndex)
                {
                    sb.Append(rowIndex + ";");
                }
                var tmp = ElementSlice(row).AsFloatCpuSpan.ToArray();
                sb.Append(string.Join(";", tmp.Select(x => x.ToString(CultureInfo.InvariantCulture))));
                sb.Append(Environment.NewLine);
                ++rowIndex;
            }
            File.WriteAllText(filePath, sb.ToString());
        }

        public CpuTensor<float> ArgMax()
        {
            var bufferShape = (int[])Shape.Clone();
            bufferShape[^1] = 1;
            var buffer = new CpuTensor<float>(bufferShape);
            ArgMax(buffer);
            return buffer;
        }

        public override void ArgMax(Tensor buffer)
        {
            var input = this;
            if (input.Shape.Length >= 3)
            {
                var input2D = input.Reshape(-1, input.Shape[^1]);
                var buffer2D = buffer.Reshape(-1, 1);
                input2D.ArgMax(buffer2D);
                return;
            }
            // input shape: (rows, numClass)
            // buffer shape: (rows, 1)
            Debug.Assert(input.Shape.Length == 2);
            Debug.Assert(buffer.Shape.Length == 2);
            Debug.Assert(buffer.Shape[1] == 1);
            Debug.Assert(input.Shape[0] == buffer.Shape[0]);
            int rows = input.Shape[0];
            int numClass = input.Shape[1];
            var inputContent = input.AsReadonlyFloatCpuSpan;
            var bufferContent = buffer.AsFloatCpuSpan;

            for (int row = 0; row < rows; row++)
            {
                int startIdx = row * numClass;
                int colArgMax = 0;
                for (int col = 1; col < numClass; col++)
                {
                    if (inputContent[startIdx + col] > inputContent[startIdx + colArgMax])
                    {
                        colArgMax = col;
                    }
                }
                bufferContent[row] = colArgMax;
            }
        }

        public void CopyToSingleRow(int srcRow, int targetRow, IList<int> srcToTargetIndexes, CpuTensor<T> targetTensor)
        {
            var srcTensor = this;
            var srcContent = srcTensor.RowSpanSlice(srcRow, 1);
            var targetContent = targetTensor.RowSpanSlice(targetRow, 1);
            for (var srcIndex = 0; srcIndex < srcToTargetIndexes.Count; srcIndex++)
            {
                var targetIndex = srcToTargetIndexes[srcIndex];
                if (targetIndex >= 0)
                {
                    targetContent[targetIndex] = srcContent[srcIndex];
                }
            }
        }
        public CpuTensor<T> ApplyRowOrder(int[] targetRowToSrcRow)
        {
            Debug.Assert(Shape.Length == 2);
            int cols = Shape[1];
            var targetTensor = new CpuTensor<T>(new []{targetRowToSrcRow.Length, cols});

            var srcContent = ReadonlyContent;
            var targetContent = targetTensor.SpanContent;

            for (int targetRow = 0; targetRow < targetRowToSrcRow.Length; ++targetRow)
            {
                var srcRow = targetRowToSrcRow[targetRow];
                for (int col = 0; col < cols; ++col)
                {
                    targetContent[col+ targetRow * cols] = srcContent[col + srcRow* cols];
                }
                //RowSlice(srcRow, 1).CopyTo(newTensor.RowSlice(newRow, 1));
            }
            return targetTensor;
        }


        public static CpuTensor<float> LoadFromBinFile(string bin_file, int[] shape)
        {
            long elementCountInfile = Utils.FileLength(bin_file) / sizeof(float);
            shape = FillMinusOneIfAny(new[] { (int)elementCountInfile }, shape);
            long elementCount = Utils.LongProduct(shape);
            if (elementCountInfile != elementCount)
            {
                throw new ArgumentException("");
            }
            float[] buffer = new float[elementCount];
            using var fs = new FileStream(bin_file, FileMode.Open, FileAccess.Read);
            using var r = new BinaryReader(fs);
            for (int i = 0; i < buffer.Length; i++)
            {
                buffer[i] = r.ReadSingle();
            }
            return new CpuTensor<float>(shape, buffer);
        }

        public static List<CpuTensor<float>> LoadTensorListFromBinFileAndStandardizeIt(string bin_file, int[] shape, float mean = 0f, float stdDev = 1f)
        {
            ISample.Log.Info($"Loading {shape[0]} tensors from {bin_file} with shape {ShapeToString(shape)}");

            long elementCountInfile = Utils.FileLength(bin_file) / sizeof(float);
            shape = FillMinusOneIfAny(new[] { (int)elementCountInfile }, shape);
            long elementCount = Utils.LongProduct(shape);
            if (elementCountInfile != elementCount)
            {
                throw new ArgumentException("");
            }
            var numberOfTensors = shape[0];
            var singleTensorShape = shape.Skip(1).ToArray();
            var singleTensorCount = Utils.Product(singleTensorShape);
            using var fs = new FileStream(bin_file, FileMode.Open, FileAccess.Read);
            using var r = new BinaryReader(fs);
            var res = new List<CpuTensor<float>>();

            var xAccBeforeStandardization = new DoubleAccumulator();
            var xAccAfterStandardization = new DoubleAccumulator();
            for (int t = 0; t < numberOfTensors; ++t)
            {
                var tensorBuffer = new float[singleTensorCount];
                for (int i = 0; i < singleTensorCount; i++)
                {
                    var f = r.ReadSingle();
                    xAccBeforeStandardization.Add(f);
                    var fNormalized = (f-mean)/stdDev;
                    xAccAfterStandardization.Add(fNormalized);
                    tensorBuffer[i] = fNormalized;
                }
                res.Add(new CpuTensor<float>(singleTensorShape, tensorBuffer));
            }
            ISample.Log.Info($"Stats before standardization: {xAccBeforeStandardization}");
            ISample.Log.Info($"Stats after standardization: {xAccAfterStandardization}");
            return res;
        }

    }
}
