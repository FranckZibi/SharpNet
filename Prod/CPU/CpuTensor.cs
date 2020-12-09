using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;
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

        public CpuTensor(int[] shape, T[] data, int typeSize) : base(shape, typeSize, false)
        {
            Content = data ?? new T[Count];
            CapacityInBytes = (ulong)(Content.Length * TypeSize);
            _ptrToOwnerPinnedMemory = IntPtr.Zero;
        }
        public CpuTensor(int[] shape, T[] data = null) : this(shape, data, Marshal.SizeOf(typeof(T)))
        {
        }
        private CpuTensor(int[] shape, CpuTensor<T> memoryOwner, int startIndex) : base(shape, memoryOwner.TypeSize, false)
        {
            Content = memoryOwner.Content.Slice(startIndex, Utils.Product(shape));
            CapacityInBytes = (ulong)(Content.Length * TypeSize);
            _ptrToOwnerPinnedMemory = memoryOwner.Pointer + TypeSize * startIndex;
        }

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

        //this (= 'y') shape :      (batchSize, embeddingDim, maxWordCountBySentence)
        //'x' shape:                (batchSize, maxWordCountBySentence)
        //'wordEmbedding' shape:    (vocabularySize, embeddingDim)
        public override void WordEmbeddingForwardPropagation(Tensor x, Tensor wordEmbedding)
        {
            var y = this;
            Debug.Assert(x.Shape.Length == 2);
            Debug.Assert(wordEmbedding.Shape.Length == 2);
            Debug.Assert(y.Shape.Length == 3);
            Debug.Assert(y.Shape[0] == x.Shape[0]); //same batch size
            Debug.Assert(y.Shape[1] == wordEmbedding.Shape[1]); //same embedding dimension
            Debug.Assert(y.Shape[2] == x.Shape[1]); //same max word count by sentence
            var maxWordCountBySentence = x.Shape[1];
            var embeddingDim = wordEmbedding.Shape[1];
            var xCpu = (CpuTensor<float>)x;
            void ProcessBatch(int batchIndex)
            {
                var ySpan = y.AsFloatCpuSpan;
                var wordEmbeddingSpan = wordEmbedding.AsReadonlyFloatCpuContent;
                for (int wordInSentence = 0; wordInSentence < maxWordCountBySentence; ++wordInSentence)
                {
                    int wordIndex = (int)(xCpu.Get(batchIndex, wordInSentence) + 0.1);
                    int indexInWordEmbedding = wordEmbedding.Idx(wordIndex, 0);
                    int indexInY = y.Idx(batchIndex, 0, wordInSentence);
                    for (int embeddingId = 0; embeddingId < embeddingDim; ++embeddingId)
                    {
                        ySpan[indexInY] = wordEmbeddingSpan[indexInWordEmbedding];
                        indexInY += maxWordCountBySentence;
                        ++indexInWordEmbedding;
                    }
                }
            }
            Parallel.For(0, x.Shape[0], ProcessBatch);
        }

        //this (= dW) shape:        (VocabularySize, EmbeddingDim)
        // x shape :                (batchSize,  maxWordCountBySentence)
        // dy shape :               (batchSize, EmbeddingDim,  maxWordCountBySentence)
        public override void WordEmbeddingBackwardPropagation(Tensor x, Tensor dy)
        {
            var dW = this;
            var xCpu = (CpuTensor<float>)x;
            var dyCpu = (CpuTensor<float>)dy;

            Debug.Assert(dW.Shape.Length == 2);
            Debug.Assert(x.Shape.Length == 2);
            Debug.Assert(dy.Shape.Length == 3);
            Debug.Assert(dy.Shape[0] == x.Shape[0]); //same batch size
            Debug.Assert(dy.Shape[1] == dW.Shape[1]); //same embedding dimension
            Debug.Assert(dy.Shape[2] == x.Shape[1]); //same max word count by sentence

            dW.ZeroMemory();
            var batchSize = dy.Shape[0];
            var embeddingDim = dy.Shape[1];
            var maxWordCountBySentence = dy.Shape[2];

            var xSpan = x.AsFloatCpuSpan;
            var dWSpan = dW.AsFloatCpuSpan;
            var dySpan = dy.AsFloatCpuSpan;

            for (int batchIndex = 0; batchIndex < batchSize; ++batchIndex)
            {
                for (int wordInSentence = 0; wordInSentence < maxWordCountBySentence; ++wordInSentence)
                {
                    int wordIndex = (int)(xSpan[xCpu.Idx(batchIndex, wordInSentence)] + 0.1);
                    int indexInDw = dW.Idx(wordIndex, 0);
                    int indexIndY = dyCpu.Idx(batchIndex, 0, wordInSentence);
                    for (int embeddingId = 0; embeddingId < embeddingDim; ++embeddingId)
                    {
                        dWSpan[indexInDw] += dySpan[indexIndY];
                        ++indexInDw;
                        indexIndY += maxWordCountBySentence;
                    }
                }
            }
        }

        /// <summary>
        /// resize the current Cpu tensor to a different shape (both bigger or smaller)
        /// </summary>
        /// <param name="newShape"></param>
        public override void Reshape(int[] newShape)
        {
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
        public override Tensor WithNewShape(int[] newShape)
        {
            AssertIsNotDisposed();
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
            var srcContent = AsReadonlyFloatCpuContent;
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
            target.Reshape(targetShape);
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
        public override void DropoutForward(Tensor y, double dropProbability, bool isTraining, Random dropoutRandom, Tensor dropoutReservedSpaceForTraining, Tensor randomNumberGeneratorStatesBufferForGPU)
        {
            var x = this;
            if (!isTraining)
            {
                x.CopyTo(y);
                return;
            }
            Debug.Assert(dropoutReservedSpaceForTraining != null);
            Debug.Assert(!dropoutReservedSpaceForTraining.UseGPU);
            Debug.Assert(randomNumberGeneratorStatesBufferForGPU == null);
            var dropProbabilityFloat = (float)dropProbability;
            Utils.RandomizeUniformDistribution(dropoutReservedSpaceForTraining.AsFloatCpuSpan, dropoutRandom, 0.0, 1.0);
            y.AsFloatCpu.BuildEntirelyFromInput(x, dropoutReservedSpaceForTraining, (prevLayer, prob) => prob < dropProbability ? 0f : prevLayer / (1 - dropProbabilityFloat));
        }
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropProbability, Tensor dropoutReserveSpace)
        {
            Debug.Assert(dropoutReserveSpace != null);
            Debug.Assert(!dropoutReserveSpace.UseGPU);
            var dropProbabilityFloat = (float)dropProbability;
            dx.AsFloatCpu.BuildEntirelyFromInput(dy, dropoutReserveSpace, (dOutput, prob) => prob < dropProbabilityFloat ? 0f : dOutput / (1 - dropProbabilityFloat));
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
                    Debug.Assert(activationParameter != null);
                    Debug.Assert(activationParameter.Dimension == 1);
                    Debug.Assert(activationParameter.Count == 1);
                    CpuTensorActivationFunctions.LeakyRelu(x, y, activationParameter.AsReadonlyFloatCpuContent[0]);
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
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY:
                    Debug.Assert(activationParameter != null);
                    Debug.Assert(activationParameter.Dimension == 1);
                    CpuTensorActivationFunctions.SoftmaxWithHierarchy(x, y, activationParameter);
                    return;

                case cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH:
                    CpuTensorActivationFunctions.Swish(x, y);
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
                    Debug.Assert(activationParameter != null);
                    Debug.Assert(activationParameter.Dimension == 1);
                    Debug.Assert(activationParameter.Count == 1);
                    CpuTensorActivationFunctions.LeakyReluGradient(y, dy, dx, activationParameter.AsReadonlyFloatCpuContent[0]);
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
                case cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX_WITH_HIERARCHY:
                    Debug.Assert(activationParameter != null);
                    Debug.Assert(activationParameter.Dimension == 1);
                    CpuTensorActivationFunctions.SoftmaxGradientWitHierarchy(y, dy, dx, activationParameter);
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
            Debug.Assert(afterDownSampling != null);
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
        public static CpuTensor<float> CreateOneHotTensor(Func<int,int> elementIdToCategoryIndex, int elementCount, int categoryCount)
        {
            var result = new CpuTensor<float>(new[] { elementCount, categoryCount });
            var yContent = result.SpanContent;
            for (int elementId = 0; elementId < elementCount; ++elementId)
            {
                var categoryIndex = elementIdToCategoryIndex(elementId);
                if (categoryIndex >= 0)
                {
                    yContent[elementId * categoryCount + categoryIndex] = 1f;
                }
            }
            return result;
        }


        // compute:     this = alpha * this
        public override void Update_Multiplying_By_Alpha(float alpha)
        {
            MKL_BLAS.cblas_sscal(Count, alpha, AsFloatPointer, 1);
        }
        #region pooling layers


        public override void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride)
        {
            var x = this;
#if DEBUG
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            Debug.Assert(x.Shape[0] == y.Shape[0]); //same batch size
            Debug.Assert(x.Shape[1] == y.Shape[1]); //same number of channels
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 3 || x.Dimension == 4);
            int hOutput = y.Shape[2];
            int hInput = x.Shape[2];
            int hExpected = (hInput - poolingHeight) / poolingStride + 1;
            Debug.Assert(hOutput == hExpected);
            if (x.Dimension == 4)
            {
                int wOutput = y.Shape[3];
                int wInput = x.Shape[3];
                int wExpected = (wInput - poolingWidth) / poolingStride + 1;
                Debug.Assert(wOutput == wExpected);
            }
#endif
            int batchSize = x.Shape[0];
            if (x.Dimension == 4)
            {
                if (PoolingLayer.IsMaxPooling(poolingMode))
                {
                    Parallel.For(0, batchSize, elementIndex => MaxPoolingForSingleElement4D(y, poolingHeight, poolingWidth, poolingStride, elementIndex ));
                }
                else
                {
                    Parallel.For(0, batchSize, elementIndex => AvgPoolingForSingleElement4D(y, poolingHeight, poolingWidth, poolingStride, elementIndex));
                }
            }
            else
            {
                Debug.Assert(x.Dimension == 3);
                Debug.Assert(poolingWidth == 1);
                if (PoolingLayer.IsMaxPooling(poolingMode))
                {
                    Parallel.For(0, batchSize, elementIndex => MaxPoolingForSingleElement3D(y, poolingHeight, poolingStride, elementIndex));
                }
                else
                {
                    Parallel.For(0, batchSize, elementIndex => AvgPoolingForSingleElement3D(y, poolingHeight, poolingStride, elementIndex));
                }
            }
        }
        private void AvgPoolingForSingleElement4D(Tensor y, int poolingHeight, int poolingWidth, int poolingStride, int elementIndex)
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
                        col_filter_start += poolingStride;
                    }
                    row_filter_start += poolingStride;
                }
            }
        }
        private void MaxPoolingForSingleElement4D(Tensor y, int poolingHeight, int poolingWidth, int poolingStride, int elementIndex)
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
                        col_filter_start += poolingStride;
                    }
                    row_filter_start += poolingStride;
                }
            }
        }
        private void AvgPoolingForSingleElement3D(Tensor y, int poolingHeight, int poolingStride, int elementIndex)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 3);
            int hOutput = y.Shape[2];
            //the first (top) point in 'y' is computed from a filter starting at (0)
            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    //we want to compute the point in y[n, channelId, row_output]
                    //it is computed by applying an avg filter located (for its top) in (row_filter_start) in the x 
                    float outputPointSum = 0f;
                    int count = 0;
                    for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingHeight); ++rowBeforePooling)
                    {
                            outputPointSum += x.AsFloatCpu.Get(elementIndex, c, rowBeforePooling);
                            ++count;
                    }
                    y.AsFloatCpu.Set(elementIndex, c, rowAfterPooling, outputPointSum / count);
                    row_filter_start += poolingStride;
                }
            }
        }
        private void MaxPoolingForSingleElement3D(Tensor y, int poolingHeight, int poolingStride, int elementIndex)
        {
            var x = this;
            Debug.Assert(AreCompatible(new List<Tensor> { x, y }));
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 3);
            int hOutput = y.Shape[2];
            //the first (top) point in 'y' is computed from a filter starting at (0)
            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    //we want to compute the point in y[n, channelId, row_output]
                    //it is computed by applying a max filter located (for its top) in (row_filter_start) in the x 
                    float outputPointResult = float.MinValue;
                    for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingHeight); ++rowBeforePooling)
                    {
                        outputPointResult = Math.Max(outputPointResult, x.AsFloatCpu.Get(elementIndex, c, rowBeforePooling));
                    }
                    y.AsFloatCpu.Set(elementIndex, c, rowAfterPooling, outputPointResult);
                    row_filter_start += poolingStride;
                }
            }
        }
        public override void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride)
        {
            int batchSize = x.Shape[0];
#if DEBUG
            var dy = this;
            Debug.Assert(AreCompatible(new List<Tensor> { dy, y, x, dx }));
            Debug.Assert(x.Dimension == y.Dimension);
            Debug.Assert(x.Dimension == 4 || x.Dimension == 3);
            Debug.Assert(x.Shape[0] == dy.Shape[0]); //same batchSize
            Debug.Assert(x.Shape[1] == dy.Shape[1]); //same number of channels
            Debug.Assert(dx.SameShape(x));
            int hOutput = dy.Shape[2];

            if (x.Dimension == 4)
            {
                int wOutput = dy.Shape[3];
                Debug.Assert(hOutput == ((x.Shape[2] - poolingHeight) / poolingStride + 1));
                Debug.Assert(wOutput == ((x.Shape[3] - poolingWidth) / poolingStride + 1));
            }
#endif
            dx.ZeroMemory();
            if( x.Dimension == 4)
            { 
                if (PoolingLayer.IsMaxPooling(poolingMode))
                {
                    Parallel.For(0, batchSize, elementIndex => MaxPoolingGradientForSingleElement4D(x, dx, poolingHeight, poolingWidth, poolingStride, elementIndex));
                }
                else
                {
                    Parallel.For(0, batchSize, elementIndex => AvgPoolingGradientForSingleElement4D(x, dx, poolingHeight, poolingWidth, poolingStride, elementIndex));
                }
            }
            else
            {
                Debug.Assert(x.Dimension == 3);
                Debug.Assert(poolingWidth == 1);
                if (PoolingLayer.IsMaxPooling(poolingMode))
                {
                    Parallel.For(0, batchSize, elementIndex => MaxPoolingGradientForSingleElement3D(x, dx, poolingHeight, poolingStride, elementIndex));
                }
                else
                {
                    Parallel.For(0, batchSize, elementIndex => AvgPoolingGradientForSingleElement3D(x, dx, poolingHeight, poolingStride, elementIndex));
                }
            }
        }
        private void AvgPoolingGradientForSingleElement4D(Tensor x, Tensor dx, int poolingHeight, int poolingWidth, int poolingStride, int elementIndex)
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
        private void MaxPoolingGradientForSingleElement4D(Tensor x, Tensor dx, int poolingHeight, int poolingWidth, int poolingStride, int elementIndex)
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
        private void AvgPoolingGradientForSingleElement3D(Tensor x, Tensor dx, int poolingHeight, int poolingStride, int elementIndex)
        {
            Debug.Assert(x.Dimension == 3);
            var dy = this;
            int hOutput = dy.Shape[2];
            double doubleMultiplier = 1.0 / (poolingHeight);
            float floatMultiplier = (float)doubleMultiplier;
            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingHeight); ++rowBeforePooling)
                    {
                        var pointGradient = dy.AsFloatCpu.Get(elementIndex, c, rowAfterPooling);
                        dx.AsFloatCpu.Set(elementIndex, c, rowBeforePooling, floatMultiplier * pointGradient);
                    }
                    row_filter_start += poolingStride;
                }
            }
        }
        //compute 'dx' from ('dy' and 'x')
        private void MaxPoolingGradientForSingleElement3D(Tensor x, Tensor dx, int poolingHeight, int poolingStride, int elementIndex)
        {
            Debug.Assert(x.Dimension == 3);
            var dy = this;
            int hOutput = dy.Shape[2];
            for (int c = 0; c < x.Shape[1]; ++c)
            {
                int row_filter_start = 0;
                for (int rowAfterPooling = 0; rowAfterPooling < hOutput; ++rowAfterPooling)
                {
                    //we retrieve the coordinate of the max value in 'x'
                    double outputPointResult = double.MinValue;
                    int maxRowBeforePooling = 0;
                    for (int rowBeforePooling = row_filter_start; rowBeforePooling < (row_filter_start + poolingHeight); ++rowBeforePooling)
                    {
                        var currentPointValue = x.AsFloatCpu.Get(elementIndex, c, rowBeforePooling);
                        if (currentPointValue > outputPointResult)
                        {
                            outputPointResult = currentPointValue;
                            maxRowBeforePooling = rowBeforePooling;
                        }
                    }
                    var pointGradient = dy.AsFloatCpu.Get(elementIndex, c, rowAfterPooling);
                    dx.AsFloatCpu.Set(elementIndex, c, maxRowBeforePooling, pointGradient);
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
        /// <param name="dW"></param>
        /// <param name="adam_vW">biased first moment estimate</param>
        /// <param name="adam_sW">biased second raw moment estimate</param>
        /// <param name="timestep"></param>
        public override void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon, Tensor dW, Tensor adam_vW, Tensor adam_sW, int timestep)
        {
            var beta1_power = Math.Pow(beta1, timestep);
            var beta2_power = Math.Pow(beta2, timestep);

            var W = this;
            //Update biased first moment estimate
            adam_vW.AsFloatCpu.Update(dW, (adam_vw, dw) => (float) (beta1 * adam_vw + (1 - beta1) * dw));
            //Update biased second raw moment estimate
            adam_sW.AsFloatCpu.Update(dW, (adam_sw, dw) => (float) (beta2 * adam_sw + (1 - beta2) * dw * dw));
            var multiplicative_factor = learningRate * (Math.Sqrt(1.0 - beta2_power) / (1.0 - beta1_power));
            //Update parameters
            W.AsFloatCpu.Update(adam_vW, adam_sW, (w, adam_vw, adam_sw) => (float) (w - multiplicative_factor * (adam_vw / (Math.Sqrt(adam_sw) + epsilon))));
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
                    cost = (-1.0 / (batchSize * categoryCount)) * yPredicted.AsFloatCpu.Merge(yExpected.AsFloatCpu, (prediction, expected) => (float)(expected * Math.Log(prediction) + (1 - expected) * Math.Log(1 - prediction))).NaNSum();
                    break;
                case NetworkConfig.LossFunctionEnum.CategoricalCrossentropy:
                    cost = (-1.0 / (batchSize)) * yPredicted.AsFloatCpu.Merge(yExpected.AsFloatCpu, (prediction, expected) => (float)(expected * Math.Log(prediction))).NaNSum();
                    break;
                case NetworkConfig.LossFunctionEnum.CategoricalCrossentropyWithHierarchy:
                    Parallel.For(0, batchSize, m => { buffer.AsFloatCpuSpan[m] = ComputeLossCategoricalCrossentropyWithHierarchy(yExpected.RowSlice(m, 1).AsReadonlyFloatCpuContent, yPredicted.RowSlice(m, 1).AsReadonlyFloatCpuContent); });
                    cost = buffer.AsReadonlyFloatCpuContent.Average();
                    break;
                case NetworkConfig.LossFunctionEnum.Huber:
                    const double huberDelta = 1.0;
                    cost = (1.0 / (batchSize)) * yPredicted.AsFloatCpu.Merge(yExpected.AsFloatCpu, (prediction, expected) => (float) ( (Math.Abs(expected-prediction)<= huberDelta) ?(0.5*Math.Pow(expected - prediction,2)):(huberDelta* Math.Abs(expected - prediction)-0.5*huberDelta*huberDelta)  )).NaNSum();
                    break;
                case NetworkConfig.LossFunctionEnum.Mse:
                    cost = (1.0 / (batchSize*yPredicted.MultDim0)) * yPredicted.AsFloatCpu.Merge(yExpected.AsFloatCpu, (prediction, expected) => (float)(Math.Pow(expected - prediction, 2))).NaNSum();
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

        private static float ComputeLossCategoricalCrossentropyWithHierarchy(ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted)
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
                        Debug.Assert(Math.Abs(expectedValue-1.0)<1e-6);
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

        public override void RandomMatrixNormalDistribution(Random rand, double mean, double stdDev)
        {
            Utils.RandomizeNormalDistribution(AsFloatCpuSpan, rand, mean, stdDev);
        }
        public override void RandomizeUniformDistribution(Random rand, double minValue, double maxValue)
        {
            Utils.RandomizeUniformDistribution(AsFloatCpuSpan, rand, minValue, maxValue);
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

        public override Tensor Clone()
        {
            var cloned = new CpuTensor<T>(Shape);
            CopyTo(cloned);
            return cloned;
        }

        //this method is only called for display / logging testing
        //this = yExpectedOneHot
        [SuppressMessage("ReSharper", "PossibleNullReferenceException")]
        public override double ComputeAccuracy(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer)
        {
            var yExpected = this;
            Debug.Assert(AreCompatible(new List<Tensor> { yExpected, yPredicted }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(!yExpected.UseGPU);
            Debug.Assert(buffer != null);
            Debug.Assert(buffer.Shape.Length == 1);
            Debug.Assert(buffer.Shape[0] == yPredicted.Shape[0]);
            int batchSize = yExpected.Shape[0];

            var yExpectedOneHotCpu = yExpected.AsFloatCpu;
            var yPredictedCpu = yPredicted.AsFloatCpu;

            var bufferPointer = (float*)buffer.Pointer;
            if (lossFunction == NetworkConfig.LossFunctionEnum.CategoricalCrossentropyWithHierarchy)
            {
                var expected = (float*)yExpected.Pointer;
                var predicted = (float*)yPredicted.Pointer;
                int nbCols = yExpected.Shape[1];
                Parallel.For(0, batchSize, i =>
                {
                    int nexIndexToCheck = 0;
                    bufferPointer[i] = IsAccuratePredictionForCategoricalCrossentropyWithHierarchy(expected + i * nbCols, predicted + i * nbCols, nbCols, & nexIndexToCheck, int.MaxValue, new List<int>()) ? 1f : 0f;
                });
            }
            else
            {
                Parallel.For(0, batchSize, m => bufferPointer[m] = ComputeSingleAccuracy(yExpectedOneHotCpu, yPredictedCpu, m, out _));
            }
            return buffer.AsReadonlyFloatCpuContent.Average();
        }

        public override double ComputeMae(Tensor yPredicted, Tensor buffer)
        {
            return ComputeMetric(yPredicted, buffer, (a, b) => Math.Abs(a - b));
        }

        public override double ComputeMse(Tensor yPredicted, Tensor buffer)
        {
            return ComputeMetric(yPredicted, buffer, (a, b) => (a - b)*(a-b));
        }
        private float ComputeMetric(Tensor yPredicted, Tensor buffer, Func<float , float, float> computeScalarMetric)
        {
            var yExpected = this;
            Debug.Assert(AreCompatible(new List<Tensor> { yExpected, yPredicted }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            Debug.Assert(!yExpected.UseGPU);
            Debug.Assert(buffer != null);
            Debug.Assert(buffer.Shape.Length == 1);
            int batchSize = yExpected.Shape[0];
            Debug.Assert(buffer.Shape[0] == batchSize);

            var bufferCpu = buffer.AsFloatCpu;

            void ComputeLine(int batchId)
            {
                float batchMetric = 0;
                var yExpectedCpu = yExpected.AsReadonlyFloatCpuContent;
                var yPredictedCpu = yPredicted.AsReadonlyFloatCpuContent;
                for (int index = batchId * yExpected.MultDim0; index < (batchId + 1) * yExpected.MultDim0; ++index)
                {
                    batchMetric += computeScalarMetric(yExpectedCpu[index], yPredictedCpu[index]);
                }
                bufferCpu[batchId] = batchMetric / yExpected.MultDim0;
            }
            Parallel.For(0, batchSize, ComputeLine);
            return buffer.ContentAsFloatArray().Average();
        }

        private static float ComputeSingleAccuracy(CpuTensor<float> yExpectedOneHot, CpuTensor<float> yPredicted, int m, out int maxIndexPredicted)
        {
            Debug.Assert(yExpectedOneHot.SameShape(yPredicted));
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

        public override void CategoricalCrossentropyWithHierarchyGradient(Tensor yExpected, Tensor yPredicted)
        {
            var loss = this;
            Debug.Assert(loss.SameShape(yExpected));
            Debug.Assert(loss.SameShape(yPredicted));
            Debug.Assert(loss.Dimension == 2);
            loss.ZeroMemory();
            Parallel.For(0, loss.Shape[0], m =>{ CategoricalCrossentropyWithHierarchyGradient(loss.RowSlice(m, 1).AsFloatCpuSpan, yExpected.RowSlice(m, 1).AsReadonlyFloatCpuContent, yPredicted.RowSlice(m, 1).AsReadonlyFloatCpuContent);});
        }

        private static void CategoricalCrossentropyWithHierarchyGradient(Span<float> loss, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted)
        {
            Debug.Assert(loss.Length == expected.Length);
            Debug.Assert(loss.Length == predicted.Length);
            for(int i = 0;i<loss.Length;++i)
            {
                var expectedValue = expected[i];
                if (Math.Abs(expectedValue) < 9.5f)
                {
                    //expectedValue contains a proba between 0 and 1
                    Debug.Assert(expectedValue>=0);
                    Debug.Assert(expectedValue<=1.0);
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
                        i += count-1; //-1 because the for(;;) loop will also increment 'i'
                    }
                }
            }
        }

        public override void HuberGradient(Tensor yExpected, Tensor yPredicted, float huberDelta)
        {
            var loss = this;
            Debug.Assert(loss.SameShape(yExpected));
            Debug.Assert(loss.SameShape(yPredicted));
            Parallel.For(0, loss.Shape[0], m => { HuberGradient(loss.RowSlice(m, 1).AsFloatCpuSpan, yExpected.RowSlice(m, 1).AsReadonlyFloatCpuContent, yPredicted.RowSlice(m, 1).AsReadonlyFloatCpuContent, huberDelta); });
        }

        private static void HuberGradient(Span<float> gradient, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted, float huberDelta)
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

        public override void HuberLoss(Tensor yExpected, Tensor yPredicted, float huberDelta)
        {
            var huberLoss = this;
            int batchSize = yExpected.Shape[0];
            Debug.Assert(huberLoss.SameShape(new[] { batchSize }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            Parallel.For(0, batchSize, batchId => { HuberLoss(batchId, huberLoss.AsFloatCpuSpan, yExpected.RowSlice(batchId, 1).AsReadonlyFloatCpuContent, yPredicted.RowSlice(batchId, 1).AsReadonlyFloatCpuContent, huberDelta); });
        }

        private static void HuberLoss(int batchId, Span<float> huberLoss, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted, float huberDelta)
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


        public override void MseGradient(Tensor yExpected, Tensor yPredicted)
        {
            var loss = this;
            Debug.Assert(loss.SameShape(yExpected));
            Debug.Assert(loss.SameShape(yPredicted));
            Parallel.For(0, loss.Shape[0], m => { MseGradient(loss.RowSlice(m, 1).AsFloatCpuSpan, yExpected.RowSlice(m, 1).AsReadonlyFloatCpuContent, yPredicted.RowSlice(m, 1).AsReadonlyFloatCpuContent); });
        }

        private static void MseGradient(Span<float> gradient, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted)
        {
            Debug.Assert(gradient.Length == expected.Length);
            Debug.Assert(gradient.Length == predicted.Length);
            for (int i = 0; i < gradient.Length; ++i)
            {
                var error = predicted[i] - expected[i];
                gradient[i] = 2*error/ gradient.Length;
            }
        }

        public override void MseLoss(Tensor yExpected, Tensor yPredicted)
        {
            var mseLoss = this;
            int batchSize = yExpected.Shape[0];
            Debug.Assert(mseLoss.SameShape(new[] { batchSize }));
            Debug.Assert(yExpected.SameShape(yPredicted));
            Parallel.For(0, batchSize, batchId => { MseLoss(batchId, mseLoss.AsFloatCpuSpan, yExpected.RowSlice(batchId, 1).AsReadonlyFloatCpuContent, yPredicted.RowSlice(batchId, 1).AsReadonlyFloatCpuContent); });
        }

        private static void MseLoss(int batchId, Span<float> mseLoss, ReadOnlySpan<float> expected, ReadOnlySpan<float> predicted)
        {
            Debug.Assert(expected.Length == predicted.Length);
            var loss = 0.0f;
            for (int i = 0; i < expected.Length; ++i)
            {
                var error = predicted[i] - expected[i];
                loss += error * error;
            }
            mseLoss[batchId] = loss / expected.Length;
        }


        /// <summary>
        /// compute the prediction embedded in the tensor (in each line the index with max value)
        /// </summary>
        /// <returns>array with prediction (=category) of each element</returns>
        //public int[] ComputePrediction()
        //{
        //    int batchSize = Shape[0];
        //    int[] categoryCount = new int[batchSize];
        //    var yPredictedCpu = AsFloatCpu;
        //    for (int m = 0; m < batchSize; ++m)
        //    {
        //        ComputeSingleAccuracy(yPredictedCpu, yPredictedCpu, m, out categoryCount[m]);
        //    }
        //    return categoryCount;
        //}

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
                MKL_BLAS.cblas_scopy(Count, AsFloatPointer, 1, b.AsFloatPointer, 1);
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
        private CpuTensor<T> Merge(CpuTensor<T> b, Func<T, T, T> func)
        {
            Debug.Assert(Dimension == b.Dimension);
            Debug.Assert(Count == b.Count);
            var content = new T[Count];
            var bSpan = b.ReadonlyContent;
            var thisSpan = ReadonlyContent;
            for (int i = 0; i < Count; ++i)
            {
                content[i] = func(thisSpan[i], bSpan[i]);
            }
            return new CpuTensor<T>(Shape, content);
        }
        private double NaNSum()
        {
            return AsReadonlyFloatCpuContent.Select(x => float.IsNaN(x) ? 0 : x).Sum();
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
    }
}
