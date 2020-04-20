using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
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
        private readonly IntPtr _ptrToOwnerMemory;
        /// <summary>
        /// used only if the tensor is the owner of the memory
        /// </summary>
        private HostPinnedMemory<T> _hostPinnedMemory;
        #endregion

        public CpuTensor(int[] shape, T[] data, int typeSize, string description) : base(shape, typeSize, false, description)
        {
            Content = data ?? new T[Count];
            CapacityInBytes = (ulong)(Content.Length * TypeSize);
            _ptrToOwnerMemory = IntPtr.Zero;
        }
        public CpuTensor(int[] shape, T[] data, string description) : this(shape, data, Marshal.SizeOf(typeof(T)), description)
        {
        }
        public CpuTensor(int[] shape, string description) : this(shape, null, description)
        {
        }
        private CpuTensor(int[] shape, CpuTensor<T> memoryOwner, int startIndex) : base(shape, memoryOwner.TypeSize, false, memoryOwner.Description)
        {
            Content = memoryOwner.Content.Slice(startIndex, Utils.Product(shape));
            CapacityInBytes = (ulong)(Content.Length * TypeSize);
            _ptrToOwnerMemory = memoryOwner.HostPointer + TypeSize * startIndex;
        }


        /// <summary>
        /// pointer to (pinned) host memory (in CPU)
        /// </summary>
        public IntPtr HostPointer
        {
            get
            {
                if (!IsOwnerOfMemory)
                {
                    Debug.Assert(_ptrToOwnerMemory != IntPtr.Zero);
                    return _ptrToOwnerMemory;
                }
                Debug.Assert(_ptrToOwnerMemory == IntPtr.Zero);
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

        public T this[int i]
        {
            get => ReadonlyContent[i];
            set => SpanContent[i] = value;
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

        public override Tensor ChangeAxis(int[] newToOldAxis)
        {
            Debug.Assert(newToOldAxis.Length == Dimension);
            Debug.Assert(newToOldAxis.Min() == 0);
            Debug.Assert(newToOldAxis.Max() == Dimension-1);

            int[] oldToNewAxis = new int[newToOldAxis.Length];
            for (int newAxis = 0; newAxis < oldToNewAxis.Length; ++newAxis)
            {
                oldToNewAxis[newToOldAxis[newAxis]] = newAxis;
            }

            var transformedShape = new int[Dimension];
            for (int newAxis = 0; newAxis < Dimension; ++newAxis)
            {
                transformedShape[newAxis] = Shape[newToOldAxis[newAxis]];
            }

            var result = new CpuTensor<T>(transformedShape, Description);

            var indexesInNewAxis =  new int[Dimension];
            for (int n = 0; n < Shape[0]; ++n)
            {
                indexesInNewAxis[oldToNewAxis[0]] = n;
                for (int c = 0; c < Shape[1]; ++c)
                {
                    indexesInNewAxis[oldToNewAxis[1]] = c;
                    for (int h = 0; h < Shape[2]; ++h)
                    {
                        indexesInNewAxis[oldToNewAxis[2]] = h;
                        for (int w = 0; w < Shape[3]; ++w)
                        {
                            indexesInNewAxis[oldToNewAxis[3]] = w;
                            result.Set(indexesInNewAxis[0], indexesInNewAxis[1], indexesInNewAxis[2], indexesInNewAxis[3], Get(n, c, h, w));
                        }
                    }
                }
            }

            return result;
        }
        public override bool IsOwnerOfMemory => _ptrToOwnerMemory == IntPtr.Zero;
        public ReadOnlySpan<T> ReadonlyContent => Content.Span;
        public Span<T> SpanContent => Content.Span;

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
        public override Tensor Clone(GPUWrapper notUsed)
        {
            return new CpuTensor<T>((int[])Shape.Clone(), Content.ToArray(), Description);
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
        public override void DropoutForward(Tensor y, double dropProbability, bool isTraining, Random dropoutRandom,
            Tensor dropoutMaskBufferForCpu, ref Tensor randomNumberGeneratorStatesBufferForGPU,
            ref Tensor dropoutReserveSpaceForGPU, ref IntPtr dropoutDescriptorForGPU, TensorMemoryPool memoryPool)
        {
            var x = this;
            Debug.Assert(dropoutMaskBufferForCpu != null);
            Debug.Assert(randomNumberGeneratorStatesBufferForGPU == null);
            Debug.Assert(dropoutReserveSpaceForGPU == null);
            Debug.Assert(dropoutDescriptorForGPU == IntPtr.Zero);
            if (!isTraining)
            {
                x.CopyTo(y);
                return;
            }
            var dropProbabilityFloat = (float)dropProbability;
            Utils.Randomize(dropoutMaskBufferForCpu.AsFloatCpuSpan, dropoutRandom, 0.0, 1.0);
            y.AsFloatCpu.BuildEntirelyFromInput(x, dropoutMaskBufferForCpu, (prevLayer, prob) => prob < dropProbability ? 0f : prevLayer / (1 - dropProbabilityFloat));
        }
        public override void DropoutBackward(Tensor dy, Tensor dx, double dropProbability,
            Tensor dropoutMaskBufferForCpu, Tensor randomNumberGeneratorStatesBufferForGPU,
            Tensor dropoutReserveSpaceForGPU, IntPtr dropoutDescriptorForGPU)
        {
            Debug.Assert(dropoutMaskBufferForCpu != null);
            Debug.Assert(randomNumberGeneratorStatesBufferForGPU == null);
            Debug.Assert(dropoutReserveSpaceForGPU == null);
            Debug.Assert(dropoutDescriptorForGPU == IntPtr.Zero);
            var _dropProbabilityFloat = (float)dropProbability;
            dx.AsFloatCpu.BuildEntirelyFromInput(dy, dropoutMaskBufferForCpu, (dOutput, prob) => prob < _dropProbabilityFloat ? 0f : dOutput / (1 - _dropProbabilityFloat));
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

        public override void MultiplyTensor(Tensor a, Tensor x)
        {
            Debug.Assert(this.SameShape(a));
            Debug.Assert(a.Count >= x.Count);
            Debug.Assert(Count % x.Count == 0);

            var aFloat = a.AsFloatCpuSpan;
            var xFloat = x.AsFloatCpuSpan;
            var thisFloat = AsFloatCpuSpan;
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
            ZeroPadding_and_Unpadding(unpaddedTensor, paddingTop, paddingBottom, paddingLeft, paddingRight, false);
        }
        public override void ZeroUnpadding(Tensor paddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight)
        {
            ((CpuTensor<T>)paddedTensor).ZeroPadding_and_Unpadding(this, paddingTop, paddingBottom, paddingLeft, paddingRight, true);
        }

        private void ZeroPadding_and_Unpadding(Tensor unpaddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight, bool isUnpadding)
        {
            var paddedTensor = this;
            Debug.Assert(AreCompatible(new List<Tensor> { paddedTensor, unpaddedTensor }));
            Debug.Assert(paddedTensor.Dimension == 4);
            Debug.Assert(paddedTensor.Dimension == unpaddedTensor.Dimension);
            Debug.Assert(paddedTensor.Shape[0] == unpaddedTensor.Shape[0]); //same batch size
            Debug.Assert(paddedTensor.Shape[1] == unpaddedTensor.Shape[1]); //same number of channels
            Debug.Assert(paddedTensor.Shape[2] == (paddingTop + unpaddedTensor.Shape[2] + paddingBottom)); //valid height for destination
            Debug.Assert(paddedTensor.Shape[3] == (paddingLeft + unpaddedTensor.Shape[3] + paddingRight)); //valid width destination
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
            System.Threading.Tasks.Parallel.For(0, unpaddedTensor.Shape[0] * unpaddedTensor.Shape[1] * unpaddedTensor.Shape[2], ApplyZeroPaddingForRowId);
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
        public static CpuTensor<float> CreateOneHotTensor(Func<int,int> elementIdToCategoryIndex, int elementCount, int categoryCount)
        {
            var yShape = new[] { elementCount, categoryCount };
            var yContent = new float[Utils.Product(yShape)];
            for (int elementId = 0; elementId < elementCount; ++elementId)
            {
                var categoryIndex = elementIdToCategoryIndex(elementId);
                if (categoryIndex >= 0)
                {
                    yContent[elementId * categoryCount + categoryIndex] = 1f;
                }
            }
            return new CpuTensor<float>(yShape, "YOneHot");
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
            int F = convolution.Shape[2];
            Debug.Assert(F == convolution.Shape[3]);
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
        //this = Weights or Bias
        public override void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon, Tensor dW, Tensor adam_vW, Tensor adam_sW, int timestep)
        {
            var beta1_power = Math.Pow(beta1, timestep);
            var beta2_power = Math.Pow(beta2, timestep);
            var multiplicative_factor = learningRate * (Math.Sqrt(1.0 - beta2_power) / (1.0 - beta1_power));

            var W = this;
            adam_vW.AsFloatCpu.Update(dW, (adam_vw, dw) => (float) (beta1 * adam_vw + (1 - beta1) * dw));
            adam_sW.AsFloatCpu.Update(dW, (adam_sw, dw) => (float) (beta2 * adam_sw + (1 - beta2) * dw * dw));
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
                    cost = (-1.0 / (batchSize * categoryCount)) * yPredicted.AsFloatCpu.Merge(yExpected.AsFloatCpu, (prediction, expected) => (float)(expected * Math.Log(prediction) + (1 - expected) * Math.Log(1 - prediction)), "BinaryCrossentropy").NaNSum();
                    break;
                case NetworkConfig.LossFunctionEnum.CategoricalCrossentropy:
                    cost = (-1.0 / (batchSize)) * yPredicted.AsFloatCpu.Merge(yExpected.AsFloatCpu, (prediction, expected) => (float)(expected * Math.Log(prediction)), "CategoricalCrossentropy").NaNSum();
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
            var categoryIndexes = AsCpu<int>().ReadonlyContent;
            Debug.Assert(yPredictedTensor != null);
            Debug.Assert(!yPredictedTensor.UseGPU);
            var batchSize = yPredictedTensor.Shape[0];
            Debug.Assert(categoryIndexes.Length == batchSize);
            var categoryCount = yPredictedTensor.Shape[1];
            var yPredicted = yPredictedTensor.AsFloatCpuSpan;

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
                    //cost = (-1.0 / (batchSize)) * yPredicted.AsFloatCpu.Merge(categoryIndexes.AsCpu<int>(), (prediction, expected) => (float)(expected * Math.Log(prediction)), "CategoricalCrossentropy").NaNSum();
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
            Utils.RandomizeNormalDistribution(AsFloatCpuSpan, rand, mean, stdDev);
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

            var yExpectedOneHotCpu = yExpectedOneHot.AsFloatCpu;
            var yPredictedCpu = yPredicted.AsFloatCpu;
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
            var categoryIndexes = AsCpu<int>().ReadonlyContent;
            int batchSize = yPredicted.Shape[0];
            Debug.Assert(batchSize == categoryIndexes.Length);
            Debug.Assert(!yPredicted.UseGPU);
            int result = 0;

            var yPredictedCpu = yPredicted.AsFloatCpu;
            for (int m = 0; m < batchSize; ++m)
            {
                result += ComputeSingleAccuracyCountFromCategoryIndexes(categoryIndexes, yPredictedCpu, m, out _);
            }
            return ((double)result) / Shape[0];
        }

        /// <summary>
        /// compute the prediction embedded in the tensor (in each line the index with max value)
        /// </summary>
        /// <returns>array with prediction (=category) of each element</returns>
        public int[] ComputePrediction()
        {
            int batchSize = Shape[0];
            int[] categoryCount = new int[batchSize];
            var yPredictedCpu = AsFloatCpu;
            for (int m = 0; m < batchSize; ++m)
            {
                ComputeSingleAccuracyCount(yPredictedCpu, yPredictedCpu, m, out categoryCount[m]);
            }
            return categoryCount;
        }

        public override void CopyTo(Tensor b)
        {
            Debug.Assert(Count == b.Count);
            if (b.UseGPU)
            {
                //copy from CPU ('this' tensor) to GPU ('b' tensor)
                ((GPUTensor<T>)b).CopyToDevice(Content);
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
        public override Tensor ExtractSubTensor(int startRowIndex, int nbRows)
        {
            Debug.Assert(Shape.Length >= 2);
            Debug.Assert(startRowIndex >= 0);
            Debug.Assert(startRowIndex < Shape[0]);
            Debug.Assert(startRowIndex + nbRows - 1 < Shape[0]);
            var extractedShape = (int[])Shape.Clone();
            extractedShape[0] = nbRows; //news number of rows
            return new CpuTensor<T>(extractedShape, this, Idx(startRowIndex));
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
            return AsFloatCpu.ReadonlyContent.Select(x => float.IsNaN(x) ? 0 : x).Sum();
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
            var aCpu = a.AsCpu<T>().ReadonlyContent;
            var bCpu = b.AsCpu<T>().ReadonlyContent;
            for (int i = 0; i < a.Count; ++i)
            {
                this[i] = funcInput(aCpu[i], bCpu[i]);
            }
        }
        public void BuildEntirelyFromInput(Tensor a, Tensor b, Tensor c, Func<T, T, T, T> funcInput)
        {
            Debug.Assert(AreCompatible(new List<Tensor> { this, a, b, c }));
            Debug.Assert(SameShape(a, b));
            var aCpu = a.AsCpu<T>().ReadonlyContent;
            var bCpu = b.AsCpu<T>().ReadonlyContent;
            var cCpu = c.AsCpu<T>().ReadonlyContent;
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
        private static int ComputeSingleAccuracyCountFromCategoryIndexes(ReadOnlySpan<int> categoryIndexes, CpuTensor<float> yPredicted, int m, out int maxIndexPredicted)
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
