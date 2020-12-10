using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;

namespace SharpNet.Data
{
    [DebuggerDisplay("{ToString(true)}")]
    public abstract unsafe class Tensor : IDisposable
    {
        #region fields
        public int[] Shape { get; protected set; }
        public int MultDim0 { get; private set; }
        public int MultDim1 { get; private set; }
        private int _multDim2;
        public bool UseGPU { get; }
        public int TypeSize { get; }
        #endregion

        #region constructors
        protected Tensor(int[] shape, int typeSize, bool useGpu)
        {
            Debug.Assert(shape.Length >= 1);
            Debug.Assert(shape.Length <= 4);
            Debug.Assert(shape.Min() >= 1);
            Shape = shape;
            UseGPU = useGpu;
            TypeSize = typeSize;
            RecomputeMultDim();
        }
        #endregion


        public bool SameShape(params Tensor[] b) { return b.Where(x=>x!=null).All(SameShape); }
        public bool SameShape(Tensor b) {return SameShape(b.Shape);}
        public bool SameShapeExceptFirstDimension(Tensor b) { return SameShapeExceptFirstDimension(b.Shape); }
        public override string ToString()
        {
            return ToString(false);
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Idx(int n) { return MultDim0 * n; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Idx(int n, int c)
        {
            return MultDim0 * n + MultDim1 * c;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Idx(int n, int c, int h, int w) { return MultDim0 * n + MultDim1 * c + _multDim2 * h + w; }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Idx(int n, int c, int h) { return MultDim0 * n + MultDim1 * c + h; }

        // this = a*b
        public void Dot(Tensor a, Tensor b) { Dot(a, false, b, false, 1, 0); }

        /// <summary>
        /// this (=y) shape is (batchSize, embeddingDim, maxWordCountBySentence)
        /// </summary>
        /// <param name="x">tensor of shape (batchSize, maxWordCountBySentence)
        /// row is the sentence Id
        /// col is the word position in this sentence
        /// x[row, col] contains the wordIndex (in the word embedding tensor) of the word at position 'row' in sentence 'col'
        /// each wordIndex is in [1, VocabularySize-1]
        /// </param>
        /// <param name="wordEmbedding">tensor of shape (vocabularySize, associateWordEmbedding)
        /// the element at row 'wordIndex' is the embedding associate with the word at index 'wordIndex'
        /// vocabularySize = 1+number of distinct words in the embedding
        /// </param>
        public abstract void WordEmbeddingForwardPropagation(Tensor x, Tensor wordEmbedding);

        /// <summary>
        /// Initialize the 'this' tensor with the gradient of the word embedding weights (= dW)
        /// 'this' shape is (VocabularySize, EmbeddingDim) (same as word embeddings)
        /// </summary>
        /// <param name="x">tensor of shape (batchSize, maxWordCountBySentence)
        /// row is the sentence Id
        /// col is the word position in this sentence
        /// x[row, col] contains the wordIndex (in the word embedding tensor) of the word at position 'row' in sentence 'col'
        /// each wordIndex is in [1, VocabularySize-1]
        /// </param>
        /// <param name="dy">tensor of shape (batchSize, EmbeddingDim, maxWordCountBySentence)
        /// </param>
        public abstract void WordEmbeddingBackwardPropagation(Tensor x, Tensor dy);

        public int Count => Shape[0] * MultDim0;
        public int Dimension => Shape.Length;
        protected ulong ReallyNeededMemoryInBytesForShape(int[] shape) { return (ulong)Utils.Product(shape) * (ulong)TypeSize; }
        public CpuTensor<float> AsFloatCpu => AsCpu<float>();
        public Span<float> AsFloatCpuSpan => AsFloatCpu.SpanContent;
        public float*  AsFloatPointer => (float*)AsFloatCpu.Pointer;

        public ReadOnlySpan<float> AsReadonlyFloatCpuContent => AsCpu<float>().ReadonlyContent;
        public string ContentStats()
        {
            int naNCount = 0;
            int infinityCount = 0;
            int count = 0;
            double sum = 0;
            double sumSquare = 0;
            double minValue = double.MaxValue;
            double maxValue = double.MinValue;
            foreach (var d in ContentAsFloatArray())
            {
                if (float.IsNaN(d))
                {
                    ++naNCount;
                    continue;
                }
                if (float.IsInfinity(d))
                {
                    ++infinityCount;
                    continue;
                }
                minValue = Math.Min(minValue, d);
                maxValue = Math.Max(maxValue, d);
                sum += d;
                sumSquare += d * d;
                ++count;
            }
            string result = "";
            const int decimalsForRounding = 6;
            if (count != 0)
            {
                var mean = (sum / count);
                var variance = (sumSquare / count) - mean * mean;
                if (Math.Abs(variance) < 1e-6)
                {
                    variance = 0;
                }
                if (Math.Abs(maxValue - minValue) < 1e-6)
                {
                    result = "Const: " + Math.Round(minValue, decimalsForRounding);
                }
                else
                {
                    result = "Min: " + Math.Round(minValue, decimalsForRounding) + "; Max: " + Math.Round(maxValue, decimalsForRounding) + "; Avg: " + Math.Round(mean, decimalsForRounding) + "; Vol: " + Math.Round(Math.Sqrt(variance), decimalsForRounding);
                }
                result += "; Count: " + Count;
            }
            if ((naNCount != 0) || (infinityCount != 0))
            {
                result += " (";
                if (naNCount != 0)
                {
                    result += naNCount + " NaN";
                }
                if (infinityCount != 0)
                {
                    result += " " + infinityCount + " infinite";
                }
                result += ")";
            }
            return result;
        }
        public static implicit operator IntPtr(Tensor t)
        {
            return (t==null)?IntPtr.Zero:t.Pointer;
        }
        public CpuTensor<T> AsCpu<T>()
        {
            if (this is CpuTensor<T> result)
            {
                return result;
            }
            throw new Exception("fail to convert " + this + " this to a CpuTensor<" + typeof(T)+">");
        }

        protected virtual int DeviceId => -1;

        public GPUTensor<T> ToGPU<T>(GPUWrapper gpuWrapper)
        {
            return UseGPU ? AsGPU<T>() : new GPUTensor<T>(Shape, AsCpu<T>().Content, gpuWrapper);
        }
        public CpuTensor<float> ToCpuFloat()
        {
            if (this is CpuTensor<float>)
            {
                return (CpuTensor<float>) this;
            }
            return new CpuTensor<float>(Shape, ContentAsFloatArray());
        }
        public abstract void Reshape(int[] newShape);

        /// <summary>
        /// return a reference of the this Tensor changing only its shape
        /// </summary>
        /// <param name="newShape"></param>
        /// <returns></returns>
        public abstract Tensor WithNewShape(int[] newShape);

        public static string ShapeToString(int[] shape)
        {
            return "(" + string.Join(", ", shape) + ")";
        }
        public GPUTensor<T> AsGPU<T>()
        {
            if (this is GPUTensor<T> result)
            {
                return result;
            }
            throw new Exception("fail to convert " + this + " this to a GPUTensor<" + typeof(T) + ">");
        }

        /// <summary>
        /// reshape the current tensor to 'newShape' in a thread safe way
        /// </summary>
        /// <param name="newShape">the target shape</param>
        public void Reshape_ThreadSafe(int[] newShape)
        {
            if (SameShape(newShape))
            {
                return;
            }
            lock (this)
            {
                if (!SameShape(newShape))
                {
                    Reshape(newShape);
                }
            }
        }




        /// <summary>
        /// Ensure that all tensors are stored in the same device (Cpu or GPU)
        /// </summary>
        /// <returns>true if all tensors are stored in the same device
        /// false if some tensors are stored and Cpu and other on GPU</returns>
        public static bool AreCompatible(List<Tensor> a)
        {
            a.RemoveAll(x => x == null);
            for (int i = 1; i < a.Count; ++i)
            {

                if (!a[0].IsCompatible(a[i]))
                {
                    return false;
                }
            }
            return true;
        }
        public ulong CapacityInBytes { get; protected set; }

        public bool HasEnoughCapacityForTensor(int[] tensorShape)
        {
            return ReallyNeededMemoryInBytesForShape(tensorShape) <= CapacityInBytes;
        }

        public abstract void ZeroMemory();
        /// <summary>
        /// this = alpha a*b + beta*this 
        /// </summary>
        /// <param name="a"></param>
        /// <param name="transposeA"></param>
        /// <param name="b"></param>
        /// <param name="transposeB"></param>
        /// <param name="alpha"></param>
        /// <param name="beta"></param>
        public abstract void Dot(Tensor a, bool transposeA, Tensor b, bool transposeB, float alpha, float beta);

        /// <summary>
        /// Compute the element wise multiplication:
        ///     [out] this = a (element_wise_multiplication) Diag(diagonalMatrix)
        ///     where 'diagonalMatrix' is a vector containing a diagonal matrix
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="a">[in] a matrix</param>
        /// <param name="diagonalMatrix">[in] a vector containing a diagonal matrix
        /// (only the diagonal of the diagonal matrix is contained in vector 'diagonalMatrix'</param>
        public abstract void MultiplyTensor(Tensor a, Tensor diagonalMatrix);


        /// <summary>
        /// this = y [out] output tensor
        /// upSample the tensor 'tensorToUpSample' by multiplying the number of rows by 'rowMultiplier' and the number of columns by 'colMultiplier'
        /// stores the result in 'this' tensor
        /// </summary>
        /// <param name="tensorBeforeUpSampling">[in] the tensor to up sample. must be of shape (n, c, h, w)</param>
        /// <param name="rowFactor">row multiplier</param>
        /// <param name="colFactor">col multiplier</param>
        /// <param name="interpolation">the type of interpolation (nearest or bilinear)</param>
        public abstract void UpSampling2D(Tensor tensorBeforeUpSampling, int rowFactor, int colFactor, UpSampling2DLayer.InterpolationEnum interpolation);

        /// <summary>
        /// this = x [out] input tensor
        /// down sample the tensor 'upSampledTensor' by dividing the number of rows by 'rowMultiplier' and the number of columns by 'colMultiplier'
        /// and summing each element
        /// stores the result in 'this' tensor
        /// </summary>
        /// <param name="tensorBeforeDownSampling">[in] the tensor to down sample. must be of shape (n, c, h, w)</param>
        /// <param name="rowFactor">row multiplier</param>
        /// <param name="colFactor">col multiplier</param>
        public abstract void DownSampling2D(Tensor tensorBeforeDownSampling, int rowFactor, int colFactor);



        /// <summary>
        /// this = [out] zero padded version of the 'unpaddedTensor' tensor received as input
        /// </summary>
        /// <param name="unpaddedTensor">[in] the tensor to which we want to add zero padding</param>
        /// <param name="paddingTop">padding to add to the top of 'src' tensor</param>
        /// <param name="paddingBottom">padding to add to the bottom of 'src' tensor</param>
        /// <param name="paddingLeft">padding to add to the left of 'src' tensor</param>
        /// <param name="paddingRight">padding to add to the right of 'src' tensor</param>
        public abstract void ZeroPadding(Tensor unpaddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight);

        public abstract void ZeroUnpadding(Tensor paddedTensor, int paddingTop, int paddingBottom, int paddingLeft, int paddingRight);

        /// <summary>
        /// this = this (element wise multiplication) x
        /// Update the value of the 'this' tensor by multiplying it by 'x'
        /// if 'this' and 'x' have the same size:
        ///     will perform an element wise multiplication of vector 'this' and vector 'x' (and store the result in 'this')
        /// else
        ///     will consider 'x' has a vector containing the diagonal of a diagonal matrix,
        ///     and will multiply 'this' with the associated diagonal matrix
        /// </summary>
        /// <param name="x"></param>
        public void Update_Multiply_By_x(Tensor x)
        {
            MultiplyTensor(this, x);
        }

        /// <summary>
        /// this = [out] a vector to store the result of the element wise product
        /// For each row of matrix a and b , compute the element wise product and this row, and store the result in this[row]
        /// this = [out] a vector of size (m)
        /// </summary>
        /// <param name="a">[in] matrix of size (m,n) </param>
        /// <param name="b">[in] matrix of size (m,n) </param>
        public abstract void MultiplyEachRowIntoSingleValue(Tensor a, Tensor b);

        public abstract void BroadcastAddVectorToOutput(Tensor y);


        /// <summary>
        /// transform the content of the 'this' tensor from shape (a,b,...) to shape (b,a,...)
        /// </summary>
        /// <param name="target"></param>
        public abstract void Switch_First_2_axis(Tensor target);


        /// <summary>
        /// create a copy of the 'this' tensor with the axis updated
        /// </summary>
        public virtual Tensor ChangeAxis(int[] targetAxisToSrcAxis)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// extract channel 'channel' from 'this' tensor and stores it in 'tensor_NH'
        /// </summary>
        /// <param name="tensor_NH"></param>
        /// <param name="channel"></param>
        public void From_NCH_to_NH(Tensor tensor_NH, int channel)
        {
            var tensor_NCH = this;
            Debug.Assert(tensor_NCH.Shape.Length == 3);
            Debug.Assert(tensor_NH.Shape.Length == 2);
            int batchSize = tensor_NCH.Shape[0];
            int h = tensor_NCH.Shape[2];
            Debug.Assert(batchSize == tensor_NH.Shape[0]);
            Debug.Assert(h == tensor_NH.Shape[1]);
            Debug.Assert(channel < tensor_NCH.Shape[1]);
            for (int n = 0; n < batchSize; ++n)
            {
                tensor_NCH.CopyTo(tensor_NCH.Idx(n, channel, 0), tensor_NH, n * h, h);
            }
        }
     
        /// <summary>
        /// compute: this += alpha * x
        /// </summary>
        /// <param name="alpha"></param>
        /// <param name="x"></param>
        public abstract void Update_Adding_Alpha_X(float alpha, Tensor x);

        /// <summary>
        /// compute: this = alpha * x + beta * this 
        /// </summary>
        /// <param name="alpha"></param>
        /// <param name="x"></param>
        /// <param name="beta"></param>
        public abstract void AddTensor(float alpha, Tensor x, float beta);

        /// <summary>
        /// compute: this = a * x + b 
        /// </summary>
        /// <param name="a">the slope of the linear function</param>
        /// <param name="x">a tensor with the same shape as the 'this' tensor </param>
        /// <param name="b">the constant to add in the linear function</param>
        public abstract void LinearFunction(float a, Tensor x, float b);


        /// <summary>
        /// Concatenate all tensors (through the 'Channel' dimension) into the 'this' tensor.
        /// those tensors must have exactly the same geometry apart from the number of channels (at index 1)
        /// 'this' : Tensor of Dimension (N, C_1+C_2+ .... +C_t, H, W)
        /// </summary>
        /// <param name="tensors">'T' Tensors of Dimension (N, C_t, H, W)</param>
        public abstract void Concatenate(IList<Tensor> tensors);

        /// <summary>
        /// Split the this tensor into the tensors 'tensors'
        /// those 'tensors'  must have exactly the same geometry apart from the number of channels (at index 1)
        /// 'this' : Tensor of Dimension (N, C_1+C_2+ ... C_t, H, W)
        /// </summary>
        /// <param name="tensors">'T' Tensor of Dimension (N, C_i, H, W)</param>
        public abstract void Split(IList<Tensor> tensors);

        public abstract void Update_Multiplying_By_Alpha(float alpha);

        /// <summary>
        /// this = x = [in] input
        /// </summary>
        /// <param name="activationType">the king of activation</param>
        /// <param name="activationParameter">use for Leaky Activation and for SoftmaxWithHierarchy, null otherwise</param>
        /// <param name="y">[out] output after activation</param>
        public abstract void ActivationForward(cudnnActivationMode_t activationType, Tensor activationParameter, Tensor y);

        /// <summary>
        /// this  = dx = [out] gradient of the output
        /// </summary>
        /// <param name="activationType"></param>
        /// <param name="activationParameter"></param>
        /// <param name="dy">[in] gradient of the output</param>
        /// <param name="x">[in] input</param>
        /// <param name="y">[in] output</param>
        public abstract void ActivationBackward(cudnnActivationMode_t activationType, Tensor activationParameter, Tensor dy, Tensor x, Tensor y);

        #region Convolution
        /// <summary>
        /// this = x (N, inputChannels, x.H, x.W)
        /// if isDepthwiseConvolution is true
        ///             Compute:      y = x (depthwise convolution) convolution (with padding / stride)
        ///             Both, x (= this), depthwiseConvolution and y must have the same number of channels.
        /// else
        ///             Compute:      y = x (convolution) convolution (with padding / stride)
        /// <param name="convolution">
        /// if isDepthwiseConvolution is true
        ///             convolution shape is (depthMultiplier=1, inputChannels, f1, f2)
        ///             outputChannels = inputChannels*depthMultiplier
        /// else
        ///             convolution shape is (filtersCount=outputChannels, inputChannels, f1,f2)
        /// </param>
        /// <param name="paddingTop">zero-padding height: number of rows of zeros implicitly concatenated onto the top of input images</param>
        /// <param name="paddingBottom">zero-padding height: number of rows of zeros implicitly concatenated onto the bottom of input images</param>
        /// <param name="paddingLeft">zero-padding width: number of columns of zeros implicitly concatenated onto the left of input images</param>
        /// <param name="paddingRight">zero-padding width: number of columns of zeros implicitly concatenated onto the right of input images</param>
        /// <param name="isDepthwiseConvolution">
        /// true if depthwise convolution, false for standard convolution
        /// </param>
        /// <param name="y">
        /// if isDepthwiseConvolution is true
        ///             y shape is (N, depthMultiplier*inputChannels, y.H, y.W)
        /// else
        ///             y shape is (N, outputChannels, y.H, y.W)
        /// </param>
        /// </summary>
        ///
        public abstract void Convolution(Tensor convolution, int paddingTop, int paddingBottom, int paddingLeft,
            int paddingRight, int stride, Tensor y, bool isDepthwiseConvolution,
            GPUWrapper.ConvolutionAlgoPreference forwardAlgoPreference, TensorMemoryPool memoryPool);

        /// <summary>
        /// this = bias tensor of dimension (1, channels, 1, 1)
        /// For each channel, will retrieve the single associated value and add it to each element of 'y' in the same channel 
        /// </summary>
        /// <param name="y"></param>
        public abstract void BroadcastConvolutionBiasToOutput(Tensor y);

        /// <summary>
        /// this = dy, a tensor of dimension (n, channels, h, w)
        /// For each channel:
        ///     1/ compute the sum of all elements of 'y' in this channel
        ///     2/ add this sum to the channel bias (there is one bias scalar value by channel)
        /// </summary>
        /// <param name="bias">the bias tensor to update, with dimension (1, channels, 1, 1) </param>
        public abstract void ConvolutionBackwardBias(Tensor bias);

        /// <summary>
        ///  this = [in] x : the input tensor
        /// </summary>
        /// <param name="convolution">[in] convolution weights</param>
        /// <param name="dy">[in] gradient of the output tensor 'y'</param>
        /// <param name="paddingTop">zero-padding height: number of rows of zeros implicitly concatenated onto the top of input images</param>
        /// <param name="paddingBottom">zero-padding height: number of rows of zeros implicitly concatenated onto the bottom of input images</param>
        /// <param name="paddingLeft">zero-padding width: number of columns of zeros implicitly concatenated onto the left of input images</param>
        /// <param name="paddingRight">zero-padding width: number of columns of zeros implicitly concatenated onto the right of input images</param>
        /// <param name="stride"></param>
        /// <param name="dx">[out] gradient of the input tensor 'x'</param>
        /// <param name="convGradient">[out] gradient of the convolution weights</param>
        /// <param name="isDepthwiseConvolution">
        ///     true for depth wise convolution
        ///     false for standard convolution
        /// </param>
        /// <param name="backwardAlgoPreference"></param>
        /// <param name="memoryPool"></param>
        public abstract void ConvolutionGradient(Tensor convolution, Tensor dy, int paddingTop, int paddingBottom,
            int paddingLeft, int paddingRight, int stride, Tensor dx, Tensor convGradient, bool isDepthwiseConvolution,
            GPUWrapper.ConvolutionAlgoPreference backwardAlgoPreference, TensorMemoryPool memoryPool);
        #endregion

        //this = x
        public abstract void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride);
        //this = dy
        public abstract void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode,
            int poolingHeight,
            int poolingWidth, int poolingStride);
        public abstract void CopyTo(Tensor b);
        public abstract void CopyTo(int startElement, Tensor other, int otherStartElement, int elementCount);
        //this = dy
        public abstract void Compute_BiasGradient_from_dy(Tensor biasGradient);
        //this = Weights or B
        public abstract void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon, Tensor dW, Tensor adam_vW, Tensor adam_sW, int timestep);
        //this = Weights or B
        public abstract void UpdateSGDOptimizer(double learningRate, double momentum, bool usenesterov, Tensor dW, Tensor velocity);
        public Tensor RowSlice(int startRowIndex, int nbRows)
        {
            Debug.Assert(Shape.Length >= 2);
            Debug.Assert(startRowIndex >= 0);
            Debug.Assert(startRowIndex < Shape[0]);
            Debug.Assert(startRowIndex + nbRows - 1 < Shape[0]);
            var extractedShape = (int[])Shape.Clone();
            extractedShape[0] = nbRows; //new number of rows
            return Slice(Idx(startRowIndex), extractedShape);
        }
        public Tensor ElementSlice(int elementIndex)
        {
            return RowSlice(elementIndex, 1);
        }
        public Tensor ElementSlice(int elementIndex, int[] newShape)
        {
            return Slice(Idx(elementIndex), newShape);
        }

        /// <summary>
        /// [out) y = output tensor of shape (n, 3*h*w, c/3)
        /// </summary>
        /// <param name="x">[in] input tensor with shape (n, c, h , w)</param>
        /// <param name="anchors"></param>
        /// <param name="inputImageHeight"></param>
        /// <param name="inputImageWidth"></param>
        /// <returns></returns>
        public abstract void YOLOV3Forward(Tensor x, int inputImageHeight, int inputImageWidth, int[] anchors);
        public abstract Tensor Slice(int startIndex, int[] sliceShape);

        /// <summary>
        /// true if the tensor is the single owner of the associated memory (it will have to dispose this memory when collected)
        /// false if the memory associated with the tensor is not owned by the 'this' tensor :
        ///     the 'tensor' should not collected the memory when disposed
        ///     the tensor is only a slice on another tensor memory
        /// </summary>
        public abstract bool IsOwnerOfMemory { get; }

        #region Dispose pattern
        protected bool _disposed;
        public abstract void Dispose();
        /// <summary>
        /// ensure the this object is not disposed (will throw an exception if the object is already disposed)
        /// </summary>
        public abstract void AssertIsNotDisposed();
        #endregion

        /// <summary>
        /// this = x [in] unnormalized input
        /// </summary>
        /// <param name="y">[out] normalized output</param>
        /// <param name="scale">[in] scale (=gammas) tensor</param>
        /// <param name="bias">[in] bias (=betas = offset) tensor</param>
        /// <param name="exponentialAverageSmoothingFactor">
        ///the smoothing factor used to compute the running mean and running variance (= 1 - momentum)
        ///     runningMean[t] = exponentialAverageSmoothingFactor * currentMean  +  (1-exponentialAverageSmoothingFactor) * runningMean[t-1]
        ///     (see https://en.wikipedia.org/wiki/Exponential_smoothing)
        /// </param>
        /// <param name="runningInputMean">weighted mean of all the inputs
        /// is isTraining=true
        ///     [in,out]  it will be updated by this method
        /// else (isTraining=false)
        ///     [in] will ony be read by the method
        /// </param>
        /// <param name="runningInputVariance">weighted variance of all the inputs
        /// is isTraining=true
        ///     [in,out]  it will be updated by this method
        /// else (isTraining=false)
        ///     [in] will ony be read by the method
        /// </param>
        /// <param name="mode"></param>
        /// <param name="epsilon"></param>
        /// <param name="meanBuffer">[out] buffer where to store the mean of the input 'x' tensor
        /// only used if isTraining=true</param>
        /// <param name="invertOfUnbiasedVolatilityBuffer">[out] buffer where to store the invert of the unbiased volatility of the input 'x' tensor
        /// only used if isTraining=true</param>
        /// <param name="isTraining">
        /// true if we are training the network
        /// false for inference</param>
        public abstract void BatchNormalization(Tensor y, Tensor scale, Tensor bias, double exponentialAverageSmoothingFactor, Tensor runningInputMean, Tensor runningInputVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer, bool isTraining);

        /// <summary>
        /// this = x [in] unnormalized input
        /// </summary>
        /// <param name="dy">[in] gradient of the output 'y' tensor</param>
        /// <param name="dx">[out] gradient of the input</param>
        /// <param name="scale">[in] scale (=gammas) tensor</param>
        /// <param name="scaleGradient">[out] gradient of the 'scale' tensor</param>
        /// <param name="biasGradient">[out] gradient of the 'bias' tensor</param>
        /// <param name="mode"></param>
        /// <param name="epsilon"></param>
        /// <param name="meanBuffer">[in] mean of the input 'x' tensor</param>
        /// <param name="invertOfUnbiasedVolatilityBuffer">[in] invert of the unbiased volatility of the input 'x' tensor</param>
        public abstract void BatchNormalizationBackward(Tensor dy, Tensor dx, Tensor scale, Tensor scaleGradient, Tensor biasGradient, cudnnBatchNormMode_t mode, double epsilon, Tensor meanBuffer, Tensor invertOfUnbiasedVolatilityBuffer);

        /// <summary>
        /// this = x
        /// </summary>
        /// <param name="y"></param>
        /// <param name="dropProbability"></param>
        /// <param name="isTraining"></param>
        /// <param name="dropoutRandom"></param>
        /// <param name="dropoutReservedSpaceForTraining">a reserved space used only for training (null for inference)</param>
        /// <param name="randomNumberGeneratorStatesBufferForGPU"></param>
        public abstract void DropoutForward(Tensor y, double dropProbability, bool isTraining, Random dropoutRandom,
            Tensor dropoutReservedSpaceForTraining,
            Tensor randomNumberGeneratorStatesBufferForGPU);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dy"></param>
        /// <param name="dx"></param>
        /// <param name="dropProbability"></param>
        /// <param name="dropoutReserveSpace"></param>
        public abstract void DropoutBackward(Tensor dy, Tensor dx, double dropProbability, Tensor dropoutReserveSpace);

        /// <summary>
        /// this = yExpected in one-hot encoding (in each row there are exactly one '1' , all other values being 0)
        /// </summary>
        /// <param name="yPredicted">what has been predicted by the NN (in each row the biggest value is the NN favorite)</param>
        /// <param name="lossFunction"></param>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public abstract double ComputeAccuracy(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer);

        /// <summary>
        /// this = yExpected
        /// </summary>
        /// <param name="yPredicted">what has been predicted by the NN</param>
        /// <param name="buffer"></param>
        /// <returns> The Mean Absolute Error (mae) between 'yExpected'( =this) and 'yPredicted'  </returns>
        public abstract double ComputeMae(Tensor yPredicted, Tensor buffer);

        /// <summary>
        /// this = yExpected
        /// </summary>
        /// <param name="yPredicted">what has been predicted by the NN</param>
        /// <param name="buffer"></param>
        /// <returns> The Mean Squared Error (mse) between 'yExpected'( =this) and 'yPredicted'  </returns>
        public abstract double ComputeMse(Tensor yPredicted, Tensor buffer);


        /// <summary>
        /// Compute the output gradient when we are using categorical hierarchy for categories
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        public abstract void CategoricalCrossentropyWithHierarchyGradient(Tensor yExpected, Tensor yPredicted);

        /// <summary>
        /// Compute the output gradient when are using Huber loss (see https://en.wikipedia.org/wiki/Huber_loss)
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        /// <param name="huberDelta"></param>
        public abstract void HuberGradient(Tensor yExpected, Tensor yPredicted, float huberDelta);

        /// <summary>
        /// Compute the Huber loss (see https://en.wikipedia.org/wiki/Huber_loss)
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        /// <param name="huberDelta"></param>
        public abstract void HuberLoss(Tensor yExpected, Tensor yPredicted, float huberDelta);

        /// <summary>
        /// Compute the output gradient when are using Men Squared Error loss
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        public abstract void MseGradient(Tensor yExpected, Tensor yPredicted);

        /// <summary>
        /// Compute the Mean Squared Error loss (see https://en.wikipedia.org/wiki/Mean_squared_error)
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        public abstract void MseLoss(Tensor yExpected, Tensor yPredicted);

        /// <summary>
        /// pointer to (device or host) pinned memory
        /// </summary>
        public abstract IntPtr Pointer { get; }

        /// <summary>
        /// this = expected (true) values
        /// </summary>
        /// <param name="yPredicted">what has been predicted by the ML</param>
        /// <param name="lossFunction"></param>
        /// <param name="buffer">a temporary buffer</param>
        /// <returns></returns>
        public abstract double ComputeLoss(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer);

        public abstract void RandomizeUniformDistribution(Random rand, double minValue, double maxValue);
        public abstract void RandomMatrixNormalDistribution(Random rand, double mean, double stdDev);

        /// <summary>
        /// set the same value 'sameValue' in the entire tensor
        /// </summary>
        /// <param name="sameValue"></param>
        public abstract void SetValue(float sameValue);
        public abstract float[] ContentAsFloatArray();

        // ReSharper disable once UnusedMemberInSuper.Global
        public abstract Tensor Clone();

        public ulong ReallyNeededMemoryInBytes => (ulong)(Count*TypeSize);
        protected void CheckConcatenate(IList<Tensor> tensors)
        {
            Debug.Assert(Shape.Length >= 2);
            //same number of elements
            Debug.Assert(Shape[1] == tensors.Select(a=>a.Shape[1]).Sum());
            Debug.Assert(Count == tensors.Select(a=>a.Count).Sum());
            foreach (var t in tensors)
            {
                Debug.Assert(Shape.Length == t.Shape.Length);
                Debug.Assert(Shape[0] == t.Shape[0]);
                Debug.Assert(Shape.Skip(2).SequenceEqual(t.Shape.Skip(2)));
            }
        }
        protected bool SameShape(int[] shape) { return Shape.SequenceEqual(shape); }
        public bool SameShapeExceptFirstDimension(int[] shape) { return Shape.Skip(1).SequenceEqual(shape.Skip(1)); }
        protected void RecomputeMultDim()
        {
            _multDim2 = Shape.Length >= 4 ? Shape[3] : 1;
            MultDim1 = Shape.Length >= 3 ? Shape[2] * _multDim2  : 1;
            MultDim0 = Shape.Length >= 2 ? Shape[1] * MultDim1 : 1;
        }

        private bool IsCompatible(Tensor a)
        {
            if (a == null)
            {
                return false;
            }
            if (UseGPU != a.UseGPU)
            {
                return false;
            }
            if (UseGPU && DeviceId != a.DeviceId)
            {
                return false; //tensors must be stored in the same GPU
            }
            return true;
        }
        private string ToString(bool displayStartOfTensor)
        {
            var result = ShapeToString(Shape);
            result += UseGPU ? "" : (this is MockTensor<float>?"Mock":"CPU");
            if (displayStartOfTensor && !UseGPU && (this is CpuTensor<float>))
            {
                //TODO re enable line below
                //result += "(" + string.Join(",", AsFloatCpuContent.Take(3)) + ",...)";
            }

            return result;
        }

        public static CpuTensor<float> SingleFloat(float f)
        {
            return new CpuTensor<float>(new []{1}, new[] {f});
        }
    }
}
