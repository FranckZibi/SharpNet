using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNet.Layers;
using SharpNet.Networks;

namespace SharpNet.Data
{
    //[DebuggerDisplay("{ToString(true)}")]
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
            Debug.Assert(shape.Min() >= 0);
            Shape = shape;
            UseGPU = useGpu;
            TypeSize = typeSize;
            RecomputeMultDim();
        }
        #endregion
        public bool SameShape(params Tensor[] b) { return b.Where(x=>x!=null).All(SameShape); }
        public bool SameShape(Tensor b) {return SameShape(b.Shape);}
        protected bool SameShape(int[] shape) { return Shape.SequenceEqual(shape); }

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
        /// compute the transpose of 'this' tensor and stores it in 'output'
        /// </summary>
        /// <param name="transposed"></param>
        public abstract void Transpose(Tensor transposed);

        /// <summary>
        /// Orthogonal initializer for Weights
        /// See: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Orthogonal
        /// </summary>
        /// <param name="rand"></param>
        public abstract void Orthogonal(Random rand);


        /// <summary>
        /// length of the buffer needed to compute the QR Factorization of 'this' tensor
        /// </summary>
        /// <returns></returns>
        public abstract int QRFactorization_FloatBufferLength();

        /// <summary>
        /// compute A (= this) = Q R factorization
        /// this : the A matrix (in row major order) of shape (m, n) (with m>=n)
        /// </summary>
        /// <param name="Q">the orthogonal 'Q' matrix of shape (m, n)</param>
        /// <param name="R">the upper triangular matrix 'R' of shape (n, n)</param>
        /// <param name="buffer">a float tensor of length returned by 'QRFactorization_FloatBufferLength'</param>
        public abstract void QRFactorization(Tensor Q, Tensor R, Tensor buffer);


        /// <summary>
        /// if the 'this' matrix is an orthogonal matrix, then transpose(this) * this = Identity matrix
        /// return the max error between the expected result (identity matrix) and the observed result of transpose(this) * this
        /// </summary>
        /// <returns></returns>
        public float MaxErrorIfOrthogonalMatrix()
        {
            var n = MultDim0;
            var a = ToCpuFloat();
            //var aTranspose = new CpuTensor<float>(new [] { n, m });
            var multResult =  new CpuTensor<float>(new [] { n, n });
            multResult.Dot(a, true, a, false, 1.0f, 0.0f);
            var spanResult = multResult.AsReadonlyFloatCpuSpan;
            float maxError = 0.0f;
            for(int row=0;row<n;++row )
            for (int col = 0; col < n; ++col)
            {
                var expectedResult = (col == row) ? 1.0f : 0.0f;
                var observedError = Math.Abs(spanResult[col + n * row] - expectedResult);
                maxError = Math.Max(maxError, observedError);
            }
            return maxError;
        }

        /// <summary>
        /// set to 0 all the elements below the main diagonal of the matrix
        /// (all elements with row index strictly less then column index)
        /// </summary>
        public abstract void SetToZeroAllElementsBelowMainDiagonal();


        /// <summary>
        /// set to 'valueForElementsAboveMainDiagonal' all the elements strictly above the main diagonal
        /// (all elements with row index strictly higher then column index)
        /// if 'this' if a 2D Tensor of shape (rows_by_matrix, cols_by_matrix)
        ///     each element above the main diagonal will be set to 'valueForElementsAboveMainDiagonal'
        /// if 'this' if a 3D Tensor of shape (matrices_count, rows_by_matrix, cols_by_matrix)
        ///     it will be considered as a list of 'matrices_count' matrices each with shape (rows_by_matrix, cols_by_matrix)
        ///     each individual matrix will be updated
        /// </summary>
        public abstract void SetAllElementsAboveMainDiagonal(float valueForElementsAboveMainDiagonal);

        /// <summary>
        /// set the 'this' square matrix to an identity matrix (1 on diagonals, 0 everywhere else)
        /// constraints: 'this must be a squared matrix (rows == cols)
        /// </summary>
        public abstract void SetIdentityMatrix();




        /// <summary>
        /// this (= 'y') shape :
        ///      (batchSize, timeSteps, outputSize)
        /// </summary>
        /// <param name="x">
        /// 'x' shape:
        ///      (batchSize, timeSteps, inputSize)
        /// </param>
        /// <param name="wordEmbedding">
        ///  'wordEmbedding' shape:
        ///     (vocabularySize, embeddingDim)
        ///     vocabularySize = 1+number of distinct words in the embedding
        /// </param>
        /// <param name="xIndexInLastDimensionToUse">the index in the last dimension of the 'x' tensor that contains the wordIndex to embed</param>
        /// <param name="yIndexInLastDimensionToUse">the index in the last dimension of the 'y' tensor where we'll store the embedding associated with the wordIndex above </param>
        /// <param name="copyCountBeforeIndex">number of values before the index of the embedding in 'x' tensor to copy before the embedding in 'y' tensor</param>
        /// <param name="copyCountAfterIndex">number of values after the index of the embedding in 'x' tensor to copy after the embedding in 'y' tensor</param>
        public abstract void WordEmbeddingForwardPropagation( /*in*/ Tensor x, /*in*/ Tensor wordEmbedding, int xIndexInLastDimensionToUse, int yIndexInLastDimensionToUse, int copyCountBeforeIndex,  int copyCountAfterIndex);




        /// <summary>
        /// Initialize :
        ///     'this' tensor (= dWordEmbedding) with the gradient of the word embedding weights
        ///     'dx' tensor with the gradient of input 'x'
        /// 'this' shape is (VocabularySize, EmbeddingDim) (same as word embeddings)
        /// </summary>
        /// <param name="x">
        /// 'x' shape:
        ///      (batchSize, timeSteps, inputSize)
        /// </param>
        /// <param name="dx">gradient of input 'x'
        ///  same shape as 'x'
        /// </param>
        /// <param name="dy">
        /// 'dy' shape:
        ///      (batchSize, timeSteps, outputShape)
        /// </param>
        /// <param name="dxIndexInLastDimensionToUse">the index in the last dimension of the 'x' tensor that contains the wordIndex that was embedded</param>
        /// <param name="dyIndexInLastDimensionToUse">the index in the last dimension of the 'dy' tensor that contains the gradient of the embedding associated with the wordIndex above </param>
        /// <param name="copyCountBeforeIndex">number of values before the embedding in 'dy' tensor to copy before the index of the embedding in 'dx' tensor</param>
        /// <param name="copyCountAfterIndex">number of values after the embedding in 'dy' tensor to copy after the index of the embedding in 'dx' tensor</param>
        public abstract void WordEmbeddingBackwardPropagation( /*in*/ Tensor x, /*out*/ Tensor dx, /*in*/ Tensor dy, int dxIndexInLastDimensionToUse, int dyIndexInLastDimensionToUse, int copyCountBeforeIndex, int copyCountAfterIndex);

        public int Count => Shape[0] * MultDim0;
        public int Dimension => Shape.Length;
        protected ulong ReallyNeededMemoryInBytesForShape(int[] shape) { return (ulong)Utils.Product(shape) * (ulong)TypeSize; }
        public CpuTensor<float> AsFloatCpu => AsCpu<float>();
        public CpuTensor<string> AsStringCpu => AsCpu<string>();
        public CpuTensor<int> AsIntCpu => AsCpu<int>();
        public Span<float> AsFloatCpuSpan => AsFloatCpu.SpanContent;
        public float*  AsFloatPointer => (float*)AsFloatCpu.Pointer;

        public ReadOnlySpan<float> AsReadonlyFloatCpuSpan => AsCpu<float>().ReadonlyContent;
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
            // ReSharper disable once MergeConditionalExpression
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
        public abstract void ReshapeInPlace(params int[] newShape);

        /// <summary>
        /// return a reference of the this Tensor changing only its shape
        /// </summary>
        /// <param name="newShape"></param>
        /// <returns></returns>
        public abstract Tensor Reshape(params int[] newShape);

        public static string ShapeToString(int[] shape)
        {
            if (shape==null)
            {
                return "(null)";
            }
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
                    ReshapeInPlace(newShape);
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

        protected static bool SameDimension(List<Tensor> a)
        {
            a.RemoveAll(x => x == null);
            for (int i = 1; i < a.Count; ++i)
            {
                if (a[0].Shape.Length != a[i].Shape.Length)
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

        /// <summary>
        /// Update this tensor with positional encoding using the formula in 'Attention is All You Need'
        /// 'this' tensor of shape (batchSize, timeSteps, embeddingDim)
        /// </summary>
        /// <param name="n">The 'n' described in the paper</param>
        public abstract void UpdateWithPositionalEncoding_AttnIsAllYouNeed(int n);
        
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
        /// Given 2 lists of 2D matrices (contained in the 3D tensors a_3D and b_3D)
        /// compute the dot product of each matrix in a_3D & b_3D
        ///     this[i] = alpha a[i]*b|i] + beta*this[i] 
        /// </summary>
        /// <param name="a_3D">a 3D tensor containing 'a_3D.Shape[0]' 2D matrices
        /// the first dimension of this tensor (a_3D.Shape[0]) must be the same as the first dimension of b_3D
        /// </param>
        /// <param name="transposeA">if we should transpose each 2D matrices contained in a_3D</param>
        /// <param name="b_3D">a 3D tensor containing 'b_3D.Shape[0]' 2D matrices
        /// the first dimension of this tensor (b_3D.Shape[0]) must be the same as the first dimension of a_3D
        /// </param>
        /// <param name="transposeB">if we should transpose each 2D matrices contained in b_3D</param>
        /// <param name="alpha"></param>
        /// <param name="beta"></param>
        public abstract void BatchMatrixMultiplication(Tensor a_3D, bool transposeA, Tensor b_3D, bool transposeB, float alpha, float beta);
        
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
        /// <param name="interpolation">the type of interpolation (nearest or bi-linear)</param>
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

        /// <summary>
        /// clip all values in the tensor in [lower, upper] range
        /// </summary>
        /// <param name="lower">minimum allowed value</param>
        /// <param name="upper">maximum allowed value</param>
        public abstract void Clip(float lower, float upper);

        public abstract void BroadcastAddVectorToOutput(Tensor y);
           
        /// <summary>
        /// transform the content of the 'this' tensor from shape (a,b,...) to shape (b,a,...)
        /// </summary>
        /// <param name="target"></param>
        public abstract void Switch_First_2_axis(Tensor target);

        /// <summary>
        /// transform the content of the 'this' tensor from shape [N, C, H] (or [N, C, H, 1]) to shape [N, H, C]
        /// and store it in target
        /// </summary>
        /// <param name="target"></param>
        public abstract void SwitchSecondAndThirdDimension(/*[OUT]*/ Tensor target);


        /// <summary>
        /// transform the content of the 'this' tensor from shape [A, B, C, *] to shape [A, C, B, *]
        /// and stores it in target
        /// 'this' tensor must be at least a 3D tensor 
        /// </summary>
        /// <param name="target"></param>
        public abstract void TransposeSecondAndThirdDimension(/*[OUT]*/ Tensor target);


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
        // ReSharper disable once UnusedMember.Global
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
        /// compute: this = beta * x + alpha
        /// </summary>
        /// <param name="slope">the slope of the linear function</param>
        /// <param name="x">a tensor with the same shape as the 'this' tensor </param>
        /// <param name="intercept">the constant to add in the linear function</param>
        public abstract void LinearFunction(float slope, Tensor x, float intercept);


        /// <summary>
        /// compute: this = slope * x + intercept
        ///  tensors 'this', slope, x, intercept must have same shape
        /// </summary>
        /// <param name="slope">the slope of the linear function</param>
        /// <param name="x">a tensor with the same shape as the 'this' tensor </param>
        /// <param name="intercept">optional parameter. The constant to add in the linear function</param>
        public void LinearFunction([NotNull] Tensor slope, Tensor x, [CanBeNull] Tensor intercept)
        {
            Debug.Assert(SameShape(slope));
            Debug.Assert(SameShape(x));
            MultiplyTensor(slope, x);            // y = slope * x
            if (intercept != null)
            {
                Debug.Assert(SameShape(intercept));
                Update_Adding_Alpha_X(1f, intercept); // y += intercept
            }
        }


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
        public abstract void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride);
        //this = dy
        public abstract void PoolingGradient(Tensor yNotUsed, Tensor x4D, Tensor dx4D, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int verticalStride, int horizontalStride);
        public abstract void CopyTo(Tensor b);
        public abstract void CopyTo(int startElement, Tensor other, int otherStartElement, int elementCount);
        //this = dy
        public abstract void Compute_BiasGradient_from_dy(Tensor biasGradient);
        //this = Weights or B
        public abstract void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon, double adamW_l2Regularization, Tensor dW, Tensor adam_vW, Tensor adam_sW, int timeStep);
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
        public Tensor GetSubTensor(int startRowIndex)
        {
            return GetSubTensor(startRowIndex, Shape.Skip(1).ToArray());
        }
        public Tensor GetSubTensor(int startRowIndex, int[] subTensorShape)
        {
            Debug.Assert(Shape.Length >= 2);
            Debug.Assert(startRowIndex >= 0);
            Debug.Assert(startRowIndex < Shape[0]);
            return Slice(Idx(startRowIndex), subTensorShape);
        }
        public static List<Tensor> RowSlice(List<CpuTensor<float>> d, int startRowIndex, int nbRows)
        {
            return d.Select(t => t.RowSlice(startRowIndex, nbRows)).ToList();
        }

        public Tensor ElementSlice(int elementIndex)
        {
            return RowSlice(elementIndex, 1);
        }
        // ReSharper disable once UnusedMember.Global
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

        public abstract void Compute_Row_Mean_Variance(Tensor row_mean, Tensor row_variance, bool unbiasedVariance);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="row_mean"></param>
        /// <param name="row_variance"></param>
        /// <param name="axis">
        ///if 0 : we'll standardize each column (each column will have a mean of 0 and a variance of 1)
        ///if 1 : we'll standardize each row (each row will have a mean of 0 and a variance of 1)
        /// </param>
        /// <param name="epsilon"></param>
        public abstract void StandardizeInPlace(Tensor row_mean, Tensor row_variance, int axis, float epsilon);


        ///  <summary>
        ///  we'll standardize each row (each row will have a mean of 0 and a variance of 1)
        ///  </summary>
        ///  <param name="row_mean"></param>
        ///  <param name="row_variance"></param>
        ///  <param name="epsilon"></param>
        ///  <param name="col_gammas"></param>
        ///  <param name="col_betas"></param>
        public abstract void StandardizeRowsInPlaceBroadcastGammasBetas([NotNull] Tensor row_mean, [NotNull] Tensor row_variance, float epsilon, [NotNull] Tensor col_gammas, [NotNull] Tensor col_betas);


        /// <summary>
        /// this = x [in] unnormalized input
        /// </summary>
        /// <param name="dy">[in] gradient of the output 'y' tensor</param>
        /// <param name="dx">[out] gradient of the input</param>
        /// <param name="col_gammas">[in] scale (=gammas) tensor</param>
        /// <param name="row_mean"></param>
        /// <param name="row_variance"></param>
        /// <param name="epsilon"></param>
        /// <param name="dmean_row"></param>
        /// <param name="dvariance_row"></param>
        public abstract void LayerNormalizationBackward(/* in */ Tensor dy, /* out */ Tensor dx, /* in */ Tensor col_gammas, /* in */ Tensor row_mean, /* in */ Tensor row_variance, float epsilon, /* out */ Tensor dmean_row, /* out */ Tensor dvariance_row);



        /// <summary>
        /// perform the forward propagation for Normalization Layer
        ///  this = [in] x : the tensor to normalize, of shape (rows, cols)
        /// </summary>
        /// <param name="y">the normalized version of input tensor 'x'
        /// y = gammas *  (x - mean ) / sqrt(variance+epsilon) + betas
        /// </param>
        /// <param name="gammas">a tensor of shape (1,cols)</param>
        /// <param name="betas">a tensor of shape (1,cols)</param>
        /// <param name="mean">the mean for each row of tensor 'x' , with shape (rows,1)</param>
        /// <param name="variance">the variance for each row of tensor 'x' , with shape (rows,1)</param>
        /// <param name="epsilon">used for numerical stability</param>
        public void LayerNormalization(/* out */ Tensor y, /* in */ Tensor gammas, /* in */ Tensor betas, /* in */  Tensor mean, /* in */ Tensor variance, float epsilon)
        {
            var x = this;
            x.CopyTo(y);
            y.StandardizeRowsInPlaceBroadcastGammasBetas(mean, variance, epsilon, gammas, betas);

        }


        /// <summary>
        /// this = [in] x , a tensor of shape (rows,cols)
        /// if axis == 1
        ///     copy the sum of each row of 'x' into tensor 'sum_result' of shape (rows,1)
        /// if axis == 0
        ///     copy the sum of each col of 'x' into tensor 'sum_result' of shape (1,cols)
        /// </summary>
        /// <param name="sum_result">a tensor to store the result of sum</param>
        /// <param name="axis">
        /// 1 to sum row by row
        /// 0 to sum column by column
        /// </param>
        public abstract void numpy_sum(Tensor sum_result, int axis);


        /// <summary>
        /// this = x
        /// </summary>
        /// <param name="y"></param>
        /// <param name="dropoutRate"></param>
        /// <param name="isTraining"></param>
        /// <param name="dropoutRandom"></param>
        /// <param name="dropoutReservedSpaceForTraining">a reserved space used only for training (null for inference)</param>
        public abstract void DropoutForward(Tensor y, double dropoutRate, bool isTraining, Random dropoutRandom, Tensor dropoutReservedSpaceForTraining);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dy"></param>
        /// <param name="dx"></param>
        /// <param name="dropoutRate"></param>
        /// <param name="dropoutReserveSpace"></param>
        public abstract void DropoutBackward(Tensor dy, Tensor dx, double dropoutRate, Tensor dropoutReserveSpace);


        #region Compute of Loss and Metrics

        /// <summary>
        /// this = a temporary buffer
        /// this method is only called for display / logging testing
        /// </summary>
        /// <param name="yExpected">expected (true) values</param>
        /// <param name="yPredicted">what has been predicted by the ML</param>
        /// <param name="evaluationMetric"></param>
        /// <returns></returns>
        public double ComputeEvaluationMetric(Tensor yExpected, Tensor yPredicted, EvaluationMetricEnum evaluationMetric)
        {
            var buffer = this;
            switch (evaluationMetric)
            {
                case EvaluationMetricEnum.PearsonCorrelation:
                    return yExpected.ComputePearsonCorrelation(yPredicted);
                case EvaluationMetricEnum.SpearmanCorrelation:
                    return yExpected.ComputeSpearmanCorrelation(yPredicted);
                case EvaluationMetricEnum.F1Micro:
                    return buffer.F1PrecisionRecallMicro(yExpected, yPredicted).f1;
                default:
                    ComputeBufferForEvaluationMetric(yExpected, yPredicted, evaluationMetric);
                    return BufferToEvaluationMetric(evaluationMetric, Count);
            }
        }

        #region BinaryCrossentropy & CategoricalCrossentropy &  SparseCategoricalCrossentropy
        public abstract void BinaryCrossentropyLossBuffer([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted);
        public double BinaryCrossentropyLoss(Tensor yExpected, Tensor yPredicted)
        {
            ComputeBufferForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.BinaryCrossentropy);
            return BufferToEvaluationMetric(EvaluationMetricEnum.BinaryCrossentropy, Count);
        }
        /// <summary>
        /// this = a buffer a vector of shape (batch_size)
        /// </summary>
        /// <param name="yExpectedOneHot"> yExpected one hot encoded
        /// in each row there is the list of all associated class)
        /// shape is (batch_size, numClass)</param>
        /// <param name="yPredicted">
        /// what has been predicted by the NN (in each row the biggest value is the NN favorite)
        /// shape is (batch_size, numClass) </param>
        /// <returns></returns>
        public abstract void CategoricalCrossentropyLossBuffer([NotNull] Tensor yExpectedOneHot, [NotNull] Tensor yPredicted);
        public double CategoricalCrossentropyLoss([NotNull] Tensor yExpectedOneHot, [NotNull] Tensor yPredicted)
        {
            ComputeBufferForEvaluationMetric(yExpectedOneHot, yPredicted, EvaluationMetricEnum.CategoricalCrossentropy);
            return BufferToEvaluationMetric(EvaluationMetricEnum.CategoricalCrossentropy, Count);
        }
        public abstract void CategoricalCrossentropyWithHierarchyLossBuffer([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted);
        public double CategoricalCrossentropyWithHierarchyLoss(Tensor yExpected, Tensor yPredicted)
        {
            ComputeBufferForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy);
            return BufferToEvaluationMetric(EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy, Count);
        }
        /// <summary>
        /// this = a buffer a vector of shape (batch_size)
        /// </summary>
        /// <param name="yExpectedSparse"> yExpected in a Sparse Matrix (in each row there is the index of the associated class)
        ///        the shape is (batch_size, 1)</param>
        /// <param name="yPredicted">
        /// what has been predicted by the NN (in each row the biggest value is the NN favorite)
        /// shape is (batch_size, numClass) </param>
        /// <returns></returns>
        public abstract void SparseCategoricalCrossentropyLossBuffer([NotNull] Tensor yExpectedSparse, [NotNull] Tensor yPredicted);
        public double SparseCategoricalCrossentropyLoss([NotNull] Tensor yExpectedSparse, [NotNull] Tensor yPredicted)
        {
            ComputeBufferForEvaluationMetric(yExpectedSparse, yPredicted, EvaluationMetricEnum.SparseCategoricalCrossentropy);
            return BufferToEvaluationMetric(EvaluationMetricEnum.SparseCategoricalCrossentropy, Count);
        }
        #endregion

        #region Accuracy & SparceAccuracy & AccuracyCategoricalCrossentropyWithHierarchy
        /// <summary>
        /// this = a vector of shape (batch_size)
        /// </summary>
        /// <param name="yExpected">
        /// yExpected in a Sparse Matrix (in each row there is the index of the associated class)
        ///        the shape is (batch_size, 1) </param>
        /// <param name="yPredicted">
        /// what has been predicted by the NN (in each row the biggest value is the NN favorite)
        /// shape is (batch_size, numClass) </param>
        /// <returns></returns>
        public abstract void ComputeSparseAccuracyBuffer([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted);
        public double ComputeSparseAccuracy(Tensor yExpectedSparse, Tensor yPredicted)
        {
            ComputeBufferForEvaluationMetric(yExpectedSparse, yPredicted, EvaluationMetricEnum.SparseAccuracy);
            return BufferToEvaluationMetric(EvaluationMetricEnum.SparseAccuracy, Count);
        }
        public abstract void ComputeAccuracyCategoricalCrossentropyWithHierarchyBuffer([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted);
        public double ComputeAccuracyCategoricalCrossentropyWithHierarchy(Tensor yExpected, Tensor yPredicted)
        {
            ComputeBufferForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy);
            return BufferToEvaluationMetric(EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy, Count);
        }
        /// <summary>
        /// this = buffer 
        /// </summary>
        /// <param name="yExpected">yExpected in one-hot encoding (in each row there are exactly one '1' , all other values being 0)</param>
        /// <param name="yPredicted">what has been predicted by the NN (in each row the biggest value is the NN favorite)</param>
        /// <returns></returns>
        public abstract void ComputeAccuracyBuffer([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted);
        public double ComputeAccuracy([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted)
        {
            ComputeBufferForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.Accuracy);
            return BufferToEvaluationMetric(EvaluationMetricEnum.Accuracy, Count);
        }
        private static bool IsSparseMetric(EvaluationMetricEnum metricEnum)
        {
            return metricEnum == EvaluationMetricEnum.SparseCategoricalCrossentropy
                   || metricEnum == EvaluationMetricEnum.SparseAccuracy;
        }
        #endregion
        
        #region Huber
        /// <summary>
        /// this = buffer
        /// Compute the Huber loss (see https://en.wikipedia.org/wiki/Huber_loss)
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        /// <param name="huberDelta"></param>
        public abstract void HuberLossBuffer(Tensor yExpected, Tensor yPredicted, float huberDelta);
        // ReSharper disable once UnusedParameter.Global
        public double HuberLoss(Tensor yExpected, Tensor yPredicted, float huberDelta)
        {
            ComputeBufferForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.Huber);
            return BufferToEvaluationMetric(EvaluationMetricEnum.Huber, Count);
        }
        #endregion
        
        #region Mae
        public double MaeLoss(Tensor yExpected, Tensor yPredicted)
        {
            ComputeBufferForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.Mae);
            return BufferToEvaluationMetric(EvaluationMetricEnum.Mae, Count);
        }
        /// <summary>
        /// Mean Absolute Error
        /// </summary>
        /// <param name="yExpected"></param>
        /// <param name="yPredicted"></param>
        public abstract void MaeLossBuffer([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted);
        #endregion

        #region Mse / Rmse
        public double MseLoss(Tensor yExpected, Tensor yPredicted)
        {
            ComputeBufferForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.Mse);
            return BufferToEvaluationMetric(EvaluationMetricEnum.Mse, Count);
        }
        public double RmseLoss(Tensor yExpected, Tensor yPredicted)
        {
            ComputeBufferForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.Mse);
            return BufferToEvaluationMetric(EvaluationMetricEnum.Rmse, Count);
        }
        /// <summary>
        /// MSE
        /// </summary>
        /// <param name="yExpected"></param>
        /// <param name="yPredicted"></param>
        public abstract void MseLossBuffer([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted);

        #endregion

        #region MseOfLogLoss
        /// <summary>
        /// this = buffer
        /// Compute the Mean Squared Error of log loss (MseOfLog loss) and stores it in the 'this' tensor
        /// This loss is defined by:
        ///     loss  = ( log( max(predicted,epsilon) ) - log(expected) ) ^2
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        /// <param name="epsilon">minimum allowed value for a prediction</param>
        public abstract void MseOfLogLossBuffer(Tensor yExpected, Tensor yPredicted, float epsilon);

        public double MseOfLogLoss(Tensor yExpected, Tensor yPredicted, float epsilon)
        {
            ComputeBufferForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.MseOfLog);
            return BufferToEvaluationMetric(EvaluationMetricEnum.MseOfLog, Count);
        }
        #endregion

        #region MeanSquaredLogError
        public abstract void MeanSquaredLogErrorLossBuffer([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted);
        public double MeanSquaredLogErrorLoss(Tensor yExpected, Tensor yPredicted)
        {
            ComputeBufferForEvaluationMetric(yExpected, yPredicted, EvaluationMetricEnum.MeanSquaredLogError);
            return BufferToEvaluationMetric(EvaluationMetricEnum.MeanSquaredLogError, Count);
        }
        #endregion

        #region F1 score
        public abstract (float f1, float precision, float recall) F1PrecisionRecallMicro(Tensor yExpected, Tensor yPredicted);
        #endregion

        #region Cosine Similarity
        /// <summary>
        /// Compute the Cosine Similarity Loss (see https://en.wikipedia.org/wiki/Cosine_similarity)
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        /// <param name="timeSeriesLength"></param>
        public abstract void CosineSimilarityLossBuffer(Tensor yExpected, Tensor yPredicted, int timeSeriesLength);
        #endregion

        #region Pearson & Spearman Correlation
        public abstract double ComputePearsonCorrelation([NotNull] Tensor y_pred);
        public abstract double ComputeSpearmanCorrelation([NotNull] Tensor y_pred);
        #endregion

        #region AUC
        public abstract void ComputeAUCBuffer([NotNull] Tensor yExpected, [NotNull] Tensor yPredicted);
        public double ComputeAUC(Tensor yExpectedSparse, Tensor yPredicted)
        {
            var buffer = this;
            Debug.Assert(buffer.Count == 1);
            ComputeBufferForEvaluationMetric(yExpectedSparse, yPredicted, EvaluationMetricEnum.AUC);
            return buffer.ContentAsFloatArray()[0];
        }
        #endregion

        // ReSharper disable once UnusedParameter.Global
        public double BufferToEvaluationMetric(EvaluationMetricEnum evaluationMetric, int elementCountInBuffer)
        {
            var buffer = this;
            if (evaluationMetric == EvaluationMetricEnum.Rmse)
            {
                return Math.Sqrt(buffer.BufferToEvaluationMetric(EvaluationMetricEnum.Mse, elementCountInBuffer));
            }
            return buffer.ContentAsFloatArray().Sum() / elementCountInBuffer;
        }

        public void ComputeBufferForEvaluationMetric(Tensor yExpected, Tensor yPredicted, EvaluationMetricEnum evaluationMetric)
        {
            var buffer = this;
            switch (evaluationMetric)
            {
                case EvaluationMetricEnum.Accuracy:
                    buffer.ComputeAccuracyBuffer(yExpected, yPredicted);
                    return;
                case EvaluationMetricEnum.SparseAccuracy:
                    buffer.ComputeSparseAccuracyBuffer(yExpected, yPredicted);
                    return;
                case EvaluationMetricEnum.SparseCategoricalCrossentropy:
                    buffer.SparseCategoricalCrossentropyLossBuffer(yExpected, yPredicted);
                    return;
                case EvaluationMetricEnum.PearsonCorrelation:
                    throw new NotImplementedException();
                //!D return yExpected.ComputePearsonCorrelation(yPredicted);
                case EvaluationMetricEnum.SpearmanCorrelation:
                    throw new NotImplementedException();
                //!D return yExpected.ComputeSpearmanCorrelation(yPredicted);
                case EvaluationMetricEnum.AUC:
                    buffer.ComputeAUCBuffer(yExpected, yPredicted);
                    return;
                case EvaluationMetricEnum.AccuracyCategoricalCrossentropyWithHierarchy:
                    buffer.ComputeAccuracyCategoricalCrossentropyWithHierarchyBuffer(yExpected, yPredicted);
                    return;
                case EvaluationMetricEnum.BinaryCrossentropy:
                    buffer.BinaryCrossentropyLossBuffer(yExpected, yPredicted);
                    return;
                case EvaluationMetricEnum.CategoricalCrossentropy:
                    buffer.CategoricalCrossentropyLossBuffer(yExpected, yPredicted);
                    return;
                case EvaluationMetricEnum.CategoricalCrossentropyWithHierarchy:
                    buffer.CategoricalCrossentropyWithHierarchyLossBuffer(yExpected, yPredicted);
                    return;
                case EvaluationMetricEnum.Huber:
                    const float huberDelta = 1.0f;
                    buffer.HuberLossBuffer(yExpected, yPredicted, huberDelta);
                    return;
                case EvaluationMetricEnum.F1Micro:
                    throw new NotImplementedException();
                //!D return buffer.F1PrecisionRecallMicro(yExpected, yPredicted).f1;
                case EvaluationMetricEnum.Mse:
                    buffer.MseLossBuffer(yExpected, yPredicted);
                    return;
                case EvaluationMetricEnum.Rmse:
                    //RMSE and MSE are using the same buffer
                    buffer.MseLossBuffer(yExpected, yPredicted);
                    return;
                case EvaluationMetricEnum.Mae:
                    buffer.MaeLossBuffer(yExpected, yPredicted);
                    return;
                case EvaluationMetricEnum.MseOfLog:
                    buffer.MseOfLogLossBuffer(yExpected, yPredicted, NetworkSample.Default_MseOfLog_Loss);
                    return;
                case EvaluationMetricEnum.MeanSquaredLogError:
                    buffer.MeanSquaredLogErrorLossBuffer(yExpected, yPredicted);
                    return;
                default:
                    throw new NotImplementedException("don't know how to calculate cost for " + evaluationMetric);
            }
        }

        public int[] ComputeMetricBufferShape(EvaluationMetricEnum metricEnum)
        {
            if (metricEnum == EvaluationMetricEnum.AUC)
            {
                return new[] { 1 };
            }
            if (IsSparseMetric(metricEnum))
            {
                return new[] { Count };
            }
            return new[] { Shape[0] };
        }

        /// <summary>
        /// if yPredicted is a 3D Tensors:
        ///     we'll assume the following shapes for all tensors:
        ///             yExpectedSparse:            (batchSize, timeSteps)
        ///             yPredicted:                 (batchSize, timeSteps, numClass)
        ///             yPredictedGradientIfAny:    (batchSize, timeSteps, numClass)
        ///    and we'll reshape them to:
        ///             yExpectedSparse:            (batchSize*timeSteps, 1)
        ///             yPredicted:                 (batchSize*timeSteps, numClass)
        ///             yPredictedGradientIfAny:    (batchSize*timeSteps, numClass)
        ///    and return those new shapes
        /// else (if yPredicted is a 2D Tensors):
        ///     we'll assume all inputs tensors are 2D tensors with following shapes:
        ///             yExpectedSparse:            (batchSize*timeSteps, 1)
        ///             yPredicted:                 (batchSize*timeSteps, numClass)
        ///             yPredictedGradientIfAny:    (batchSize*timeSteps, numClass)
        ///    and will do nothing (we'll return the exact same tensors received as input)
        /// </summary>
        /// <param name="yExpectedSparse"></param>
        /// <param name="yPredicted"></param>
        /// <param name="yPredictedGradientIfAny"></param>
        /// <returns></returns>
        protected static (Tensor, Tensor, Tensor) ReformatTo2DTensorsSparse(Tensor yExpectedSparse, Tensor yPredicted, Tensor yPredictedGradientIfAny = null)
        {
            if (yPredicted.Shape.Length == 3)
            {
                //yExpectedSparse shape:    (batchSize, timeSteps)
                //yPredicted shape:         (batchSize, timeSteps, numClass)
                yExpectedSparse = yExpectedSparse.Reshape(-1, 1);
                yPredicted = yPredicted.Reshape(-1, yPredicted.Shape[^1]);
                yPredictedGradientIfAny = yPredictedGradientIfAny?.Reshape(yPredicted.Shape);
                //yExpectedSparse shape:    (batchSize*timeSteps, 1)
                //yPredicted shape:         (batchSize*timeSteps, numClass)
            }
            Debug.Assert(AreCompatible(new List<Tensor> { yExpectedSparse, yPredicted }));
            Debug.Assert(yExpectedSparse.Shape.Length == 2);
            Debug.Assert(yExpectedSparse.Shape[1] == 1);
            Debug.Assert(yPredicted.Shape.Length == 2);
            Debug.Assert(yExpectedSparse.Shape[0] == yPredicted.Shape[0]);
            return (yExpectedSparse, yPredicted, yPredictedGradientIfAny);
        }


        #endregion




        #region Compute of Gradients (for backward propagation)
        /// <summary>
        /// Compute the output gradient when we are using categorical hierarchy for categories
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        public abstract void CategoricalCrossentropyWithHierarchyGradient(Tensor yExpected, Tensor yPredicted);

        /// <summary>
        /// Compute the output gradient when using Cosine Similarity Loss
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        /// <param name="timeSeriesLength"></param>
        public abstract void CosineSimilarityGradient(Tensor yExpected, Tensor yPredicted, int timeSeriesLength);

        /// <summary>
        /// Compute the output gradient when using Huber loss (see https://en.wikipedia.org/wiki/Huber_loss)
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        /// <param name="huberDelta"></param>
        public abstract void HuberGradient(Tensor yExpected, Tensor yPredicted, float huberDelta);

        /// <summary>
        /// Compute the output gradient when are using Mean Squared Error loss
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        public abstract void MseGradient(Tensor yExpected, Tensor yPredicted);


        /// <summary>
        /// Compute the output gradient when are using Sparse Categorical Crossentropy loss
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpectedSparse">the expected values for the prediction with shape:
        ///     (batchSize, 1)              (when yPredicted is a 2D Tensor)
        /// or
        ///     (batchSize, timeSteps)      (when yPredicted is a 3D Tensor)
        /// each element is the index of the expected class</param>
        /// <param name="yPredicted">the observed values for the prediction with shape:
        ///     (bachSize, numClass)
        /// or
        ///     (batchSize, timeSteps, numClass)
        /// </param>
        public abstract void SparseCategoricalCrossentropyGradient(Tensor yExpectedSparse, Tensor yPredicted);
        

        /// <summary>
        /// Compute the output gradient when are using Mean Absolute Error loss
        /// and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        public abstract void MaeGradient(Tensor yExpected, Tensor yPredicted);

        /// <summary>
        /// Compute the output gradient when we are using MseOfLog loss (see above) and stores it in the 'this' tensor
        /// </summary>
        /// <param name="yExpected">the expected values for the prediction</param>
        /// <param name="yPredicted">the observed values for the prediction</param>
        /// <param name="epsilon">minimum allowed value for a prediction</param>
        public abstract void MseOfLogGradient(Tensor yExpected, Tensor yPredicted, float epsilon);
        #endregion

        /// <summary>
        /// pointer to (device or host) pinned memory
        /// </summary>
        public abstract IntPtr Pointer { get; }
        
        public abstract void UniformDistribution(Random rand, double minValue, double maxValue);
        public abstract void NormalDistribution(Random rand, double mean, double stdDev);

        /// <summary>
        /// Glorot Uniform Initializer for Weights
        /// See: https://www.tensorflow.org/api_docs/python/tf/compat/v1/keras/initializers/glorot_uniform
        /// </summary>
        /// <param name="rand"></param>
        public void GlorotUniform(Random rand)
        {
            int fanIn = Shape[0];  //number of input units in the weight tensor
            int fanOut = MultDim0; //number of output units in the weight tensor
            var limit = Math.Sqrt(6.0 / (fanIn + fanOut));
            UniformDistribution(rand, -limit, limit);
        }

        public void PytorchUniform(Random rand)
        {
            int fanIn = Shape[0];  //number of input units in the weight tensor
            var limit = Math.Sqrt(1.0 / (fanIn));
            UniformDistribution(rand, -limit, limit);
        }
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
        public bool SameShapeExceptFirstDimension(int[] shape) { return Shape.Skip(1).SequenceEqual(shape.Skip(1)); }
        protected void RecomputeMultDim()
        {
            _multDim2 = Shape.Length >= 4 ? Shape[3] : 1;
            MultDim1 = Shape.Length >= 3 ? Shape[2] * _multDim2  : 1;
            MultDim0 = Shape.Length >= 2 ? Shape[1] * MultDim1 : 1;
        }


        /// <summary>
        /// We want to reshape a tensor to another shape (with the same number of elements)
        /// The original tensor shape is 'originalShape'
        /// The target shape is 'newShapeWithPossibleMinusOne'
        /// If the target shape contains a -1, it will replace this -1 by the valid value for this dimension
        /// to keep the same number of elements
        /// </summary>
        /// <param name="originalShape">the original shape of the tensor</param>
        /// <param name="newShapeWithPossibleMinusOne">the tatget shape of the tensor
        /// this array may contain (at most) one -1</param>
        /// <returns>the target shape of the tensor without -1, so that it has the same number of elements as in the original shape</returns>
        /// <exception cref="ArgumentException"></exception>
        public static int[] FillMinusOneIfAny(int[] originalShape, int[] newShapeWithPossibleMinusOne)
        {
            int indexMinusOne = -1;
            for (int i = 0; i < newShapeWithPossibleMinusOne.Length; i++)
            {
                if (newShapeWithPossibleMinusOne[i] == -1)
                {
                    if (indexMinusOne != -1)
                    {
                        throw new ArgumentException("Only one -1 is allowed in the new shape");
                    }
                    indexMinusOne = i;
                }
            }
            if (indexMinusOne == -1)
            {
                return newShapeWithPossibleMinusOne;
            }
            int oldShapeCount = Utils.Product(originalShape);
            int newShapeCount = -Utils.Product(newShapeWithPossibleMinusOne);
            if (oldShapeCount % newShapeCount != 0)
            {
                throw new ArgumentException("The new shape is not compatible with the old shape");
            }
            var newShapeWithoutMinusOne = (int[])newShapeWithPossibleMinusOne.Clone();
            newShapeWithoutMinusOne[indexMinusOne] = oldShapeCount / newShapeCount;
            return newShapeWithoutMinusOne;
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
            result += UseGPU ? "" : "CPU";
            if (displayStartOfTensor && !UseGPU && (this is CpuTensor<float>))
            {
                //TODO re enable line below
                //result += "(" + string.Join(",", AsFloatCpuContent.Take(3)) + ",...)";
            }

            return result;
        }

        /// <summary>
        /// this: a tensor of shape (a,b,...,y,z,numClass)
        /// buffer: a tensor of shape (a,b,...,y,z,1)
        /// for each row of the input tensor, find the index of the max element of the row
        /// and store it in the buffer tensor
        /// </summary>
        /// <param name="buffer">a tensor of shape (a,b,...,y,z,1) </param>
        public abstract void ArgMax(Tensor buffer);

        public static int[] ToPooling4D(int[] shape3D)
        {
            Debug.Assert(shape3D.Length == 3);
            return new[] { shape3D[0], 1, shape3D[1], shape3D[2] };
        }
        public static CpuTensor<float> SingleFloat(float f)
        {
            return new CpuTensor<float>(new []{1}, new[] {f});
        }


        /// <summary>
        /// When 'this' is a tensor with >=3 dimension         (ex:  (a, b, c, d))
        ///     if flattenInputTensorOnLastDimension == true
        ///             we'll change its shape to a 2D Matrix       (ex:  (a*b*c, d) )
        ///             so that the last dimension of the matrix (ex: d) is preserved
        ///     else (flattenInputTensorOnLastDimension == false)
        ///             we'll change its shape to a 2D Matrix       (ex:  (a, b*c*d) )
        ///             so that the first dimension of the matrix (ex: a) is preserved
        /// </summary>
        /// <param name="flattenInputTensorOnLastDimension"></param>
        /// <returns>A 2D Matrix</returns>
        public Tensor As2DTensor(bool flattenInputTensorOnLastDimension)
        {
            if (Shape.Length == 2)
            {
                return this;
            }
            return flattenInputTensorOnLastDimension 
                ? Reshape(-1, Shape[^1]) 
                : Reshape(Shape[0], -1);
        }

    }
}
