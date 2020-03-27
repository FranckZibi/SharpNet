using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNet.Networks;

namespace SharpNet.Data
{
    [DebuggerDisplay("{ToString(true)}")]
    public abstract class Tensor : IDisposable
    {
        #region fields
        public int[] Shape { get; protected set; }
        public int MultDim0 { get; private set; }
        public int MultDim1 { get; private set; }
        private int _multDim2;
        public bool UseGPU { get; }
        public string Description { get; }
        public int TypeSize { get; }
        #endregion

        public bool SameShape(params Tensor[] b) { return b.Where(x=>x!=null).All(SameShape); }
        public bool SameShape(Tensor b) {return Shape.SequenceEqual(b.Shape);}
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
        // this = a*b
        public void Dot(Tensor a, Tensor b) { Dot(a, false, b, false, 1, 0); }
        public int Count => Shape[0] * MultDim0;
        public int Dimension => Shape.Length;
        protected ulong ReallyNeededMemoryInBytesForShape(int[] shape) { return (ulong)Utils.Product(shape) * (ulong)TypeSize; }
        public CpuTensor<float> AsFloatCpu => AsCpu<float>();
        public float[] AsFloatCpuContent => AsCpu<float>().Content;
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
            return t.DevicePointer;
        }
        public CpuTensor<T> AsCpu<T>() where T : struct
        {
            if (this is CpuTensor<T> result)
            {
                return result;
            }
            throw new Exception("fail to convert " + this + " this to a CpuTensor<" + typeof(T)+">");
        }

        public GPUTensor<T> ToGPU<T>(GPUWrapper gpuWrapper) where T : struct
        {
            return UseGPU ? AsGPU<T>() : new GPUTensor<T>(Shape, AsCpu<T>().HostPointer, Description, gpuWrapper);
        }
        public CpuTensor<float> ToCpuFloat()
        {
            if (this is CpuTensor<float>)
            {
                return (CpuTensor<float>) this;
            }
            return new CpuTensor<float>(Shape, ContentAsFloatArray(), Description);
        }
        public abstract void Reshape(int[] newShape);

        public static ulong OccupiedMemoryInBytes(IEnumerable<Tensor> tensors)
        {
            return (ulong)tensors.Select(x => (long) (x?.CapacityInBytes ?? 0) ).Sum();
        }

        /// <summary>
        /// Ensure that all tensors are stored in the same device (Cpu or GPU)
        /// </summary>
        /// <param name="a">lit of tensors</param>
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
        ///     this = a (element_wise_multiplication) Diag(c)
        ///     where c is a vector containing a diagonal matrix
        /// </summary>
        /// <param name="a">a matrix</param>
        /// <param name="x">a vector containing a diagonal matrix
        /// (only the diagonal of the diagonal matrix is contained in vector 'x'</param>
        public abstract void MultiplyTensor(Tensor a, Tensor x);

        /// <summary>
        /// Update the value of the 'this( tensor by multiplying it by 'x'
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
        /// For each row of matrix a and b , compute the element wise product and this row, and store the result in this[row]
        /// this = a vector of size (m)
        /// </summary>
        /// <param name="a">a matrix of size (m,n) </param>
        /// <param name="b">a matrix of size (m,n) </param>
        public abstract void MultiplyEachRowIntoSingleValue(Tensor a, Tensor b);

        /// <summary>
        /// this = singleLineMatrix to add to y
        /// </summary>
        /// <param name="y"></param>
        public abstract void BroadcastAddVectorToOutput(Tensor y);

        /// <summary>
        /// extract channel 'channel' from this tensor and store it in 'tensor_NH'
        /// </summary>
        /// <param name="tensor_NH"></param>
        /// <param name="channel"></param>
        public abstract void From_NCH_to_NH(Tensor tensor_NH, int channel);

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
        /// Concatenate the 2 tensors 'a' & 'b'  (through the 'Channel' dimension) into the 'this' tensor.
        /// 'this', 'a' and 'b'  must have exactly the same geometry apart from the number of channels (at index 1)
        /// 'this' : Tensor of Dimension (N, Ca+Cb, H, W)
        /// </summary>
        /// <param name="a">Tensor of Dimension (N, Ca, H, W)</param>
        /// <param name="b">Tensor of Dimension (N, Cb, H, W)</param>
        public abstract void Concatenate(Tensor a, Tensor b);

        /// <summary>
        /// Clone
        /// </summary>
        /// <param name="gpuWrapper"></param>
        /// <returns></returns>
        public abstract Tensor Clone(GPUWrapper gpuWrapper);

        /// <summary>
        /// Split the this tensor into the tensors 'a' & 'b'.
        /// 'this', 'a' and 'b' must have exactly the same geometry apart from the number of channels (at index 1)
        /// 'this' : Tensor of Dimension (N, Ca+Cb, H, W)
        /// </summary>
        /// <param name="a">Tensor of Dimension (N, Ca, H, W)</param>
        /// <param name="b">Tensor of Dimension (N, Cb, H, W)</param>
        public abstract void Split(Tensor a, Tensor b);
        public abstract void Update_Multiplying_By_Alpha(float alpha);

        //this = x
        public abstract void ActivationForward(cudnnActivationMode_t activationType, Tensor y);


        #region Convolution
        /// <summary>
        /// this = x (N, C, x.H, x.W)
        /// if isDepthwiseConvolution is true
        ///             Compute:      y = x (depthwise convolution) convolution (with padding / stride)
        ///             Both, x (= this), depthwiseConvolution and y must have the same number of channels.
        /// else
        ///             Compute:      y = x (convolution) convolution (with padding / stride)
        /// <param name="depthwiseConvolution">
        /// if isDepthwiseConvolution is true
        ///             (depthMultiplier=1, inputChannels=outputChannels, f1, f2)
        /// else
        ///             (filtersCount=outputChannels, inputChannels, f1,f2)
        /// </param>
        /// <param name="y">
        /// if isDepthwiseConvolution is true
        ///             (N, depthMultiplier*C, y.H, y.W)
        /// else
        ///             (N, conv.filtersCount, y.H, y.W)
        /// </param>
        /// </summary>
        public abstract void Convolution(Tensor convolution, int padding, int stride, Tensor y, bool isDepthwiseConvolution);

        /// <summary>
        /// this = bias tensor of dimension (1, channels, 1, 1)
        /// For each channel, will retrieve the single associated value and add it to each element of 'y' in the same channel 
        /// </summary>
        /// <param name="convolutionBackwardBias">a bias tensor (1, channel, 1, 1) </param>
        public abstract void BroadcastConvolutionBiasToOutput(Tensor y);

        /// <summary>
        /// this = dy, a tensor of dimension (n, channels, h, w)
        /// For each channel:
        ///     1/ compute the sum of all elements of 'y' in this channel
        ///     2/ add this sum to the channel bias (there is one bias scalar value by channel)
        /// </summary>
        /// <param name="bias">the bias tensor to update, with dimension (1, channels, 1, 1) </param>
        /// <summary>
        public abstract void ConvolutionBackwardBias(Tensor bias);

        //this = x
        public abstract void ConvolutionGradient(Tensor convolution, Tensor dy, int padding, int stride, Tensor dx, Tensor convGradient, bool isDepthwiseConvolution);
        #endregion

        //this = x
        public abstract void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingHeight, int poolingWidth, int poolingStride);
        //this = dy
        public abstract void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode,
            int poolingHeight,
            int poolingWidth, int poolingStride);
        public abstract void CopyTo(Tensor b);
        public abstract void CopyTo(int startElement, Tensor other, int otherStartElement, int elementCount);
        //this  = y
        public abstract void ActivationBackward(Tensor dy, Tensor x, cudnnActivationMode_t activationType, Tensor dx);
        //this = dy
        public abstract void Compute_BiasGradient_from_dy(Tensor biasGradient);
        //this = Weights or B
        public abstract void UpdateAdamOptimizer(double learningRate, double beta1, double beta2, double epsilon, Tensor dW, Tensor adam_vW, Tensor adam_sW, int timestep);
        //this = Weights or B
        public abstract void UpdateSGDOptimizer(double learningRate, double momentum, bool usenesterov, Tensor dW, Tensor velocity);
        public abstract Tensor ExtractSubTensor(int startRowIndex, int nbRows);

        #region Dispose pattern
        protected bool _disposed;
        public abstract void Dispose();
        /// <summary>
        /// ensure the this object is not disposed (will throw an exception if the object is already disposed)
        /// </summary>
        public abstract void AssertIsNotDisposed();
        #endregion

        public abstract Tensor Transpose();

        /// <summary>
        /// this = x
        /// </summary>
        /// <param name="y"></param>
        /// <param name="bnScale"></param>
        /// <param name="bnBias"></param>
        /// <param name="exponentialAverageFactor"></param>
        /// <param name="resultRunningMean"></param>
        /// <param name="resultRunningVariance"></param>
        /// <param name="mode"></param>
        /// <param name="epsilon"></param>
        /// <param name="resultSaveMean"></param>
        /// <param name="resultSaveVariance"></param>
        /// <param name="isTraining"></param>
        public abstract void BatchNormalization(Tensor y, Tensor bnScale, Tensor bnBias, double exponentialAverageFactor, Tensor resultRunningMean, Tensor resultRunningVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor resultSaveMean, Tensor resultSaveVariance, bool isTraining);

        /// <summary>
        /// this = x
        /// </summary>
        /// <param name="dy"></param>
        /// <param name="dx"></param>
        /// <param name="bnScale"></param>
        /// <param name="resultBnScaleDiff"></param>
        /// <param name="resultBnBiasDiff"></param>
        /// <param name="mode"></param>
        /// <param name="epsilon"></param>
        /// <param name="resultSaveMean"></param>
        /// <param name="resultSaveVariance"></param>
        public abstract void BatchNormalizationBackward(Tensor dy, Tensor dx, Tensor bnScale, Tensor resultBnScaleDiff, Tensor resultBnBiasDiff, cudnnBatchNormMode_t mode, double epsilon, Tensor resultSaveMean, Tensor resultSaveVariance);

        /// <summary>
        /// this = x
        /// </summary>
        /// <param name="y"></param>
        /// <param name="dropProbability"></param>
        /// <param name="isTraining"></param>
        /// <param name="dropoutRandom"></param>
        /// <param name="dropoutMaskBuffer"></param>
        public abstract void DropoutForward(Tensor y, double dropProbability, bool isTraining, Random dropoutRandom, Tensor dropoutMaskBuffer);
        public abstract void DropoutBackward(Tensor dy, Tensor dx, double dropProbability, Tensor usedDropoutMask);

        /// <summary>
        /// this = yExpected in one-hot encoding (in each row there are exactly one '1' , all other values being 0)
        /// </summary>
        /// <param name="yPredicted">what has been predicted by the ML (in each row the biggest value is the ML favorite)</param>
        /// <param name="notUsedBuffer"></param>
        /// <returns></returns>
        public abstract double ComputeAccuracy(Tensor yPredicted, Tensor notUsedBuffer);


        /// <summary>
        /// this = expected category index for each element
        /// </summary>
        /// <param name="yPredicted">what has been predicted by the ML (in each row the biggest value is the ML favorite)</param>
        /// <param name="notUsedBuffer"></param>
        /// <returns></returns>
        public abstract double ComputeAccuracyFromCategoryIndexes(Tensor yPredicted, Tensor notUsedBuffer);

        protected abstract IntPtr DevicePointer { get; }


        /// <summary>
        /// this = yExpected in one-hot encoding (in each row there are exactly one '1' , all other values being 0)
        /// </summary>
        /// <param name="yPredicted">what has been predicted by the ML (in each row the biggest value is the ML favorite)</param>
        /// <param name="lossFunction"></param>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public abstract double ComputeLoss(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer);

        /// <summary>
        /// this = expected Category Indexes (int Tensor)
        /// </summary>
        /// <param name="yPredicted">what has been predicted by the ML (in each row the biggest value is the ML favorite)</param>
        /// <param name="lossFunction"></param>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public abstract double ComputeLossFromCategoryIndexes(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer);

        public abstract void RandomMatrixNormalDistribution(Random rand, double mean, double stdDev);
        public abstract void NewSameValueTensor(double sameValue);
        public abstract float[] ContentAsFloatArray();
        protected Tensor(int[] shape, int typeSize, bool useGpu, string description)
        {
            Debug.Assert(shape.Length >= 1);
            Debug.Assert(shape.Length <= 4);
            Debug.Assert(shape.Min() >= 1);
            Shape = shape;
            UseGPU = useGpu;
            Description = description;
            TypeSize = typeSize;
            RecomputeMultDim();
        }
        protected ulong ReallyNeededMemoryInBytes => (ulong)(Count*TypeSize);
        protected void CheckConcatenate(Tensor a, Tensor b)
        {
            Debug.Assert(Shape.Length >= 2);
            Debug.Assert(Shape.Length == a.Shape.Length);
            Debug.Assert(Shape.Length == b.Shape.Length);
            //same number of elements
            Debug.Assert(Shape[0] == a.Shape[0]);
            Debug.Assert(Shape[0] == b.Shape[0]);
            Debug.Assert(Shape[1] == (a.Shape[1] + b.Shape[1]));
            Debug.Assert(Shape.Skip(2).SequenceEqual(a.Shape.Skip(2)));
            Debug.Assert(Shape.Skip(2).SequenceEqual(b.Shape.Skip(2)));
        }
        protected int Idx(int n, int c, int h) { return MultDim0 * n + MultDim1 * c + h; }

        protected void RecomputeMultDim()
        {
            _multDim2 = Shape.Length >= 4 ? Shape[3] : 1;
            MultDim1 = Shape.Length >= 3 ? Shape[2] * _multDim2 : 1;
            MultDim0 = Shape.Length >= 2 ? Shape[1] * MultDim1 : 1;
        }
        private bool IsCompatible(Tensor a)
        {
            return (a != null && UseGPU == a.UseGPU);
        }
        private string ToString(bool displayStartOfTensor)
        {
            var result = Description + "(" + string.Join(", ", Shape) + ")";
            result += UseGPU ? "" : "CPU";
            if (displayStartOfTensor && !UseGPU)
            {
                result += "(" + string.Join(",", AsCpu<float>().Content.Take(3)) + ",...)";
            }

            return result;
        }
        public GPUTensor<T> AsGPU<T>() where T : struct
        {
            if (this is GPUTensor<T> result)
            {
                return result;
            }
            throw new Exception("fail to convert " + this + " this to a GPUTensor<" + typeof(T) + ">");
        }
    }
}
