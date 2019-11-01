using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
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
        public int Idx(int n) { return MultDim0 * n; }

        public int Idx(int n, int c)
        {
            return MultDim0 * n + MultDim1 * c;
        }
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
            return t.AsGPU<float>().DevicePointer;
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
        /// this = dy
        /// </summary>
        /// <param name="convolutionBackwardBias"></param>
        public abstract void ConvolutionBackwardBias(Tensor convolutionBackwardBias);

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
        /// compute: this = alpha * x + this 
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
        //this = Tensor<T> convolutionBiasVector
        public abstract void BroadcastConvolutionBiasToOutput(Tensor y);
        //this = x
        public abstract void ActivationForward(cudnnActivationMode_t activationType, Tensor y);
        //this = x
        public abstract void Convolution(Tensor convolution, int padding, int stride, Tensor y);
        //this = x
        public abstract void ConvolutionGradient(Tensor conv, Tensor dy, int padding, int stride, Tensor dx, Tensor convGradient);
        //this = x
        public abstract void Pooling(Tensor y, cudnnPoolingMode_t poolingMode, int poolingSize, int poolingStride);
        //this = dy
        public abstract void PoolingGradient(Tensor y, Tensor x, Tensor dx, cudnnPoolingMode_t poolingMode, int poolingSize, int poolingStride);
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
        /// this = yExpected in one-hot encoding (in each row there are exactly one '1' , all other values being 0)
        /// </summary>
        /// <param name="yPredicted">what has been predicted by the ML (in each row the biggest value is the ML favorite)</param>
        /// <param name="lossFunction"></param>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public abstract double ComputeLoss(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer);
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
