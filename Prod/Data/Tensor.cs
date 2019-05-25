using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.CPU;
using SharpNet.GPU;

namespace SharpNet.Data
{
    [DebuggerDisplay("{ToString(true)}")]
    public abstract class Tensor : IDisposable
    {
        #region fields
        public int[] Shape { get; private set; }
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
        public void Dot(Tensor a, Tensor b) { Dot(a, false, b, false, 1.0, 0.0); }
        public int Height => Shape[0];
        public int Width => Shape.Length == 1 ? 1 : Shape[1];
        public int Count => Shape[0] * MultDim0;
        public int Dimension => Shape.Length;
        public bool UseDoublePrecision => TypeSize == 8;
        public bool UseSinglePrecision => !UseDoublePrecision;
        public ulong ReallyNeededMemoryInBytesForShape(int[] shape) { return (ulong)Utils.Product(shape) * (ulong)TypeSize; }
        public CpuTensor<double> AsDoubleCpu => AsCpu<double>();
        public CpuTensor<float> AsFloatCpu => AsCpu<float>();
        public double[] AsDoubleCpuContent => AsCpu<double>().Content;
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
            foreach (var d in ContentAsDoubleArray())
            {
                if (double.IsNaN(d))
                {
                    ++naNCount;
                    continue;
                }
                if (double.IsInfinity(d))
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
                    result += " " + infinityCount + " infinites";
                }
                result += ")";
            }
            return result;
        }
        public static implicit operator IntPtr(Tensor t)
        {
            return t.UseDoublePrecision ? t.AsGPU<double>().DevicePointer : t.AsGPU<float>().DevicePointer;
        }
        public CpuTensor<T> AsCpu<T>() where T : struct
        {
            if (this is CpuTensor<T> result)
            {
                return result;
            }
            throw new Exception("fail to convert " + this + " this to a CpuTensor<" + typeof(T)+">");
        }
        public GPUTensor<T> AsGPU<T>() where T : struct
        {
            if (this is GPUTensor<T> result)
            {
                return result;
            }
            throw new Exception("fail to convert " + this + " this to a GPUTensor<" + typeof(T)+">");
        }
        public GPUTensor<T> ToGPU<T>(GPUWrapper gpuWrapper) where T : struct
        {
            return UseGPU ? AsGPU<T>() : new GPUTensor<T>(Shape, AsCpu<T>().HostPointer, Description, gpuWrapper);
        }
        public void Reshape(int[] newShape)
        {
            if (Shape.SequenceEqual(newShape))
            {
                return;
            }
            Debug.Assert(ReallyNeededMemoryInBytesForShape(newShape) <= CapacityInBytes);
            Shape = newShape;
            RecomputeMultDim();
        }

        public static ulong OccupiedMemoryInBytes(IEnumerable<Tensor> tensors)
        {
            return (ulong)tensors.Select(x => (long) (x?.CapacityInBytes ?? 0) ).Sum();
        }
        //shapeInput                [batchSize x channelDepth x x.H x x.Weights]
        //shapeConvolution          [filtersCount x channelDepth x f x f]
        //shapeOutput               [batchSize x filtersCount x H[o] x Weights[o]]
        public static int[] ConvolutionOutputShape(int[] shapeInput, int[] shapeConvolution, int padding, int stride)
        {
            Debug.Assert(shapeInput.Length == 4);
            Debug.Assert(shapeConvolution.Length == 4);
            Debug.Assert(padding >= 0);
            Debug.Assert(stride >= 1);
            Debug.Assert(shapeInput[1] == shapeConvolution[1]); //same channel depth for x and convolution
            Debug.Assert(shapeConvolution[2] == shapeConvolution[3]); //convolution height == convolution width
            var f = shapeConvolution[2];
            Debug.Assert(f % 2 == 1); // F must be odd
            var hInput = shapeInput[2];
            var hOutput = (hInput - f + 2 * padding) / stride + 1;
            var wInput = shapeInput[3];
            var wOutput = (wInput - f + 2 * padding) / stride + 1;
            var batchSize = shapeInput[0];
            var filtersCount = shapeConvolution[0];
            return new[] { batchSize, filtersCount, hOutput, wOutput };
        }
        //shapeInput                (x.N, x.C, x.H, x.W)
        //pooling                   (poolingSize, poolingSize) with stride 'poolingStride'
        //shapeOutput               (x.N, x.C, y.H, y.W)
        public static int[] PoolingOutputShape(int[] shapeInput, int poolingSize, int poolingStride)
        {
            Debug.Assert(shapeInput.Length == 4);
            Debug.Assert(poolingStride >= 1);
            var hInput = shapeInput[2];
            var hOutput = (hInput - poolingSize) / poolingStride + 1;
            var wInput = shapeInput[3];
            var wOutput = (wInput - poolingSize) / poolingStride + 1;
            return new[] { shapeInput[0], shapeInput[1], hOutput, wOutput };
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
        public abstract ulong CapacityInBytes { get; }
        public abstract void ZeroMemory();
        //this = dy
        public abstract void ConvolutionBackwardBias(Tensor convolutionBackwardBias);
        // this = alpha a*b + beta*this
        public abstract void Dot(Tensor a, bool transposeA, Tensor b, bool transposeB, double alpha, double beta);
        //this = singleLineMatrix to add to y
        public abstract void BroadcastAddVectorToOutput(Tensor y);
        // compute: this = alpha * x + this
        public abstract void Update_Adding_Alpha_X(double alpha, Tensor x);
        // compute: this = alpha * x + beta * this
        public abstract void AddTensor(double alpha, Tensor x, double beta);
        /// <summary>
        /// Concatenate the 2 tensors 'a' & 'b'  (through the 'Channel' dimension) into the 'this' tensor.
        /// They must have exactly the same geometry apart from the number of channels (at index 1)
        /// 'this' : Tensor of Dimension [N, Ca+Cb, H, W]
        /// </summary>
        /// <param name="a">Tensor of Dimension [N, Ca, H, W]</param>
        /// <param name="b">Tensor of Dimension [N, Cb, H, W]</param>
        public abstract void Concatenate(Tensor a, Tensor b);
        /// <summary>
        /// Clone
        /// </summary>
        /// <param name="gpuWrapper"></param>
        /// <returns></returns>
        public abstract Tensor Clone(GPUWrapper gpuWrapper);
        /// <summary>
        /// Split the this tensor into the tensors 'a' & 'b'.
        /// They must have exactly the same geometry apart from the number of channels (at index 1)
        /// 'this' : Tensor of Dimension [N, Ca+Cb, H, W]
        /// </summary>
        /// <param name="a">Tensor of Dimension [N, Ca, H, W]</param>
        /// <param name="b">Tensor of Dimension [N, Cb, H, W]</param>
        public abstract void Split(Tensor a, Tensor b);
        public abstract void Update_Multiplying_By_Alpha(double alpha);
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
        public abstract void Dispose();
        //this = x
        public abstract void BatchNormalization(Tensor y, Tensor bnScale, Tensor bnBias, double exponentialAverageFactor, Tensor resultRunningMean, Tensor resultRunningVariance, cudnnBatchNormMode_t mode, double epsilon, Tensor resultSaveMean, Tensor resultSaveVariance, bool isTraining);
        //this = x
        public abstract void BatchNormalizationBackward(Tensor dy, Tensor dx, Tensor bnScale, Tensor resultBnScaleDiff, Tensor resultBnBiasDiff, cudnnBatchNormMode_t mode, double epsilon, Tensor resultSaveMean, Tensor resultSaveVariance);
        //this = x
        public abstract void DropoutForward(Tensor y, double dropProbability, bool isTraining, Random dropoutRandom, Tensor dropoutMaskBuffer);
        public abstract void DropoutBackward(Tensor dy, Tensor dx, double dropProbability, Tensor usedDropoutMask);
        //this = yExpected in one-hot encoding (in each row there are exactly one '1' , all other values being 0)
        //yPredicted : what has been predicted by the ML (in each row the biggest value is the ML favorite)
        public abstract int ComputeAccuracy(Tensor yPredicted, Tensor buffer);
        //this = yExpected in one-hot encoding (in each row there are exactly one '1' , all other values being 0)
        //yPredicted : what has been predicted by the ML (in each row the biggest value is the ML favorite)
        public abstract double ComputeLoss(Tensor yPredicted, NetworkConfig.LossFunctionEnum lossFunction, Tensor buffer);
        public abstract void RandomMatrixNormalDistribution(Random rand, double mean, double stdDev);
        public abstract void NewSameValueTensor(double sameValue);
        public abstract double[] ContentAsDoubleArray();
        public abstract float[] ContentAsFloatArray();
        protected static double[] ToDoubleArray(float[] data)
        {
            var result = new double[data.Length];
            for (int i = 0; i < data.Length; ++i)
            {
                result[i] = data[i];
            }
            return result;
        }
        protected static float[] ToFloatArray(double[] data)
        {
            var result = new float[data.Length];
            for (int i = 0; i < data.Length; ++i)
            {
                result[i] = (float)data[i];
            }
            return result;
        }
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
        protected void RecomputeMultDim()
        {
            _multDim2 = Shape.Length >= 4 ? Shape[3] : 1;
            MultDim1 = Shape.Length >= 3 ? Shape[2] * _multDim2 : 1;
            MultDim0 = Shape.Length >= 2 ? Shape[1] * MultDim1 : 1;
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

        private bool IsCompatible(Tensor a)
        {
            return (a != null && UseDoublePrecision == a.UseDoublePrecision && UseGPU == a.UseGPU);
        }
        private string ToString(bool displayStartofTensor)
        {
            var result = Description + "(" + string.Join(", ", Shape) + ")";
            result += UseGPU ? "" : "CPU";
            result += UseSinglePrecision ? "" : "x2";
            if (displayStartofTensor && !UseGPU)
            {
                if (UseDoublePrecision)
                {
                    result += "(" + string.Join(",", AsCpu<double>().Content.Take(3)) + ",...)";
                }
                else
                {
                    result += "(" + string.Join(",", AsCpu<float>().Content.Take(3)) + ",...)";
                }
            }

            return result;
        }
    }
}
