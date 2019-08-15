using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Pictures;

namespace SharpNet.Datasets
{
    public abstract class AbstractDataSetLoader<T> : IDataSetLoader<T> where T : struct
    {
        #region private & protected fields
        private CpuTensor<T> xInputCpuChunkBuffer = new CpuTensor<T>(new[] { 1 }, nameof(xInputCpuChunkBuffer));
        private CpuTensor<T> xOutputCpuChunkBuffer = new CpuTensor<T>(new[] { 1 }, nameof(xOutputCpuChunkBuffer));
        private CpuTensor<T> yOutputCpuChunkBuffer = new CpuTensor<T>(new[] { 1 }, nameof(yOutputCpuChunkBuffer));
        protected readonly string[] _categoryIdToDescription;
        protected int[] _elementIdToCategoryIdOrMinusOneIfUnknown;
        #endregion

        #region constructor
        protected AbstractDataSetLoader(int channels, int categories, string[] categoryIdToDescription)
        {
            Channels = channels;
            Categories = categories;
            _categoryIdToDescription = categoryIdToDescription;
            TypeSize = Marshal.SizeOf(typeof(T));
        }
        #endregion

        public abstract void LoadAt(int elementId, int indexInBuffer, CpuTensor<T> buffer);
        public void Load(int epoch, bool isTraining, int indexFirstElement, int miniBatchSize, IReadOnlyList<int> orderInCurrentEpoch, ImageDataGenerator imageDataGenerator, ref Tensor xChunkBuffer, ref Tensor yChunkBuffer)
        {
            Debug.Assert(xChunkBuffer.Shape[0] == miniBatchSize);
            Debug.Assert(xChunkBuffer.TypeSize == TypeSize);
            Debug.Assert(AreCompatible_X_Y(xChunkBuffer, yChunkBuffer));

            xInputCpuChunkBuffer.Reshape(XChunk_Shape(miniBatchSize));
            if (xChunkBuffer.UseGPU)
            {
                //we'll first create mini batch input in CPU, then copy them in GPU
                xOutputCpuChunkBuffer.Reshape(XChunk_Shape(miniBatchSize));
                yOutputCpuChunkBuffer.Reshape(YChunk_Shape(miniBatchSize));
            }
            else
            {
                xOutputCpuChunkBuffer = xChunkBuffer.AsCpu<T>();
                yOutputCpuChunkBuffer = yChunkBuffer.AsCpu<T>();
            }
            Debug.Assert(AreCompatible_X_Y(xOutputCpuChunkBuffer, yOutputCpuChunkBuffer));
            //for (int inputPictureIndex = 0; inputPictureIndex < miniBatchSize; ++inputPictureIndex){_imageDataGenerator.CreateSingleInputForEpoch(epoch, isTraining, _x, _y, xCpuChunkBuffer, yCpuChunkBuffer, shouldShuffle?orderInCurrentEpoch[indexFirstElement + inputPictureIndex]:(indexFirstElement + inputPictureIndex), inputPictureIndex, TypeSize);}
            Parallel.For(0, miniBatchSize, pictureIndexInOutputBuffer => imageDataGenerator.CreateSingleInputForEpoch(epoch, isTraining, this, orderInCurrentEpoch[indexFirstElement + pictureIndexInOutputBuffer], xInputCpuChunkBuffer, _elementIdToCategoryIdOrMinusOneIfUnknown, Y, pictureIndexInOutputBuffer, xOutputCpuChunkBuffer, yOutputCpuChunkBuffer, TypeSize));
            if (xChunkBuffer.UseGPU)
            {
                xChunkBuffer.AsGPU<T>().CopyToDevice(xOutputCpuChunkBuffer.HostPointer);
                yChunkBuffer.AsGPU<T>().CopyToDevice(yOutputCpuChunkBuffer.HostPointer);
            }

            /*
            //uncomment to store data augmented pictures
            if (isTraining&&(indexFirstElement == 0))
            { 
                var meanAndVolatilityOfEachChannel = new List<Tuple<double, double>> { Tuple.Create(125.306918046875, 62.9932192781369), Tuple.Create(122.950394140625, 62.0887076400142), Tuple.Create(113.865383183594, 66.7048996406309) };
                var xCpuChunkBytes = (xCpuChunkBuffer as CpuTensor<float>)?.Select((n, c, val) => (byte)((val*meanAndVolatilityOfEachChannel[c].Item2+ meanAndVolatilityOfEachChannel[c].Item1)));
                for (int i = indexFirstElement; i < Math.Min((indexFirstElement + miniBatchSize),100); ++i)
                {
                    PictureTools.SaveBitmap(xCpuChunkBytes, i, System.IO.Path.Combine(@"C:\Users\fzibi\AppData\Local\SharpNet\Train"), i.ToString("D5")+"_epoch_" + epoch, "");
                }
            }
            */

        }

        public abstract int Count { get; }
        public int Channels { get; }
        public int Categories { get; } 
        public string CategoryIdToDescription(int categoryId)
        {
            if (_categoryIdToDescription == null)
            {
                return categoryId.ToString();
            }
            return _categoryIdToDescription[categoryId];
        }
        public int ElementIdToCategoryId(int elementId)
        {
            return _elementIdToCategoryIdOrMinusOneIfUnknown[elementId];
        }
        /// <summary>
        /// retrieve the category associated with a specific element
        /// </summary>
        /// <param name="elementId">the id of the element, int the range [0, Count-1] </param>
        /// <returns>the associated category id, or -1 if the category is not known</returns>
        public virtual IDataSetLoader<float> ToSinglePrecision()
        {
            if (this is IDataSetLoader<float>)
            {
                return (IDataSetLoader<float>)this;
            }
            throw new NotImplementedException();
        }
        public virtual IDataSetLoader<double> ToDoublePrecision()
        {
            if (this is IDataSetLoader<double>)
            {
                return (IDataSetLoader<double>)this;
            }
            throw new NotImplementedException();
        }
        public abstract int CurrentHeight { get; }
        public abstract int CurrentWidth { get; }
        public int[] Y_Shape => new []{Count, Categories};

        public void Dispose()
        {
            xInputCpuChunkBuffer?.Dispose();
            xInputCpuChunkBuffer = null;
            xOutputCpuChunkBuffer?.Dispose();
            xOutputCpuChunkBuffer = null;
            yOutputCpuChunkBuffer?.Dispose();
            yOutputCpuChunkBuffer = null;
        }
        public int[] XChunk_Shape(int miniBatchSize)
        {
            return new []{ miniBatchSize , Channels, CurrentHeight, CurrentWidth};
        }
        public int[] YChunk_Shape(int miniBatchSize)
        {
            return new [] { miniBatchSize, Categories};
        }
        public int TypeSize { get; }
        public static bool AreCompatible_X_Y(Tensor X, Tensor Y)
        {
            if (X == null && Y == null)
            {
                return true;
            }
            return (X != null) && (Y != null)
                               && (X.UseDoublePrecision == Y.UseDoublePrecision)
                               && (X.UseGPU == Y.UseGPU)
                               && (X.Shape[0] == Y.Shape[0]) //same number of tests
                               && (Y.Shape.Length == 2);
        }
        public abstract CpuTensor<T> Y { get; }

      
        protected static bool IsValidYSet(Tensor data)
        {
            Debug.Assert(!data.UseGPU);
            if (data.UseDoublePrecision)
            {
                return data.AsDoubleCpuContent.All(IsValidY);
            }
            return data.AsFloatCpuContent.All(x => IsValidY(x));
        }

        private static bool IsValidY(double x)
        {
            return Math.Abs(x) <= 1e-9 || Math.Abs(x - 1.0) <= 1e-9;
        }
    }
}