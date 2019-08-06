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
    public class InMemoryDataSetLoader<T> : IDataSetLoader<T> where T : struct
    {
        #region private fields
        private readonly CpuTensor<T> _x;
        private readonly CpuTensor<T> _y;
        private readonly Random _randForShuffle = new Random(0);
        private readonly ImageDataGenerator _imageDataGenerator;
        private readonly List<int> orderInCurrentEpoch;
        private CpuTensor<T> xCpuChunkBuffer = new CpuTensor<T>(new[] { 1 }, nameof(xCpuChunkBuffer));
        private CpuTensor<T> yCpuChunkBuffer = new CpuTensor<T>(new[] { 1 }, nameof(yCpuChunkBuffer));
        private int epochPreviousCall = -1;
        #endregion

        public CpuTensor<T> Y => _y;
        public InMemoryDataSetLoader(CpuTensor<T> x, CpuTensor<T> y, ImageDataGenerator imageDataGenerator)
        {
            Debug.Assert(AreCompatible_X_Y(x, y));
            _x = x;
            _y = y;
            if (!IsValidYSet(_y))
            {
                throw new Exception("Invalid Training Set 'y' : must contain only 0 and 1");
            }
            _imageDataGenerator = imageDataGenerator;
            orderInCurrentEpoch = Enumerable.Range(0, Count).ToList();
            TypeSize = Marshal.SizeOf(typeof(T));
        }
        public void Load(int epoch, bool isTraining, int indexFirstElement, int miniBatchSize, bool randomizeOrder, ref Tensor xChunkBuffer, ref Tensor yChunkBuffer)
        {
            Debug.Assert(xChunkBuffer.Shape[0] == miniBatchSize);
            Debug.Assert(xChunkBuffer.TypeSize == TypeSize);
            Debug.Assert(AreCompatible_X_Y(xChunkBuffer,yChunkBuffer));


            bool shouldShuffle = epoch >= 2 && randomizeOrder && isTraining;

            //change of epoch
            if (epochPreviousCall != epoch)
            {
                //if the first epoch (numEpoch == 1), there is no need to shuffle data
                if (shouldShuffle)
                {
                    Utils.Shuffle(orderInCurrentEpoch, _randForShuffle);
                }

                epochPreviousCall = epoch;
            }

            if (xChunkBuffer.UseGPU)
            {
                //we'll first create mini batch input in CPU, then copy them in GPU
                xCpuChunkBuffer.Reshape(XChunk_Shape(miniBatchSize));
                yCpuChunkBuffer.Reshape(YChunk_Shape(miniBatchSize));
            }
            else
            {
                xCpuChunkBuffer = xChunkBuffer.AsCpu<T>();
                yCpuChunkBuffer = yChunkBuffer.AsCpu<T>();
            }

            Debug.Assert(AreCompatible_X_Y(xCpuChunkBuffer, yCpuChunkBuffer));


            //for (int inputPictureIndex = 0; inputPictureIndex < miniBatchSize; ++inputPictureIndex){_imageDataGenerator.CreateSingleInputForEpoch(epoch, isTraining, _x, _y, xCpuChunkBuffer, yCpuChunkBuffer, shouldShuffle?orderInCurrentEpoch[indexFirstElement + inputPictureIndex]:(indexFirstElement + inputPictureIndex), inputPictureIndex, TypeSize);}
            Parallel.For(0, miniBatchSize, inputPictureIndex => _imageDataGenerator.CreateSingleInputForEpoch(epoch, isTraining, _x, _y, xCpuChunkBuffer, yCpuChunkBuffer, shouldShuffle?orderInCurrentEpoch[indexFirstElement + inputPictureIndex]:(indexFirstElement + inputPictureIndex), inputPictureIndex, TypeSize));

            if (xChunkBuffer.UseGPU)
            {
                xChunkBuffer.AsGPU<T>().CopyToDevice(xCpuChunkBuffer.HostPointer);
                yChunkBuffer.AsGPU<T>().CopyToDevice(yCpuChunkBuffer.HostPointer);
            }


            //uncomment to store data augmented pictures
            /*
            if (indexFirstElement == 0)
            { 
                var meanAndVolatilityOfEachChannel = new List<Tuple<double, double>> { Tuple.Create(125.306918046875, 62.9932192781369), Tuple.Create(122.950394140625, 62.0887076400142), Tuple.Create(113.865383183594, 66.7048996406309) };
                var xCpuChunkBytes = (xCpuChunkBuffer as CpuTensor<float>)?.Select((n, c, val) => (byte)((val*meanAndVolatilityOfEachChannel[c].Item2+ meanAndVolatilityOfEachChannel[c].Item1)));
                for (int i = indexFirstElement; i < Math.Min((indexFirstElement + miniBatchSize),10); ++i)
                {
                    PictureTools.SaveBitmap(xCpuChunkBytes, i, Path.Combine(@"C:\Users\fzibi\AppData\Local\SharpNet\Train"), i.ToString("D5")+"_epoch_" + epoch, "");
                }
            }
            */

        }
        public int Count => _x.Shape[0];
        public int TypeSize { get; }
        public int Categories => _y.Shape[1];
        public int Channels => _x.Shape[1];
        public int CurrentHeight => _x.Shape[2];
        public int CurrentWidth => _x.Shape[3];
        public IDataSetLoader<float> ToSinglePrecision()
        {
            if (this is IDataSetLoader<float>)
            {
                return (IDataSetLoader<float>) this;
            }
            return new InMemoryDataSetLoader<float>(_x.ToSinglePrecision(), _y.ToSinglePrecision(), _imageDataGenerator);
        }
        public IDataSetLoader<double> ToDoublePrecision()
        {
            if (this is IDataSetLoader<double>)
            {
                return (IDataSetLoader<double>)this;
            }
            return new InMemoryDataSetLoader<double>(_x.ToDoublePrecision(), _y.ToDoublePrecision(), _imageDataGenerator);
        }
        public int[] Y_Shape => _y.Shape;
        public int[] XChunk_Shape(int miniBatchSize)
        {
            var result = (int[])_x.Shape.Clone();
            result[0] = miniBatchSize;
            return result;
        }
        public int[] YChunk_Shape(int miniBatchSize)
        {
            var result = (int[])_y.Shape.Clone();
            result[0] = miniBatchSize;
            return result;
        }
        public override string ToString()
        {
            return _x + " => " + _y;
        }
        public void Dispose()
        {
            xCpuChunkBuffer?.Dispose();
            xCpuChunkBuffer = null;
            yCpuChunkBuffer?.Dispose();
            yCpuChunkBuffer = null;
        }

        private static bool IsValidYSet(Tensor data)
        {
            Debug.Assert(!data.UseGPU);
            if (data.UseDoublePrecision)
            {
                return data.AsDoubleCpuContent.All(IsValidY);
            }
            return data.AsFloatCpuContent.All(x => IsValidY(x));
        }
        private static bool AreCompatible_X_Y(Tensor X, Tensor Y)
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
        private static bool IsValidY(double x)
        {
            return Math.Abs(x) <= 1e-9 || Math.Abs(x - 1.0) <= 1e-9;
        }
    }
}