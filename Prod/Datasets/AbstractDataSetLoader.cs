using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Pictures;

namespace SharpNet.Datasets
{
    public abstract class AbstractDataSetLoader : IDataSetLoader
    {
        #region private & protected fields
        private CpuTensor<float> xInputCpuChunkBuffer = new CpuTensor<float>(new[] { 1 }, nameof(xInputCpuChunkBuffer));
        private CpuTensor<float> xOutputCpuChunkBuffer = new CpuTensor<float>(new[] { 1 }, nameof(xOutputCpuChunkBuffer));
        private CpuTensor<float> yOutputCpuChunkBuffer = new CpuTensor<float>(new[] { 1 }, nameof(yOutputCpuChunkBuffer));
        #endregion

        #region constructor
        protected AbstractDataSetLoader(int channels, int categories)
        {
            Channels = channels;
            Categories = categories;
        }
        #endregion

        public abstract void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> buffer);
        public void Load(int epoch, bool isTraining, int indexFirstElement, IReadOnlyList<int> elementIdToOrderInCurrentEpoch,
            ImageDataGenerator imageDataGenerator, ref Tensor xChunkBuffer, ref Tensor yChunkBuffer)
        {
            var miniBatchSize = xChunkBuffer.Shape[0];
            var xMiniBatchShape = xChunkBuffer.Shape;
            var yMiniBatchShape = yChunkBuffer.Shape;

            Debug.Assert(xChunkBuffer.TypeSize == TypeSize);
            Debug.Assert(AreCompatible_X_Y(xChunkBuffer, yChunkBuffer));

            //we initialize 'xInputCpuChunkBuffer' with all the input data in the given order
            xInputCpuChunkBuffer.Reshape(xMiniBatchShape);
            //We copy the element Id in the input buffer 'xInputBufferPictures' at index 'indexInBuffer'
            Parallel.For(0, miniBatchSize, indexInBuffer => LoadAt(elementIdToOrderInCurrentEpoch[indexFirstElement + indexInBuffer], indexInBuffer, xInputCpuChunkBuffer));

            if (xChunkBuffer.UseGPU)
            {
                //we'll first create mini batch input in CPU, then copy them in GPU
                xOutputCpuChunkBuffer.Reshape(xMiniBatchShape);
                yOutputCpuChunkBuffer.Reshape(yMiniBatchShape);
            }
            else
            {
                xOutputCpuChunkBuffer = xChunkBuffer.AsCpu<float>();
                yOutputCpuChunkBuffer = yChunkBuffer.AsCpu<float>();
            }
            Debug.Assert(AreCompatible_X_Y(xOutputCpuChunkBuffer, yOutputCpuChunkBuffer));
            int IndexInMiniBatchToCategoryId(int miniBatchIdx) => ElementIdToCategoryId(elementIdToOrderInCurrentEpoch[indexFirstElement + miniBatchIdx]);

            //We compute the output y vector
            yOutputCpuChunkBuffer.ZeroMemory();
            for (int idx = 0; idx < miniBatchSize; ++idx)
            {
                yOutputCpuChunkBuffer.Set(idx, IndexInMiniBatchToCategoryId(idx), 1);
            }
            if (!imageDataGenerator.UseDataAugmentation || (epoch == 1) || !isTraining)
            {
                //we'll just copy the input picture from index 'inputPictureIndex' in 'inputEnlargedPictures' to index 'outputPictureIndex' of 'outputBufferPictures'
                xInputCpuChunkBuffer.CopyTo(xOutputCpuChunkBuffer);
            }
            else
            {
                Parallel.For(0, miniBatchSize, indexInMiniBatch => imageDataGenerator.DataAugmentationForMiniBatch(indexInMiniBatch, xInputCpuChunkBuffer, xOutputCpuChunkBuffer, yOutputCpuChunkBuffer, IndexInMiniBatchToCategoryId));
            }

            if (xChunkBuffer.UseGPU)
            {
                xChunkBuffer.AsGPU<float>().CopyToDevice(xOutputCpuChunkBuffer.HostPointer);
                yChunkBuffer.AsGPU<float>().CopyToDevice(yOutputCpuChunkBuffer.HostPointer);
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

        /// <summary>
        ///dimension of a single element in the training data (in shape (channels,height, width)
        /// </summary>
        public int[] InputShape_CHW => new[] { Channels, Height, Width};

        public abstract int Count { get; }
        public abstract string ElementIdToDescription(int elementId);
        public int Channels { get; }
        public int Categories { get; }
        public abstract string CategoryIdToDescription(int categoryId);
        /// <summary>
        /// retrieve the category associated with a specific element
        /// </summary>
        /// <param name="elementId">the id of the element, int the range [0, Count-1] </param>
        /// <returns>the associated category id, or -1 if the category is not known</returns>
        public abstract int ElementIdToCategoryId(int elementId);
        public abstract int Height { get; }
        public abstract int Width { get; }
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
        public int[] XMiniBatch_Shape(int miniBatchSize)
        {
            return new []{ miniBatchSize , Channels, Height, Width};
        }
        public int TypeSize => 4; //float size

        public IDataSet SplitIntoTrainingAndValidation(double percentageInTrainingSet)
        {
            int lastElementIdIncludedInTrainingSet = (int)(percentageInTrainingSet * Count);
            var training = new SubDataSetLoader(this, id => id < lastElementIdIncludedInTrainingSet);
            var test = new SubDataSetLoader(this, id => id >= lastElementIdIncludedInTrainingSet);
            return new DataLoader(training, test);
        }
        public AbstractDataSetLoader Shuffle(Random rand)
        {
            return new ShuffledDataSetLoader(this, rand);
        }
        public AbstractDataSetLoader Take(int count)
        {
            return new SubDataSetLoader(this, elementId => elementId<count);
        }
        public static bool AreCompatible_X_Y(Tensor X, Tensor Y)
        {
            if (X == null && Y == null)
            {
                return true;
            }
            return (X != null) && (Y != null)
                               && (X.UseGPU == Y.UseGPU)
                               && (X.Shape[0] == Y.Shape[0]) //same number of tests
                               && (Y.Shape.Length == 2);
        }
        public abstract CpuTensor<float> Y { get; }

        public static string[] DefaultGetCategoryIdToDescription(int categoryCount)
        {
            return Enumerable.Range(0, categoryCount).Select(x => x.ToString()).ToArray();
        }
        protected static bool IsValidYSet(Tensor data)
        {
            Debug.Assert(!data.UseGPU);
            return data.AsFloatCpuContent.All(x => IsValidY(x));
        }


   
        private static bool IsValidY(double x)
        {
            return Math.Abs(x) <= 1e-9 || Math.Abs(x - 1.0) <= 1e-9;
        }
    }
}