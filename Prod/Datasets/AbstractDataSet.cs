using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.Pictures;

namespace SharpNet.Datasets
{
    public abstract class AbstractDataSet : IDataSet
    {
        #region private & protected fields
        /// <summary>
        /// buffer with all original elements (no data augmentation) in the order needed for the current mini batch 
        /// </summary>
        private CpuTensor<float> xInputCpuChunkBuffer = new CpuTensor<float>(new[] { 1 }, nameof(xInputCpuChunkBuffer));
        /// <summary>
        /// buffer with all augmented elements in the order needed for the current mini batch 
        /// </summary>
        private CpuTensor<float> xOutputCpuChunkBuffer = new CpuTensor<float>(new[] { 1 }, nameof(xOutputCpuChunkBuffer));
        private CpuTensor<float> yOutputCpuChunkBuffer = new CpuTensor<float>(new[] { 1 }, nameof(yOutputCpuChunkBuffer));
        /// <summary>
        /// the mean and volatility used to normalize the 'this' DataSet
        /// will be null or empty if no normalization occured in the DataSet
        /// </summary>
        protected List<Tuple<float, float>> _meanAndVolatilityForEachChannel;
        #endregion

        #region constructor
        protected AbstractDataSet(string name, int channels, int categories, List<Tuple<float, float>> meanAndVolatilityForEachChannel)
        {
            Name = name;
            Channels = channels;
            Categories = categories;
            _meanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
        }
        #endregion

        public abstract void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> buffer);
        public void Load(int epoch, bool isTraining, int indexFirstElement, IReadOnlyList<int> indexInCurrentEpochToElementId,
            ImageDataGenerator imageDataGenerator, ref Tensor xChunkBuffer, ref Tensor yChunkBuffer)
        {
            xChunkBuffer.AssertIsNotDisposed();
            yChunkBuffer.AssertIsNotDisposed();
            var miniBatchSize = xChunkBuffer.Shape[0];
            var xMiniBatchShape = xChunkBuffer.Shape;
            var yMiniBatchShape = yChunkBuffer.Shape;

            Debug.Assert(xChunkBuffer.TypeSize == TypeSize);
            Debug.Assert(AreCompatible_X_Y(xChunkBuffer, yChunkBuffer));

            //we initialize 'xInputCpuChunkBuffer' with all the input data in the given order
            xInputCpuChunkBuffer.Reshape(xMiniBatchShape);
            //We copy the element Id in the input buffer 'xInputBufferPictures' at index 'indexInBuffer'
            Parallel.For(0, miniBatchSize, indexInBuffer => LoadAt(indexInCurrentEpochToElementId[indexFirstElement + indexInBuffer], indexInBuffer, xInputCpuChunkBuffer));

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
            int IndexInMiniBatchToCategoryId(int miniBatchIdx) => ElementIdToCategoryId(indexInCurrentEpochToElementId[indexFirstElement + miniBatchIdx]);
            ImageStatistic IndexInMiniBatchToImageStatistic(int miniBatchIdx) => ElementIdToImageStatistic(indexInCurrentEpochToElementId[indexFirstElement + miniBatchIdx]);

            //We compute the output y vector
            yOutputCpuChunkBuffer.ZeroMemory();
            for (int idx = 0; idx < miniBatchSize; ++idx)
            {
                var categoryId = IndexInMiniBatchToCategoryId(idx);
                if (categoryId >= 0)
                {
                    yOutputCpuChunkBuffer.Set(idx, categoryId, 1);
                }
            }

            if (!imageDataGenerator.UseDataAugmentation || (epoch == 1) || !isTraining)
            {
                //we'll just copy the input picture from index 'inputPictureIndex' in 'inputEnlargedPictures' to index 'outputPictureIndex' of 'outputBufferPictures'
                xInputCpuChunkBuffer.CopyTo(xOutputCpuChunkBuffer);
            }
            else
            {
                Parallel.For(0, miniBatchSize, indexInMiniBatch => imageDataGenerator.DataAugmentationForMiniBatch(indexInMiniBatch, xInputCpuChunkBuffer, xOutputCpuChunkBuffer, yOutputCpuChunkBuffer, IndexInMiniBatchToCategoryId, IndexInMiniBatchToImageStatistic, MeanAndVolatilityForEachChannel));
            }

            if (xChunkBuffer.UseGPU)
            {
                xChunkBuffer.AsGPU<float>().CopyToDevice(xOutputCpuChunkBuffer.HostPointer);
                yChunkBuffer.AsGPU<float>().CopyToDevice(yOutputCpuChunkBuffer.HostPointer);
            }

            //uncomment to store data augmented pictures
            //if (isTraining && (indexFirstElement == 0))
            //{
            //    var meanAndVolatilityOfEachChannel = new List<Tuple<double, double>> { Tuple.Create(125.306918046875, 62.9932192781369), Tuple.Create(122.950394140625, 62.0887076400142), Tuple.Create(113.865383183594, 66.7048996406309) };
            //    var xCpuChunkBytes = (xOutputCpuChunkBuffer as CpuTensor<float>)?.Select((n, c, val) => (byte)((val * meanAndVolatilityOfEachChannel[c].Item2 + meanAndVolatilityOfEachChannel[c].Item1)));
            //    for (int i = indexFirstElement; i < Math.Min((indexFirstElement + miniBatchSize), 100); ++i)
            //    {
            //        PictureTools.SaveBitmap(xCpuChunkBytes, i, Path.Combine(@"C:\Users\Franck\AppData\Local\SharpNet\Train"), i.ToString("D5") + "_epoch_" + epoch, "");
            //    }
            //}

        }

        /// <summary>
        ///dimension of a single element in the training data (in shape (channels,height, width)
        /// </summary>
        public int[] InputShape_CHW => new[] { Channels, Height, Width };


        /// <summary>
        /// original content (no data augmentation/no normalization) of the element at index 'elementId'
        /// </summary>
        /// <param name="elementId">the index of element to retrieve (between 0 and Count-1) </param>
        /// <returns>a byte tensor containing the element at index 'elementId' </returns>
        public virtual BitmapContent OriginalElementContent(int elementId)
        {
            var buffer = LoadSingleElement(elementId);
            var result = new BitmapContent(InputShape_CHW, new byte[Channels * Height * Width], ElementIdToDescription(elementId));
            for (int channel = 0; channel < Channels; ++channel)
            {
                for (int row = 0; row < Height; ++row)
                {
                    for (int col = 0; col < Width; ++col)
                    {
                        var originalValue = OriginalValue(buffer.Get(channel, row, col), channel);
                        result.Set(channel, row, col, originalValue);
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// TODO : check if we should use a cache to avoid recomputing the image stat at each epoch
        /// </summary>
        /// <param name="elementId"></param>
        /// <returns></returns>
        public ImageStatistic ElementIdToImageStatistic(int elementId)
        {
            return ImageStatistic.ValueOf(OriginalElementContent(elementId));
        }

        /// <summary>
        /// return the element at index 'elementId'
        /// </summary>
        /// <param name="elementId">index of the element to load (between 0 and Count-1) </param>
        /// <returns>the element at index 'elementId'</returns>
        private CpuTensor<float> LoadSingleElement(int elementId)
        {
            var result = new CpuTensor<float>(new[] { 1, Channels, Height, Width }, ElementIdToDescription(elementId));
            LoadAt(elementId, 0, result);
            result.Reshape(InputShape_CHW);
            return result;
        }

        /// <summary>
        /// return the original (unnormalized) value of 'normalizedValue' that is located in channel 'channel'
        /// </summary>
        /// <param name="normalizedValue"></param>
        /// <param name="channel"></param>
        /// <returns></returns>
        private byte OriginalValue(float normalizedValue, int channel)
        {
            if (!IsNormalized)
            {
                //no normalization was performed on the input
                return (byte)normalizedValue;
            }
            var originalValue = (normalizedValue) * OriginalChannelVolatility(channel) + OriginalChannelMean(channel)+0.5;
            originalValue = Math.Min(originalValue, 255);
            originalValue = Math.Max(originalValue, 0);
            return (byte)originalValue;
        }

        public abstract int Count { get; }
        public abstract string ElementIdToDescription(int elementId);
        public string Name { get; }
        public int Channels { get; }
        public int Categories { get; }

        public List<Tuple<float, float>> MeanAndVolatilityForEachChannel => _meanAndVolatilityForEachChannel;


        public double OriginalChannelMean(int channel)
        {
            Debug.Assert(IsNormalized);
            return MeanAndVolatilityForEachChannel[channel].Item1;
        }
        public double OriginalChannelVolatility(int channel)
        {
            Debug.Assert(IsNormalized);
            return MeanAndVolatilityForEachChannel[channel].Item2;
        }

        public bool IsNormalized => MeanAndVolatilityForEachChannel != null && MeanAndVolatilityForEachChannel.Count != 0;

        public abstract string CategoryIdToDescription(int categoryId);
        /// <summary>
        /// retrieve the category associated with a specific element
        /// </summary>
        /// <param name="elementId">the id of the element, int the range [0, Count-1] </param>
        /// <returns>the associated category id, or -1 if the category is not known</returns>
        public abstract int ElementIdToCategoryId(int elementId);
        public abstract int Height { get; }
        public abstract int Width { get; }
        public int[] Y_Shape => new[] { Count, Categories };
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
            return new[] { miniBatchSize, Channels, Height, Width };
        }
        public int TypeSize => 4; //float size

        public ITrainingAndTestDataSet SplitIntoTrainingAndValidation(double percentageInTrainingSet)
        {
            int lastElementIdIncludedInTrainingSet = (int)(percentageInTrainingSet * Count);
            var training = new SubDataSet(this, id => id < lastElementIdIncludedInTrainingSet);
            var test = new SubDataSet(this, id => id >= lastElementIdIncludedInTrainingSet);
            return new TrainingAndTestDataLoader(training, test, Name);
        }
        public AbstractDataSet Shuffle(Random rand)
        {
            return new ShuffledDataSet(this, rand);
        }
        public AbstractDataSet Take(int count)
        {
            return new SubDataSet(this, elementId => elementId < count);
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
        public void CreatePredictionFile(CpuTensor<float> prediction, string outputFile, string headerIfAny = null)
        {
            var categories = prediction.ComputePrediction();
            File.Delete(outputFile);
            if (!string.IsNullOrEmpty(headerIfAny))
            {
                File.AppendAllText(outputFile, headerIfAny + Environment.NewLine);
            }
            for (int elementId = 0; elementId < Count; ++elementId)
            {
                File.AppendAllText(outputFile, ElementIdToDescription(elementId) + "," + categories[elementId] + Environment.NewLine);
            }
        }

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