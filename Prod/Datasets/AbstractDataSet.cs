using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
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
        private CpuTensor<float> xOriginalNotAugmentedMiniBatchCpu = new CpuTensor<float>(new[] { 1 }, nameof(xOriginalNotAugmentedMiniBatchCpu));
        /// <summary>
        /// buffer with all augmented elements in the order needed for the current mini batch 
        /// </summary>
        private CpuTensor<float> xMiniBatchCpu = new CpuTensor<float>(new[] { 1 }, nameof(xMiniBatchCpu));
        private CpuTensor<float> yMiniBatchCpu = new CpuTensor<float>(new[] { 1 }, nameof(yMiniBatchCpu));
        /// <summary>
        /// the miniBatch Id associated with the above xMiniBatchCpu & yMiniBatchCpu tensors
        /// or -1 if those tensors are empty
        /// </summary>
        private long alreadyComputedMiniBatchId = -1;
        private readonly Random[] _rands;
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
            _rands = new Random[2 * Environment.ProcessorCount];
            for (int i = 0; i < _rands.Length; ++i)
            {
                _rands[i] = new Random(i);
            }

            if (UseBackgroundThread)
            {
                thread = new Thread(BackgroundThread);
                thread.Start();
            }
        }
        #endregion

        public abstract void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> buffer);
        public void Load(int epoch, bool isTraining,
            int[] shuffledElementId, int firstIndexInShuffledElementId,
            DataAugmentationConfig dataAugmentationConfig, Tensor xMiniBatch, Tensor yMiniBatch)
        {
            Debug.Assert(xMiniBatch != null);
            Debug.Assert(yMiniBatch != null);
            Debug.Assert(xMiniBatch.TypeSize == TypeSize);
            Debug.Assert(AreCompatible_X_Y(xMiniBatch, yMiniBatch));
            xMiniBatch.AssertIsNotDisposed();
            yMiniBatch.AssertIsNotDisposed();

            if (UseBackgroundThread)
            {
                //if a load is in progress, we'll wait until it finishes
                while (_backgroundThreadStatus != BackgroundThreadStatus.IDLE)
                {
                    Thread.Sleep(1);
                }
                //we initialize 'xMiniBatchCpu' & 'yMiniBatchCpu' tensors
                threadParameters = Tuple.Create(epoch, isTraining, shuffledElementId, firstIndexInShuffledElementId,
                    dataAugmentationConfig, xMiniBatch.Shape, yMiniBatch.Shape);
                _backgroundThreadStatus = BackgroundThreadStatus.ABOUT_TO_PROCESS_INPUT;
                while (_backgroundThreadStatus != BackgroundThreadStatus.IDLE)
                {
                    Thread.Sleep(1);
                }
            }
            else
            {
                LoadMiniBatchInCpu(epoch, isTraining, shuffledElementId, firstIndexInShuffledElementId, dataAugmentationConfig, xMiniBatch.Shape, yMiniBatch.Shape);
            }

            if (xMiniBatch.UseGPU)
            {
                xMiniBatch.AsGPU<float>().CopyToDevice(xMiniBatchCpu.HostPointer);
                yMiniBatch.AsGPU<float>().CopyToDevice(yMiniBatchCpu.HostPointer);
            }
            else
            {
                xMiniBatchCpu.CopyTo(xMiniBatch.AsCpu<float>());
                yMiniBatchCpu.CopyTo(yMiniBatch.AsCpu<float>());
            }

            //uncomment to store data augmented pictures
            //if (isTraining && (indexFirstElement == 0))
            //{
            //    var meanAndVolatilityOfEachChannel = new List<Tuple<double, double>> { Tuple.Create(125.306918046875, 62.9932192781369), Tuple.Create(122.950394140625, 62.0887076400142), Tuple.Create(113.865383183594, 66.7048996406309) };
            //    var xCpuChunkBytes = (xMiniBatchCpu as CpuTensor<float>)?.Select((n, c, val) => (byte)((val * meanAndVolatilityOfEachChannel[c].Item2 + meanAndVolatilityOfEachChannel[c].Item1)));
            //    for (int i = indexFirstElement; i < Math.Min((indexFirstElement + miniBatchSize), 100); ++i)
            //    {
            //        PictureTools.SaveBitmap(xCpuChunkBytes, i, Path.Combine(@"C:\Users\Franck\AppData\Local\SharpNet\Train"), i.ToString("D5") + "_epoch_" + epoch, "");
            //    }
            //}

            //we check if we can start to compute the next mini batch content in advance
            int firstIndexInShuffledElementIdForNextMiniBatch = firstIndexInShuffledElementId + xMiniBatch.Shape[0];
            int nextMiniBatchSize = Math.Min(shuffledElementId.Length - firstIndexInShuffledElementIdForNextMiniBatch, xMiniBatch.Shape[0]);
            if (UseBackgroundThread && nextMiniBatchSize > 0)
            {
                var xNextMiniBatchShape = (int[])xMiniBatch.Shape.Clone();
                var yNextMiniBatchShape = (int[])yMiniBatch.Shape.Clone();
                xNextMiniBatchShape[0] = yNextMiniBatchShape[0] = nextMiniBatchSize;
                threadParameters = Tuple.Create(epoch, isTraining, shuffledElementId, firstIndexInShuffledElementIdForNextMiniBatch, dataAugmentationConfig, xNextMiniBatchShape, yNextMiniBatchShape);
                _backgroundThreadStatus = BackgroundThreadStatus.ABOUT_TO_PROCESS_INPUT;
            }
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
            xOriginalNotAugmentedMiniBatchCpu?.Dispose();
            xOriginalNotAugmentedMiniBatchCpu = null;
            xMiniBatchCpu?.Dispose();
            xMiniBatchCpu = null;
            yMiniBatchCpu?.Dispose();
            yMiniBatchCpu = null;
            threadParameters = null;
            thread?.Abort();
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

        /// <summary>
        /// Load in 'xMiniBatchCpu' & 'yMiniBatchCpu' tensors the data related to the minni batch starting
        /// at 'firstIndexInShuffledElementId'
        /// </summary>
        /// <param name="epoch"></param>
        /// <param name="isTraining"></param>
        /// <param name="shuffledElementId"></param>
        /// <param name="firstIndexInShuffledElementId"></param>
        /// <param name="dataAugmentationConfig"></param>
        /// <param name="xMiniBatchShape"></param>
        /// <param name="yMiniBatchShape"></param>
        private void LoadMiniBatchInCpu(int epoch, bool isTraining,
            int[] shuffledElementId, int firstIndexInShuffledElementId,
            DataAugmentationConfig dataAugmentationConfig,
            int[] xMiniBatchShape, int[] yMiniBatchShape)
        {
            var miniBatchSize = xMiniBatchShape[0];

            var miniBatchId = ComputeMiniBatchHashId(shuffledElementId, firstIndexInShuffledElementId, miniBatchSize);
            if (miniBatchId == alreadyComputedMiniBatchId)
            {
                //nothing to do, the mini batch data is already stored in 'xMiniBatchCpu' & 'yMiniBatchCpu' tensors
                return;
            }

            //we initialize 'xOriginalNotAugmentedMiniBatchCpu' with all the original (not augmented elements)
            //contained in the mini batch
            xOriginalNotAugmentedMiniBatchCpu.Reshape(xMiniBatchShape);
            int MiniBatchIdxToElementId(int miniBatchIdx) => shuffledElementId[firstIndexInShuffledElementId + miniBatchIdx];
            Parallel.For(0, miniBatchSize, indexInBuffer => LoadAt(MiniBatchIdxToElementId(indexInBuffer), indexInBuffer, xOriginalNotAugmentedMiniBatchCpu));

            //we'll first create mini batch input in a local CPU buffer, then copy them in xMiniBatch/yMiniBatch
            xMiniBatchCpu.Reshape(xMiniBatchShape);
            yMiniBatchCpu.Reshape(yMiniBatchShape);

            Debug.Assert(AreCompatible_X_Y(xMiniBatchCpu, yMiniBatchCpu));
            int MiniBatchIdxToCategoryId(int miniBatchIdx) => ElementIdToCategoryId(MiniBatchIdxToElementId(miniBatchIdx));
            ImageStatistic MiniBatchIdxToImageStatistic(int miniBatchIdx) => ElementIdToImageStatistic(MiniBatchIdxToElementId(miniBatchIdx));

            //We compute the output y vector
            yMiniBatchCpu.ZeroMemory();
            for (int idx = 0; idx < miniBatchSize; ++idx)
            {
                var categoryId = MiniBatchIdxToCategoryId(idx);
                if (categoryId >= 0)
                {
                    yMiniBatchCpu.Set(idx, categoryId, 1);
                }
            }

            if (!dataAugmentationConfig.UseDataAugmentation || (epoch == 1) || !isTraining)
            {
                //we'll just copy the input picture from index 'inputPictureIndex' in 'inputEnlargedPictures' to index 'outputPictureIndex' of 'outputBufferPictures'
                xOriginalNotAugmentedMiniBatchCpu.CopyTo(xMiniBatchCpu);
            }
            else
            {
                var imageDataGenerator = new ImageDataGenerator(dataAugmentationConfig);
                Parallel.For(0, miniBatchSize, indexInMiniBatch => imageDataGenerator.DataAugmentationForMiniBatch(indexInMiniBatch, xOriginalNotAugmentedMiniBatchCpu, xMiniBatchCpu, yMiniBatchCpu, MiniBatchIdxToCategoryId, MiniBatchIdxToImageStatistic, MeanAndVolatilityForEachChannel, GetRandomForIndexInMiniBatch(indexInMiniBatch)));
            }
            alreadyComputedMiniBatchId = miniBatchId;
        }

        private static bool IsValidY(double x)
        {
            return Math.Abs(x) <= 1e-9 || Math.Abs(x - 1.0) <= 1e-9;
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
            var originalValue = (normalizedValue) * OriginalChannelVolatility(channel) + OriginalChannelMean(channel) + 0.5;
            originalValue = Math.Min(originalValue, 255);
            originalValue = Math.Max(originalValue, 0);
            return (byte)originalValue;
        }
        private Random GetRandomForIndexInMiniBatch(int indexInMiniBatch)
        {
            var rand = _rands[indexInMiniBatch % _rands.Length];
            return rand;
        }
        private static long ComputeMiniBatchHashId(int[] shuffledElementId, int firstIndexInShuffledElementId, int miniBatchSize)
        {
            long result = 1;
            for (int i = firstIndexInShuffledElementId; i < firstIndexInShuffledElementId + miniBatchSize; ++i)
            {
                result += (i + 1) * shuffledElementId[i];
            }
            return result;
        }

        #region Processing Thread management
        private bool UseBackgroundThread => true;
        private readonly Thread thread;
        private enum BackgroundThreadStatus { IDLE, ABOUT_TO_PROCESS_INPUT, PROCESSING_INPUT };
        private BackgroundThreadStatus _backgroundThreadStatus = BackgroundThreadStatus.IDLE;
        private Tuple<int, bool, int[], int, DataAugmentationConfig, int[], int[]> threadParameters;
        private void BackgroundThread()
        {
            for (; ; )
            {
                while (_backgroundThreadStatus != BackgroundThreadStatus.ABOUT_TO_PROCESS_INPUT)
                {
                    Thread.Sleep(1);
                }
                _backgroundThreadStatus = BackgroundThreadStatus.PROCESSING_INPUT;
                Debug.Assert(threadParameters != null);
                LoadMiniBatchInCpu(threadParameters.Item1, threadParameters.Item2, threadParameters.Item3, threadParameters.Item4, threadParameters.Item5, threadParameters.Item6, threadParameters.Item7);
                threadParameters = null;
                _backgroundThreadStatus = BackgroundThreadStatus.IDLE;
            }
        }
        #endregion
    }
}