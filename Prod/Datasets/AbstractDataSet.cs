using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http.Headers;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
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
        private CpuTensor<float> xBufferMiniBatchCpu = new CpuTensor<float>(new[] { 1 }, nameof(xBufferMiniBatchCpu));
        private CpuTensor<float> yBufferMiniBatchCpu = new CpuTensor<float>(new[] { 1 }, nameof(yBufferMiniBatchCpu));
        /// <summary>
        /// the miniBatch Id associated with the above xBufferMiniBatchCpu & yBufferMiniBatchCpu tensors
        /// or -1 if those tensors are empty
        /// </summary>
        private long alreadyComputedMiniBatchId = -1;
        private readonly Random[] _rands;
        /// <summary>
        /// the mean and volatility used to normalize the 'this' DataSet
        /// will be null or empty if no normalization occured in the DataSet
        /// </summary>
        protected List<Tuple<float, float>> _meanAndVolatilityForEachChannel;

        public Logger Logger { get; }

        #endregion

        #region constructor
        protected AbstractDataSet(string name, int channels, int categories, List<Tuple<float, float>> meanAndVolatilityForEachChannel, Logger logger)
        {
            Name = name;
            Channels = channels;
            Categories = categories;
            _meanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
            Logger = logger ?? Logger.ConsoleLogger;
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

        public abstract void LoadAt(int elementId, int indexInBuffer, [NotNull] CpuTensor<float> xBuffer, [CanBeNull] CpuTensor<float> yBuffer);
        public void LoadMiniBatch(int epoch, bool isTraining,
            int[] shuffledElementId, int firstIndexInShuffledElementId,
            DataAugmentationConfig dataAugmentationConfig, CpuTensor<float> xMiniBatch, CpuTensor<float> yMiniBatch)
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
                //we initialize 'xBufferMiniBatchCpu' & 'yBufferMiniBatchCpu' tensors
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

            xBufferMiniBatchCpu.CopyTo(xMiniBatch.AsCpu<float>());
            yBufferMiniBatchCpu.CopyTo(yMiniBatch.AsCpu<float>());

            //uncomment to store data augmented pictures
            //if (isTraining && (firstIndexInShuffledElementId == 0))
            //{
            //    //CIFAR10
            //    //var meanAndVolatilityOfEachChannel = new List<Tuple<double, double>> { Tuple.Create(125.306918046875, 62.9932192781369), Tuple.Create(122.950394140625, 62.0887076400142), Tuple.Create(113.865383183594, 66.7048996406309) };
            //    //SVHN
            //    var meanAndVolatilityOfEachChannel = new List<Tuple<float, float>> { Tuple.Create(109.8823f, 50.11187f), Tuple.Create(109.7114f, 50.57312f), Tuple.Create(113.8187f, 50.85124f) };
            //    var xCpuChunkBytes = xBufferMiniBatchCpu.Select((n, c, val) => (byte)((val * meanAndVolatilityOfEachChannel[c].Item2 + meanAndVolatilityOfEachChannel[c].Item1)));
            //    for (int i = firstIndexInShuffledElementId; i < Math.Min((firstIndexInShuffledElementId + xMiniBatch.Shape[0]), 100); ++i)
            //    {
            //        int elementId = shuffledElementId[i];
            //        var categoryIndex = ElementIdToCategoryIndex(elementId);
            //        PictureTools.SaveBitmap(xCpuChunkBytes, i, Path.Combine(@"C:\Users\Franck\AppData\Local\SharpNet\Train"), elementId.ToString("D5") + "_cat" + categoryIndex + "_epoch_" + epoch.ToString("D3"), "");
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


        public static CpuTensor<float> ToXWorkingSet(CpuTensor<byte> x, List<Tuple<float, float>> meanAndVolatilityOfEachChannel)
        {
            var xWorkingSet = x.Select((n, c, val) => (float)((val - meanAndVolatilityOfEachChannel[c].Item1) / Math.Max(meanAndVolatilityOfEachChannel[c].Item2, 1e-9)));
            //xWorkingSet = x.Select((n, c, val) => (float)val/255f);
            return xWorkingSet;
        }

        public static CpuTensor<float> ToYWorkingSet(CpuTensor<byte> categoryBytes, int categories , Func<byte, int> categoryByteToCategoryIndex)
        {
            Debug.Assert(categoryBytes.MultDim0 == 1);
            var batchSize = categoryBytes.Shape[0];
            var newShape = new[] { batchSize, categories };
            var newY = new CpuTensor<float>(newShape, categoryBytes.Description);
            for (int n = 0; n < batchSize; ++n)
            {
                newY.Set(n, categoryByteToCategoryIndex(categoryBytes.Get(n, 0)), 1f);
            }
            return newY;
        }


        /// <summary>
        /// original content (no data augmentation/no normalization) of the element at index 'elementId'
        /// </summary>
        /// <param name="elementId">the index of element to retrieve (between 0 and Count-1) </param>
        /// <returns>a byte tensor containing the element at index 'elementId' </returns>
        public virtual BitmapContent OriginalElementContent(int elementId)
        {
            var buffer = LoadSingleElement(elementId);
            var result = new BitmapContent(InputShape_CHW, new byte[Channels * Height * Width], elementId.ToString());
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

        /// <summary>
        /// retrieve the category associated with a specific element
        /// </summary>
        /// <param name="elementId">the id of the element, int the range [0, Count-1] </param>
        /// <returns>the associated category id, or -1 if the category is not known</returns>
        public abstract int ElementIdToCategoryIndex(int elementId);
        public abstract int Height { get; }
        public abstract int Width { get; }
        public int[] Y_Shape => new[] { Count, Categories };
        public void Dispose()
        {
            xOriginalNotAugmentedMiniBatchCpu?.Dispose();
            xOriginalNotAugmentedMiniBatchCpu = null;
            xBufferMiniBatchCpu?.Dispose();
            xBufferMiniBatchCpu = null;
            yBufferMiniBatchCpu?.Dispose();
            yBufferMiniBatchCpu = null;
            threadParameters = null;
            for (int i = 0; i < 1000; ++i)
            {
                _backgroundThreadStatus = BackgroundThreadStatus.TO_ABORT;
                if (thread == null || !thread.IsAlive)
                {
                    break;
                }
                Thread.Sleep(1);
                if (i + 1 == 1000)
                {
                    Logger.Info("fail to stop BackgroundThread in "+Name);
                }
            }
        }
        public int[] XMiniBatch_Shape(int miniBatchSize)
        {
            return new[] { miniBatchSize, Channels, Height, Width };
        }
        public int[] YMiniBatch_Shape(int miniBatchSize)
        {
            return new[] { miniBatchSize, Categories };
        }

        public int TypeSize => 4; //float size

        public ITrainingAndTestDataSet SplitIntoTrainingAndValidation(double percentageInTrainingSet)
        {
            int lastElementIdIncludedInTrainingSet = (int)(percentageInTrainingSet * Count);
            var training = new SubDataSet(this, id => id < lastElementIdIncludedInTrainingSet);
            var test = new SubDataSet(this, id => id >= lastElementIdIncludedInTrainingSet);
            return new TrainingAndTestDataLoader(training, test, this);
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
                File.AppendAllText(outputFile, elementId + "," + categories[elementId] + Environment.NewLine);
            }
        }

        public static string[] DefaultGetCategoryIndexToDescription(int categoryCount)
        {
            return Enumerable.Range(0, categoryCount).Select(x => x.ToString()).ToArray();
        }

        protected static bool IsValidYSet(Tensor data)
        {
            Debug.Assert(!data.UseGPU);
            return data.AsFloatCpuContent.All(x => IsValidY(x));
        }
        protected List<Tuple<float, float>> Sum_SumSquare_Count_to_ComputeMeanAndVolatilityForEachChannel(float[] sumSumSquareCountForEachChannel)
        {
            const int DistinctValuesToComputeInEachChannel = 3; //sum + sumSquare + count
            var result = new List<Tuple<float, float>>();
            for (int channel = 0; channel < Channels; ++channel)
            {
                var sum = sumSumSquareCountForEachChannel[DistinctValuesToComputeInEachChannel * channel];
                var sumSquare = sumSumSquareCountForEachChannel[DistinctValuesToComputeInEachChannel * channel + 1];
                var count = sumSumSquareCountForEachChannel[DistinctValuesToComputeInEachChannel * channel + 2];
                var mean = (sum / count);
                var variance = (sumSquare / count) - mean * mean;
                var volatility = (float)Math.Sqrt(Math.Max(0, variance));
                Logger?.Info("Mean and volatility for channel#" + channel + " : " + mean.ToString(CultureInfo.InvariantCulture) + " ; " + volatility.ToString(CultureInfo.InvariantCulture));
                result.Add(Tuple.Create(mean, volatility));
            }
            return result;
        }
        protected void UpdateStatus(ref int nbPerformed)
        {
            int delta = Math.Max(Count / 100, 1);
            var newNbPerformed = Interlocked.Increment(ref nbPerformed);
            if ((newNbPerformed % delta == 0) || (newNbPerformed == Count))
            {
                Logger.Info("Done: " + (100 * newNbPerformed) / Count + "%");
            }
        }
        protected void UpdateWith_Sum_SumSquare_Count_For_Each_Channel(int elementId, float[] _sum_SumSquare_Count_For_Each_Channel, bool ignoreZeroPixel, ref int nbPerformed)
        {
            UpdateStatus(ref nbPerformed);
            OriginalElementContent(elementId).UpdateWith_Sum_SumSquare_Count_For_Each_Channel(_sum_SumSquare_Count_For_Each_Channel, ignoreZeroPixel);
        }




        /// <summary>
        /// Load in 'xBufferMiniBatchCpu' & 'yBufferMiniBatchCpu' tensors the data related to the minni batch starting
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
                //nothing to do, the mini batch data is already stored in 'xBufferMiniBatchCpu' & 'yBufferMiniBatchCpu' tensors
                return;
            }

            //we initialize 'xOriginalNotAugmentedMiniBatchCpu' with all the original (not augmented elements)
            //contained in the mini batch
            xOriginalNotAugmentedMiniBatchCpu.Reshape(xMiniBatchShape);
            //we'll first create mini batch input in a local CPU buffer, then copy them in xMiniBatch/yMiniBatch
            xBufferMiniBatchCpu.Reshape(xMiniBatchShape);
            yBufferMiniBatchCpu.Reshape(yMiniBatchShape);
            yBufferMiniBatchCpu.ZeroMemory();

            int MiniBatchIdxToElementId(int miniBatchIdx) => shuffledElementId[firstIndexInShuffledElementId + miniBatchIdx];
            Parallel.For(0, miniBatchSize, indexInBuffer => LoadAt(MiniBatchIdxToElementId(indexInBuffer), indexInBuffer, xOriginalNotAugmentedMiniBatchCpu, yBufferMiniBatchCpu));

            Debug.Assert(AreCompatible_X_Y(xBufferMiniBatchCpu, yBufferMiniBatchCpu));
            int MiniBatchIdxToCategoryIndex(int miniBatchIdx) => ElementIdToCategoryIndex(MiniBatchIdxToElementId(miniBatchIdx));
            ImageStatistic MiniBatchIdxToImageStatistic(int miniBatchIdx) => ElementIdToImageStatistic(MiniBatchIdxToElementId(miniBatchIdx));

    

            if (!dataAugmentationConfig.UseDataAugmentation || (epoch == 1) || !isTraining)
            {
                //we'll just copy the input picture from index 'inputPictureIndex' in 'inputEnlargedPictures' to index 'outputPictureIndex' of 'outputBufferPictures'
                xOriginalNotAugmentedMiniBatchCpu.CopyTo(xBufferMiniBatchCpu);
            }
            else
            {
                var imageDataGenerator = new ImageDataGenerator(dataAugmentationConfig);
                Parallel.For(0, miniBatchSize, indexInMiniBatch => imageDataGenerator.DataAugmentationForMiniBatch(indexInMiniBatch, xOriginalNotAugmentedMiniBatchCpu, xBufferMiniBatchCpu, yBufferMiniBatchCpu, MiniBatchIdxToCategoryIndex, MiniBatchIdxToImageStatistic, MeanAndVolatilityForEachChannel, GetRandomForIndexInMiniBatch(indexInMiniBatch)));
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
            var xBuffer = new CpuTensor<float>(new[] { 1, Channels, Height, Width }, elementId.ToString());
            LoadAt(elementId, 0, xBuffer, null);
            xBuffer.Reshape(InputShape_CHW);
            return xBuffer;
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
        // same speed on CIFAR10 with UseBackgroundThread set to either true of false (tested on 5-jan-2020)
        private bool UseBackgroundThread => true;
        private readonly Thread thread;
        private enum BackgroundThreadStatus { IDLE, ABOUT_TO_PROCESS_INPUT, PROCESSING_INPUT, TO_ABORT };
        private BackgroundThreadStatus _backgroundThreadStatus = BackgroundThreadStatus.IDLE;
        private Tuple<int, bool, int[], int, DataAugmentationConfig, int[], int[]> threadParameters;
        private void BackgroundThread()
        {
            for (; ; )
            {
                while (_backgroundThreadStatus != BackgroundThreadStatus.ABOUT_TO_PROCESS_INPUT)
                {
                    if (_backgroundThreadStatus == BackgroundThreadStatus.TO_ABORT)
                    {
                        return;
                    }
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