using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using log4net;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.Pictures;

namespace SharpNet.Datasets
{
    public enum ResizeStrategyEnum
    {
        /// <summary>
        /// we do n ot resize the image from disk to the target size for training/inference
        /// we expect them to be the same
        /// an exception is thrown if it is not the case
        /// </summary>
        None,


        /// <summary>
        /// we'll simply resize the image from disk to the target size for training/inference
        /// without keeping the same proportion.
        /// It means that the picture can be distorted to fit the target size
        /// </summary>
        ResizeToTargetSize,

        /// <summary>
        /// We'll resize the image so that it will have exactly the same width as the size fo the training/inference tensor
        /// We'll keep the same proportion as in the original image (no distortion)
        /// </summary>
        // ReSharper disable once UnusedMember.Global
        ResizeToWidthSizeKeepingSameProportion,

        /// <summary>
        /// We'll resize the image so that it will have exactly the same height as the size fo the training/inference tensor
        /// We'll keep the same proportion as in the original image (no distortion)
        /// </summary>
        // ReSharper disable once UnusedMember.Global
        ResizeToHeightSizeKeepingSameProportion,

        /// <summary>
        /// We'll take the biggest crop in the original image and resize this crop to match exactly the size fo the training/inference tensor
        /// We'll keep the same proportion as in the original image (no distortion)
        /// </summary>
        // ReSharper disable once UnusedMember.Global
        BiggestCropInOriginalImageToKeepSameProportion
    }


    public abstract class AbstractDataSet : IDataSet
    {
        #region private & protected fields
        protected static readonly ILog Log = LogManager.GetLogger(typeof(AbstractDataSet));
        /// <summary>
        /// tensor with all original elements (no data augmentation) in the order needed for the current mini batch 
        /// </summary>
        private readonly CpuTensor<float> xOriginalNotAugmentedMiniBatch = new CpuTensor<float>(new[] { 1 });
        /// <summary>
        /// tensor with all augmented elements in the order needed for the current mini batch 
        /// </summary>
        private readonly CpuTensor<float> xDataAugmentedMiniBatch = new CpuTensor<float>(new[] { 1 });
        /// <summary>
        /// a temporary buffer used to construct the data augmented pictures
        /// </summary>
        private readonly CpuTensor<float> xBufferForDataAugmentedMiniBatch = new CpuTensor<float>(new[] { 1 });
        private readonly CpuTensor<float> yDataAugmentedMiniBatch = new CpuTensor<float>(new[] { 1 });
        /// <summary>
        /// the miniBatch Id associated with the above xBufferMiniBatchCpu & yBufferMiniBatchCpu tensors
        /// or -1 if those tensors are empty
        /// </summary>
        private long alreadyComputedMiniBatchId = -1;
        private readonly Random[] _rands;
        #endregion

        #region public properties
        public List<Tuple<float, float>> MeanAndVolatilityForEachChannel { get; }
        #endregion

        #region constructor
        protected AbstractDataSet(string name, int channels, string[ ] categoryDescriptions, List<Tuple<float, float>> meanAndVolatilityForEachChannel, ResizeStrategyEnum resizeStrategy)
        {
            Name = name;
            Channels = channels;
            CategoryDescriptions = categoryDescriptions;
            MeanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
            ResizeStrategy = resizeStrategy;
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

        public string[] CategoryDescriptions { get; }

        public abstract void LoadAt(int elementId, int indexInBuffer, [NotNull] CpuTensor<float> xBuffer,[CanBeNull] CpuTensor<float> yBuffer, bool withDataAugmentation);

        public void LoadMiniBatch(bool withDataAugmentation, int[] shuffledElementId, int firstIndexInShuffledElementId, DataAugmentationConfig dataAugmentationConfig, CpuTensor<float> xMiniBatch, CpuTensor<float> yMiniBatch)
        {
            Debug.Assert(xMiniBatch != null);
            Debug.Assert(yMiniBatch != null);
            Debug.Assert(xMiniBatch.Shape.Length>=1);
            Debug.Assert(xMiniBatch.TypeSize == TypeSize);
            Debug.Assert(AreCompatible_X_Y(xMiniBatch, yMiniBatch));

            xMiniBatch.AssertIsNotDisposed();
            yMiniBatch.AssertIsNotDisposed();

            if (UseBackgroundThread)
            {
                //if the background thread is working, we'll wait until it finishes
                backgroundThreadIsIdle.WaitOne();

                //the background is in idle state
                var miniBatchId = ComputeMiniBatchHashId(shuffledElementId, firstIndexInShuffledElementId, xMiniBatch.Shape[0]);
                if (miniBatchId != alreadyComputedMiniBatchId)
                {
                    LoadMiniBatchInCpu(withDataAugmentation, shuffledElementId, firstIndexInShuffledElementId, dataAugmentationConfig, xMiniBatch.Shape, yMiniBatch.Shape);
                }
                else
                {
                    //the background has already computed the current batch to process
                }
                //we know that the background thread is in idle state
                backgroundThreadIsIdle.Set();
            }
            else
            {
                LoadMiniBatchInCpu(withDataAugmentation, shuffledElementId, firstIndexInShuffledElementId, dataAugmentationConfig, xMiniBatch.Shape, yMiniBatch.Shape);
            }

            Debug.Assert(xMiniBatch.Shape.Length >= 1);
            xDataAugmentedMiniBatch.CopyTo(xMiniBatch.AsCpu<float>());
            Debug.Assert(xMiniBatch.Shape.Length >= 1);
            yDataAugmentedMiniBatch.CopyTo(yMiniBatch.AsCpu<float>());

            //uncomment to store data augmented pictures
            //if (withDataAugmentation && (firstIndexInShuffledElementId == 0))
            //{
            //    //CIFAR10
            //    var meanAndVolatilityOfEachChannel = new List<Tuple<double, double>> { Tuple.Create(125.306918046875, 62.9932192781369), Tuple.Create(122.950394140625, 62.0887076400142), Tuple.Create(113.865383183594, 66.7048996406309) };
            //    //SVHN
            //    //var meanAndVolatilityOfEachChannel = new List<Tuple<float, float>> { Tuple.Create(109.8823f, 50.11187f), Tuple.Create(109.7114f, 50.57312f), Tuple.Create(113.8187f, 50.85124f) };
            //    var xCpuChunkBytes = xDataAugmentedMiniBatch.Select((n, c, val) => (byte)((val * meanAndVolatilityOfEachChannel[c].Item2 + meanAndVolatilityOfEachChannel[c].Item1)));
            //    for (int i = firstIndexInShuffledElementId; i < Math.Min((firstIndexInShuffledElementId + xMiniBatch.Shape[0]), 100); ++i)
            //    {
            //        int elementId = shuffledElementId[i];
            //        var categoryIndex = ElementIdToCategoryIndex(elementId);
            //        PictureTools.SaveBitmap(xCpuChunkBytes, i, System.IO.Path.Combine(NetworkConfig.DefaultLogDirectory, "Train"), elementId.ToString("D5") + "_cat" + categoryIndex, "");
            //    }
            //}

            //we check if we can start compute the next mini batch content in advance
            int firstIndexInShuffledElementIdForNextMiniBatch = firstIndexInShuffledElementId + xMiniBatch.Shape[0];
            int nextMiniBatchSize = Math.Min(shuffledElementId.Length - firstIndexInShuffledElementIdForNextMiniBatch, xMiniBatch.Shape[0]);
            if (UseBackgroundThread && nextMiniBatchSize > 0)
            {
                //we will ask the background thread to compute the next mini batch
                backgroundThreadIsIdle.WaitOne();
                var xNextMiniBatchShape = (int[])xMiniBatch.Shape.Clone();
                var yNextMiniBatchShape = (int[])yMiniBatch.Shape.Clone();
                xNextMiniBatchShape[0] = yNextMiniBatchShape[0] = nextMiniBatchSize;
                threadParameters = Tuple.Create(withDataAugmentation, shuffledElementId, firstIndexInShuffledElementIdForNextMiniBatch, dataAugmentationConfig, xNextMiniBatchShape, yNextMiniBatchShape);
                backgroundThreadHasSomethingTodo.Set();
            }
        }

        public static CpuTensor<float> ToXWorkingSet(CpuTensor<byte> x, List<Tuple<float, float>> meanAndVolatilityOfEachChannel)
        {
            var xWorkingSet = x.Select((n, c, val) => (float)((val - meanAndVolatilityOfEachChannel[c].Item1) / Math.Max(meanAndVolatilityOfEachChannel[c].Item2, 1e-9)));
            //xWorkingSet = x.Select((n, c, val) => (float)val/255f);
            return xWorkingSet;
        }
        public static CpuTensor<float> ToYWorkingSet(CpuTensor<byte> categoryBytes, int categoryCount , Func<byte, int> categoryByteToCategoryIndex)
        {
            Debug.Assert(categoryBytes.MultDim0 == 1);
            var batchSize = categoryBytes.Shape[0];
            var newShape = new[] { batchSize, categoryCount };
            var newY = new CpuTensor<float>(newShape);
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
        /// <param name="targetHeight"></param>
        /// <param name="targetWidth"></param>
        /// <param name="withDataAugmentation"></param>
        /// <returns>a byte tensor containing the element at index 'elementId' </returns>
        public virtual BitmapContent OriginalElementContent(int elementId, int targetHeight, int targetWidth, bool withDataAugmentation)
        {
            var xBuffer = new CpuTensor<float>(new[] { 1, Channels, targetHeight, targetWidth });
            LoadAt(elementId, 0, xBuffer, null, withDataAugmentation);

            var inputShape_CHW = new[]{Channels, targetHeight, targetWidth};

            xBuffer.Reshape(inputShape_CHW); //from (1,c,h,w) shape to (c,h,w) shape
            var bufferContent = xBuffer.ReadonlyContent;


            var resultContent = new byte[Channels * targetHeight * targetWidth];
            int idxInResultContent = 0;
            var result = new BitmapContent(inputShape_CHW, resultContent);

            int nbBytesByChannel = targetHeight * targetWidth;
            var isNormalized = IsNormalized;
            for (int channel = 0; channel < Channels; ++channel)
            {
                var originalChannelVolatility = OriginalChannelVolatility(channel);
                var originalChannelMean = OriginalChannelMean(channel);
                for (int i = 0; i < nbBytesByChannel; ++i)
                {
                    byte originalValue;
                    float normalizedValue = bufferContent[idxInResultContent];
                    if (!isNormalized)
                    {
                        //no normalization was performed on the input
                        originalValue = (byte)normalizedValue;
                    }
                    else
                    {
                        originalValue = (byte) ((normalizedValue) * originalChannelVolatility + originalChannelMean + 0.1);
                    }
                    resultContent[idxInResultContent++] = originalValue;
                }
            }
            return result;
        }

        /// <summary>
        /// TODO : check if we should use a cache to avoid recomputing the image stat at each epoch
        /// </summary>
        /// <param name="elementId"></param>
        /// <param name="targetHeight"></param>
        /// <param name="targetWidth"></param>
        /// <returns></returns>
        public ImageStatistic ElementIdToImageStatistic(int elementId, int targetHeight, int targetWidth)
        {
            return ImageStatistic.ValueOf(OriginalElementContent(elementId, targetHeight, targetWidth, false));
        }


        public abstract int Count { get; }
        public string Name { get; }
        public ResizeStrategyEnum ResizeStrategy { get; }
        public int Channels { get; }
        public int CategoryCount => CategoryDescriptions.Length;
        /// <summary>
        /// return the mean of channel 'channel' of the original DataSet (before normalization)
        /// </summary>
        /// <param name="channel">the channel for which we want to extract the mean</param>
        /// <returns>mean of channel 'channel'</returns>
        protected double OriginalChannelMean(int channel)
        {
            Debug.Assert(IsNormalized);
            return MeanAndVolatilityForEachChannel[channel].Item1;
        }

        /// <summary>
        /// return the volatility of channel 'channel' of the original DataSet (before normalization)
        /// </summary>
        /// <param name="channel">the channel for which we want to extract the volatility</param>
        /// <returns>volatility of channel 'channel'</returns>
        protected double OriginalChannelVolatility(int channel)
        {
            Debug.Assert(IsNormalized);
            return MeanAndVolatilityForEachChannel[channel].Item2;
        }

        /// <summary>
        /// true if the current data set is normalized (with mean=0 and volatility=1 in each channel)
        /// </summary>
        private bool IsNormalized => MeanAndVolatilityForEachChannel != null && MeanAndVolatilityForEachChannel.Count != 0;

        // ReSharper disable once UnusedMember.Global
        public string CategoryDescription(int categoryIndex)
        {
            if (categoryIndex < 0 || categoryIndex >= CategoryCount || CategoryDescriptions == null)
            {
                return "";
            }
            return CategoryDescriptions[categoryIndex];
        }

        public virtual string ElementIdToDescription(int elementId)
        {
            return elementId.ToString();
        }
        /// <summary>
        /// retrieve the category associated with a specific element
        /// </summary>
        /// <param name="elementId">the id of the element, int the range [0, Count-1] </param>
        /// <returns>the associated category id, or -1 if the category is not known</returns>
        public abstract int ElementIdToCategoryIndex(int elementId);
        public abstract string ElementIdToPathIfAny(int elementId);

        public int[] Y_Shape => new[] { Count, CategoryCount };
        public virtual void Dispose()
        {
            xOriginalNotAugmentedMiniBatch?.Dispose();
            xDataAugmentedMiniBatch?.Dispose();
            xBufferForDataAugmentedMiniBatch?.Dispose();
            yDataAugmentedMiniBatch?.Dispose();
            if (UseBackgroundThread)
            {
                //we stop the background thread
                threadParameters = null;
                shouldStopBackgroundThread = true;
                backgroundThreadIsIdle.WaitOne(1000);
                backgroundThreadHasSomethingTodo.Set();
                Thread.Sleep(10);
                if (thread.IsAlive)
                {
                    Log.Info("fail to stop BackgroundThread in " + Name);
                }
            }
        }
        public int[] YMiniBatch_Shape(int miniBatchSize)
        {
            return new[] { miniBatchSize, CategoryCount };
        }

        public int TypeSize => 4; //float size
        public override string ToString() {return Name;}

        private IDataSet Slice(int firstElementId, int count)
        {
            return new SubDataSet(this, id => id >= firstElementId && id <(firstElementId+count) );
        }

        public ITrainingAndTestDataSet SplitIntoTrainingAndValidation(double percentageInTrainingSet)
        {
            int countInTrainingSet = (int)(percentageInTrainingSet * Count+0.1);
            var training = Slice(0, countInTrainingSet);
            var test = Slice(countInTrainingSet, Count- countInTrainingSet);
            return new TrainingAndTestDataLoader(training, test, this);
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
        
        protected static bool IsValidYSet(Tensor data)
        {
            Debug.Assert(!data.UseGPU);
            return data.AsReadonlyFloatCpuContent.All(x => IsValidY(x));
        }
        protected void UpdateStatus(ref int nbPerformed)
        {
            int delta = Math.Max(Count / 100, 1);
            var newNbPerformed = Interlocked.Increment(ref nbPerformed);
            if ((newNbPerformed % delta == 0) || (newNbPerformed == Count))
            {
                Log.Info("Done: " + (100 * newNbPerformed) / Count + "%");
            }
        }
        /// <summary>
        /// Load in 'xBufferMiniBatchCpu' & 'yBufferMiniBatchCpu' tensors the data related to the mini batch starting
        /// at 'firstIndexInShuffledElementId'
        /// </summary>
        /// <param name="withDataAugmentation"></param>
        /// <param name="shuffledElementId"></param>
        /// <param name="firstIndexInShuffledElementId"></param>
        /// <param name="dataAugmentationConfig"></param>
        /// <param name="xMiniBatchShape"></param>
        /// <param name="yMiniBatchShape"></param>
        private void LoadMiniBatchInCpu(bool withDataAugmentation,
            int[] shuffledElementId, int firstIndexInShuffledElementId,
            DataAugmentationConfig dataAugmentationConfig,
            int[] xMiniBatchShape, int[] yMiniBatchShape)
        {
            Debug.Assert(Channels == xMiniBatchShape[1]);
            var miniBatchSize = xMiniBatchShape[0];

            var miniBatchId = ComputeMiniBatchHashId(shuffledElementId, firstIndexInShuffledElementId, miniBatchSize);
            if (miniBatchId == alreadyComputedMiniBatchId)
            {
                //nothing to do, the mini batch data is already stored in 'xBufferMiniBatchCpu' & 'yBufferMiniBatchCpu' tensors
                return;
            }

            //we initialize 'xOriginalNotAugmentedMiniBatchCpu' with all the original (not augmented elements)
            //contained in the mini batch
            xOriginalNotAugmentedMiniBatch.Reshape(xMiniBatchShape);
            //we'll first create mini batch input in a local CPU buffer, then copy them in xMiniBatch/yMiniBatch
            xDataAugmentedMiniBatch.Reshape(xMiniBatchShape);
            xBufferForDataAugmentedMiniBatch.Reshape(xMiniBatchShape);
            yDataAugmentedMiniBatch.Reshape(yMiniBatchShape);
            yDataAugmentedMiniBatch.ZeroMemory();

            int MiniBatchIdxToElementId(int miniBatchIdx) => shuffledElementId[firstIndexInShuffledElementId + miniBatchIdx];
            Parallel.For(0, miniBatchSize, indexInBuffer => LoadAt(MiniBatchIdxToElementId(indexInBuffer), indexInBuffer, xOriginalNotAugmentedMiniBatch, yDataAugmentedMiniBatch, withDataAugmentation));

            Debug.Assert(AreCompatible_X_Y(xDataAugmentedMiniBatch, yDataAugmentedMiniBatch));
            int MiniBatchIdxToCategoryIndex(int miniBatchIdx) => ElementIdToCategoryIndex(MiniBatchIdxToElementId(miniBatchIdx));
            if (!dataAugmentationConfig.UseDataAugmentation || !withDataAugmentation)
            {
                //we'll just copy the input picture from index 'inputPictureIndex' in 'inputEnlargedPictures' to index 'outputPictureIndex' of 'outputBufferPictures'
                xOriginalNotAugmentedMiniBatch.CopyTo(xDataAugmentedMiniBatch);
            }
            else
            {
                int targetHeight = xMiniBatchShape[2];
                int targetWidth = xMiniBatchShape[3];
                Lazy<ImageStatistic> MiniBatchIdxToLazyImageStatistic(int miniBatchIdx) => new Lazy<ImageStatistic>(() => ElementIdToImageStatistic(MiniBatchIdxToElementId(miniBatchIdx), targetHeight, targetWidth));
                var imageDataGenerator = new ImageDataGenerator(dataAugmentationConfig);
                Parallel.For(0, miniBatchSize, indexInMiniBatch => imageDataGenerator.DataAugmentationForMiniBatch(indexInMiniBatch, xOriginalNotAugmentedMiniBatch, xDataAugmentedMiniBatch, yDataAugmentedMiniBatch, MiniBatchIdxToCategoryIndex, MiniBatchIdxToLazyImageStatistic, MeanAndVolatilityForEachChannel, GetRandomForIndexInMiniBatch(indexInMiniBatch), xBufferForDataAugmentedMiniBatch));
            }
            alreadyComputedMiniBatchId = miniBatchId;
        }

        private static bool IsValidY(double x)
        {
            return Math.Abs(x) <= 1e-9 || Math.Abs(x - 1.0) <= 1e-9;
        }

        public Random GetRandomForIndexInMiniBatch(int indexInMiniBatch)
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
        private bool UseBackgroundThread { get; } = true;
        private readonly Thread thread;
        private Tuple<bool, int[], int, DataAugmentationConfig, int[], int[]> threadParameters;
        private readonly AutoResetEvent backgroundThreadHasSomethingTodo = new AutoResetEvent(false);
        private readonly AutoResetEvent backgroundThreadIsIdle = new AutoResetEvent(false);
        private bool shouldStopBackgroundThread = false;

        private void BackgroundThread()
        {
            for (; ; )
            {
                //we signal that the thread is in Idle mode
                backgroundThreadIsIdle.Set();
                //we wait for the master thread to prepare something to do
                backgroundThreadHasSomethingTodo.WaitOne();
                if (shouldStopBackgroundThread)
                {
                    return;
                }
                Debug.Assert(threadParameters != null);
                // ReSharper disable once PossibleNullReferenceException
                LoadMiniBatchInCpu(threadParameters.Item1, threadParameters.Item2, threadParameters.Item3, threadParameters.Item4, threadParameters.Item5, threadParameters.Item6);
                threadParameters = null;
            }
        }
        #endregion
    }
}
