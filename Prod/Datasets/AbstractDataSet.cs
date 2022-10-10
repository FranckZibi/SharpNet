﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using log4net;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.HyperParameters;
using SharpNet.Models;
using SharpNet.Networks;
using SharpNet.Pictures;

namespace SharpNet.Datasets
{
    public abstract class AbstractDataSet : IDataSet
    {
        #region private & protected fields
        protected static readonly ILog Log = LogManager.GetLogger(typeof(AbstractDataSet));
        /// <summary>
        /// tensor with all original elements (no data augmentation) in the order needed for the current mini batch 
        /// </summary>
        private readonly List<CpuTensor<float>> all_xOriginalNotAugmentedMiniBatch = new List<CpuTensor<float>>();
        /// <summary>
        /// tensor with all augmented elements in the order needed for the current mini batch 
        /// </summary>
        private readonly List<CpuTensor<float>> all_xDataAugmentedMiniBatch = new List<CpuTensor<float>>();
        /// <summary>
        /// a temporary buffer used to construct the data augmented pictures
        /// </summary>
        private readonly List<CpuTensor<float>> all_xBufferForDataAugmentedMiniBatch = new List<CpuTensor<float>>();
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
        [NotNull] public string[] FeatureNames { get; }
        [NotNull] public string[] CategoricalFeatures { get; }
        [NotNull] public string[] IdColumns { get; }
        [NotNull] public string[] TargetLabels { get; }
        public bool UseBackgroundThreadToLoadNextMiniBatch { get; }
        public char Separator { get; }

        #endregion

        #region constructor
        protected AbstractDataSet(string name,
            Objective_enum objective,
            int channels,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            ResizeStrategyEnum resizeStrategy,
            [NotNull] string[] featureNames,
            [NotNull] string[] categoricalFeatures,
            [NotNull] string[] idColumns,
            [NotNull] string[] targetLabels,
            bool useBackgroundThreadToLoadNextMiniBatch,
            char separator)
        {
            Separator = separator;
            Name = name;
            Objective = objective;
            Channels = channels;
            MeanAndVolatilityForEachChannel = meanAndVolatilityForEachChannel;
            ResizeStrategy = resizeStrategy;
            UseBackgroundThreadToLoadNextMiniBatch = useBackgroundThreadToLoadNextMiniBatch;
            _rands = new Random[2 * Environment.ProcessorCount];
            for (int i = 0; i < _rands.Length; ++i)
            {
                _rands[i] = new Random(i);
            }
            FeatureNames = featureNames;
            CategoricalFeatures = categoricalFeatures;
            IdColumns = idColumns;
            TargetLabels = targetLabels;
            if (UseBackgroundThreadToLoadNextMiniBatch)
            {
                thread = new Thread(BackgroundThread);
                thread.Start();
            }
        }
        #endregion

        /// <summary>
        /// the type of use of the dataset : Regression or Classification
        /// </summary>
        public Objective_enum Objective { get; }

        public bool IsRegressionProblem => Objective == Objective_enum.Regression;


        public abstract void LoadAt(int elementId, int indexInBuffer, [NotNull] CpuTensor<float> xBuffer,[CanBeNull] CpuTensor<float> yBuffer, bool withDataAugmentation);

        protected virtual void LoadAt(int elementId, int indexInBuffer, [NotNull] List<CpuTensor<float>> xBuffers,
            [CanBeNull] CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            if (xBuffers.Count != 1)
            {
                throw new ArgumentException("only 1 InputLayer is supported, received "+xBuffers.Count);
            }
            LoadAt(elementId, indexInBuffer, xBuffers[0], yBuffer, withDataAugmentation);
        }


        public int LoadMiniBatch(bool withDataAugmentation, int[] shuffledElementId, int firstIndexInShuffledElementId,
            DataAugmentationSample dataAugmentationSample, [NotNull] List<CpuTensor<float>> all_xMiniBatches,
            [NotNull] CpuTensor<float> yMiniBatch)
        {
            Debug.Assert(all_xMiniBatches[0].Shape.Length>=1);
            Debug.Assert(all_xMiniBatches[0].TypeSize == TypeSize);
            Debug.Assert(AreCompatible_X_Y(all_xMiniBatches[0], yMiniBatch));


            var all_xMiniBatchesShapes = all_xMiniBatches.Select(t => t.Shape).ToList();

            all_xMiniBatches[0].AssertIsNotDisposed();
            yMiniBatch.AssertIsNotDisposed();

            int elementsActuallyLoaded;
            if (UseBackgroundThreadToLoadNextMiniBatch)
            {
                //if the background thread is working, we'll wait until it finishes
                backgroundThreadIsIdle.WaitOne();

                //the background is in idle state
                var miniBatchId = ComputeMiniBatchHashId(shuffledElementId, firstIndexInShuffledElementId, all_xMiniBatches[0].Shape[0]);
                if (miniBatchId != alreadyComputedMiniBatchId)
                {
                    elementsActuallyLoaded = LoadMiniBatchInCpu(withDataAugmentation, shuffledElementId, firstIndexInShuffledElementId, dataAugmentationSample, all_xMiniBatchesShapes, yMiniBatch.Shape);
                }
                else
                {
                    //the background thread has already computed the current batch to process
                    elementsActuallyLoaded = elementsActuallyLoadedByBackgroundThread;
                }
                //we know that the background thread is in idle state
                backgroundThreadIsIdle.Set();
            }
            else
            {
                elementsActuallyLoaded = LoadMiniBatchInCpu(withDataAugmentation, shuffledElementId, firstIndexInShuffledElementId, dataAugmentationSample, all_xMiniBatchesShapes, yMiniBatch.Shape);
            }

            Debug.Assert(all_xMiniBatches[0].Shape.Length >= 1);
            for (int x = 0; x < all_xDataAugmentedMiniBatch.Count; ++x)
            {
                Debug.Assert(all_xMiniBatches[x].Shape.Length >= 1);
                all_xDataAugmentedMiniBatch[x].CopyTo(all_xMiniBatches[x].AsCpu<float>());
            }

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
            int firstIndexInShuffledElementIdForNextMiniBatch = firstIndexInShuffledElementId + all_xMiniBatches[0].Shape[0];
            int nextMiniBatchSize = Math.Min(shuffledElementId.Length - firstIndexInShuffledElementIdForNextMiniBatch, all_xMiniBatches[0].Shape[0]);
            if (UseBackgroundThreadToLoadNextMiniBatch && nextMiniBatchSize > 0)
            {
                //we will ask the background thread to compute the next mini batch
                backgroundThreadIsIdle.WaitOne();
                var xNextMiniBatchShape = new List<int[]>();
                foreach (var shape in all_xMiniBatchesShapes)
                {
                    var clonedShape = (int[]) shape.Clone();
                    clonedShape[0] = nextMiniBatchSize;
                    xNextMiniBatchShape.Add(clonedShape);
                }
                //(int[])xMiniBatches.Shape.Clone();
                var yNextMiniBatchShape = (int[])yMiniBatch.Shape.Clone();
                yNextMiniBatchShape[0] = nextMiniBatchSize;
                threadParameters = Tuple.Create(withDataAugmentation, shuffledElementId, firstIndexInShuffledElementIdForNextMiniBatch, dataAugmentationSample, xNextMiniBatchShape, yNextMiniBatchShape);
                backgroundThreadHasSomethingTodo.Set();
            }
            return elementsActuallyLoaded;
        }

        public static CpuTensor<float> ToXWorkingSet(CpuTensor<byte> x, List<Tuple<float, float>> meanAndVolatilityOfEachChannel)
        {
            var xWorkingSet = x.Select((_, c, val) => (float)((val - meanAndVolatilityOfEachChannel[c].Item1) / Math.Max(meanAndVolatilityOfEachChannel[c].Item2, 1e-9)));
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
        public DataFrame Y_InTargetFormat()
        {
            if (Y == null)
            {
                return null;
            }
            var y_true_InTargetFormat = DataFrame.New(Y, TargetLabels);
            if (IdColumns.Length == 0)
            {
                return y_true_InTargetFormat;
            }
            return DataFrame.MergeHorizontally(ExtractIdDataFrame(), y_true_InTargetFormat);
        }

        public string Name { get; }
        public ResizeStrategyEnum ResizeStrategy { get; }
        public virtual bool ShouldCreateSnapshotForEpoch(int epoch, Network network)
        {
            return false;
        }

        public virtual void Save(IModel model, string workingDirectory, string modelName)
        {
            model.Save(workingDirectory, modelName);
        }

        public virtual DataFrame ExtractIdDataFrame()
        {
            if (IdColumns.Length == 0)
            {
                throw new ArgumentException($"can't extract id columns because the DataFrame doesn't contain any");
            }
            var columnIndexesOfIds = IdColumns.Select(id => Array.IndexOf(FeatureNames, id)).ToList();
            columnIndexesOfIds.Sort();

            var content = new float[Count * IdColumns.Length];
            int cols = FeatureNames.Length;
            using CpuTensor<float> singleRow = new(new[] { 1, cols });
            var singleRowAsSpan = singleRow.AsReadonlyFloatCpuContent;
            int nextIdx = 0;
            for (int row = 0; row < Count; ++row)
            {
                LoadAt(row, 0, singleRow, null, false);
                foreach (var colIdx in columnIndexesOfIds)
                {
                    content[nextIdx++] = singleRowAsSpan[colIdx];
                }
            }
            var idCpuTensor = new CpuTensor<float>(new[] { Count, IdColumns.Length }, content);
            return DataFrame.New(idCpuTensor, IdColumns);
        }

        public int Channels { get; }
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
        public IDataSet Resize(int targetSize, bool shuffle)
        {
            var elementIdToOriginalElementId = new List<int>(targetSize);
            for (int elementId = 0; elementId < targetSize; ++elementId)
            {
                elementIdToOriginalElementId.Add(elementId % Count);
            }
            if (shuffle)
            {
                Utils.Shuffle(elementIdToOriginalElementId, new Random(0));
            }
            return new MappedDataSet(this, elementIdToOriginalElementId);
        }

        public IDataSet Shuffle(Random r)
        {
            var elementIdToOriginalElementId = Enumerable.Range(0, Count).ToList();
            Utils.Shuffle(elementIdToOriginalElementId, r);
            return new MappedDataSet(this, elementIdToOriginalElementId);
        }

        public virtual IDataSet SubDataSet(double percentageToKeep)
        {
            return SubDataSet(_ => _rands[0].NextDouble() < percentageToKeep);
        }

        public virtual double PercentageToUseForLossAndAccuracyFastEstimate => 0.1;

        public IDataSet SubDataSet(Func<int, bool> elementIdInOriginalDataSetToIsIncludedInSubDataSet)
        {
            var subElementIdToOriginalElementId = new List<int>();
            for (int originalElementId = 0; originalElementId < Count; ++originalElementId)
            {
                if (elementIdInOriginalDataSetToIsIncludedInSubDataSet(originalElementId))
                {
                    subElementIdToOriginalElementId.Add(originalElementId);
                }
            }
            if (subElementIdToOriginalElementId.Count == 0)
            {
                return null;
            }
            return new MappedDataSet(this, subElementIdToOriginalElementId);
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

        public virtual void Dispose()
        {
            all_xOriginalNotAugmentedMiniBatch.ForEach(t=>t.Dispose());
            all_xOriginalNotAugmentedMiniBatch.Clear();
            all_xDataAugmentedMiniBatch.ForEach(t => t.Dispose());
            all_xDataAugmentedMiniBatch.Clear();
            all_xBufferForDataAugmentedMiniBatch.ForEach(t => t.Dispose());
            all_xBufferForDataAugmentedMiniBatch.Clear();
            yDataAugmentedMiniBatch?.Dispose();
            if (UseBackgroundThreadToLoadNextMiniBatch)
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
            var yMiniBatchShape = (int[])Y.Shape.Clone();
            yMiniBatchShape[0] = miniBatchSize;
            return yMiniBatchShape;
        }

        public virtual List<int[]> XMiniBatch_Shape(int[] shapeForFirstLayer)
        {
            return new List<int[]> {shapeForFirstLayer};
        }

        public int TypeSize => 4; //float size
        public override string ToString() {return Name;}

        public virtual ITrainingAndTestDataSet SplitIntoTrainingAndValidation(double percentageInTrainingSet)
        {
            int countInTrainingSet = (int)(percentageInTrainingSet * Count+0.1);
            return IntSplitIntoTrainingAndValidation(countInTrainingSet);
        }
        public virtual ITrainingAndTestDataSet IntSplitIntoTrainingAndValidation(int countInTrainingSet)
        {
            var training = SubDataSet(id => id < countInTrainingSet);
            var test = SubDataSet(id => id >= countInTrainingSet);
            return new TrainingAndTestDataset(training, test, Name);
        }

        public static bool AreCompatible_X_Y(Tensor X, Tensor Y)
        {
            if (X == null && Y == null)
            {
                return true;
            }

            return (X != null) && (Y != null)
                               && (X.UseGPU == Y.UseGPU)
                               && (X.Shape[0] == Y.Shape[0]); //same number of tests
        }
        public abstract CpuTensor<float> Y { get; }

        public CpuTensor<float> Y_InModelFormat(int numClasses)
        {
            if (Y == null)
            {
                return null;
            }
            return IsRegressionProblem ? Y : CpuTensor<float>.FromClassIndexToProba(Y, numClasses);
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


        private TimeSeriesDataAugmentation TimeSeriesDataAugmentation;

        /// <summary>
        /// Load in 'xBufferMiniBatchCpu' & 'yBufferMiniBatchCpu' tensors the data related to the mini batch starting
        /// at 'firstIndexInShuffledElementId'
        /// </summary>
        /// <returns>the number of actually loaded elements,in the range [1, xMiniBatchShape[0] ]  </returns>
        private int LoadMiniBatchInCpu(bool withDataAugmentation,
            int[] shuffledElementId, int firstIndexInShuffledElementId,
            DataAugmentationSample dataAugmentationSample,
            List<int[]> all_xMiniBatchShape, int[] yMiniBatchShape)
        {
            Debug.Assert(Channels == all_xMiniBatchShape[0][1]);
            var miniBatchSize = all_xMiniBatchShape[0][0];

            int maxElementsToLoad = GetMaxElementsToLoad(shuffledElementId, firstIndexInShuffledElementId, miniBatchSize);

            var miniBatchId = ComputeMiniBatchHashId(shuffledElementId, firstIndexInShuffledElementId, miniBatchSize);
            if (miniBatchId == alreadyComputedMiniBatchId)
            {
                //nothing to do, the mini batch data is already stored in 'xBufferMiniBatchCpu' & 'yBufferMiniBatchCpu' tensors
                return maxElementsToLoad;
            }

            //we initialize 'xOriginalNotAugmentedMiniBatchCpu' with all the original (not augmented elements)
            //contained in the mini batch

            //we'll first create mini batch input in a local CPU buffer, then copy them in xMiniBatch/yMiniBatch
            for (int x = 0; x < all_xMiniBatchShape.Count; ++x)
            {
                foreach(var l in new[] { all_xOriginalNotAugmentedMiniBatch , all_xDataAugmentedMiniBatch , all_xBufferForDataAugmentedMiniBatch })
                { 
                    if (l.Count <= x)
                    {
                        l.Add(new CpuTensor<float>(all_xMiniBatchShape[x]));
                    }
                    else
                    {
                        l[x].Reshape(all_xMiniBatchShape[x]);
                    }
                }
            }

            yDataAugmentedMiniBatch.Reshape(yMiniBatchShape);
            yDataAugmentedMiniBatch.ZeroMemory();

            int MiniBatchIdxToElementId(int miniBatchIdx) => shuffledElementId[ (firstIndexInShuffledElementId+miniBatchIdx)%shuffledElementId.Length ];
            Parallel.For(0, miniBatchSize, indexInBuffer => LoadAt(MiniBatchIdxToElementId(indexInBuffer%maxElementsToLoad), indexInBuffer, all_xOriginalNotAugmentedMiniBatch, yDataAugmentedMiniBatch, withDataAugmentation));

            Debug.Assert(AreCompatible_X_Y(all_xDataAugmentedMiniBatch[0], yDataAugmentedMiniBatch));
            int MiniBatchIdxToCategoryIndex(int miniBatchIdx) => ElementIdToCategoryIndex(MiniBatchIdxToElementId(miniBatchIdx));
            if (!dataAugmentationSample.UseDataAugmentation || !withDataAugmentation)
            {
                //no Data Augmentation: we'll just copy the input element
                for (int x = 0; x < all_xOriginalNotAugmentedMiniBatch.Count; ++x)
                {
                    all_xOriginalNotAugmentedMiniBatch[x].CopyTo(all_xDataAugmentedMiniBatch[x]);
                }
            }
            else if (dataAugmentationSample.DataAugmentationType == ImageDataGenerator.DataAugmentationEnum.TIME_SERIES)
            {
                //Data Augmentation for time series
                for (int x = 0; x < all_xOriginalNotAugmentedMiniBatch.Count; ++x)
                {
                    all_xOriginalNotAugmentedMiniBatch[x].CopyTo(all_xDataAugmentedMiniBatch[x]);
                }
                if (TimeSeriesDataAugmentation == null)
                {
                    var featuresCount = all_xOriginalNotAugmentedMiniBatch[0].Shape[2];
                    TimeSeriesDataAugmentation = new TimeSeriesDataAugmentation(dataAugmentationSample, (ITimeSeriesDataSet) this, featuresCount);
                }
                Parallel.For(0, miniBatchSize, indexInMiniBatch => TimeSeriesDataAugmentation.DataAugmentationForMiniBatch(indexInMiniBatch % maxElementsToLoad, all_xDataAugmentedMiniBatch[0], GetRandomForIndexInMiniBatch(indexInMiniBatch)));
            }
            else
            {
                //Data Augmentation for images
                Debug.Assert(all_xMiniBatchShape.Count == 1);
                int targetHeight = all_xMiniBatchShape[0][2];
                int targetWidth = all_xMiniBatchShape[0][3];
                Lazy<ImageStatistic> MiniBatchIdxToLazyImageStatistic(int miniBatchIdx) => new Lazy<ImageStatistic>(() => ElementIdToImageStatistic(MiniBatchIdxToElementId(miniBatchIdx), targetHeight, targetWidth));
                var imageDataGenerator = new ImageDataGenerator(dataAugmentationSample);
                Parallel.For(0, miniBatchSize, indexInMiniBatch => imageDataGenerator.DataAugmentationForMiniBatch(indexInMiniBatch % maxElementsToLoad, all_xOriginalNotAugmentedMiniBatch[0], all_xDataAugmentedMiniBatch[0], yDataAugmentedMiniBatch, MiniBatchIdxToCategoryIndex, MiniBatchIdxToLazyImageStatistic, MeanAndVolatilityForEachChannel, GetRandomForIndexInMiniBatch(indexInMiniBatch), all_xBufferForDataAugmentedMiniBatch[0]));
            }

            //TODO: ensure that there is no NaN or Infinite in xDataAugmentedMiniBatch and yDataAugmentedMiniBatch

            alreadyComputedMiniBatchId = miniBatchId;
            return maxElementsToLoad;
        }

        protected virtual int GetMaxElementsToLoad(int[] shuffledElementId, int firstIndexInShuffledElementId, int batchSize)
        {
            return Math.Min(batchSize, shuffledElementId.Length - firstIndexInShuffledElementId);
        }

        public Random GetRandomForIndexInMiniBatch(int indexInMiniBatch)
        {
            var rand = _rands[indexInMiniBatch % _rands.Length];
            return rand;
        }

        /// <summary>
        /// save the dataset in path 'path' in 'LightGBM' format.
        /// if addTargetColumnAsFirstColumn == true:
        ///     first column is the label 'y' (to predict)
        ///     all other columns are the features
        /// else
        ///     save only 'x' (feature) tensor
        /// </summary>
        /// <param name="path">the path where to save the dataset (it contains both the directory and the filename)</param>
        /// <param name="addTargetColumnAsFirstColumn"></param>
        /// <param name="includeIdColumns"></param>
        /// <param name="overwriteIfExists">overwrite the file if it already exists</param>
        /// <param name="separator"></param>
        private void to_csv([NotNull] string path, char separator, bool addTargetColumnAsFirstColumn, bool includeIdColumns, bool overwriteIfExists)
        {
            if (File.Exists(path) && !overwriteIfExists)
            {
                Log.Debug($"No need to save dataset {Name} in path {path} : it already exists");
                return;
            }
            Log.Debug($"Saving dataset {Name} in path {path} (addTargetColumnAsFirstColumn =  {addTargetColumnAsFirstColumn})");

            var sb = new StringBuilder();
            if (addTargetColumnAsFirstColumn)
            {
                if (TargetLabels.Length != 1)
                {
                    var errorMsg = $"invalid number of target features, expecting 1, found {TargetLabels.Length}: {string.Join(' ', TargetLabels)}";
                    Log.Error(errorMsg);
                    throw new Exception(errorMsg);
                }
                sb.Append(TargetLabels[0] + separator);
            }
            // ReSharper disable once PossibleNullReferenceException
            var yDataAsSpan = (addTargetColumnAsFirstColumn&&Y!=null) ? Y.AsFloatCpuSpan : null;


            List<int> validIdxColumns;
            if (includeIdColumns)
            {
                validIdxColumns = Enumerable.Range(0, FeatureNames.Length).ToList();
            }
            else
            {
                validIdxColumns = new List<int>();
                for (var index = 0; index < FeatureNames.Length; index++)
                {
                    if (Array.IndexOf(IdColumns, FeatureNames[index]) == -1)
                    {
                        validIdxColumns.Add(index);
                    }
                }
            }

            //we construct the header
            sb.Append(string.Join(separator, validIdxColumns.Select(idx => FeatureNames[idx])) + Environment.NewLine);

            using CpuTensor<float> singleRow = new(new[] { 1, FeatureNames.Length});
            var singleRowAsSpan = singleRow.AsReadonlyFloatCpuContent;
            for (int row = 0; row < Count; ++row)
            {
                if (addTargetColumnAsFirstColumn)
                {
                    var yValue = yDataAsSpan==null?AbstractSample.DEFAULT_VALUE:yDataAsSpan[row];
                    sb.Append(yValue.ToString(CultureInfo.InvariantCulture) + separator);
                }
                LoadAt(row, 0, singleRow, null, false);
                foreach (var colIdx in validIdxColumns)
                {
                    sb.Append(singleRowAsSpan[colIdx].ToString(CultureInfo.InvariantCulture));
                    sb.Append( colIdx == validIdxColumns.Last() ? Environment.NewLine  : separator);
                }
            }

            var fileContent = sb.ToString();
            lock (Lock_to_csv)
            {
                File.WriteAllText(path, fileContent);
            }
            Log.Debug($"Dataset {Name} saved in path {path} (addTargetColumnAsFirstColumn =  {addTargetColumnAsFirstColumn})");
        }

        public string to_csv_in_directory(string directory, bool addTargetColumnAsFirstColumn, bool includeIdColumns, bool overwriteIfExists)
        {
            var datasetPath = Path.Combine(directory, ComputeUniqueDatasetName(this, addTargetColumnAsFirstColumn, includeIdColumns) + ".csv");
            to_csv(datasetPath, Separator, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            return datasetPath;
        }


        private static readonly object Lock_to_csv = new();

        private static long ComputeMiniBatchHashId(int[] shuffledElementId, int firstIndexInShuffledElementId, int miniBatchSize)
        {
            long result = 1;
            for (int i = firstIndexInShuffledElementId; i < firstIndexInShuffledElementId + miniBatchSize; ++i)
            {
                result += (i + 1) * shuffledElementId[i%shuffledElementId.Length];
            }
            return result;
        }
        private static string ComputeUniqueDatasetName(IDataSet dataset, bool addTargetColumnAsFirstColumn, bool includeIdColumns)
        {
            var desc = ComputeDescription(dataset);
            if (addTargetColumnAsFirstColumn)
            {
                desc += '_' + ComputeDescription(dataset.Y);
            }
            if (!includeIdColumns)
            {
                desc += "_without_ids";
            }
            return Utils.ComputeHash(desc, 10);
        }
        private static string ComputeDescription(Tensor tensor)
        {
            if (tensor == null || tensor.Count == 0)
            {
                return "";
            }
            Debug.Assert(tensor.Shape.Length == 2);
            var xDataSpan = tensor.AsReadonlyFloatCpuContent;
            var desc = string.Join('_', tensor.Shape);
            for (int col = 0; col < tensor.Shape[1]; ++col)
            {
                int row = ((tensor.Shape[0] - 1) * col) / Math.Max(1, tensor.Shape[1] - 1);
                var val = xDataSpan[row * tensor.Shape[1] + col];
                desc += '_' + Math.Round(val, 6).ToString(CultureInfo.InvariantCulture);
            }
            return desc;
        }
        private static string ComputeDescription(IDataSet dataset)
        {
            if (dataset == null || dataset.Count == 0)
            {
                return "";
            }
            int rows = dataset.Count;
            int cols = dataset.FeatureNames.Length;
            var desc = rows + "_" + cols;
            using CpuTensor<float> xBuffer = new(new[] { 1, cols });
            var xDataSpan = xBuffer.AsReadonlyFloatCpuContent;
            for (int col = 0; col < cols; ++col)
            {
                int row = ((rows - 1) * col) / Math.Max(1, cols - 1);
                dataset.LoadAt(row, 0, xBuffer, null, false);
                var val = xDataSpan[col];
                desc += '_' + Math.Round(val, 6).ToString(CultureInfo.InvariantCulture);
            }
            return desc;
        }

        #region Processing Thread management
        /// <summary>
        /// true if we should use a separate thread to load the content of the next mini batch
        /// while working on the current mini batch
        /// </summary>
        private readonly Thread thread;
        private Tuple<bool, int[], int, DataAugmentationSample, List<int[]>, int[]> threadParameters;
        private readonly AutoResetEvent backgroundThreadHasSomethingTodo = new AutoResetEvent(false);
        private readonly AutoResetEvent backgroundThreadIsIdle = new AutoResetEvent(false);
        private bool shouldStopBackgroundThread = false;
        /// <summary>
        /// number of elements actually loaded by the background thread
        /// </summary>
        private int elementsActuallyLoadedByBackgroundThread = 0;
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
                elementsActuallyLoadedByBackgroundThread = LoadMiniBatchInCpu(threadParameters.Item1, threadParameters.Item2, threadParameters.Item3, threadParameters.Item4, threadParameters.Item5, threadParameters.Item6);
                threadParameters = null;
            }
        }
        #endregion
    }
}
