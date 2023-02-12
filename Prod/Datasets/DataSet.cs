using System;
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
    public abstract class DataSet : IDisposable
    {
        #region private & protected fields
        protected static readonly ILog Log = LogManager.GetLogger(typeof(DataSet));
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
        private readonly CpuTensor<float> yDataAugmentedMiniBatch = new(new[] { 1 });
        /// <summary>
        /// the miniBatch Id associated with the above xBufferMiniBatchCpu & yBufferMiniBatchCpu tensors
        /// or -1 if those tensors are empty
        /// </summary>
        private long alreadyComputedMiniBatchId = -1;

        private readonly Random[] _rands;
        #endregion

        #region public properties
        public List<Tuple<float, float>> MeanAndVolatilityForEachChannel { get; }
        /// <summary>
        /// the entire list of columns in the training DataSet (with all Id Columns & Features), but without the Target Columns
        /// </summary>
        [NotNull] public string[] ColumnNames { get; }
        [NotNull] public string[] CategoricalFeatures { get; }
        [NotNull] public string[] IdColumns { get; }
        public bool UseBackgroundThreadToLoadNextMiniBatch { get; }
        public char Separator { get; }
        #endregion

        #region constructor
        protected DataSet(string name,
            Objective_enum objective,
            int channels,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel,
            ResizeStrategyEnum resizeStrategy,
            [NotNull] string[] columnNames,
            [NotNull] string[] categoricalFeatures,
            [NotNull] string[] idColumns,
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
            ColumnNames = columnNames;

            var invalidCategoricalFeatures = Utils.Without(categoricalFeatures, ColumnNames);
            if (invalidCategoricalFeatures.Count != 0)
            {
                Log.Error($"{invalidCategoricalFeatures.Count} invalid CategoricalFeatures: {string.Join(' ', invalidCategoricalFeatures)} => IGNORING");
                categoricalFeatures = Utils.Intersect(categoricalFeatures, ColumnNames).ToArray();
            }
            CategoricalFeatures = categoricalFeatures;

            if (idColumns.Length != 0)
            {
                var foundIdColumns = Utils.Intersect(idColumns, ColumnNames);
                if (foundIdColumns.Count < idColumns.Length)
                {
                    var missingIdColumns = Utils.Without(idColumns, foundIdColumns);

                    var errorMsg = $" {missingIdColumns.Count} missing IdColumns: {string.Join(' ', missingIdColumns)}";
                    Log.Error(errorMsg);
                    throw new ArgumentException(errorMsg);
                }
            }
            IdColumns = idColumns;

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
        public bool IsClassificationProblem => Objective == Objective_enum.Classification;

        /// <summary>
        /// Load the element 'elementId' in the buffer 'buffer' at index 'indexInBuffer'
        /// </summary>
        /// <param name="elementId">id of element to store, in range [0, Count-1] </param>
        /// <param name="indexInBuffer">where to store the element in the buffer</param>
        /// <param name="xBuffer">buffer where to store elementId (with a capacity of 'xBuffer.Shape[0]' elements) </param>
        /// <param name="yBuffer">buffer where to store the associate category (with a capacity of 'yBuffer.Shape[0]' elements) </param>
        /// <param name="withDataAugmentation"></param>
        /// <param name="isTraining"></param>
        public abstract void LoadAt(int elementId, int indexInBuffer, [NotNull] CpuTensor<float> xBuffer,
            [CanBeNull] CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining);

        protected virtual void LoadAt(int elementId, int indexInBuffer, [NotNull] List<CpuTensor<float>> xBuffers,
            [CanBeNull] CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
        {
            if (xBuffers.Count != 1)
            {
                throw new ArgumentException("only 1 InputLayer is supported, received "+xBuffers.Count);
            }
            LoadAt(elementId, indexInBuffer, xBuffers[0], yBuffer, withDataAugmentation, isTraining);
        }


        /// <summary>
        /// Load 'miniBatch' (= xMiniBatch.Shape[0](') elements from the 'this' DataSet and copy them into 'xMiniBatch' (and 'yMiniBatch') tensors
        /// The indexes in the 'this' dataset of those 'miniBatch' elements to copy into 'xMiniBatch' are:
        ///     shuffledElementId[firstIndexInShuffledElementId]
        ///     shuffledElementId[firstIndexInShuffledElementId+1]
        ///     .../...
        ///     shuffledElementId[firstIndexInShuffledElementId+'miniBatch'-1 ]
        /// </summary>
        /// <param name="withDataAugmentation">true if data augmentation should be used
        ///     if false will return the original (not augmented) input</param>
        /// <param name="isTraining"></param>
        /// <param name="shuffledElementId">list of all elementId in 'random' (shuffled) order</param>
        /// <param name="firstIndexInShuffledElementId"></param>
        /// <param name="dataAugmentationSample"></param>
        /// <param name="all_xMiniBatches"></param>
        /// <param name="yMiniBatch">buffer where all categoryCount (associated with the mini batch) will be stored</param>
        /// <returns>number of actual items loaded,
        /// in range [1, miniBatchSize ]
        ///     with xMiniBatch.Shape[0] = xMiniBatch.Shape[0] </returns>
        public int LoadMiniBatch(bool withDataAugmentation, bool isTraining, int[] shuffledElementId,
            int firstIndexInShuffledElementId, NetworkSample dataAugmentationSample,
            [NotNull] List<CpuTensor<float>> all_xMiniBatches, [NotNull] CpuTensor<float> yMiniBatch)
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
                    elementsActuallyLoaded = LoadMiniBatchInCpu(withDataAugmentation, isTraining, shuffledElementId, firstIndexInShuffledElementId, dataAugmentationSample, all_xMiniBatchesShapes, yMiniBatch.Shape);
                    if (elementsActuallyLoaded == shuffledElementId.Length)
                    {
                        elementsActuallyLoadedByBackgroundThread = elementsActuallyLoaded;
                    }
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
                elementsActuallyLoaded = LoadMiniBatchInCpu(withDataAugmentation, isTraining, shuffledElementId, firstIndexInShuffledElementId, dataAugmentationSample, all_xMiniBatchesShapes, yMiniBatch.Shape);
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
            //    //var meanAndVolatilityOfEachChannel = new List<Tuple<double, double>> { Tuple.Create(125.306918046875, 62.9932192781369), Tuple.Create(122.950394140625, 62.0887076400142), Tuple.Create(113.865383183594, 66.7048996406309) };
            //    //SVHN
            //    //var meanAndVolatilityOfEachChannel = new List<Tuple<float, float>> { Tuple.Create(109.8823f, 50.11187f), Tuple.Create(109.7114f, 50.57312f), Tuple.Create(113.8187f, 50.85124f) };
            //    //EffiScience95
            //    var meanAndVolatilityOfEachChannel = new List<Tuple<float, float>> { Tuple.Create(128.42516f, 79.42157f), Tuple.Create(107.48822f, 74.195564f), Tuple.Create(97.46115f, 73.76817f) };
            //    var xCpuChunkBytes = all_xDataAugmentedMiniBatch[0].Select((_, c, val) => (byte)((val * meanAndVolatilityOfEachChannel[c].Item2 + meanAndVolatilityOfEachChannel[c].Item1)));
            //    for (int i = firstIndexInShuffledElementId; i < (firstIndexInShuffledElementId + all_xDataAugmentedMiniBatch[0].Shape[0]); ++i)
            //    {
            //        int elementId = shuffledElementId[i];
            //        var categoryIndex = ElementIdToCategoryIndex(elementId);
            //        PictureTools.SaveBitmap(xCpuChunkBytes, i- firstIndexInShuffledElementId, Path.Combine(NetworkSample.DefaultWorkingDirectory, "Train"), elementId.ToString("D5") + "_cat" + categoryIndex, "");
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
                threadParameters = Tuple.Create(withDataAugmentation, isTraining, shuffledElementId, firstIndexInShuffledElementIdForNextMiniBatch, dataAugmentationSample, xNextMiniBatchShape, yNextMiniBatchShape);
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
        /// <param name="isTraining"></param>
        /// <returns>a byte tensor containing the element at index 'elementId' </returns>
        public virtual BitmapContent OriginalElementContent(int elementId, int targetHeight, int targetWidth,
            bool withDataAugmentation, bool isTraining)
        {
            var xBuffer = new CpuTensor<float>(new[] { 1, Channels, targetHeight, targetWidth });
            LoadAt(elementId, 0, xBuffer, null, withDataAugmentation, isTraining);

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
            return ImageStatistic.ValueOf(OriginalElementContent(elementId, targetHeight, targetWidth, false, false));
        }


        /// <summary>
        /// number of elements in DataSet
        /// </summary>
        public abstract int Count { get; }
        public virtual DataFrame AddIdColumnsAtLeftIfNeeded(DataFrame df)
        {
            if (IdColumns.Length == 0)
            {
                return df;
            }
            var intersection = Utils.Intersect(df.Columns, IdColumns);
            if (intersection.Count == 0)
            {
                return DataFrame.MergeHorizontally(ExtractIdDataFrame(), df);
            }

            if (intersection.Count == IdColumns.Length)
            {
                // all Id Columns are already in the DataFrame, nothing to do
                return df;
            }
            //some Id Columns are in the DataFrame and some other are not
            var errorMsg = $"found only a part {string.Join(' ', intersection)} of Id Columns ({string.Join(' ', IdColumns)})";
            Log.Error(errorMsg);
            throw new Exception(errorMsg);
        }

        public string Name { get; }
        public ResizeStrategyEnum ResizeStrategy { get; }
        /// <summary>
        /// check if we should save the network for the current epoch
        /// </summary>
        public virtual bool ShouldCreateSnapshotForEpoch(int epoch, Network network)
        {
            return false;
        }

        public virtual void Save(Model model, string workingDirectory, string modelName)
        {
            model.Save(workingDirectory, modelName);
        }

        public virtual DataFrame ExtractIdDataFrame()
        {
            if (IdColumns.Length == 0)
            {
                // can't extract id columns because the DataFrame doesn't contain any
                return null;
            }
            var columnIndexesOfIds = IdColumns.Select(id => Array.IndexOf(ColumnNames, id)).ToList();
            columnIndexesOfIds.Sort();

            var content = new float[Count * IdColumns.Length];
            int cols = ColumnNames.Length;
            using CpuTensor<float> singleRow = new(new[] { 1, cols });
            var singleRowAsSpan = singleRow.AsReadonlyFloatCpuContent;
            int nextIdx = 0;
            for (int row = 0; row < Count; ++row)
            {
                LoadAt(row, 0, singleRow, null, false, false);
                foreach (var colIdx in columnIndexesOfIds)
                {
                    content[nextIdx++] = singleRowAsSpan[colIdx];
                }
            }
            return DataFrame.New(content, IdColumns);
        }

        public (DataFrame X, DataFrame Y) ExtractXYDataFrame()
        {
            int cols = ColumnNames.Length;
            CpuTensor<float> X = new(new[] { Count, cols });
            for (int row = 0; row < Count; ++row)
            {
                LoadAt(row, row, X, null, false, false);
            }
            return (DataFrame.New(X, ColumnNames), DataFrame.New(Y, new List<string>{"y"}));
        }


        /// <summary>
        /// number of channels of each elements
        /// </summary>
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
        public DataSet Resize(int targetSize, bool shuffle)
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

        //public DataSet Shuffle(Random r)
        //{
        //    var elementIdToOriginalElementId = Enumerable.Range(0, Count).ToList();
        //    Utils.Shuffle(elementIdToOriginalElementId, r);
        //    return new MappedDataSet(this, elementIdToOriginalElementId);
        //}

        /// <summary>
        /// return a data set keeping only 'percentageToKeep' elements of the current data set
        /// </summary>
        /// <param name="percentageToKeep">percentage to keep between 0 and 1.0</param>
        /// <returns></returns>
        public virtual DataSet SubDataSet(double percentageToKeep)
        {
            return SubDataSet(_ => _rands[0].NextDouble() < percentageToKeep);
        }

        public virtual double PercentageToUseForLossAndAccuracyFastEstimate => 0.1;

        public DataSet SubDataSet(Func<int, bool> elementIdInOriginalDataSetToIsIncludedInSubDataSet)
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
        public virtual string ElementIdToPathIfAny(int elementId)
        {
            return "";
        }
        
        #region Dispose pattern
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        // ReSharper disable once RedundantDefaultMemberInitializer
        public bool Disposed { get; set; } = false;
        protected virtual void Dispose(bool disposing)
        {
            if (Disposed)
            {
                return;
            }
            Disposed = true;
            //Release Unmanaged Resources
            if (disposing)
            {
                //Release Managed Resources
                all_xOriginalNotAugmentedMiniBatch.ForEach(t => t.Dispose());
                all_xOriginalNotAugmentedMiniBatch.Clear();
                all_xDataAugmentedMiniBatch.ForEach(t => t.Dispose());
                all_xDataAugmentedMiniBatch.Clear();
                all_xBufferForDataAugmentedMiniBatch.ForEach(t => t.Dispose());
                all_xBufferForDataAugmentedMiniBatch.Clear();
                yDataAugmentedMiniBatch?.Dispose();
            }
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
        ~DataSet()
        {
            Dispose(false);
        }
        #endregion

        public virtual int[] YMiniBatch_Shape(int miniBatchSize)
        {
            var yMiniBatchShape = (int[])Y.Shape.Clone();
            yMiniBatchShape[0] = miniBatchSize;
            return yMiniBatchShape;
        }

        /// <summary>
        /// the length of the returned list is:
        ///     2 for Encoder Decoder (first is the shape of the network input, second is the shape of the decoder input)
        ///     1 for all other kind of networks (with the shape of the network input)
        /// </summary>
        /// <param name="shapeForFirstLayer"></param>
        /// <returns></returns>
        public virtual List<int[]> XMiniBatch_Shape(int[] shapeForFirstLayer)
        {
            return new List<int[]> {shapeForFirstLayer};
        }

        // ReSharper disable once MemberCanBeMadeStatic.Global
        public int TypeSize => 4; //float size
        public override string ToString() {return Name;}

        public virtual ITrainingAndTestDataset SplitIntoTrainingAndValidation(double percentageInTrainingSet)
        {
            int countInTrainingSet = (int)(percentageInTrainingSet * Count+0.1);
            return IntSplitIntoTrainingAndValidation(countInTrainingSet);
        }
        public virtual ITrainingAndTestDataset IntSplitIntoTrainingAndValidation(int countInTrainingSet)
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
        /// <summary>
        /// Y shape for
        ///     regression:                 (_, 1)
        ///     binary classification:      (_, 1)  where each element is a probability in [0, 1] range
        ///     multi class classification  (_, NumClasses) with each element being the probability of belonging to this class
        /// </summary>
        public abstract CpuTensor<float> Y { get; }

        /// <summary>
        /// Y_InModelFormat shape for
        ///     regression:                 (_, 1)
        ///     binary classification:      (_, 1)  where each element is a probability in [0, 1) range
        ///     multi class classification  (_, NumClasses) with each element being the probability of belonging to this class
        /// </summary>
        /// <returns></returns>
        public DataFrame Y_InModelFormat()
        {
            return Y == null ? null : DataFrame.New(Y);
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
        /// <returns>the number of actually loaded elements,in the range [1, xMiniBatchShape[0] ]  </returns>
        private int LoadMiniBatchInCpu(bool withDataAugmentation,
            bool isTraining,
            int[] shuffledElementId, int firstIndexInShuffledElementId,
            NetworkSample dataAugmentationSample,
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
            Parallel.For(0, miniBatchSize, indexInBuffer => LoadAt(MiniBatchIdxToElementId(indexInBuffer%maxElementsToLoad), indexInBuffer, all_xOriginalNotAugmentedMiniBatch, yDataAugmentedMiniBatch, withDataAugmentation, isTraining));

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
        /// save the dataset in path 'path' in 'libsvm' format.
        /// </summary>
        /// <param name="path">the path where to save the dataset (it contains both the directory and the filename)</param>
        /// <param name="overwriteIfExists">overwrite the file if it already exists</param>
        /// <param name="separator"></param>
        private void to_libsvm(string path, char separator, bool overwriteIfExists)
        {
            lock (Lock_to_csv)
            {
                if (File.Exists(path) && !overwriteIfExists)
                {
                    //Log.Debug($"No need to save dataset {Name} in path {path} : it already exists");
                    return;
                }

                var sw = Stopwatch.StartNew();
                Log.Debug($"Saving dataset {Name} in path {path}");

                // ReSharper disable once PossibleNullReferenceException
                var yDataAsSpan = Y != null ? Y.AsFloatCpuSpan : null;

                List<int> validIdxColumns = new();
                for (var index = 0; index < ColumnNames.Length; index++)
                {
                    if (Array.IndexOf(IdColumns, ColumnNames[index]) == -1)
                    {
                        validIdxColumns.Add(index);
                    }
                }

                var sb = new StringBuilder();
                using var singleRow = new CpuTensor<float>(new[] { 1, ColumnNames.Length });
                var singleRowAsSpan = singleRow.AsReadonlyFloatCpuContent;
                int idx = 0;
                for (int row = 0; row < Count; ++row)
                {
                    float yValue;
                    if (yDataAsSpan == null) { yValue = AbstractSample.DEFAULT_VALUE; }
                    else if (IsRegressionProblem) { yValue = yDataAsSpan[row]; }
                    else { yValue = ElementIdToCategoryIndex(row); }
                    sb.Append(yValue.ToString(CultureInfo.InvariantCulture)+" ");
                    LoadAt(row, 0, singleRow, null, false, false);
                    for (var index = 0; index < validIdxColumns.Count; index++)
                    {
                        var val = singleRowAsSpan[validIdxColumns[index]];
                        if (!float.IsNaN(val) && !float.IsInfinity(val))
                        {
                            sb.Append($"{1 + index}:{val.ToString(CultureInfo.InvariantCulture)} ");
                        }
                    }

                    sb.Append(Environment.NewLine);
                }

                File.WriteAllText(path, sb.ToString());
                Log.Debug($"Dataset {Name} saved in path {path} in {Math.Round(sw.Elapsed.TotalSeconds, 1)}s)");
            }

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
            lock (Lock_to_csv)
            {
                if (File.Exists(path) && !overwriteIfExists)
                {
                    //Log.Debug($"No need to save dataset {Name} in path {path} : it already exists");
                    return;
                }

                var sw = Stopwatch.StartNew();
                Log.Debug($"Saving dataset {Name} in path {path} (addTargetColumnAsFirstColumn =  {addTargetColumnAsFirstColumn})");

                // ReSharper disable once PossibleNullReferenceException
                var yDataAsSpan = (addTargetColumnAsFirstColumn && Y != null) ? Y.AsFloatCpuSpan : null;

                List<int> validIdxColumns;
                if (includeIdColumns)
                {
                    validIdxColumns = Enumerable.Range(0, ColumnNames.Length).ToList();
                }
                else
                {
                    validIdxColumns = new List<int>();
                    for (var index = 0; index < ColumnNames.Length; index++)
                    {
                        if (Array.IndexOf(IdColumns, ColumnNames[index]) == -1)
                        {
                            validIdxColumns.Add(index);
                        }
                    }
                }

                //we construct the header
                var header = new List<string>();
                if (addTargetColumnAsFirstColumn)
                {
                    header.Add("y");
                }
                header.AddRange(validIdxColumns.Select(idx => ColumnNames[idx]));

                int rows = Count;
                int cols = (addTargetColumnAsFirstColumn ? 1 : 0) + validIdxColumns.Count;
                using var floatTensor = new CpuTensor<float>(new[] { rows, cols });
                using var singleRow = new CpuTensor<float>(new[] { 1, ColumnNames.Length });
                var floatTensorSpan = floatTensor.SpanContent;
                var singleRowAsSpan = singleRow.AsReadonlyFloatCpuContent;
                int idx = 0;
                for (int row = 0; row < Count; ++row)
                {
                    if (addTargetColumnAsFirstColumn)
                    {
                        float yValue;
                        if (yDataAsSpan == null) { yValue = AbstractSample.DEFAULT_VALUE; }
                        else if (IsRegressionProblem) { yValue = yDataAsSpan[row]; }
                        else  { yValue = ElementIdToCategoryIndex(row); }
                        floatTensorSpan[idx++] = yValue;
                    }
                    LoadAt(row, 0, singleRow, null, false, false);
                    foreach (var colIdx in validIdxColumns)
                    {
                        floatTensorSpan[idx++] = singleRowAsSpan[colIdx];
                    }
                }

                var df = DataFrame.New(floatTensor, header);
                df.to_csv(path, separator, true);
                Log.Debug($"Dataset {Name} saved in path {path} in {Math.Round(sw.Elapsed.TotalSeconds, 1)}s (addTargetColumnAsFirstColumn =  {addTargetColumnAsFirstColumn})");
            }
        }

        /// <summary>
        /// save the dataset in directory 'directory' in 'LightGBM' format.
        /// if addTargetColumnAsFirstColumn == true:
        ///     first column is the label 'y' (to predict)
        ///     all other columns are the features
        /// else
        ///     save only 'x' (feature) tensor
        /// </summary>
        /// <param name="directory">the directory where to save to dataset</param>
        /// <param name="addTargetColumnAsFirstColumn"></param>
        /// <param name="includeIdColumns"></param>
        /// <param name="overwriteIfExists"></param>
        /// <returns>the path (directory+filename) where the dataset has been saved</returns>
        public virtual string to_csv_in_directory(string directory, bool addTargetColumnAsFirstColumn, bool includeIdColumns, bool overwriteIfExists)
        {
            if (ColumnNames.Length == 0)
            {
                return ""; //nothing to save
            }
            var datasetPath = Path.Combine(directory, ComputeUniqueDatasetName(this, addTargetColumnAsFirstColumn, includeIdColumns) + ".csv");
            to_csv(datasetPath, Separator, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            return datasetPath;
        }

        /// <summary>
        /// save the dataset in directory 'directory' in 'libsvm' format.
        /// </summary>
        /// <param name="directory">the directory where to save to dataset</param>
        /// <param name="overwriteIfExists"></param>
        /// <returns>the path (directory+filename) where the dataset has been saved</returns>
        public virtual string to_libsvm_in_directory(string directory, bool overwriteIfExists)
        {
            if (ColumnNames.Length == 0)
            {
                return ""; //nothing to save
            }
            var datasetPath = Path.Combine(directory, ComputeUniqueDatasetName(this, true, false) + ".csv");
            to_libsvm(datasetPath, Separator, overwriteIfExists);
            return datasetPath;
        }

      
        public virtual List<TrainingAndTestDataset> KFoldSplit(int kfold, int countMustBeMultipleOf)
        {
            var validationIntervalForKfold = KFoldModel.KFoldIntervals(kfold, Count, countMustBeMultipleOf);
            List<TrainingAndTestDataset> res = new();
            for (var index = 0; index < validationIntervalForKfold.Count; index++)
            {
                var intervalForValidation = validationIntervalForKfold[index];
                var training = SubDataSet(id => id < intervalForValidation.Item1 || id > intervalForValidation.Item2);
                var test = SubDataSet(id => id >= intervalForValidation.Item1 && id <= intervalForValidation.Item2);
                res.Add(new TrainingAndTestDataset(training, test, KFoldModel.KFoldModelNameEmbeddedModelName(Name, index)));
            }
            return res;
        }


        private static readonly object Lock_to_csv = new();
        private static readonly object Lock_to_libsvm = new();

        private static long ComputeMiniBatchHashId(int[] shuffledElementId, int firstIndexInShuffledElementId, int miniBatchSize)
        {
            long result = 1;
            for (int i = firstIndexInShuffledElementId; i < firstIndexInShuffledElementId + miniBatchSize; ++i)
            {
                result += (i + 1) * shuffledElementId[i%shuffledElementId.Length];
            }
            return result;
        }
        private static string ComputeUniqueDatasetName(DataSet dataset, bool addTargetColumnAsFirstColumn, bool includeIdColumns)
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
        private static string ComputeDescription(DataSet dataset)
        {
            if (dataset == null || dataset.Count == 0)
            {
                return "";
            }
            int rows = dataset.Count;
            int cols = dataset.ColumnNames.Length;
            var desc = rows + "_" + cols;
            using CpuTensor<float> xBuffer = new(new[] { 1, cols });
            var xDataSpan = xBuffer.AsReadonlyFloatCpuContent;
            for (int col = 0; col < cols; ++col)
            {
                int row = ((rows - 1) * col) / Math.Max(1, cols - 1);
                dataset.LoadAt(row, 0, xBuffer, null, false, false);
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
        private Tuple<bool, bool, int[], int, NetworkSample, List<int[]>, int[]> threadParameters;
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
                elementsActuallyLoadedByBackgroundThread = LoadMiniBatchInCpu(threadParameters.Item1, threadParameters.Item2, threadParameters.Item3, threadParameters.Item4, threadParameters.Item5, threadParameters.Item6, threadParameters.Item7);
                threadParameters = null;
            }
        }
        #endregion
    }
}
