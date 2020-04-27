using System;
using System.Collections.Generic;
using SharpNet.CPU;
using SharpNet.DataAugmentation;

namespace SharpNet.Datasets
{
    public interface IDataSet : IDisposable
    {
        /// <summary>
        /// Load 'xMiniBatch.Shape[0]' elements from the 'this' DataSet and copy them into 'xMiniBatch' (and 'yMiniBatch') tensors
        /// The indexes in the 'this' dataset of those 'xMiniBatch.Shape[0]' elements to copy into 'xMiniBatch' are:
        ///     shuffledElementId[firstIndexInShuffledElementId]
        ///     shuffledElementId[firstIndexInShuffledElementId+1]
        ///     .../...
        ///     shuffledElementId[firstIndexInShuffledElementId+xMiniBatch.Shape[0]-1 ]
        /// </summary>
        /// <param name="withDataAugmentation">true if data augmentation should be used
        /// if false will return the original (not augmented) input</param>
        /// <param name="shuffledElementId">list of all elementId in 'random' (shuffled) order</param>
        /// <param name="firstIndexInShuffledElementId"></param>
        /// <param name="dataAugmentationConfig"></param>
        /// <param name="xMiniBatch">buffer where all elements (associated with the mini batch) will be stored</param>
        /// <param name="yMiniBatch">buffer where all categoryCount (associated with the mini batch) will be stored</param>
        /// <returns></returns>
        void LoadMiniBatch(bool withDataAugmentation, 
            int[] shuffledElementId, int firstIndexInShuffledElementId,
            DataAugmentationConfig dataAugmentationConfig, CpuTensor<float> xMiniBatch, CpuTensor<float> yMiniBatch);

        /// <summary>
        /// Load the element 'elementId' in the buffer 'buffer' at index 'indexInBuffer'
        /// </summary>
        /// <param name="elementId">id of element to store, in range [0, Count-1] </param>
        /// <param name="indexInBuffer">where to store the element in the buffer</param>
        /// <param name="xBuffer">buffer where to store elementId (with a capacity of 'xBuffer.Shape[0]' elements) </param>
        /// <param name="yBuffer">buffer where to store the associate category (with a capacity of 'yBuffer.Shape[0]' elements) </param>
        void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer);


        /// <summary>
        /// number of elements in DataSet
        /// </summary>
        int Count { get; }
        int TypeSize { get; }

        /// <summary>
        ///  number of distinct categoryCount in the DataSet
        /// </summary>
        int CategoryCount { get; }

        /// <summary>
        /// return the mean of channel 'channel' of the original DataSet (before normalization)
        /// </summary>
        /// <param name="channel">the channel for which we want to extract the mean</param>
        /// <returns>mean of channel 'channel'</returns>
        double OriginalChannelMean(int channel);

        /// <summary>
        /// return the volatility of channel 'channel' of the original DataSet (before normalization)
        /// </summary>
        /// <param name="channel">the channel for which we want to extract the volatility</param>
        /// <returns>volatility of channel 'channel'</returns>
        double OriginalChannelVolatility(int channel);

        List<Tuple<float, float>> MeanAndVolatilityForEachChannel { get; }
        Logger Logger { get; }

        /// <summary>
        /// true if the current data set is normalized (with mean=0 and volatility=1 in each channel)
        /// </summary>
        bool IsNormalized { get; }

        /// <summary>
        /// element id to associated category index
        /// </summary>
        /// <param name="elementId">id of the element (between 0 and 'Count-1')</param>
        /// <returns>index of the associated category (between 0 and 'CategoryCount-1')</returns>
        int ElementIdToCategoryIndex(int elementId);

        ImageStatistic ElementIdToImageStatistic(int elementId);

        /// <summary>
        /// number of channels of each elements
        /// </summary>
        int Channels { get; }
        /// <summary>
        /// Current height of elements to load
        /// </summary>
        int Height { get; }
        /// <summary>
        /// Current width of elements to load
        /// </summary>
        int Width { get; }

        int[] InputShape_CHW {get;}
        int[] Y_Shape { get; }
        int[] XMiniBatch_Shape(int miniBatchSize);
        int[] YMiniBatch_Shape(int miniBatchSize);
        CpuTensor<float> Y { get; }

        string Name { get; }

        void CreatePredictionFile(CpuTensor<float> prediction, string outputFile, string headerIfAny = null);
    }
}
