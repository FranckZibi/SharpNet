using System;
using System.Collections.Generic;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;

namespace SharpNet.Datasets
{
    public interface IDataSet : IDisposable
    {
        /// <summary>
        /// Load 'x.Shape[0]' elements from the 'this' DataSet and copy them into 'x' (and 'y') tensors
        /// The indexes in the 'this' dataset of those 'x.Shape[0]' elements to copy into 'x' are:
        ///     indexInCurrentEpochToElementId[indexFirstElement]
        ///     indexInCurrentEpochToElementId[indexFirstElement+1]
        ///     .../...
        ///     indexInCurrentEpochToElementId[indexFirstElement+x.Shape[0]-1 ]
        /// </summary>
        /// <param name="epoch">index of epoch. The first epoch is 1</param>
        /// <param name="isTraining"></param>
        /// <param name="indexFirstElement">The index of the first element to load (the very first element fo the data set is at index 0</param>
        /// <param name="indexInCurrentEpochToElementId"></param>
        /// <param name="dataAugmentationConfig"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        void Load(int epoch, bool isTraining, int indexFirstElement, IReadOnlyList<int> indexInCurrentEpochToElementId,
            DataAugmentationConfig dataAugmentationConfig, ref Tensor x, ref Tensor y);

        /// <summary>
        /// Load the element 'elementId' in the buffer 'buffer' at index 'indexInBuffer'
        /// </summary>
        /// <param name="elementId">id of element to store, in range [0, Count-1] </param>
        /// <param name="indexInBuffer">where to store the element in the buffer</param>
        /// <param name="buffer">buffer where to store elementId (with a capacity of 'buffer.Shape[0]' elements) </param>
        void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> buffer);


        /// <summary>
        /// number of elements in DataSet
        /// </summary>
        int Count { get; }
        int TypeSize { get; }

        /// <summary>
        ///  number of distinct categories in the DataSet
        /// </summary>
        int Categories { get; }

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

        /// <summary>
        /// true if the current data set is normalized (with mean=0 and volatility=1 in each channel)
        /// </summary>
        bool IsNormalized { get; }

        /// <summary>
        /// category id to associated description
        /// </summary>
        /// <param name="categoryId">id of the category (between 0 and 'Categories-1')</param>
        /// <returns></returns>
        string CategoryIdToDescription(int categoryId);
        /// <summary>
        /// element id to associated category id 
        /// </summary>
        /// <param name="elementId">id of the element (between 0 and 'Count-1')</param>
        /// <returns>id of the associated category (between 0 and 'Categories-1')</returns>
        int ElementIdToCategoryId(int elementId);
        /// <summary>
        /// element id to associated description
        /// </summary>
        /// <param name="elementId">id of the element (between 0 and 'Count-1')</param>
        /// <returns>description of the associated element id</returns>
        string ElementIdToDescription(int elementId);

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
        int[] Y_Shape { get; }
        int[] XMiniBatch_Shape(int miniBatchSize);
        CpuTensor<float> Y { get; }

        string Name { get; }

        void CreatePredictionFile(CpuTensor<float> prediction, string outputFile, string headerIfAny = null);
    }
}
