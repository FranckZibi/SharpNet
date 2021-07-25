﻿using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.Networks;

namespace SharpNet.Datasets
{
    public interface ITimeSeriesDataSet
    {
        void SetBatchPredictionsForInference(int[] batchElementIds, Tensor batchPredictions);
        Tuple<double, double, double, double, double, double>  GetFeatureStatistics(int featureId);
    }


    public interface IDataSet : IDisposable
    {
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
        /// <param name="shuffledElementId">list of all elementId in 'random' (shuffled) order</param>
        /// <param name="firstIndexInShuffledElementId"></param>
        /// <param name="dataAugmentationConfig"></param>
        /// <param name="all_xMiniBatches"></param>
        /// <param name="yMiniBatch">buffer where all categoryCount (associated with the mini batch) will be stored</param>
        /// <returns>number of actual items loaded,
        /// in range [1, miniBatchSize ]
        ///     with xMiniBatch.Shape[0] = xMiniBatch.Shape[0] </returns>
        int LoadMiniBatch(bool withDataAugmentation,
            int[] shuffledElementId, int firstIndexInShuffledElementId,
            DataAugmentationConfig dataAugmentationConfig, List<CpuTensor<float>> all_xMiniBatches,
            CpuTensor<float> yMiniBatch);

        /// <summary>
        /// Load the element 'elementId' in the buffer 'buffer' at index 'indexInBuffer'
        /// </summary>
        /// <param name="elementId">id of element to store, in range [0, Count-1] </param>
        /// <param name="indexInBuffer">where to store the element in the buffer</param>
        /// <param name="xBuffer">buffer where to store elementId (with a capacity of 'xBuffer.Shape[0]' elements) </param>
        /// <param name="yBuffer">buffer where to store the associate category (with a capacity of 'yBuffer.Shape[0]' elements) </param>
        /// <param name="withDataAugmentation"></param>
        void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer,
            bool withDataAugmentation);

        /// <summary>
        /// number of elements in DataSet
        /// </summary>
        int Count { get; }
        int TypeSize { get; }

        List<Tuple<float, float>> MeanAndVolatilityForEachChannel { get; }

        /// <summary>
        /// element id to associated category index
        /// </summary>
        /// <param name="elementId">id of the element (between 0 and 'Count-1')</param>
        /// <returns>index of the associated category (between 0 and 'CategoryCount-1')</returns>
        int ElementIdToCategoryIndex(int elementId);

        string ElementIdToDescription(int elementId);
        string ElementIdToPathIfAny(int elementId);

        /// <summary>
        /// number of channels of each elements
        /// </summary>
        int Channels { get; }


        List<int[]> XMiniBatch_Shape(int[] shapeForFirstLayer);
        int[] YMiniBatch_Shape(int miniBatchSize);
        [NotNull] CpuTensor<float> Y { get; }

        string Name { get; }

        ITrainingAndTestDataSet SplitIntoTrainingAndValidation(double percentageInTrainingSet);
        // ReSharper disable once UnusedMember.Global
        IDataSet Resize(int targetSize, bool shuffle);
        // ReSharper disable once UnusedMember.Global
        IDataSet Shuffle(Random r);
        IDataSet SubDataSet(Func<int, bool> elementIdInOriginalDataSetToIsIncludedInSubDataSet);



        ResizeStrategyEnum ResizeStrategy { get; }

        /// <summary>
        /// check if we should save the network for the current epoch
        /// </summary>
        bool ShouldCreateSnapshotForEpoch(int epoch, Network network);
        void SaveModelAndParameters(Network network, string modelFilePath, string parametersFilePath);
    }
}
