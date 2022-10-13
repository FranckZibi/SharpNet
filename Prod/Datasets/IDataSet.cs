using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;
using SharpNet.Models;
using SharpNet.Networks;

namespace SharpNet.Datasets
{
    public interface ITimeSeriesDataSet
    {
        void SetBatchPredictionsForInference(int[] batchElementIds, Tensor batchPredictions);
        Tuple<double, double, double, double, double, double>  GetEncoderFeatureStatistics(int featureId);
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
        /// <param name="dataAugmentationSample"></param>
        /// <param name="all_xMiniBatches"></param>
        /// <param name="yMiniBatch">buffer where all categoryCount (associated with the mini batch) will be stored</param>
        /// <returns>number of actual items loaded,
        /// in range [1, miniBatchSize ]
        ///     with xMiniBatch.Shape[0] = xMiniBatch.Shape[0] </returns>
        int LoadMiniBatch(bool withDataAugmentation,
            int[] shuffledElementId, int firstIndexInShuffledElementId,
            DataAugmentationSample dataAugmentationSample, List<CpuTensor<float>> all_xMiniBatches,
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
        bool UseBackgroundThreadToLoadNextMiniBatch { get; }

        List<Tuple<float, float>> MeanAndVolatilityForEachChannel { get; }

        /// <summary>
        /// element id to associated category index
        /// </summary>
        /// <param name="elementId">id of the element (between 0 and 'Count-1')</param>
        /// <returns>index of the associated category (between 0 and 'CategoryCount-1')</returns>
        int ElementIdToCategoryIndex(int elementId);

        string ElementIdToDescription(int elementId);
        string ElementIdToPathIfAny(int elementId);


        DataFrame AddIdColumnsAtLeftIfNeeded(DataFrame df);

        bool IsRegressionProblem {get;}
        
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
        string to_csv_in_directory([NotNull] string directory, bool addTargetColumnAsFirstColumn, bool includeIdColumns, bool overwriteIfExists);


        /// <summary>
        /// all column names in the Training DataSet, including Id Columns (if any) and  all Features Columns
        /// this will never include the target
        /// </summary>
        string[] ColumnNames { get; }

        string[] CategoricalFeatures { get; }
        string[] IdColumns { get; }
        string[] TargetLabels { get; }
        char Separator { get; }

        /// <summary>
        /// the type of use of the dataset : Regression or Classification
        /// </summary>
        Objective_enum Objective { get; }

        /// <summary>
        /// number of channels of each elements
        /// </summary>
        int Channels { get; }


        /// <summary>
        /// the length of the returned list is:
        ///     2 for Encoder Decoder (first is the shape of the network input, second is the shape of the decoder input)
        ///     1 for all other kind of networks (with the shape of the network input)
        /// </summary>
        /// <param name="shapeForFirstLayer"></param>
        /// <returns></returns>
        List<int[]> XMiniBatch_Shape(int[] shapeForFirstLayer);
        int[] YMiniBatch_Shape(int miniBatchSize);
        [CanBeNull] CpuTensor<float> Y { get; }
        public DataFrame Y_InModelFormat(int numClasses, bool includeIdColumns);
        DataFrame Y_InTargetFormat(bool includeIdColumns);

        string Name { get; }

        ITrainingAndTestDataSet SplitIntoTrainingAndValidation(double percentageInTrainingSet);
        ITrainingAndTestDataSet IntSplitIntoTrainingAndValidation(int countInTrainingSet);
        // ReSharper disable once UnusedMember.Global
        IDataSet Resize(int targetSize, bool shuffle);
        // ReSharper disable once UnusedMember.Global
        IDataSet Shuffle(Random r);

        /// <summary>
        /// return a data set keeping only 'percentageToKeep' elements of the current data set
        /// </summary>
        /// <param name="percentageToKeep">percentage to keep between 0 and 1.0</param>
        /// <returns></returns>
        IDataSet SubDataSet(double percentageToKeep);

        IDataSet SubDataSet(Func<int, bool> elementIdInOriginalDataSetToIsIncludedInSubDataSet);

        double PercentageToUseForLossAndAccuracyFastEstimate { get; }

        ResizeStrategyEnum ResizeStrategy { get; }

        /// <summary>
        /// check if we should save the network for the current epoch
        /// </summary>
        bool ShouldCreateSnapshotForEpoch(int epoch, Network network);
        void Save(IModel network, string workingDirectory, string modelName);
    }
}
