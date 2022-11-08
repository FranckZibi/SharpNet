using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;

namespace SharpNet.Datasets;

public class InMemoryDataSetV2 : DataSet
{
    // ReSharper disable once PrivateFieldCanBeConvertedToLocalVariable
    private readonly AbstractDatasetSample _datasetSample;

    #region private fields
    private readonly int[] _elementIdToCategoryIndex;
    private CpuTensor<float> _x => XDataFrame.FloatCpuTensor();
    #endregion

    
    public InMemoryDataSetV2(
        AbstractDatasetSample datasetSample,
        [NotNull] DataFrame x_df,
        [CanBeNull] DataFrame y_df,
        bool useBackgroundThreadToLoadNextMiniBatch = false)
        : base(datasetSample.Name,
            datasetSample.GetObjective(),
            x_df.Shape[1],
            null,
            ResizeStrategyEnum.None,
            x_df.Columns,
            Utils.Intersect(x_df.Columns, datasetSample.CategoricalFeatures).ToArray(),
            Utils.Intersect(x_df.Columns, datasetSample.IdColumns).ToArray(),
            datasetSample.TargetLabels,
            useBackgroundThreadToLoadNextMiniBatch,
            datasetSample.GetSeparator())
    {
        _datasetSample = datasetSample;
        Debug.Assert(y_df == null || AreCompatible_X_Y(x_df.FloatCpuTensor(), y_df.FloatCpuTensor()));

        if (IsRegressionProblem || y_df == null)
        {
            _elementIdToCategoryIndex = null;
        }
        else
        {
            _elementIdToCategoryIndex = y_df.Shape[1] == 1 
                ? y_df.FloatCpuTensor().ReadonlyContent.Select(f => Utils.NearestInt(f)).ToArray() 
                : y_df.FloatCpuTensor().ArgMax().ReadonlyContent.Select(f => Utils.NearestInt(f)).ToArray();
        }

        if (IsClassificationProblem && y_df != null && y_df.Shape[1] == 1 && DatasetSample.NumClass>=2)
        {
            var yFloat = CpuTensor<float>.CreateOneHotTensor(ElementIdToCategoryIndex, y_df.Shape[0], DatasetSample.NumClass);
            y_df = DataFrame.New(yFloat);
        }

        XDataFrame = x_df;
        YDataFrame_InModelFormat = y_df;

    }
    public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation)
    {
        Debug.Assert(indexInBuffer >= 0 && indexInBuffer < xBuffer.Shape[0]);
        //same number of channels / same height  / same width
        //only the first dimension (batch size) can be different
        Debug.Assert(_x.SameShapeExceptFirstDimension(xBuffer));
        _x.CopyTo(_x.Idx(elementId), xBuffer, xBuffer.Idx(indexInBuffer), xBuffer.MultDim0);
        if (yBuffer != null && Y != null)
        {
            Debug.Assert(Y.SameShapeExceptFirstDimension(yBuffer));
            Y.CopyTo(Y.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
        }
    }

    public override int[] YMiniBatch_Shape(int miniBatchSize)
    {
        return new[] { miniBatchSize, DatasetSample.NumClass };
    }

    public (int[] vocabularySizes, int[] embeddingDims, int[] indexesInLastDimensionToUse) EmbeddingDescription(int defaultEmbeddingSize)
    {
        List<int> vocabularySizes = new();
        List<int> embeddingDims = new();
        List<int> indexesInLastDimensionToUse = new();

        for (var i = 0; i < ColumnNames.Length; i++)
        {
            var column = ColumnNames[i];
            
            if (Array.IndexOf(IdColumns, column) >= 0)
            {
                //we'll discard Id columns
                indexesInLastDimensionToUse.Add(i);
                embeddingDims.Add(0); //0 embedding dim :  the feature will be discarded
                vocabularySizes.Add(1);
                continue;
            }

            if (Array.IndexOf(CategoricalFeatures, column) < 0)
            {
                continue;
            }
            indexesInLastDimensionToUse.Add(i);
            embeddingDims.Add(defaultEmbeddingSize);
            var columnStats = _datasetSample.DatasetEncoder[column];
            vocabularySizes.Add(1+columnStats.GetDistinctCategoricalValues().Count);
        }


        return (vocabularySizes.ToArray(), embeddingDims.ToArray(), indexesInLastDimensionToUse.ToArray());
    }
    public override int Count => _x.Shape[0];

    public override int ElementIdToCategoryIndex(int elementId)
    {
        if (IsRegressionProblem)
        {
            throw new Exception("can't return a category index for regression");
        }
        return _elementIdToCategoryIndex[elementId];
    }
    public AbstractDatasetSample DatasetSample => _datasetSample;
    public DataFrame XDataFrame { get; }
    public DataFrame YDataFrame_InModelFormat { get; }
    public override CpuTensor<float> Y => YDataFrame_InModelFormat?.FloatCpuTensor();
    public override string ToString()
    {
        return XDataFrame + " => " + YDataFrame_InModelFormat;
    }
}