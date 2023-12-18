using System;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;

namespace SharpNet.Datasets;

public class DataFrameDataSet : DataSet
{
    // ReSharper disable once PrivateFieldCanBeConvertedToLocalVariable

    #region private fields and properties
    private readonly int[] _elementIdToCategoryIndex;
    private CpuTensor<float> _x => XDataFrame.FloatTensor;
    [CanBeNull] private CpuTensor<float> yDataFrameDataSet => YDataFrame_InModelFormat?.FloatTensor;
    private DataFrame YDataFrame_InModelFormat { get; }
    #endregion

    #region public fields and properties
    public DataFrame XDataFrame { get; }
    public AbstractDatasetSample DatasetSample { get; }
    #endregion

    public DataFrameDataSet(
        AbstractDatasetSample datasetSample,
        [NotNull] DataFrame x_df,
        [CanBeNull] DataFrame y_df,
        [CanBeNull] string[] y_IDs)
        : base(datasetSample.Name,
            datasetSample.GetObjective(),
            null,
            ResizeStrategyEnum.None,
            x_df.Columns,
            Utils.Intersect(x_df.Columns, datasetSample.CategoricalFeatures).ToArray(),
            y_IDs,
            datasetSample.IdColumn, 
            datasetSample.GetSeparator())
    {
        DatasetSample = datasetSample;
        Debug.Assert(y_df == null || AreCompatible_X_Y(x_df.FloatTensor, y_df.FloatTensor));

        if (IsRegressionProblem || y_df == null)
        {
            _elementIdToCategoryIndex = null;
        }
        else
        {
            _elementIdToCategoryIndex = y_df.Shape[1] == 1 
                ? y_df.FloatTensor.ReadonlyContent.Select(f => Utils.NearestInt(f)).ToArray() 
                : y_df.FloatTensor.ArgMax().ReadonlyContent.Select(f => Utils.NearestInt(f)).ToArray();
        }

        XDataFrame = x_df;
        YDataFrame_InModelFormat = y_df;
    }

    public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
    {
        if (xBuffer != null)
        {
            //same number of channels / same height  / same width
            //only the first dimension (batch size) can be different
            Debug.Assert(_x.SameShapeExceptFirstDimension(xBuffer));
            Debug.Assert(indexInBuffer >= 0 && indexInBuffer < xBuffer.Shape[0]);
            _x.CopyTo(_x.Idx(elementId), xBuffer, xBuffer.Idx(indexInBuffer), xBuffer.MultDim0);
        }

        if (yBuffer != null && yDataFrameDataSet != null)
        {
            Debug.Assert(indexInBuffer >= 0 && indexInBuffer < yBuffer.Shape[0]);
            Debug.Assert(yDataFrameDataSet.SameShapeExceptFirstDimension(yBuffer));
            yDataFrameDataSet.CopyTo(yDataFrameDataSet.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
        }
    }

    public override int[] X_Shape(int batchSize) => Utils.CloneShapeWithNewCount(_x.Shape, batchSize);
    public override int[] Y_Shape(int batchSize) => Utils.CloneShapeWithNewCount(yDataFrameDataSet?.Shape, batchSize);

    public override CpuTensor<float> LoadFullY()
    {
        return yDataFrameDataSet;
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
    public override string ToString()
    {
        return XDataFrame + " => " + YDataFrame_InModelFormat;
    }
    public override AbstractDatasetSample GetDatasetSample()
    {
        return DatasetSample;
    }
}