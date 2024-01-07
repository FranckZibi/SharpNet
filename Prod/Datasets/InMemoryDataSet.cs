using System;
using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.CPU;

namespace SharpNet.Datasets;

public class InMemoryDataSet : DataSet
{
    #region private fields
    private readonly int[] _elementIdToCategoryIndex;
    private readonly CpuTensor<float> _x;
    [CanBeNull] private readonly CpuTensor<float> _yInMemoryDataSet;
    private ConstDatasetSample constDatasetSample;
    private CpuTensor<float> x;
    private CpuTensor<float> y;
    private string name;
    private List<Tuple<float, float>> meanAndVolatilityForEachChannel;
    private ResizeStrategyEnum none;
    private string[] strings;
    private string[] y_IDs;
    private char separator;
    #endregion


    private static ConstDatasetSample NewConstDatasetSample(
        [NotNull] CpuTensor<float> x,
        [CanBeNull] CpuTensor<float> y,
        Objective_enum objective,
        Func<string, bool> isCategoricalColumn,
        [CanBeNull] string idColumn)
    {
        string[] targetLabels = {"y"};
        int[] x_shape_for_1_batchSize = (int[])x.Shape.Clone();
        int[] y_shape_for_1_batchSize = (y==null)?(new[]{1,1}): (int[])y.Shape.Clone();
        y_shape_for_1_batchSize[0] = 1;
        int numClass = y_shape_for_1_batchSize[^1]; 
        return new ConstDatasetSample(idColumn, targetLabels, x_shape_for_1_batchSize, y_shape_for_1_batchSize, numClass, objective, isCategoricalColumn);
    }

    public InMemoryDataSet(
        [NotNull] AbstractDatasetSample datasetSample,
        [NotNull] CpuTensor<float> x,
        [CanBeNull] CpuTensor<float> y,
        string name = "",
        List<Tuple<float, float>> meanAndVolatilityForEachChannel = null,
        string[] columnNames = null,
        [CanBeNull] string[] y_IDs = null,
        char separator = ',')
        : base(name,
            datasetSample,
            meanAndVolatilityForEachChannel,
            ResizeStrategyEnum.None,
            columnNames ?? new string[0],
            y_IDs,
            separator)
    {
        Debug.Assert(x != null);
        Debug.Assert(y==null || AreCompatible_X_Y(x, y));
        //Debug.Assert(x.Shape[1] == columnNames.Length);

        _x = x;
        _yInMemoryDataSet = y;

        if (IsRegressionProblem || y == null)
        {
            _elementIdToCategoryIndex = null;
        }
        else
        {
            _elementIdToCategoryIndex = new int[y.Shape[0]];
            var ySpan = y.AsReadonlyFloatCpuSpan;
            for (int elementId = 0; elementId < y.Shape[0]; ++elementId)
            {
                int startIndex = elementId * y.Shape[1];
                for (int categoryIdx = 0; categoryIdx < y.Shape[1]; ++categoryIdx)
                {
                    if (ySpan[startIndex + categoryIdx] > 0.9f)
                    {
                        _elementIdToCategoryIndex[elementId] = categoryIdx;
                    }
                }
            }
        }
    }


    public InMemoryDataSet([NotNull] CpuTensor<float> x,
        [CanBeNull] CpuTensor<float> y,
        string name = "",
        Objective_enum objective = Objective_enum.Regression,
        List<Tuple<float, float>> meanAndVolatilityForEachChannel = null,
        string[] columnNames = null,
        Func<string, bool> isCategoricalColumn = null,
        [CanBeNull] string[] y_IDs = null,
        [CanBeNull] string idColumn = null,
        char separator = ',')
        : this(
            NewConstDatasetSample(x, y, objective, isCategoricalColumn, idColumn), 
            x,
            y,
            name,
            meanAndVolatilityForEachChannel,
            columnNames ?? new string[0],
            y_IDs,
            separator)
    {
    }

    public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
    {
        if (xBuffer != null)
        {
            Debug.Assert(indexInBuffer >= 0 &&  indexInBuffer < xBuffer.Shape[0]);
            //same number of channels / same height  / same width
            //only the first dimension (batch size) can be different
            Debug.Assert(_x.MultDim0 == xBuffer.MultDim0);
            _x.CopyTo(_x.Idx(elementId), xBuffer, xBuffer.Idx(indexInBuffer), xBuffer.MultDim0);
        }
        if (yBuffer != null)
        {
            if (_yInMemoryDataSet == null)
            {
                yBuffer.ZeroMemory();
            }
            else
            {
                Debug.Assert(_yInMemoryDataSet.SameShapeExceptFirstDimension(yBuffer));
                _yInMemoryDataSet.CopyTo(_yInMemoryDataSet.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
            }
        }
    }

    public override CpuTensor<float> LoadFullY()
    {
        return _yInMemoryDataSet;
    }

    public override int[] X_Shape(int batchSize) => Utils.CloneShapeWithNewCount(_x.Shape, batchSize);
    public override int[] Y_Shape(int batchSize) => Utils.CloneShapeWithNewCount(_yInMemoryDataSet?.Shape, batchSize);

    public override int Count => _x.Shape[0];

    public override int ElementIdToCategoryIndex(int elementId)
    {
        if (IsRegressionProblem)
        {
            throw new Exception("can't return a category index for regression");
        }
        return _elementIdToCategoryIndex[elementId];
    }

    public CpuTensor<float> X => _x;
    public override string ToString()
    {
        return _x + " => " + _yInMemoryDataSet;
    }
}