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
    #endregion
    
    public InMemoryDataSet([NotNull] CpuTensor<float> x,
        [CanBeNull] CpuTensor<float> y,
        string name = "",
        Objective_enum objective = Objective_enum.Regression,
        List<Tuple<float, float>> meanAndVolatilityForEachChannel = null,
        string[] columnNames = null,
        string[] categoricalFeatures = null,
        string idColumn = "",
        [CanBeNull] string[] yIDs = null,
        char separator = ',')
        : base(name,
            objective,
            meanAndVolatilityForEachChannel, 
            ResizeStrategyEnum.None,
            columnNames ?? new string[0],
            categoricalFeatures ??new string[0],
            idColumn ?? "",
            yIDs,
            separator)
    {
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

    public override int[] Y_Shape()
    {
        return _yInMemoryDataSet?.Shape;
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

    public CpuTensor<float> X => _x;
    public override AbstractDatasetSample GetDatasetSample()
    {
        return null;
    }
    public override string ToString()
    {
        return _x + " => " + _yInMemoryDataSet;
    }
}