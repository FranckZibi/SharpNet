using System;
using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.CPU;

namespace SharpNet.Datasets;

public class TensorListDataSet : DataSet
{
    #region private fields
    private readonly int[] _elementIdToCategoryIndex;
    private readonly List<CpuTensor<float>> _xList;
    [CanBeNull] private readonly CpuTensor<float> _yInMemoryDataSet;
    #endregion

    public TensorListDataSet([NotNull] List<CpuTensor<float>> xList,
        [CanBeNull] CpuTensor<float> y,
        string name = "",
        Objective_enum objective = Objective_enum.Regression,
        List<Tuple<float, float>> meanAndVolatilityForEachChannel = null,
        string[] columnNames = null,
        string[] categoricalFeatures = null,
        [CanBeNull] string[] y_IDs = null,
        [CanBeNull] string idColumn = null,
        char separator = ',')
        : base(name,
            objective,
            meanAndVolatilityForEachChannel,
            ResizeStrategyEnum.None,
            columnNames ?? new string[0],
            categoricalFeatures ?? new string[0],
            y_IDs,
            idColumn,
            separator)
    {
        _xList = xList;
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
            Debug.Assert(indexInBuffer >= 0 && indexInBuffer < xBuffer.Shape[0]);
            //same number of channels / same height  / same width
            //only the first dimension (batch size) can be different
            //Debug.Assert(_xList[elementId].Count == xBuffer.MultDim0);
            _xList[elementId].CopyTo(0, xBuffer, xBuffer.Idx(indexInBuffer), xBuffer.MultDim0);
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

    public override int Count => _xList.Count;

    public override int ElementIdToCategoryIndex(int elementId)
    {
        if (IsRegressionProblem)
        {
            throw new Exception("can't return a category index for regression");
        }
        return _elementIdToCategoryIndex[elementId];
    }

    public override AbstractDatasetSample GetDatasetSample()
    {
        return null;
    }
    public override string ToString()
    {
        return " => " + _yInMemoryDataSet;
    }
}