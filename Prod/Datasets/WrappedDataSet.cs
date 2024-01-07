using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using SharpNet.CPU;

namespace SharpNet.Datasets;

public abstract class WrappedDataSet : DataSet
{
    protected readonly DataSet _original;

    protected WrappedDataSet(DataSet original, [CanBeNull] string[] Y_IDs)
        : base(original.Name,
            original.DatasetSample,
            original.MeanAndVolatilityForEachChannel,
            original.ResizeStrategy,
            original.ColumnNames,
            Y_IDs,
            original.Separator)
    {
        _original = original;
    }

    public override string ToString()
    {
        return _original.ToString();
    }

    public override void LoadAt(int subElementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
    {
        _original.LoadAt(subElementId, indexInBuffer, xBuffer, yBuffer, withDataAugmentation, isTraining);
    }
    public override int Count => _original.Count;
    public override int[] X_Shape(int batchSize) => _original.X_Shape(batchSize);
    public override int[] Y_Shape(int batchSize) => _original.Y_Shape(batchSize);
    public override int ElementIdToCategoryIndex(int elementId) => _original.ElementIdToCategoryIndex(elementId);
    public override bool CanBeSavedInCSV => _original.CanBeSavedInCSV;
    public override bool UseRowIndexAsId => _original.UseRowIndexAsId;
    public override List<int[]> XMiniBatch_Shape(int[] shapeForFirstLayer) => _original.XMiniBatch_Shape(shapeForFirstLayer);
    public override int[] IdToValidationKFold(int n_splits, int countMustBeMultipleOf)
    {
        if (Count != _original.Count)
        {
            throw new ArgumentException($"{Count} != {_original.Count}");
        }
        return _original.IdToValidationKFold(n_splits, countMustBeMultipleOf);
    }

    #region Dispose pattern
    protected override void Dispose(bool disposing)
    {
        Disposed = true;
#pragma warning disable CA1816
        GC.SuppressFinalize(this);
#pragma warning restore CA1816
    }
    #endregion
}