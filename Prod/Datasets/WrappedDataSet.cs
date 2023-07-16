using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using SharpNet.CPU;

namespace SharpNet.Datasets;

public abstract class WrappedDataSet : DataSet
{
    private readonly DataSet _original;

    protected WrappedDataSet(DataSet original, [CanBeNull] string[] Y_IDs)
        : base(original.Name,
            original.Objective,
            original.MeanAndVolatilityForEachChannel,
            original.ResizeStrategy,
            original.ColumnNames,
            original.CategoricalFeatures,
            original.IdColumn,
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

    public override int ElementIdToCategoryIndex(int elementId) { return _original.ElementIdToCategoryIndex(elementId); }
    public override bool CanBeSavedInCSV => _original.CanBeSavedInCSV;
    public override bool UseRowIndexAsId => _original.UseRowIndexAsId;
    public override List<int[]> XMiniBatch_Shape(int[] shapeForFirstLayer) { return _original.XMiniBatch_Shape(shapeForFirstLayer); }

    public override List<TrainingAndTestDataset> KFoldSplit(int kfold, int countMustBeMultipleOf)
    {
        return _original.KFoldSplit(kfold, countMustBeMultipleOf);
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