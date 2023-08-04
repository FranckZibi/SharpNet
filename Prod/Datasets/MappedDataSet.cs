using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.Datasets;

public sealed class MappedDataSet : WrappedDataSet
{
    private readonly DataSet _original;
    private readonly IReadOnlyList<int> _elementIdToOriginalElementId;
        
    public MappedDataSet(DataSet original, IReadOnlyList<int> elementIdToOriginalElementId) : base(original, null)
    {
        _original = original;
        _elementIdToOriginalElementId = new List<int>(elementIdToOriginalElementId);
    }

    public override string ToString()
    {
        return $"{Count} elements Map of: "+ _original;
    }

    public override void LoadAt(int subElementId, int indexInBuffer, CpuTensor<float> xBuffer,
        CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
    {
        _original.LoadAt(_elementIdToOriginalElementId[subElementId], indexInBuffer, xBuffer, yBuffer, withDataAugmentation, isTraining);
    }

    public override int[] Y_Shape()
    {
        var mapped_y_shape = (int[])_original.Y_Shape().Clone();
        mapped_y_shape[0] = Count;
        return mapped_y_shape;
    }

    public override int Count => _elementIdToOriginalElementId.Count;

    public override int ElementIdToCategoryIndex(int elementId)
    {
        return _original.ElementIdToCategoryIndex(_elementIdToOriginalElementId[elementId]);
    }
    public override string Y_ID_row_InTargetFormat(int Y_row_InTargetFormat)
    {
        if (_original.Y_IDs == null)
        {
            return null; // there is no Y_ID column in the prediction file in target format
        }

        if (_original.Y_IDs.Length != _original.Count)
        {
            int mod = _original.Y_IDs.Length % _original.Count;
            if (mod != 0)
            {
                throw new ArgumentException($"RowInTargetFormatPredictionToId.Length={_original.Y_IDs.Length} is not a multiple of Count={_original.Count}");
            }
            int mult = _original.Y_IDs.Length / _original.Count;
            var elementId = Y_row_InTargetFormat/ mult;
            var originalElementId = _elementIdToOriginalElementId[elementId];
            var originalRowInTargetFormatPrediction = originalElementId * mult + Y_row_InTargetFormat % mult;
            return _original.Y_ID_row_InTargetFormat(originalRowInTargetFormatPrediction);
        }
        return _original.Y_ID_row_InTargetFormat(_elementIdToOriginalElementId[Y_row_InTargetFormat]);
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