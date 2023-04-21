using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.Datasets;

public sealed class MappedDataSet : WrappedDataSet
{
    private readonly DataSet _original;
    private readonly IReadOnlyList<int> _elementIdToOriginalElementId;
    private readonly CpuTensor<float> yMappedDataSet;
        
    public MappedDataSet(DataSet original, IReadOnlyList<int> elementIdToOriginalElementId) : base(original, false)
    {
        _original = original;
        _elementIdToOriginalElementId = new List<int>(elementIdToOriginalElementId);

        //We compute Y 
        var originalYIfAny = original.Y;
        if (originalYIfAny != null)
        {
            var mapped_Y_shape = (int[])originalYIfAny.Shape.Clone();
            mapped_Y_shape[0] = elementIdToOriginalElementId.Count; // mapped batch size
            yMappedDataSet = new CpuTensor<float>(mapped_Y_shape);
            for (int elementId = 0; elementId < elementIdToOriginalElementId.Count; ++elementId)
            {
                originalYIfAny.CopyTo(originalYIfAny.Idx(elementIdToOriginalElementId[elementId]), yMappedDataSet, yMappedDataSet.Idx(elementId), originalYIfAny.MultDim0);
            }
        }
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
    public override int Count => _elementIdToOriginalElementId.Count;

    public override int ElementIdToCategoryIndex(int elementId)
    {
        return _original.ElementIdToCategoryIndex(_elementIdToOriginalElementId[elementId]);
    }
    public override string ElementIdToPathIfAny(int elementId)
    {
        return _original.ElementIdToPathIfAny(_elementIdToOriginalElementId[elementId]);
    }
    public override string ID_Y_row_InTargetFormat(int Y_row_InTargetFormat)
    {
        if (_original.Y_IDs != null && _original.Y_IDs.Length != _original.Count)
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
            return _original.ID_Y_row_InTargetFormat(originalRowInTargetFormatPrediction);
        }

        return _original.ID_Y_row_InTargetFormat(_elementIdToOriginalElementId[Y_row_InTargetFormat]);
    }




    public override CpuTensor<float> Y => yMappedDataSet;

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