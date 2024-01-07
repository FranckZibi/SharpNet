﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;

namespace SharpNet.Datasets;

public sealed class RandomizeColumnDataSet : WrappedDataSet
{
    private readonly List<string> _columnNameToRandomize;
    private readonly Random _r;
    private CpuTensor<float> tmp = null;


    private CpuTensor<float> GetBuffer(int[] expectedShape)
    {
        if (tmp == null)
        {
            tmp = new CpuTensor<float>(expectedShape);
        }
        else
        {
            tmp.ReshapeInPlace(expectedShape);
        }
        return tmp;
    }
    

    public RandomizeColumnDataSet(DataSet original, List<string> columnNameToRandomize, Random r)
        : base(original, original.Y_IDs)
    {
        _columnNameToRandomize = columnNameToRandomize;
        _r = r;

    }

    public override void LoadAt(int subElementId, int indexInBuffer, CpuTensor<float> xBuffer,
        CpuTensor<float> yBuffer, bool withDataAugmentation, bool isTraining)
    {
        _original.LoadAt(subElementId, indexInBuffer, xBuffer, yBuffer, withDataAugmentation, isTraining);
        if (xBuffer != null)
        {
            if (xBuffer.Shape.Length != 2)
            {
                throw new ArgumentException($"xBuffer.Shape.Length={xBuffer.Shape.Length}!=2");
            }
            int cols = xBuffer.Shape[1];
            if (cols != ColumnNames.Length)
            {
                throw new ArgumentException($"cols={cols}!=ColumnNames.Length={ColumnNames.Length}");
            }
            Debug.Assert(xBuffer.Shape.Length == 2);
            var xBufferSpan = xBuffer.SpanContent;

            var bufferShape = (int[])xBuffer.Shape.Clone();
            bufferShape[0] = 1;
            var buffer = GetBuffer(bufferShape);
            foreach (string c in _columnNameToRandomize)
            {
                int indexColumn = Array.IndexOf(ColumnNames, c);
                if (indexColumn < 0)
                {
                    throw new Exception($"invalid column name {c}");
                }
                int randomSubElementId = _r.Next(0, Count);
                _original.LoadAt(randomSubElementId, 0, buffer, null, withDataAugmentation, isTraining);
                xBufferSpan[xBuffer.Idx(indexInBuffer, indexColumn)] = buffer.SpanContent[buffer.Idx(0, indexColumn)];
            }
        }
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