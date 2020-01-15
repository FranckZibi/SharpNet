﻿using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class SubDataSet : AbstractDataSet
    {
        private readonly IDataSet _original;
        private readonly List<int> subElementIdToOriginalElementId = new List<int>();

        public SubDataSet(IDataSet original, Func<int,bool> elementIdInOriginalDataSetToIsIncludedInSubDataSet) 
            : base(original.Name, original.Channels, original.Categories, original.MeanAndVolatilityForEachChannel, original.Logger)
        {
            _original = original;
            for (int originalElementId = 0; originalElementId < _original.Count; ++originalElementId)
            {
                if (elementIdInOriginalDataSetToIsIncludedInSubDataSet(originalElementId))
                {
                    subElementIdToOriginalElementId.Add(originalElementId);
                }
            }
            //We compute Y 
            Y = CpuTensor<float>.CreateOneHotTensor(ElementIdToCategoryIndex, subElementIdToOriginalElementId.Count, Categories);
        }
        public override void LoadAt(int subElementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer)
        {
            _original.LoadAt(subElementIdToOriginalElementId[subElementId], indexInBuffer, xBuffer, yBuffer);
        }
        public override int Count => subElementIdToOriginalElementId.Count;
        public override int ElementIdToCategoryIndex(int elementId)
        {
            return _original.ElementIdToCategoryIndex(subElementIdToOriginalElementId[elementId]);
        }
        public override int Height => _original.Height;
        public override int Width => _original.Width;
        public override CpuTensor<float> Y { get; }
    }
}
