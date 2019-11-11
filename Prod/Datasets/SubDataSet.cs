using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class SubDataSet : AbstractDataSet
    {
        private readonly IDataSet _original;
        private readonly List<int> subElementIdToOriginalElementId = new List<int>();

        public SubDataSet(IDataSet original, Func<int,bool> elementIdInOriginalDataSetToIsIncludedInSubDataSet) 
            : base(original.Name, original.Channels, original.Categories, original.MeanAndVolatilityForEachChannel)
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
            Y = CpuTensor<float>.CreateOneHotTensor(ElementIdToCategoryId, subElementIdToOriginalElementId.Count, Categories);
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> buffer)
        {
            _original.LoadAt(subElementIdToOriginalElementId[elementId], indexInBuffer, buffer);
        }
        public override string CategoryIdToDescription(int categoryId)
        {
            return _original.CategoryIdToDescription(categoryId);
        }
        public override int Count => subElementIdToOriginalElementId.Count;
        public override int ElementIdToCategoryId(int elementId)
        {
            return _original.ElementIdToCategoryId(subElementIdToOriginalElementId[elementId]);
        }
        public override string ElementIdToDescription(int elementId)
        {
            return _original.ElementIdToDescription(subElementIdToOriginalElementId[elementId]);
        }

        public override int Height => _original.Height;
        public override int Width => _original.Width;
        public override CpuTensor<float> Y { get; }
    }
}
