using System;
using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class SubDataSetLoader<T> : AbstractDataSetLoader<T> where T : struct
    {
        private readonly IDataSetLoader<T> _original;

        public SubDataSetLoader(IDataSetLoader<T> original, Func<int,bool> elemetIdInOriginaDataSetToIsIncludedInSubDataSet) 
            : base(original.Channels, original.Categories)
        {
            _original = original;
            for (int originalElementId = 0; originalElementId < _original.Count; ++originalElementId)
            {
                if (elemetIdInOriginaDataSetToIsIncludedInSubDataSet(originalElementId))
                {
                    subElementIdToOriginalElementId.Add(originalElementId);
                }
            }
            //We compute Y 
            Y = CpuTensor<T>.CreateOneHotTensor(ElementIdToCategoryId, subElementIdToOriginalElementId.Count, Categories);
        }

        private readonly List<int> subElementIdToOriginalElementId = new List<int>();

        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<T> buffer)
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

        public override int Height => _original.Height;
        public override int Width => _original.Width;
        public override CpuTensor<T> Y { get; }
    }
}
