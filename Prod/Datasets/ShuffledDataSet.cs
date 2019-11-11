using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class ShuffledDataSet : AbstractDataSet
    {
        private readonly IDataSet _original;
        private readonly List<int> _shuffledElementIdToOriginalElementId;

        public ShuffledDataSet(IDataSet original, Random rand)
            : base(original.Name, original.Channels, original.Categories, original.MeanAndVolatilityForEachChannel)
        {
            _original = original;
            _shuffledElementIdToOriginalElementId = Enumerable.Range(0, original.Count).ToList();
            Utils.Shuffle(_shuffledElementIdToOriginalElementId, rand);
            //We compute Y 
            Y = CpuTensor<float>.CreateOneHotTensor(ElementIdToCategoryId, _shuffledElementIdToOriginalElementId.Count, Categories);
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> buffer)
        {
            _original.LoadAt(_shuffledElementIdToOriginalElementId[elementId], indexInBuffer, buffer);
        }
        public override string CategoryIdToDescription(int categoryId)
        {
            return _original.CategoryIdToDescription(categoryId);
        }
        public override int Count => _shuffledElementIdToOriginalElementId.Count;
        public override int ElementIdToCategoryId(int elementId)
        {
            return _original.ElementIdToCategoryId(_shuffledElementIdToOriginalElementId[elementId]);
        }
        public override string ElementIdToDescription(int elementId)
        {
            return _original.ElementIdToDescription(_shuffledElementIdToOriginalElementId[elementId]);
        }

        public override int Height => _original.Height;
        public override int Width => _original.Width;
        public override CpuTensor<float> Y { get; }
    }
}