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
            : base(original.Name, original.Channels, original.Categories, original.MeanAndVolatilityForEachChannel, original.Logger)
        {
            _original = original;
            _shuffledElementIdToOriginalElementId = Enumerable.Range(0, original.Count).ToList();
            Utils.Shuffle(_shuffledElementIdToOriginalElementId, rand);
            //We compute Y 
            Y = CpuTensor<float>.CreateOneHotTensor(ElementIdToCategoryIndex, _shuffledElementIdToOriginalElementId.Count, Categories);
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer)
        {
            _original.LoadAt(_shuffledElementIdToOriginalElementId[elementId], indexInBuffer, xBuffer, yBuffer);
        }
        public override int Count => _shuffledElementIdToOriginalElementId.Count;
        public override int ElementIdToCategoryIndex(int elementId)
        {
            return _original.ElementIdToCategoryIndex(_shuffledElementIdToOriginalElementId[elementId]);
        }
        public override int Height => _original.Height;
        public override int Width => _original.Width;
        public override CpuTensor<float> Y { get; }
    }
}