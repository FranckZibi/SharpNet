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
            : base(original.Name, original.Channels, ((AbstractDataSet)original).CategoryDescriptions, original.MeanAndVolatilityForEachChannel, original.ResizeStrategy, original.HierarchyIfAny)
        {
            _original = original;
            _shuffledElementIdToOriginalElementId = Enumerable.Range(0, original.Count).ToList();
            Utils.Shuffle(_shuffledElementIdToOriginalElementId, rand);
            //We compute Y 
            Y = CpuTensor<float>.CreateOneHotTensor(ElementIdToCategoryIndex, _shuffledElementIdToOriginalElementId.Count, CategoryCount);
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
        public override string ElementIdToDescription(int elementId)
        {
            return _original.ElementIdToDescription(_shuffledElementIdToOriginalElementId[elementId]);
        }
        public override string ElementIdToPathIfAny(int elementId)
        {
            return _original.ElementIdToPathIfAny(_shuffledElementIdToOriginalElementId[elementId]);
        }
        
        public override CpuTensor<float> Y { get; }
    }
}