using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public sealed class MappedDataSet : AbstractDataSet
    {
        private readonly IDataSet _original;
        private readonly IReadOnlyList<int> _elementIdToOriginalElementId;
        
        public MappedDataSet(IDataSet original, IReadOnlyList<int> elementIdToOriginalElementId) 
            : base(original.Name, 
                original.Objective, 
                original.Channels, 
                ((AbstractDataSet)original).CategoryDescriptions, 
                original.MeanAndVolatilityForEachChannel, 
                original.ResizeStrategy, 
                null, 
                true)
        {
            _original = original;
            this._elementIdToOriginalElementId = new List<int>(elementIdToOriginalElementId);

            //We compute Y 
            Y = new CpuTensor<float>(original.YMiniBatch_Shape(elementIdToOriginalElementId.Count));
            for (int elementId = 0; elementId < elementIdToOriginalElementId.Count; ++elementId)
            {
                original.Y.CopyTo(original.Y.Idx(elementIdToOriginalElementId[elementId]), Y, Y.Idx(elementId), original.Y.MultDim0);
            }
        }

        public override void LoadAt(int subElementId, int indexInBuffer, CpuTensor<float> xBuffer,
            CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            _original.LoadAt(_elementIdToOriginalElementId[subElementId], indexInBuffer, xBuffer, yBuffer, withDataAugmentation);
        }
        public override int Count => _elementIdToOriginalElementId.Count;

        public override string ColIdToFeatureName(int colId)
        {
            return _original.ColIdToFeatureName(colId);
        }

        public override int ElementIdToCategoryIndex(int elementId)
        {
            return _original.ElementIdToCategoryIndex(_elementIdToOriginalElementId[elementId]);
        }
        public override string ElementIdToDescription(int elementId)
        {
            return _original.ElementIdToDescription(_elementIdToOriginalElementId[elementId]);
        }
        public override string ElementIdToPathIfAny(int elementId)
        {
            return _original.ElementIdToPathIfAny(_elementIdToOriginalElementId[elementId]);
        }

        public override CpuTensor<float> Y { get; }
    }
}
