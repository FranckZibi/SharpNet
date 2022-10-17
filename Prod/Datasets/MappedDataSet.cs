using System.Collections.Generic;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public sealed class MappedDataSet : DataSet
    {
        private readonly DataSet _original;
        private readonly IReadOnlyList<int> _elementIdToOriginalElementId;
        
        public MappedDataSet(DataSet original, IReadOnlyList<int> elementIdToOriginalElementId) 
            : base(original.Name, 
                original.Objective, 
                original.Channels, 
                original.MeanAndVolatilityForEachChannel, 
                original.ResizeStrategy,
                original.ColumnNames,
                original.CategoricalFeatures, 
                original.IdColumns, 
                original.TargetLabels, 
                original.UseBackgroundThreadToLoadNextMiniBatch,
                original.Separator)
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
