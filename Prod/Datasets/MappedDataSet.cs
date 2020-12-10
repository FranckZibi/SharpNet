using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public sealed class MappedDataSet : AbstractDataSet
    {
        private readonly IDataSet _original;
        private readonly IReadOnlyList<int> _elementIdToOriginalElementId;

        public static MappedDataSet SubDataSet(IDataSet original, Func<int, bool> elementIdInOriginalDataSetToIsIncludedInSubDataSet)
        {
            var subElementIdToOriginalElementId = new List<int>();
            for (int originalElementId = 0; originalElementId < original.Count; ++originalElementId)
            {
                if (elementIdInOriginalDataSetToIsIncludedInSubDataSet(originalElementId))
                {
                    subElementIdToOriginalElementId.Add(originalElementId);
                }
            }
            return new MappedDataSet(original, subElementIdToOriginalElementId);
        }

        public static MappedDataSet Shuffle(IDataSet original, Random r)
        {
            var elementIdToOriginalElementId = Enumerable.Range(0, original.Count).ToList();
            Utils.Shuffle(elementIdToOriginalElementId, r);
            return new MappedDataSet(original, elementIdToOriginalElementId);
        }

        // ReSharper disable once UnusedMember.Global
        public static MappedDataSet Resize(IDataSet original, int targetSize, bool shuffle)
        {
            var elementIdToOriginalElementId = new List<int>(targetSize);
            for (int elementId = 0; elementId < targetSize; ++elementId)
            {
                elementIdToOriginalElementId.Add(elementId%original.Count);
            }
            if (shuffle)
            {
                Utils.Shuffle(elementIdToOriginalElementId, new Random(0));
            }
            return new MappedDataSet(original, elementIdToOriginalElementId);
        }

        private MappedDataSet(IDataSet original, IReadOnlyList<int> elementIdToOriginalElementId) 
            : base(original.Name, original.Channels, ((AbstractDataSet)original).CategoryDescriptions, original.MeanAndVolatilityForEachChannel, original.ResizeStrategy)
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
