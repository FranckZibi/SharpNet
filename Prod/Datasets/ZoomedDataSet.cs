using SharpNet.CPU;
using SharpNet.Layers;

namespace SharpNet.Datasets
{
    public class ZoomedDataSet : AbstractDataSet
    {
        #region private fields
        private readonly IDataSet _original;
        private readonly int[] _originalShape_CHW;
        private readonly int _rowFactor;
        private readonly int _colFactor;
        private readonly CpuTensor<float> _xBufferBeforeZoom = new CpuTensor<float>(new[] { 1 });
        #endregion

        public ZoomedDataSet(IDataSet original, int[] originalShape_CHW, int rowFactor, int colFactor)
            : base(original.Name, original.Channels, ((AbstractDataSet)original).CategoryDescriptions, original.MeanAndVolatilityForEachChannel, original.ResizeStrategy)
        {
            _original = original;
            _originalShape_CHW = originalShape_CHW;
            _rowFactor = rowFactor;
            _colFactor = colFactor;
        }
        public override void LoadAt(int subElementId, int indexInBuffer, CpuTensor<float> xBuffer,
            CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            _xBufferBeforeZoom.Reshape_ThreadSafe(new []{xBuffer.Shape[0], _originalShape_CHW[1], _originalShape_CHW [2], _originalShape_CHW[3]});
            _original.LoadAt(subElementId, indexInBuffer, _xBufferBeforeZoom, yBuffer, withDataAugmentation);
            var tensorBeforeUpSampling = _xBufferBeforeZoom.ElementSlice(indexInBuffer);
            var tensorAfterUpSampling = xBuffer.ElementSlice(indexInBuffer);
            tensorAfterUpSampling.UpSampling2D(tensorBeforeUpSampling, _rowFactor, _colFactor, UpSampling2DLayer.InterpolationEnum.Nearest);
        }

        public override int Count => _original.Count;
        public override int ElementIdToCategoryIndex(int elementId)
        {
            return _original.ElementIdToCategoryIndex(elementId);
        }
        public override string ElementIdToDescription(int elementId)
        {
            return _original.ElementIdToDescription(elementId);
        }
        public override string ElementIdToPathIfAny(int elementId)
        {
            return _original.ElementIdToPathIfAny(elementId);
        }

        public override CpuTensor<float> Y => _original.Y;

        public override string ToString()
        {
            return _original+" Zoom [x"+ _rowFactor+", x"+ _colFactor+"]";
        }
    }
}
