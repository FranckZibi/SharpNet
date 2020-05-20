using SharpNet.CPU;
using SharpNet.Layers;

namespace SharpNet.Datasets
{
    public class ZoomedDataSet : AbstractDataSet
    {
        #region private fields
        private readonly IDataSet _original;
        private readonly int _rowFactor;
        private readonly int _colFactor;
        private readonly CpuTensor<float> _xBufferBeforeZoom = new CpuTensor<float>(new[] { 1 });
        #endregion

        public ZoomedDataSet(IDataSet original, int rowFactor, int colFactor)
            : base(original.Name, original.Channels, original.CategoryCount, original.MeanAndVolatilityForEachChannel)
        {
            _original = original;
            _rowFactor = rowFactor;
            _colFactor = colFactor;
        }
        public override void LoadAt(int subElementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer)
        {
            _xBufferBeforeZoom.Reshape_ThreadSafe(_original.XMiniBatch_Shape(xBuffer.Shape[0]));
            _original.LoadAt(subElementId, indexInBuffer, _xBufferBeforeZoom, yBuffer);
            var tensorBeforeUpSampling = _xBufferBeforeZoom.ElementSlice(indexInBuffer);
            var tensorAfterUpSampling = xBuffer.ElementSlice(indexInBuffer);
            tensorAfterUpSampling.UpSampling2D(tensorBeforeUpSampling, _rowFactor, _colFactor, UpSampling2DLayer.InterpolationEnum.Nearest);
        }

        public override int Count => _original.Count;
        public override int ElementIdToCategoryIndex(int elementId)
        {
            return _original.ElementIdToCategoryIndex(elementId);
        }
        public override int Height => _rowFactor*_original.Height;
        public override int Width => _colFactor*_original.Width;
        public override CpuTensor<float> Y => _original.Y;

        public override string ToString()
        {
            return _original+" Zoom [x"+ _rowFactor+", x"+ _colFactor+"]";
        }
    }
}
