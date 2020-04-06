using System.Diagnostics;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class ZoomedDataSet : AbstractDataSet
    {
        private readonly IDataSet _original;
        private readonly int _heightMultiplier;
        private readonly int _widthMultiplier;
        private readonly CpuTensor<float> xBufferBeforeZoom;

        public ZoomedDataSet(IDataSet original, int heightMultiplier, int widthMultiplier)
            : base(original.Name, original.Channels, original.CategoryCount, original.MeanAndVolatilityForEachChannel, original.Logger)
        {
            _original = original;
            _heightMultiplier = heightMultiplier;
            _widthMultiplier = widthMultiplier;
            xBufferBeforeZoom =  new CpuTensor<float>(new []{1,Channels, original.Height, original.Width }, nameof(xBufferBeforeZoom));
        }
        public override void LoadAt(int subElementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer)
        {
            _original.LoadAt(subElementId, 0, xBufferBeforeZoom, yBuffer);
            CopyIntoWithZoom(xBufferBeforeZoom, _heightMultiplier, _widthMultiplier, indexInBuffer, xBuffer);
        }

        /// <summary>
        /// copy a 'zoomed' version of 'originalTensor' into index 'indexInBuffer' of 'xBuffer'
        /// The originalTensor will have its height multiplied by 'heightMultiplier' and width multiplied by 'widthMultiplier'
        /// </summary>
        /// <param name="originalTensor"></param>
        /// <param name="heightMultiplier"></param>
        /// <param name="widthMultiplier"></param>
        /// <param name="indexInBuffer"></param>
        /// <param name="xBuffer"></param>
        private static void CopyIntoWithZoom(CpuTensor<float> originalTensor, int heightMultiplier, int widthMultiplier,int indexInBuffer, CpuTensor<float> xBuffer)
        {
            Debug.Assert(originalTensor.Shape[1] == xBuffer.Shape[1]);
            Debug.Assert(originalTensor.Shape[2]* heightMultiplier == xBuffer.Shape[2]);
            Debug.Assert(originalTensor.Shape[3]* widthMultiplier == xBuffer.Shape[3]);
            for (int c = 0; c < xBuffer.Shape[1]; ++c)
            for (int row = 0; row < xBuffer.Shape[2]; ++row)
            for (int col = 0; col < xBuffer.Shape[3]; ++col)
            {
                xBuffer.Set(indexInBuffer,c,row,col, originalTensor.Get(0,c,row/heightMultiplier, col/widthMultiplier));
            }
        }

        public override int Count => _original.Count;
        public override int ElementIdToCategoryIndex(int elementId)
        {
            return _original.ElementIdToCategoryIndex(elementId);
        }
        public override int Height => _heightMultiplier*_original.Height;
        public override int Width => _widthMultiplier*_original.Width;
        public override CpuTensor<float> Y { get; }

        public override string ToString()
        {
            return _original+" Zoom [x"+ _heightMultiplier+", x"+ _widthMultiplier+"]";
        }
    }
}