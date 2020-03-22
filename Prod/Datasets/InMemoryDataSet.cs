using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class InMemoryDataSet : AbstractDataSet
    {
        #region private fields
        private readonly CpuTensor<float> _x;
        private readonly int[] _elementIdToCategoryIndex;
        #endregion

        public InMemoryDataSet(CpuTensor<float> x, CpuTensor<float> y, int[] elementIdToCategoryIndex,
            string name, List<Tuple<float, float>> meanAndVolatilityForEachChannel)
            : base(name, x.Shape[1], y.Shape[1], meanAndVolatilityForEachChannel, null)
        {
            Debug.Assert(AreCompatible_X_Y(x, y));
            if (elementIdToCategoryIndex == null)
            {
                throw new ArgumentException("elementIdToCategoryIndex must be provided");
            }
            if (!IsValidYSet(y))
            {
                throw new Exception("Invalid Training Set 'y' : must contain only 0 and 1");
            }

            _x = x;
            Y = y;
            _elementIdToCategoryIndex = elementIdToCategoryIndex;
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer)
        {
            Debug.Assert(indexInBuffer >= 0 &&  indexInBuffer < xBuffer.Shape[0]);
            Debug.Assert(_x.Shape[1] == xBuffer.Shape[1]); //same number of channels
            Debug.Assert(_x.Shape[2] == xBuffer.Shape[2]); //same height
            Debug.Assert(_x.Shape[3] == xBuffer.Shape[3]); //same width
            var pictureInputIdx = _x.Idx(elementId);
            var pictureOutputIdx = xBuffer.Idx(indexInBuffer);
            Buffer.BlockCopy(_x.Content, pictureInputIdx * TypeSize, xBuffer.Content, pictureOutputIdx * TypeSize, xBuffer.MultDim0 * TypeSize);

            //we update yBuffer
            var categoryIndex = ElementIdToCategoryIndex(elementId);
            for (int cat = 0; cat < Categories; ++cat)
            {
                yBuffer?.Set(indexInBuffer, cat, (cat == categoryIndex) ? 1f : 0f);
            }
        }

        public override int Count => _x.Shape[0];
        public override int Height => _x.Shape[2];
        public override int Width => _x.Shape[3];
        public override int ElementIdToCategoryIndex(int elementId)
        {
            return _elementIdToCategoryIndex[elementId];
        }

        public override CpuTensor<float> Y { get; }
        public override string ToString()
        {
            return _x + " => " + Y;
        }
    }
}
