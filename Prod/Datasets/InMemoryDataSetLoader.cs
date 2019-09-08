using System;
using System.Diagnostics;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class InMemoryDataSetLoader : AbstractDataSetLoader
    {
        #region private fields
        private readonly CpuTensor<float> _x;
        private readonly int[] _elementIdToCategoryId;
        private readonly string[] _categoryIdToDescription;
        #endregion

        public InMemoryDataSetLoader(CpuTensor<float> x, CpuTensor<float> y, int[] elementIdToCategoryId, string[] categoryIdToDescription, string name)
            : base(name, x.Shape[1], y.Shape[1])
        {
            Debug.Assert(AreCompatible_X_Y(x, y));
            Debug.Assert(elementIdToCategoryId != null);
            if (!IsValidYSet(y))
            {
                throw new Exception("Invalid Training Set 'y' : must contain only 0 and 1");
            }

            _x = x;
            Y = y;
            _elementIdToCategoryId = elementIdToCategoryId;
            _categoryIdToDescription = categoryIdToDescription;
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> buffer)
        {
            Debug.Assert(indexInBuffer >= 0 &&  indexInBuffer < buffer.Shape[0]);
            Debug.Assert(_x.Shape[1] == buffer.Shape[1]); //same number of channels
            Debug.Assert(_x.Shape[2] == buffer.Shape[2]); //same height
            Debug.Assert(_x.Shape[3] == buffer.Shape[3]); //same width
            var pictureInputIdx = _x.Idx(elementId);
            var pictureOutputIdx = buffer.Idx(indexInBuffer);
            Buffer.BlockCopy(_x.Content, pictureInputIdx * TypeSize, buffer.Content, pictureOutputIdx * TypeSize, buffer.MultDim0 * TypeSize);
        }

        public override int Count => _x.Shape[0];
        public override int Height => _x.Shape[2];
        public override int Width => _x.Shape[3];
        public override int ElementIdToCategoryId(int elementId)
        {
            return _elementIdToCategoryId[elementId];
        }

        public override string ElementIdToDescription(int elementId)
        {
            return elementId.ToString();
        }

        public override string CategoryIdToDescription(int categoryId)
        {
            if (_categoryIdToDescription == null)
            {
                return categoryId.ToString();
            }
            return _categoryIdToDescription[categoryId];
        }

        public override CpuTensor<float> Y { get; }
        public override string ToString()
        {
            return _x + " => " + Y;
        }
    }
}
