using System;
using System.Diagnostics;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class InMemoryDataSetLoader<T> : AbstractDataSetLoader<T> where T : struct
    {
        #region private fields
        private readonly CpuTensor<T> _x;
        #endregion

        public InMemoryDataSetLoader(CpuTensor<T> x, CpuTensor<T> y, int[] elementIdToCategoryId, string[] categoryIdToDescription)
            : base(x.Shape[1], y.Shape[1], categoryIdToDescription)
        {
            Debug.Assert(AreCompatible_X_Y(x, y));
            Debug.Assert(elementIdToCategoryId != null);
            if (!IsValidYSet(y))
            {
                throw new Exception("Invalid Training Set 'y' : must contain only 0 and 1");
            }

            _x = x;
            Y = y;
            _elementIdToCategoryIdOrMinusOneIfUnknown = elementIdToCategoryId;
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<T> buffer)
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
        public override int CurrentHeight => _x.Shape[2];
        public override int CurrentWidth => _x.Shape[3];
        public override IDataSetLoader<float> ToSinglePrecision()
        {
            if (this is IDataSetLoader<float>)
            {
                return (IDataSetLoader<float>) this;
            }
            return new InMemoryDataSetLoader<float>(_x.ToSinglePrecision(), Y.ToSinglePrecision(), _elementIdToCategoryIdOrMinusOneIfUnknown, _categoryIdToDescription);
        }
        public override IDataSetLoader<double> ToDoublePrecision()
        {
            if (this is IDataSetLoader<double>)
            {
                return (IDataSetLoader<double>)this;
            }
            return new InMemoryDataSetLoader<double>(_x.ToDoublePrecision(), Y.ToDoublePrecision(), _elementIdToCategoryIdOrMinusOneIfUnknown, _categoryIdToDescription);
        }

        public override CpuTensor<T> Y { get; }
        public override string ToString()
        {
            return _x + " => " + Y;
        }
    }
}
