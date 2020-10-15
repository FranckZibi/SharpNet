﻿using System;
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

        /// <summary>
        /// TODO : remove 'elementIdToCategoryIndex' from input
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="elementIdToCategoryIndex"></param>
        /// <param name="categoryDescriptions"></param>
        /// <param name="name"></param>
        /// <param name="meanAndVolatilityForEachChannel"></param>
        public InMemoryDataSet(CpuTensor<float> x, CpuTensor<float> y, int[] elementIdToCategoryIndex, string[] categoryDescriptions,
            string name, List<Tuple<float, float>> meanAndVolatilityForEachChannel)
            : base(name, x.Shape[1], categoryDescriptions, meanAndVolatilityForEachChannel, ResizeStrategyEnum.None)
        {
            Debug.Assert(AreCompatible_X_Y(x, y));
            Debug.Assert(categoryDescriptions.Length == y.Shape[1]);
            if (elementIdToCategoryIndex == null)
            {
                throw new ArgumentException("elementIdToCategoryIndex must be provided");
            }

            _x = x;
            Y = y;
            _elementIdToCategoryIndex = elementIdToCategoryIndex;
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer,
            CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            Debug.Assert(indexInBuffer >= 0 &&  indexInBuffer < xBuffer.Shape[0]);
            //same number of channels / same height  / same width
            //only the first dimension (batch size) can be different
            Debug.Assert(_x.SameShapeExceptFirstDimension(xBuffer)); 
            var pictureInputIdx = _x.Idx(elementId);
            var pictureOutputIdx = xBuffer.Idx(indexInBuffer);
            _x.CopyTo(pictureInputIdx, xBuffer, pictureOutputIdx, xBuffer.MultDim0);


            if (CategoryCount == 1)
            {
                yBuffer?.Set(indexInBuffer, 0, Y.Get(elementId, 0));
            }
            else
            {
                //we update yBuffer
                var categoryIndex = ElementIdToCategoryIndex(elementId);
                for (int cat = 0; cat < CategoryCount; ++cat)
                {
                    yBuffer?.Set(indexInBuffer, cat, (cat == categoryIndex) ? 1f : 0f);
                }
            }

        }

        public override int Count => _x.Shape[0];
        public override int ElementIdToCategoryIndex(int elementId)
        {
            return _elementIdToCategoryIndex[elementId];
        }
        public override string ElementIdToPathIfAny(int elementId)
        {
            return "";
        }

        public override CpuTensor<float> Y { get; }
        public override string ToString()
        {
            return _x + " => " + Y;
        }
    }
}
