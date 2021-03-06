﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public sealed class UnivariateTimeSeriesDataSet : AbstractDataSet
    {
        #region private fields
        private readonly Memory<float> _univariateTimeSeries;
        private readonly int _timeSteps;
        private readonly int _stride;
        #endregion

        public UnivariateTimeSeriesDataSet(Memory<float> univariateTimeSeries, int timeSteps, int stride, 
            string name = "", List<Tuple<float, float>> meanAndVolatilityForEachChannel = null)
            : base(name,
                timeSteps,
                new []{"NA"},
                meanAndVolatilityForEachChannel,
                ResizeStrategyEnum.None)
        {
            _univariateTimeSeries = univariateTimeSeries;
            _timeSteps = timeSteps;
            _stride = stride;
            var totalCount = (_univariateTimeSeries.Length - _timeSteps - 1) / _stride + 1;
            Count = totalCount;
            if (_univariateTimeSeries.Length < (_timeSteps + 1))
            {
                throw new ArgumentException("time series is too short ("+ _univariateTimeSeries.Length+") with timeSteps="+_timeSteps);
            }

            //We build 'Y' field
            Y = new CpuTensor<float>(new []{ totalCount, 1});
            var yAsSpan = Y.AsFloatCpuSpan;
            var timeSeriesAsSpan = _univariateTimeSeries.Span;
            for (int elementId = 0; elementId < totalCount; ++elementId)
            {
                yAsSpan[elementId] = timeSeriesAsSpan[timeSteps + elementId * _stride];
            }
        }

        /// <param name="elementId"></param>
        /// <param name="indexInBuffer"></param>
        /// <param name="xBuffer">
        /// shape: (batchSize, timeSteps, inputSize = 1)
        /// </param>
        /// <param name="yBuffer">
        /// shape: (batchSize, outputSize = 1)
        /// </param>
        /// <param name="withDataAugmentation"></param>
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            Debug.Assert(indexInBuffer >= 0 && indexInBuffer < xBuffer.Shape[0]);
            Debug.Assert(xBuffer.SameShapeExceptFirstDimension(X_Shape));
            Debug.Assert(xBuffer.Shape[0] == yBuffer.Shape[0]); //same batch size
            Debug.Assert(yBuffer == null || yBuffer.SameShapeExceptFirstDimension(Y.Shape));

            var xSrc = _univariateTimeSeries.Span.Slice(elementId*_stride, xBuffer.MultDim0);
            var xDest = xBuffer.AsFloatCpuSpan.Slice(indexInBuffer * xBuffer.MultDim0, xBuffer.MultDim0);
            Debug.Assert(xSrc.Length == xDest.Length);
            xSrc.CopyTo(xDest);
            if (yBuffer != null)
            {
                Y.CopyTo(Y.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
            }
        }

        public override int Count {get;}

        public override int ElementIdToCategoryIndex(int elementId)
        {
            return -1;
        }
        public override string ElementIdToPathIfAny(int elementId)
        {
            return "";
        }

        // ReSharper disable once UnusedMember.Global
        public float DefaultMae
        {
            get
            {
                float result = 0;
                var timeSeriesAsSpan = _univariateTimeSeries.Span;
                for (int i = 1; i < timeSeriesAsSpan.Length; ++i)
                {
                    result += Math.Abs(timeSeriesAsSpan[i] - timeSeriesAsSpan[i - 1]);
                }
                return result / (timeSeriesAsSpan.Length - 1);
            }

        }

        private int[] X_Shape => new []{Count, _timeSteps, 1};

        public override CpuTensor<float> Y { get; }
        public override string ToString()
        {
            return X_Shape + " => " + Y;
        }
    }
}