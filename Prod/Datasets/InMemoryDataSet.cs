using System;
using System.Collections.Generic;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class InMemoryDataSet : DataSet
    {
        #region private fields
        private readonly int[] _elementIdToCategoryIndex;
        private readonly CpuTensor<float> _x;
        #endregion




        public static InMemoryDataSet MergeVertically(InMemoryDataSet top, InMemoryDataSet bottom)
        {
            if (top == null)
            {
                return bottom;
            }
            if (bottom == null)
            {
                return top;
            }
            return new InMemoryDataSet(
                CpuTensor<float>.MergeVertically(top._x, bottom._x),
                CpuTensor<float>.MergeVertically(top.Y, bottom.Y),
                top.Name,
                top.Objective,
                top.MeanAndVolatilityForEachChannel,
                top.ColumnNames,
                top.CategoricalFeatures,
                top.IdColumns,
                top.TargetLabels,
                top.UseBackgroundThreadToLoadNextMiniBatch,
                top.Separator);
        }

        public InMemoryDataSet([NotNull] CpuTensor<float> x,
            [CanBeNull] CpuTensor<float> y,
            string name = "",
            Objective_enum objective = Objective_enum.Regression,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel = null,
            string[] columnNames = null,
            string[] categoricalFeatures = null,
            string[] idColumns = null, 
            string[] targetLabels = null,
            bool useBackgroundThreadToLoadNextMiniBatch = true,
            char separator = ',')
            : base(name,
                objective,
                x.Shape[1], 
                meanAndVolatilityForEachChannel, 
                ResizeStrategyEnum.None,
                columnNames ?? new string[0],
                categoricalFeatures ??new string[0],
                idColumns ?? new string[0],
                targetLabels ?? new string[0],
                useBackgroundThreadToLoadNextMiniBatch,
                separator)
        {
            Debug.Assert(y==null || AreCompatible_X_Y(x, y));
            //Debug.Assert(x.Shape[1] == columnNames.Length);

            _x = x;
            Y = y;

            if (IsRegressionProblem || y == null)
            {
                _elementIdToCategoryIndex = null;
            }
            else
            {
                _elementIdToCategoryIndex = new int[y.Shape[0]];
                var ySpan = y.AsReadonlyFloatCpuContent;
                for (int elementId = 0; elementId < y.Shape[0]; ++elementId)
                {
                    int startIndex = elementId * y.Shape[1];
                    for (int categoryIdx = 0; categoryIdx < y.Shape[1]; ++categoryIdx)
                    {
                        if (ySpan[startIndex + categoryIdx] > 0.9f)
                        {
                            _elementIdToCategoryIndex[elementId] = categoryIdx;
                        }
                    }
                }
            }
        }
        public override void LoadAt(int elementId, int indexInBuffer, CpuTensor<float> xBuffer, CpuTensor<float> yBuffer, bool withDataAugmentation)
        {
            Debug.Assert(indexInBuffer >= 0 &&  indexInBuffer < xBuffer.Shape[0]);
            //same number of channels / same height  / same width
            //only the first dimension (batch size) can be different
            Debug.Assert(_x.SameShapeExceptFirstDimension(xBuffer));
            _x.CopyTo(_x.Idx(elementId), xBuffer, xBuffer.Idx(indexInBuffer), xBuffer.MultDim0);
            if (yBuffer != null)
            {
                Debug.Assert(Y.SameShapeExceptFirstDimension(yBuffer));
                Y.CopyTo(Y.Idx(elementId), yBuffer, yBuffer.Idx(indexInBuffer), yBuffer.MultDim0);
            }
        }

        public override int Count => _x.Shape[0];

        public override int ElementIdToCategoryIndex(int elementId)
        {
            if (IsRegressionProblem)
            {
                throw new Exception("can't return a category index for regression");
            }
            return _elementIdToCategoryIndex[elementId];
        }

        public override string ElementIdToPathIfAny(int elementId)
        {
            return "";
        }

        public DataFrame XDataFrame => DataFrame.New(_x, ColumnNames);
        public override CpuTensor<float> Y { get; }
        public override string ToString()
        {
            return _x + " => " + Y;
        }
    }
}
