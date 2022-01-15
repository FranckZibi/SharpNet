using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using SharpNet.CPU;

namespace SharpNet.Datasets
{
    public class InMemoryDataSet : AbstractDataSet
    {
        #region private fields
        private readonly int[] _elementIdToCategoryIndex;
        private readonly CpuTensor<float> _x;
        #endregion

        /// <summary>
        /// TODO : remove 'elementIdToCategoryIndex' from input
        /// </summary>
        public InMemoryDataSet([NotNull] CpuTensor<float> x, 
            [NotNull] CpuTensor<float> y,
            string name = "",
            Objective_enum? objective = null,
            List<Tuple<float, float>> meanAndVolatilityForEachChannel = null, 
            string[] categoryDescriptions = null,
            string[] featureNames = null,
            bool useBackgroundThreadToLoadNextMiniBatch = true)
            : base(name,
                objective,
                x.Shape[1], 
                categoryDescriptions ?? Enumerable.Range(0, y.Shape[1]).Select(i => i.ToString()).ToArray(), 
                meanAndVolatilityForEachChannel, 
                ResizeStrategyEnum.None,
                featureNames, 
                useBackgroundThreadToLoadNextMiniBatch)
        {
            Debug.Assert(AreCompatible_X_Y(x, y));

            _x = x;
            Y = y;

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
            return _elementIdToCategoryIndex[elementId];
        }
        public override string ElementIdToPathIfAny(int elementId)
        {
            return "";
        }

        public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation(double percentageInTrainingSet)
        {
            int rowsInTrainingSet = (int)(percentageInTrainingSet * Count + 0.1);
            int rowsInValidationSet = Count - rowsInTrainingSet;
            var training = new InMemoryDataSet((CpuTensor<float>)_x.RowSlice(0, rowsInTrainingSet), (CpuTensor<float>)Y.RowSlice(0, rowsInTrainingSet), Name,  Objective, MeanAndVolatilityForEachChannel, CategoryDescriptions, FeatureNamesIfAny, _useBackgroundThreadToLoadNextMiniBatch);
            var test = new InMemoryDataSet((CpuTensor<float>)_x.RowSlice(rowsInTrainingSet, rowsInValidationSet), (CpuTensor<float>)Y.RowSlice(rowsInTrainingSet, rowsInValidationSet), Name, Objective, MeanAndVolatilityForEachChannel, CategoryDescriptions, FeatureNamesIfAny, _useBackgroundThreadToLoadNextMiniBatch);
            return new TrainingAndTestDataLoader(training, test, Name);
        }

        public override CpuTensor<float> X_if_available => _x;

        public override CpuTensor<float> Y { get; }
        public override string ToString()
        {
            return _x + " => " + Y;
        }
    }
}
