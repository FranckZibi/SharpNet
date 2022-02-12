using System;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.Datasets;

namespace SharpNet.DataAugmentation
{
    public class TimeSeriesDataAugmentation
    {
        private readonly DataAugmentationSample _sample;
        ///for each featureId:
        ///     Item1:  feature minimum
        ///     Item2:  feature maximum
        ///     Item3:  feature mean
        ///     Item4:  feature volatility
        ///     Item5:  feature correlation with label
        ///     Item6:  feature importance
        private readonly Tuple<double, double, double, double, double, double>[] FeatureStatistics;

        public TimeSeriesDataAugmentation(DataAugmentationSample sample, ITimeSeriesDataSet dataSet, int featuresCount)
        {
            _sample = sample;
            FeatureStatistics = new Tuple<double, double, double, double, double, double>[featuresCount];
            for (int featureId = 0; featureId < featuresCount; ++featureId)
            {
                FeatureStatistics[featureId] = dataSet.GetEncoderFeatureStatistics(featureId);
            }
        }

        public void DataAugmentationForMiniBatch(
            int indexInMiniBatch,
            CpuTensor<float> xDataAugmentedMiniBatch,
            Random rand
        )
        {
            //expecting 'x' shape     (batchSize, timeSteps, input_length)
            Debug.Assert(xDataAugmentedMiniBatch.Shape.Length == 3);
            int timeSteps = xDataAugmentedMiniBatch.Shape[1];
            int featureLength = xDataAugmentedMiniBatch.Shape[2]; //features length
            Debug.Assert(featureLength == FeatureStatistics.Length);

            if (_sample.AugmentedFeaturesPercentage < 1e-6)
            {
                return;
            }
            var xAugmentedSpan = xDataAugmentedMiniBatch.ElementSlice(indexInMiniBatch).AsFloatCpuSpan;
            int featureToBeAugmented = (int) Math.Ceiling(_sample.AugmentedFeaturesPercentage * featureLength);
            Debug.Assert(featureToBeAugmented >= 1);
            if (_sample.UseContinuousFeatureInEachTimeStep)
            {
                int firstFeatureId = rand.Next(featureLength);
                for (int timeStep = 0; timeStep < timeSteps; ++timeStep)
                {
                    if (!_sample.SameAugmentedFeaturesForEachTimeStep)
                    {
                        firstFeatureId = rand.Next(featureLength);
                    }
                    for (int f = firstFeatureId; f < firstFeatureId + featureToBeAugmented; ++f)
                    {
                        int featureId = (f % featureLength);
                        var idx = featureLength * timeStep + featureId;
                        xAugmentedSpan[idx] = GetAugmentedValue(xAugmentedSpan[idx], featureId, rand);
                    }
                }
            }
            else
            {
                while (featureToBeAugmented-- > 0)
                {
                    int featureId = rand.Next(featureLength);
                    for (int timeStep = 0; timeStep < timeSteps; ++timeStep)
                    {
                        var idx = featureLength * timeStep + featureId;
                        xAugmentedSpan[idx] = GetAugmentedValue(xAugmentedSpan[idx], featureId, rand);
                    }
                }
            }
        }
        private float GetAugmentedValue(float originalValue, int featureId, Random rand)
        {
            var stats = FeatureStatistics[featureId];
            if (_sample.TimeSeriesDataAugmentationType == DataAugmentationSample.TimeSeriesDataAugmentationEnum.NOTHING || stats == null)
            {
                return originalValue;
            }
            if (_sample.TimeSeriesDataAugmentationType == DataAugmentationSample.TimeSeriesDataAugmentationEnum.REPLACE_BY_ZERO)
            {
                return 0f;
            }
            if (_sample.TimeSeriesDataAugmentationType == DataAugmentationSample.TimeSeriesDataAugmentationEnum.REPLACE_BY_MEAN)
            {
                return (float)stats.Item3;
            }
            if (_sample.TimeSeriesDataAugmentationType == DataAugmentationSample.TimeSeriesDataAugmentationEnum.ADD_NOISE)
            {
                Debug.Assert(_sample.NoiseInPercentageOfVolatility > 1e-6);
                var featureMinimum = stats.Item1;
                var featureMaximum = stats.Item2;
                var featureVolatility = stats.Item4;
                var augmentedValue = originalValue + (2 * rand.NextDouble() - 1) * featureVolatility * _sample.NoiseInPercentageOfVolatility;
                augmentedValue = Math.Min(augmentedValue, featureMaximum);
                augmentedValue = Math.Max(augmentedValue, featureMinimum);
                return (float)augmentedValue;
            }
            throw new Exception("not implemented " + _sample.TimeSeriesDataAugmentationType);
        }
    }
}