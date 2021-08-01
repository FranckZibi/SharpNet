using System;
using System.Diagnostics;
using SharpNet.CPU;
using SharpNet.Datasets;

namespace SharpNet.DataAugmentation
{
    public class TimeSeriesDataAugmentation
    {
        private readonly DataAugmentationConfig Config;
        ///for each featureId:
        ///     Item1:  feature minimum
        ///     Item2:  feature maximum
        ///     Item3:  feature mean
        ///     Item4:  feature volatility
        ///     Item5:  feature correlation with label
        ///     Item6:  feature importances
        private readonly Tuple<double, double, double, double, double, double>[] FeatureStatistics;

        public TimeSeriesDataAugmentation(DataAugmentationConfig config, ITimeSeriesDataSet dataSet, int featuresCount)
        {
            Config = config;
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

            if (Config.AugmentedFeaturesPercentage < 1e-6)
            {
                return;
            }
            var xAugmentedSpan = xDataAugmentedMiniBatch.ElementSlice(indexInMiniBatch).AsFloatCpuSpan;
            int featureToBeAugmented = (int) Math.Ceiling(Config.AugmentedFeaturesPercentage * featureLength);
            Debug.Assert(featureToBeAugmented >= 1);
            if (Config.UseContinuousFeatureInEachTimeStep)
            {
                int firstFeatureId = rand.Next(featureLength);
                for (int timeStep = 0; timeStep < timeSteps; ++timeStep)
                {
                    if (!Config.SameAugmentedFeaturesForEachTimeStep)
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
            if (Config.TimeSeriesDataAugmentationType == DataAugmentationConfig.TimeSeriesDataAugmentationEnum.NOTHING || stats == null)
            {
                return originalValue;
            }
            if (Config.TimeSeriesDataAugmentationType == DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_ZERO)
            {
                return 0f;
            }
            if (Config.TimeSeriesDataAugmentationType == DataAugmentationConfig.TimeSeriesDataAugmentationEnum.REPLACE_BY_MEAN)
            {
                return (float)stats.Item3;
            }
            if (Config.TimeSeriesDataAugmentationType == DataAugmentationConfig.TimeSeriesDataAugmentationEnum.ADD_NOISE)
            {
                Debug.Assert(Config.NoiseInPercentageOfVolatility > 1e-6);
                var featureMinimum = stats.Item1;
                var featureMaximum = stats.Item2;
                var featureVolatility = stats.Item4;
                var augmentedValue = originalValue + (2 * rand.NextDouble() - 1) * featureVolatility * Config.NoiseInPercentageOfVolatility;
                augmentedValue = Math.Min(augmentedValue, featureMaximum);
                augmentedValue = Math.Max(augmentedValue, featureMinimum);
                return (float)augmentedValue;
            }
            throw new Exception("not implemented " + Config.TimeSeriesDataAugmentationType);
        }
    }
}