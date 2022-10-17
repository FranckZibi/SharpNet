using System;
using SharpNet.Data;

namespace SharpNet.Datasets
{
    public interface ITimeSeriesDataSet
    {
        void SetBatchPredictionsForInference(int[] batchElementIds, Tensor batchPredictions);
        Tuple<double, double, double, double, double, double>  GetEncoderFeatureStatistics(int featureId);
    }
}
