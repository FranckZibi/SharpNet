using SharpNet.Data;

namespace SharpNet.Datasets
{
    public interface ITimeSeriesDataSet
    {
        void SetBatchPredictionsForInference(int[] batchElementIds, Tensor batchPredictions);
    }
}
