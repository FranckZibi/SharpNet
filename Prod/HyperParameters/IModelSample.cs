using SharpNet.Datasets;
using System.Collections.Generic;

namespace SharpNet.HyperParameters;

public interface IModelSample : ISample
{
    EvaluationMetricEnum GetLoss();
    void FillSearchSpaceWithDefaultValues(IDictionary<string, object> existingHyperParameterValues, AbstractDatasetSample datasetSample);
    public static IModelSample LoadModelSample(string workingDirectory, string sampleName)
    {
        return (IModelSample)Load(workingDirectory, sampleName);
    }
}
