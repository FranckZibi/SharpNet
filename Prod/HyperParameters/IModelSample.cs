using SharpNet.Datasets;
using SharpNet.Models;
using System.Collections.Generic;

namespace SharpNet.HyperParameters;

public interface IModelSample : ISample
{
    EvaluationMetricEnum GetLoss();
    void FillSearchSpaceWithDefaultValues(IDictionary<string, object> existingHyperParameterValues, AbstractDatasetSample datasetSample);
    Model NewModel(AbstractDatasetSample datasetSample, string workingDirectory, string modelName);
    public static IModelSample LoadModelSample(string workingDirectory, string sampleName)
    {
        return (IModelSample)Load(workingDirectory, sampleName);
    }



}
