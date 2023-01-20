using SharpNet.Datasets;
using SharpNet.Models;
using System.Collections.Generic;

namespace SharpNet.HyperParameters;

public interface IModelSample : ISample
{
    EvaluationMetricEnum GetLoss();
    void FillSearchSpaceWithDefaultValues(IDictionary<string, object> existingHyperParameterValues, AbstractDatasetSample datasetSample);
    Model NewModel(AbstractDatasetSample datasetSample, string workingDirectory, string modelName);
    void Use_All_Available_Cores();

    public static IModelSample LoadModelSample(string workingDirectory, string sampleName, bool useAllAvailableCores)
    {
        IModelSample sample = (IModelSample)Load(workingDirectory, sampleName);
        if (useAllAvailableCores)
        {
            sample.Use_All_Available_Cores();
        }
        return sample;
    }



}
