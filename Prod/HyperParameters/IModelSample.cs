using SharpNet.Datasets;
using SharpNet.Models;
using System.Collections.Generic;

namespace SharpNet.HyperParameters;

public interface IModelSample : ISample
{
    EvaluationMetricEnum GetLoss();
    
    /// <summary>
    /// the evaluation metric used to rank the final submission
    /// depending on the evaluation metric, higher (ex: Accuracy) or lower (ex: Rmse) may be better
    /// </summary>
    /// <returns></returns>
    EvaluationMetricEnum GetRankingEvaluationMetric();

    void FillSearchSpaceWithDefaultValues(IDictionary<string, object> existingHyperParameterValues, AbstractDatasetSample datasetSample);

    Model NewModel(AbstractDatasetSample datasetSample, string workingDirectory, string modelName);
}

