using SharpNet.Datasets;
using SharpNet.Models;

namespace SharpNet.HyperParameters;

public interface IModelSample : ISample
{
    // ReSharper disable once UnusedMemberInSuper.Global
    EvaluationMetricEnum GetLoss();
    
    /// <summary>
    /// the evaluation metric used to rank the final submission
    /// depending on the evaluation metric, higher (ex: Accuracy) or lower (ex: Rmse) may be better
    /// </summary>
    /// <returns></returns>
    // ReSharper disable once UnusedMemberInSuper.Global
    EvaluationMetricEnum GetRankingEvaluationMetric();

    Model NewModel(AbstractDatasetSample datasetSample, string workingDirectory, string modelName);
}

