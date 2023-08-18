using System.Collections.Generic;

namespace SharpNet.HyperParameters;

public interface IMetricConfig
{
    EvaluationMetricEnum GetLoss();
    EvaluationMetricEnum GetRankingEvaluationMetric();
    List<EvaluationMetricEnum> Metrics { get; }
    float Get_MseOfLog_Epsilon();
    float Get_Huber_Delta();
    float Get_BCEWithFocalLoss_PercentageInTrueClass();
    float Get_BCEWithFocalLoss_Gamma();
}