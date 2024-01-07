using System.Collections.Generic;
using SharpNet;
using SharpNet.Hyperparameters;

namespace SharpNetTests;

public class TestMetricConfig : IMetricConfig
{
    private readonly EvaluationMetricEnum rankingEvaluationMetric;
    private readonly EvaluationMetricEnum lossMetric;
    private readonly float mseOfLog_Epsilon;
    private readonly float bceWithFocalLoss_PercentageInTrueClass;
    private readonly float bceWithFocalLoss_Gamma;
    private readonly float huber_Delta;
    public TestMetricConfig(List<EvaluationMetricEnum> metrics = null, EvaluationMetricEnum lossMetric = EvaluationMetricEnum.BinaryCrossentropy, EvaluationMetricEnum rankingEvaluationMetric = EvaluationMetricEnum.BinaryCrossentropy,
        float mseOfLog_Epsilon = float.NaN, float huber_Delta = float.NaN,
        float bceWithFocalLossPercentageInTrueClass = float.NaN, float bceWithFocalLoss_Gamma = float.NaN
    )
    {
        this.rankingEvaluationMetric = rankingEvaluationMetric;
        this.lossMetric = lossMetric;
        Metrics = metrics;
        this.mseOfLog_Epsilon = mseOfLog_Epsilon;
        this.huber_Delta = huber_Delta;
        this.bceWithFocalLoss_PercentageInTrueClass = bceWithFocalLossPercentageInTrueClass;
        this.bceWithFocalLoss_Gamma = bceWithFocalLoss_Gamma;
    }
    public EvaluationMetricEnum GetLoss() => lossMetric;
    public EvaluationMetricEnum GetRankingEvaluationMetric() => rankingEvaluationMetric;
    public List<EvaluationMetricEnum> Metrics { get; }
    public float Get_MseOfLog_Epsilon() => mseOfLog_Epsilon;
    public float Get_Huber_Delta() => huber_Delta;
    public float Get_BCEWithFocalLoss_PercentageInTrueClass() => bceWithFocalLoss_PercentageInTrueClass;
    public float Get_BCEWithFocalLoss_Gamma() => bceWithFocalLoss_Gamma;
}