using System;
using System.Collections.Generic;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
// ReSharper disable FieldCanBeMadeReadOnly.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Models;

public class KFoldSample : AbstractModelSample
{
    #region constructors
    // ReSharper disable once UnusedMember.Local
    private KFoldSample() : base(new HashSet<string>())
    {
    }
    public KFoldSample(int nSplits, string embeddedModelWorkingDirectory, string embeddedModelName, EvaluationMetricEnum embeddedModelLoss, EvaluationMetricEnum rankingEvaluationMetric, int countMustBeMultipleOf) : base(new HashSet<string>())
    {
        if (nSplits <= 1)
        {
            throw new ArgumentException($"Invalid KFold splits {nSplits} , must be >= 2");
        }
        n_splits = nSplits;
        EmbeddedModelWorkingDirectory = embeddedModelWorkingDirectory;
        EmbeddedModelName = embeddedModelName;
        EmbeddedModelLoss = embeddedModelLoss;
        EmbeddedModelRankingEvaluationMetric = rankingEvaluationMetric;
        CountMustBeMultipleOf = countMustBeMultipleOf;
    }
    #endregion

    #region Hyper-Parameters
    public int n_splits = 5;
    // ReSharper disable once MemberCanBePrivate.Global
    public string EmbeddedModelWorkingDirectory;
    // ReSharper disable once NotAccessedField.Global
    public string EmbeddedModelName;
    public EvaluationMetricEnum EmbeddedModelLoss = EvaluationMetricEnum.DEFAULT_VALUE;
    public EvaluationMetricEnum EmbeddedModelRankingEvaluationMetric = EvaluationMetricEnum.DEFAULT_VALUE;
    // ReSharper disable once MemberCanBePrivate.Global
    //public EvaluationMetricEnum EmbeddedModelLoss;
    public int CountMustBeMultipleOf = 1;
    public bool Should_Use_All_Available_Cores = false;
    #endregion
    public override EvaluationMetricEnum GetLoss() => EmbeddedModelLoss;
    public override EvaluationMetricEnum GetRankingEvaluationMetric() => EmbeddedModelRankingEvaluationMetric;
    public override List<EvaluationMetricEnum> GetAllEvaluationMetrics()
    {
        return new List<EvaluationMetricEnum> { GetRankingEvaluationMetric() };
    }
    public override void Use_All_Available_Cores()
    {
        Should_Use_All_Available_Cores = true;
    }

    public override Model NewModel(AbstractDatasetSample datasetSample, string workingDirectory, string modelName)
    {
        return new KFoldModel(this, workingDirectory, modelName, datasetSample);
    }
}
