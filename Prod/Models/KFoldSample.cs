using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

[SuppressMessage("ReSharper", "FieldCanBeMadeReadOnly.Global")]
[SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
[SuppressMessage("ReSharper", "NotAccessedField.Global")]
public class KFoldSample : AbstractSample, IModelSample
{
    #region constructors
    private KFoldSample() : base(new HashSet<string>())
    {
    }
    public KFoldSample(int nSplits, string embeddedModelName, bool useFullTrainingDataset, int countMustBeMultipleOf, MetricEnum metric, LossFunctionEnum loss) : base(new HashSet<string>())
    {
        this.n_splits = nSplits;
        EmbeddedModelName = embeddedModelName;
        UseFullTrainingDataset = useFullTrainingDataset;
        CountMustBeMultipleOf = countMustBeMultipleOf;
        Metric = metric;
        Loss = loss;
    }
    public static KFoldSample LoadKFoldSample(string workingDirectory, string modelName)
    {
        return (KFoldSample)ISample.LoadConfigIntoSample(() => new KFoldSample(), workingDirectory, modelName);
    }
    #endregion

    #region Hyper-Parameters
    public int n_splits = 5;
    public string EmbeddedModelName = DEFAULT_VALUE_STR;
    public bool UseFullTrainingDataset;
    public int CountMustBeMultipleOf = 1;
    public MetricEnum Metric = MetricEnum.Rmse;
    // ReSharper disable once MemberCanBePrivate.Global
    public LossFunctionEnum Loss = LossFunctionEnum.Rmse;
    #endregion

    public MetricEnum GetMetric()
    {
        return Metric;
    }
    public LossFunctionEnum GetLoss()
    {
        return Loss;
    }
    public void Use_All_Available_Cores()
    {
    }
}

