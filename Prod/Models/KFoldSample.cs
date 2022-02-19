using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

[SuppressMessage("ReSharper", "FieldCanBeMadeReadOnly.Global")]
public class KFoldSample : AbstractSample, IModelSample
{
    #region constructors
    private KFoldSample() : base(new HashSet<string>())
    {
    }
    public KFoldSample(int nSplits, MetricEnum metric, LossFunctionEnum loss) : base(new HashSet<string>())
    {
        this.n_splits = nSplits;
        Metric = metric;
        Loss = loss;
    }
    public static KFoldSample ValueOf(string workingDirectory, string modelName)
    {
        return (KFoldSample)ISample.LoadConfigIntoSample(() => new KFoldSample(), workingDirectory, modelName);
    }
    #endregion

    #region Hyper-Parameters
    public int n_splits = 5;
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
}
