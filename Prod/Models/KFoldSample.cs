using System.Collections.Generic;
using SharpNet.HyperParameters;
// ReSharper disable FieldCanBeMadeReadOnly.Global

namespace SharpNet.Models;

public class KFoldSample : AbstractSample, IModelSample
{
    #region private fields
    private IModelSample _embeddedModelSample;
    #endregion

    #region constructors
    // ReSharper disable once UnusedMember.Local
    private KFoldSample() : base(new HashSet<string>())
    {
    }
    public KFoldSample(int nSplits, string embeddedModelName, string embeddedModelDirectory, int countMustBeMultipleOf) : base(new HashSet<string>())
    {
        n_splits = nSplits;
        EmbeddedModelName = embeddedModelName;
        EmbeddedModelDirectory = embeddedModelDirectory;
        CountMustBeMultipleOf = countMustBeMultipleOf;
        _embeddedModelSample = GetEmbeddedModelSample();
    }
    #endregion

    #region Hyper-Parameters
    public int n_splits = 5;
    // ReSharper disable once MemberCanBePrivate.Global
    public string EmbeddedModelName;
    // ReSharper disable once MemberCanBePrivate.Global
    public string EmbeddedModelDirectory;
    public int CountMustBeMultipleOf = 1;
    #endregion
    public MetricEnum GetMetric()
    {
        return GetEmbeddedModelSample().GetMetric();
    }
    public LossFunctionEnum GetLoss()
    {
        return GetEmbeddedModelSample().GetLoss();
    }
    public IModelSample GetEmbeddedModelSample()
    {
        if (_embeddedModelSample == null)
        {
            _embeddedModelSample = IModelSample.LoadModelSample(EmbeddedModelDirectory, EmbeddedModelName);
        }
        return _embeddedModelSample;
    }
}
