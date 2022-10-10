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
    public KFoldSample(int nSplits, string embeddedModelName, string embeddedModelWorkingDirectory, int countMustBeMultipleOf) : base(new HashSet<string>())
    {
        n_splits = nSplits;
        EmbeddedModelName = embeddedModelName;
        EmbeddedModelWorkingDirectory = embeddedModelWorkingDirectory;
        CountMustBeMultipleOf = countMustBeMultipleOf;
        _embeddedModelSample = GetEmbeddedModelSample();
    }
    public KFoldSample(int nSplits, string embeddedModelName, string embeddedModelWorkingDirectory, IModelSample embeddedModelSample, int countMustBeMultipleOf) : base(new HashSet<string>())
    {
        n_splits = nSplits;
        EmbeddedModelName = embeddedModelName;
        EmbeddedModelWorkingDirectory = embeddedModelWorkingDirectory;
        CountMustBeMultipleOf = countMustBeMultipleOf;
        _embeddedModelSample = embeddedModelSample;
    }
    #endregion

    #region Hyper-Parameters
    public int n_splits = 5;
    // ReSharper disable once MemberCanBePrivate.Global
    public string EmbeddedModelName;
    // ReSharper disable once MemberCanBePrivate.Global
    public string EmbeddedModelWorkingDirectory;
    public int CountMustBeMultipleOf = 1;
    #endregion
    public EvaluationMetricEnum GetLoss()
    {
        return GetEmbeddedModelSample().GetLoss();
    }
    public IModelSample GetEmbeddedModelSample()
    {
        if (_embeddedModelSample == null)
        {
            _embeddedModelSample = IModelSample.LoadModelSample(EmbeddedModelWorkingDirectory, EmbeddedModelName);
        }
        return _embeddedModelSample;
    }
}
