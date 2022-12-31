using System;
using System.Collections.Generic;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
// ReSharper disable FieldCanBeMadeReadOnly.Global

namespace SharpNet.Models;

public class KFoldSample : AbstractSample, IModelSample
{
    #region constructors
    // ReSharper disable once UnusedMember.Local
    private KFoldSample() : base(new HashSet<string>())
    {
    }
    public KFoldSample(int nSplits, string embeddedModelWorkingDirectory, string embeddedModelName, EvaluationMetricEnum embeddedModelLoss, int countMustBeMultipleOf) : base(new HashSet<string>())
    {
        if (nSplits <= 1)
        {
            throw new ArgumentException($"Invalid KFold splits {nSplits} , must be >= 2");
        }
        n_splits = nSplits;
        EmbeddedModelWorkingDirectory = embeddedModelWorkingDirectory;
        EmbeddedModelName = embeddedModelName;
        EmbeddedModelLoss = embeddedModelLoss;
        CountMustBeMultipleOf = countMustBeMultipleOf;
    }
    #endregion

    #region Hyper-Parameters
    public int n_splits = 5;
    // ReSharper disable once MemberCanBePrivate.Global
    public string EmbeddedModelWorkingDirectory;
    public string EmbeddedModelName;
    // ReSharper disable once MemberCanBePrivate.Global
    public EvaluationMetricEnum EmbeddedModelLoss;
    public int CountMustBeMultipleOf = 1;
    #endregion
    public EvaluationMetricEnum GetLoss()
    {
        return EmbeddedModelLoss;
    }

    public void FillSearchSpaceWithDefaultValues(IDictionary<string, object> existingHyperParameterValues, AbstractDatasetSample datasetSample)
    {
        throw new NotImplementedException();
    }
}
