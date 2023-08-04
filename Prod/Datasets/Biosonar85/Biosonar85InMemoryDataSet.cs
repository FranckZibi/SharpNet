using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using SharpNet.CPU;

namespace SharpNet.Datasets.Biosonar85;

public class Biosonar85InMemoryDataSet : InMemoryDataSet
{
    public Biosonar85InMemoryDataSet([NotNull] CpuTensor<float> x, [CanBeNull] CpuTensor<float> y, string name, List<Tuple<float, float>> meanAndVolatilityForEachChannel = null, [CanBeNull] string[] yIDs = null)
        : base(x, y, name, Objective_enum.Classification, meanAndVolatilityForEachChannel, null /* columnNames*/, new string[0], yIDs, "id", ',')
    {
    }

    public override int[] IdToValidationKFold(int n_splits, int countMustBeMultipleOf)
    {
        if (n_splits != 2)
        {
            throw new ArgumentException($"only KFold == 2 is supported, not {n_splits}");
        }
        var validationIntervalForKfold = new int[Count];
        string IdToSite(string id) { return id.Split(new[] { '-', '.' })[1]; }
        for (int i = 0; i < Y_IDs.Length; ++i)
        {
            var site = IdToSite(Y_IDs[i]);
            validationIntervalForKfold[i] = (site.StartsWith("GUA") || site.StartsWith("JAM") || site.StartsWith("BERMUDE"))?1:0;
        }
        return validationIntervalForKfold;
    }
}
