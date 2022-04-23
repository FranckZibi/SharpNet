using System;
using System.Diagnostics.CodeAnalysis;
using SharpNet.CatBoost;
using SharpNet.HPO;
using SharpNet.LightGBM;
using SharpNet.Models;

namespace SharpNet.HyperParameters;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public interface IModelSample : ISample
{
    MetricEnum GetMetric();
    LossFunctionEnum GetLoss();

    public static IModelSample LoadModelSample(string workingDirectory, string sampleName)
    {
        try { return LoadSample<LightGBMSample>(workingDirectory, sampleName); } catch { }
        try { return LoadSample<CatBoostSample>(workingDirectory, sampleName); } catch { }
        try { return LoadSample<KFoldSample>(workingDirectory, sampleName); } catch { }
        try { return LoadSample<WeightsOptimizerSample>(workingDirectory, sampleName); } catch { }
        throw new Exception($"can't load sample from model {sampleName} in directory {workingDirectory}");
    }
}

