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
    void Use_All_Available_Cores();

    public static IModelSample LoadModelSample(string workingDirectory, string sampleName)
    {
        try { return LightGBMSample.LoadLightGBMSample(workingDirectory, sampleName); } catch { }
        try { return CatBoostSample.LoadCatBoostSample(workingDirectory, sampleName); } catch { }
        try { return KFoldSample.LoadKFoldSample(workingDirectory, sampleName); } catch { }
        try { return WeightsOptimizerSample.LoadWeightsOptimizerSample(workingDirectory, sampleName); } catch { }
        throw new Exception($"can't load sample from model {sampleName} in directory {workingDirectory}");
    }
}

