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

    public static IModelSample ValueOfModelSample(string workingDirectory, string modelName)
    {
        try { return LightGBMSample.ValueOf(workingDirectory, modelName); } catch { }
        try { return CatBoostSample.ValueOf(workingDirectory, modelName); } catch { }
        try { return KFoldSample.Load(workingDirectory, modelName); } catch { }
        try { return WeightsOptimizerSample.ValueOf(workingDirectory, modelName); } catch { }
        throw new Exception($"can't load sample from model {modelName} in directory {workingDirectory}");
    }
}

