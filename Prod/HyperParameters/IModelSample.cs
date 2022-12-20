using System.Diagnostics.CodeAnalysis;

namespace SharpNet.HyperParameters;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public interface IModelSample : ISample
{
    EvaluationMetricEnum GetLoss();

    public static IModelSample LoadModelSample(string workingDirectory, string sampleName)
    {
        return (IModelSample)Load(workingDirectory, sampleName);
    }
}

