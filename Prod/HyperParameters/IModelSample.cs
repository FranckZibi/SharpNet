namespace SharpNet.HyperParameters;

public interface IModelSample : ISample
{
    MetricEnum GetMetric();
    LossFunctionEnum GetLoss();
}