using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.Networks;

namespace SharpNet.Models;

public interface IModel
{
    void Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny);
    CpuTensor<float> Predict(IDataSet dataset);

    string WorkingDirectory { get; }
    string ModelName { get; }

    void Save(string workingDirectory, string modelName);



    public static string MetricsToString(IDictionary<NetworkConfig.Metric, double> metrics, string prefix)
    {
        return string.Join(" - ", metrics.OrderBy(x => x.Key).Select(e => prefix + e.Key + ": " + Math.Round(e.Value, 4))).ToLowerInvariant();
    }
    public static string TrainingAndValidationMetricsToString(IDictionary<NetworkConfig.Metric, double> trainingMetrics, IDictionary<NetworkConfig.Metric, double> validationMetrics)
    {
        return MetricsToString(trainingMetrics, "") + " - " + MetricsToString(validationMetrics, "val_");
    }

}