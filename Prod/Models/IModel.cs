using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Datasets;

namespace SharpNet.Models;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public interface IModel
{
    void Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny);
    CpuTensor<float> Predict(IDataSet dataset);
    void Save(string workingDirectory, string modelName);
    float ComputeScore(CpuTensor<float> y_true, CpuTensor<float> y_pred);
    string WorkingDirectory { get; }
    string ModelName { get; }

    public static string MetricsToString(IDictionary<MetricEnum, double> metrics, string prefix)
    {
        return string.Join(" - ", metrics.OrderBy(x => x.Key).Select(e => prefix + e.Key + ": " + Math.Round(e.Value, 4))).ToLowerInvariant();
    }
    public static string TrainingAndValidationMetricsToString(IDictionary<MetricEnum, double> trainingMetrics, IDictionary<MetricEnum, double> validationMetrics)
    {
        return MetricsToString(trainingMetrics, "") + " - " + MetricsToString(validationMetrics, "val_");
    }
}