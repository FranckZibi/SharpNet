using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Networks;

namespace SharpNet.Models;

public interface IModel
{
    void Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny);
    CpuTensor<float> Predict(IDataSet dataset);
    void Save(string workingDirectory, string modelName);
    float ComputeScore(CpuTensor<float> y_true, CpuTensor<float> y_pred);

    int GetNumEpochs();
    string GetDeviceName();
    double GetLearningRate();
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

    public static AbstractModel ValueOf(string workingDirectory, string modelName)
    {
        try { return KFoldModel.ValueOf(workingDirectory, modelName); } catch { }
        try { return Network.ValueOf(workingDirectory, modelName); } catch { }
        try { return LightGBMModel.ValueOf(workingDirectory, modelName); } catch { }
        try { return CatBoostModel.ValueOf(workingDirectory, modelName); } catch { }
        throw new ArgumentException($"cant' load model {modelName} from {workingDirectory}");
    }

    public static AbstractModel NewModel(ISample sample, string workingDirectory, string modelName)
    {
        if (sample is CatBoostSample catBoostSample)
        {
            return new CatBoostModel(catBoostSample, workingDirectory, modelName);
        }
        if (sample is LightGBMSample lightGBMSample)
        {
            return new LightGBMModel(lightGBMSample, workingDirectory, modelName);
        }
        if (sample is NetworkSample networkSample)
        {
            return new Network(networkSample, workingDirectory, modelName);
        }
        throw new ArgumentException($"cant' load model {modelName} from {workingDirectory} for sample type {sample.GetType()}");
    }
}