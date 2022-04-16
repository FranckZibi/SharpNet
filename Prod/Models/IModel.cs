using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using log4net;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Networks;

namespace SharpNet.Models;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public interface IModel
{
    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(IModel));
    IModelSample ModelSample { get; }
    string WorkingDirectory { get; }
    string ModelName { get; }
    #endregion


    #region constructor
    public static IModel NewModel(IModelSample sample, string workingDirectory, string modelName)
    {
        if (sample is CatBoostSample catBoostSample)
        {
            return new CatBoostModel(catBoostSample, workingDirectory, modelName);
        }
        if (sample is LightGBMSample lightGBMSample)
        {
            return new LightGBMModel(lightGBMSample, workingDirectory, modelName);
        }
        //if (sample is WeightsOptimizerSample weightsOptimizerSample)
        //{
        //    return new WeightsOptimizer(weightsOptimizerSample, workingDirectory, modelName);
        //}
        //if (sample is KFoldSample)
        //{
        //    return KFoldModel.LoadTrainedKFoldModel(workingDirectory, modelName);
        //}
        if (sample is NetworkSample networkSample)
        {
            return new Network(networkSample, workingDirectory, modelName);
        }
        throw new ArgumentException($"cant' load model {modelName} from {workingDirectory} for sample type {sample.GetType()}");
    }
    protected static IModel LoadTrainedAbstractModel(string workingDirectory, string modelName)
    {
        //try { return ModelAndDataset.LoadAutoTrainableModel(workingDirectory, modelName); } catch { }
        //try { return KFoldModel.LoadTrainedKFoldModel(workingDirectory, modelName); } catch { }
        try { return Network.LoadTrainedNetworkModel(workingDirectory, modelName); } catch { }
        try { return LightGBMModel.LoadTrainedLightGBMModel(workingDirectory, modelName); } catch { }
        try { return CatBoostModel.LoadTrainedCatBoostModel(workingDirectory, modelName); } catch { }
        throw new ArgumentException($"can't load model {modelName} from {workingDirectory}");
    }
    #endregion

    (string train_XDatasetPath, string train_YDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath) Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny);
    CpuTensor<float> Predict(IDataSet dataset);
    (CpuTensor<float> predictions, string predictionPath) PredictWithPath(IDataSet dataset);
    void Save(string workingDirectory, string modelName);
    float ComputeScore(CpuTensor<float> y_true, CpuTensor<float> y_pred);
    List<string> ModelFiles();
    bool NewScoreIsBetterTheReferenceScore(float newScore, float referenceScore);
    void AddResumeToCsv(double trainingTimeInSeconds, float trainScore, float validationScore, string csvPath);

    public static string MetricsToString(IDictionary<MetricEnum, double> metrics, string prefix)
    {
        return string.Join(" - ", metrics.OrderBy(x => x.Key).Select(e => prefix + e.Key + ": " + Math.Round(e.Value, 4))).ToLowerInvariant();
    }
    public static string TrainingAndValidationMetricsToString(IDictionary<MetricEnum, double> trainingMetrics, IDictionary<MetricEnum, double> validationMetrics)
    {
        return MetricsToString(trainingMetrics, "") + " - " + MetricsToString(validationMetrics, "val_");
    }
}