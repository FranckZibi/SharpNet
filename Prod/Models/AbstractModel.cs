﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using log4net;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

public abstract class AbstractModel : IModel
{
    #region private & protected fields
    public ISample Sample { get; }
    #endregion

    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(AbstractModel));
    public abstract double GetLearningRate();
    public string WorkingDirectory { get; }
    public string ModelName { get; }
    public string LastDatasetPathUsedForTraining { get; protected set; } = "";
    public string LastDatasetPathUsedForPrediction { get; protected set; } = "";
    // ReSharper disable once MemberCanBeProtected.Global
    // ReSharper disable once UnusedAutoPropertyAccessor.Global
    public string LastDatasetPathUsedForValidation { get; protected set; } = "";
    #endregion

    #region constructor
    protected AbstractModel(ISample sample, string workingDirectory, string modelName)
    {
        WorkingDirectory = workingDirectory;
        ModelName = modelName;

        if (sample is not IMetricFunction)
        {
            throw new ArgumentException($"Sample {sample.GetType()} must implement {nameof(IMetricFunction)}");
        }

        Sample = sample;
    }
    #endregion

    public abstract void Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny);
    public abstract CpuTensor<float> Predict(IDataSet dataset);
    public abstract void Save(string workingDirectory, string modelName);
    public List<string> AllFiles()
    {
        var res = SampleFiles();
        res.AddRange(ModelFiles());
        return res;
    }
    public static string DatasetPath(IDataSet dataset, bool addTargetColumnAsFirstColumn, string rootDatasetPath) => Path.Combine(rootDatasetPath, ComputeUniqueDatasetName(dataset, addTargetColumnAsFirstColumn) + ".csv");

    /// <param name="savePredictions"></param>
    /// <param name="UnNormalizeYIfNeeded"></param>
    /// <param name="trainDataset"></param>
    /// <param name="validationDatasetIfAny"></param>
    /// <param name="testDatasetIfAny"></param>
    /// <returns>the cost associated with the model</returns>
    public (string, float, string, float, string) ComputePredictions(
        IDataSet trainDataset,
        IDataSet validationDatasetIfAny,
        IDataSet testDatasetIfAny,
        Action<CpuTensor<float>, string> savePredictions,
        Func<CpuTensor<float>, CpuTensor<float>> UnNormalizeYIfNeeded = null
        )
    {
        UnNormalizeYIfNeeded ??= c => c;

        Log.Debug($"Computing Model '{ModelName}' predictions for Training Dataset");
        var trainPredictions = UnNormalizeYIfNeeded(Predict(trainDataset));
        Log.Debug("Computing Model score on Training");
        var trainScore = ComputeScore(UnNormalizeYIfNeeded(trainDataset.Y), trainPredictions);
        Log.Info($"Model '{ModelName}' score on training: {trainScore}");


        var validationPredictionsPath = "";
        float validationScore = float.NaN;
        CpuTensor<float> validationPredictions = null;
        if (validationDatasetIfAny != null)
        {
            Log.Debug($"Computing Model '{ModelName}' predictions for Validation Dataset");
            validationPredictions = UnNormalizeYIfNeeded(Predict(validationDatasetIfAny));
            Log.Debug($"Computing Model '{ModelName}' score on Validation");
            validationScore = ComputeScore(UnNormalizeYIfNeeded(validationDatasetIfAny.Y), validationPredictions);
            Log.Info($"Model '{ModelName}' score on Validation: {validationScore}");
        }

        var trainPredictionsPath = Path.Combine(WorkingDirectory, ModelName + "_predict_train_" + Math.Round(trainScore, 5) + ".csv");
        Log.Info($"Saving Model '{ModelName}' predictions for Training Dataset");
        savePredictions(trainPredictions, trainPredictionsPath);

        if (!float.IsNaN(validationScore))
        {
            Log.Info($"Saving Model '{ModelName}' predictions for Validation Dataset");
            validationPredictionsPath = Path.Combine(WorkingDirectory, ModelName + "_predict_valid_" + Math.Round(validationScore, 5) + ".csv");
            savePredictions(validationPredictions, validationPredictionsPath);
        }

        var testPredictionsPath = "";
        if (testDatasetIfAny != null)
        {
            Log.Debug($"Computing Model '{ModelName}' predictions for Test Dataset");
            var testPredictions = Predict(testDatasetIfAny);
            Log.Info("Saving predictions for Test Dataset");
            testPredictionsPath = Path.Combine(WorkingDirectory, ModelName + "_predict_test_.csv");
            savePredictions(testPredictions, testPredictionsPath);
        }

        return (trainPredictionsPath, trainScore, validationPredictionsPath, validationScore, testPredictionsPath);
    }

    public float ComputeScore(CpuTensor<float> y_true, CpuTensor<float> y_pred)
    {
        using var buffer = new CpuTensor<float>(new[] { y_true.Shape[0] });
        var metricEnum = ((IMetricFunction)Sample).GetMetric();
        var lossFunctionEnum = ((IMetricFunction)Sample).GetLoss();
        return (float)y_true.ComputeMetric(y_pred, metricEnum, lossFunctionEnum, buffer);
    }

    public bool NewScoreIsBetterTheReferenceScore(float newScore, float referenceScore)
    {
        var metricEnum = ((IMetricFunction)Sample).GetMetric();
        switch (metricEnum)
        {
            case MetricEnum.Accuracy: 
                return newScore > referenceScore; // highest is better
            case MetricEnum.Loss:
            case MetricEnum.Mae:
            case MetricEnum.Mse:
            case MetricEnum.Rmse:
                return newScore < referenceScore; // lowest is better
            default:
                throw new NotImplementedException($"unknown metric : {metricEnum}");
        }
    }

    public abstract int GetNumEpochs();
    public abstract string GetDeviceName();

    public abstract List<string> ModelFiles();

    private static string ComputeDescription(Tensor tensor)
    {
        if (tensor == null || tensor.Count == 0)
        {
            return "";
        }
        Debug.Assert(tensor.Shape.Length == 2);
        var xDataSpan = tensor.AsReadonlyFloatCpuContent;
        var desc = string.Join('_', tensor.Shape);
        for (int col = 0; col < tensor.Shape[1]; ++col)
        {
            int row = ((tensor.Shape[0] - 1) * col) / Math.Max(1, tensor.Shape[1] - 1);
            var val = xDataSpan[row * tensor.Shape[1] + col];
            desc += '_' + Math.Round(val, 6).ToString(CultureInfo.InvariantCulture);
        }
        return desc;
    }


    private static string ComputeDescription(IDataSet dataset)
    {
        if (dataset == null || dataset.Count == 0)
        {
            return "";
        }
        var desc = string.Join('_', dataset.Count);
        int rows = dataset.Count;
        int cols = dataset.FeatureNamesIfAny.Length;
        using CpuTensor<float> xBuffer = new (new []{1, cols});
        var xDataSpan = xBuffer.AsReadonlyFloatCpuContent;
        for (int col = 0; col < cols; ++col)
        {
            int row = ((rows - 1) * col) / Math.Max(1, cols - 1);
            dataset.LoadAt(row, 0, xBuffer, null, false);
            var val = xDataSpan[col];
            desc += '_' + Math.Round(val, 6).ToString(CultureInfo.InvariantCulture);
        }
        return desc;
    }

    private static string ComputeUniqueDatasetName(IDataSet dataset, bool addTargetColumnAsFirstColumn)
    {
        var desc = ComputeDescription(dataset);
        if (addTargetColumnAsFirstColumn)
        {
            desc += '_' + ComputeDescription(dataset.Y);
        }
        return Utils.ComputeHash(desc, 10);
    }
    private List<string> SampleFiles()
    {
        return Sample.SampleFiles(WorkingDirectory, ModelName);
    }
}