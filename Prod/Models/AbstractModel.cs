using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public abstract class AbstractModel : IModel
{
    #region private & protected fields
    private static readonly object LockUpdateFileObject = new();
    #endregion

    #region public fields & properties
    public IModelSample ModelSample { get; }
    public string WorkingDirectory { get; }
    public string ModelName { get; }
    #endregion


    #region constructor
    protected AbstractModel(IModelSample modelSample, string workingDirectory, string modelName)
    {
        WorkingDirectory = workingDirectory;
        ModelName = modelName;
        ModelSample = modelSample;
    }
    #endregion

    public List<string> AllFiles()
    {
        var res = ModelSample.SampleFiles(WorkingDirectory, ModelName);
        res.AddRange(ModelFiles());
        return res;
    }
    public virtual (CpuTensor<float> predictions, string predictionPath) PredictWithPath(IDataSet dataset)
    {
        return (Predict(dataset), "");
    }
    public float ComputeScore(CpuTensor<float> y_true, CpuTensor<float> y_pred)
    {
        using var buffer = new CpuTensor<float>(new[] { y_true.Shape[0] });
        var metricEnum = ModelSample.GetMetric();
        var lossFunctionEnum = ModelSample.GetLoss();
        return (float)y_true.ComputeMetric(y_pred, metricEnum, lossFunctionEnum, buffer);
    }
    public bool NewScoreIsBetterTheReferenceScore(float newScore, float referenceScore)
    {
        var metricEnum = ModelSample.GetMetric();
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
    public void AddResumeToCsv(double trainingTimeInSeconds, float trainScore, float validationScore, string csvPath)
    {
        var line = "";
        try
        {
            int numEpochs = GetNumEpochs();
            //We save the results of the net
            line = DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";"
                + ModelName.Replace(';', '_') + ";"
                + DeviceName() + ";"
                + TotalParams() + ";" //should be: datasetSample.X_Shape(1)[1];
                + numEpochs + ";"
                + "-1" + ";"
                + GetLearningRate() + ";"
                + trainingTimeInSeconds + ";"
                + (trainingTimeInSeconds / numEpochs) + ";"
                + trainScore + ";"
                + "NaN" + ";"
                + validationScore + ";"
                + "NaN" + ";"
                + Environment.NewLine;
            lock (LockUpdateFileObject)
            {
                File.AppendAllText(csvPath, line);
            }
        }
        catch (Exception e)
        {
            IModel.Log.Error("fail to add line in file:" + Environment.NewLine + line + Environment.NewLine + e);
        }
    }
    public abstract (string train_XDatasetPath, string train_YDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath) Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny);
    public abstract CpuTensor<float> Predict(IDataSet dataset);
    public abstract void Save(string workingDirectory, string modelName);
    public abstract string DeviceName();
    public abstract int TotalParams();
    public abstract List<string> ModelFiles();

    protected abstract int GetNumEpochs();
    protected static string DatasetPath(IDataSet dataset, bool addTargetColumnAsFirstColumn, string rootDatasetPath) => Path.Combine(rootDatasetPath, ComputeUniqueDatasetName(dataset, addTargetColumnAsFirstColumn) + ".csv");
    protected abstract double GetLearningRate();

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
        int rows = dataset.Count;
        int cols = dataset.FeatureNamesIfAny.Length;
        var desc = rows+"_"+cols;
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

    protected static void LogDebug(string message) { IModel.Log.Debug(message); }
    protected static void LogInfo(string message) { IModel.Log.Info(message); }
    protected static void LogWarn(string message) { IModel.Log.Warn(message); }
    protected static void LogError(string message) { IModel.Log.Error(message); }
}
