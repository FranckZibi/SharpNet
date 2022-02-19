using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using log4net;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Networks;

namespace SharpNet.Models;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public abstract class AbstractModel : IModel
{
    #region private & protected fields
    private static readonly object LockUpdateFileObject = new();
    #endregion

    #region public fields & properties
    public IModelSample Sample { get; }
    public static readonly ILog Log = LogManager.GetLogger(typeof(AbstractModel));
    public string WorkingDirectory { get; }
    public string ModelName { get; }
    public string LastDatasetPathUsedForTraining { get; protected set; } = "";
    public string LastDatasetPathUsedForPrediction { get; protected set; } = "";
    // ReSharper disable once MemberCanBeProtected.Global
    // ReSharper disable once UnusedAutoPropertyAccessor.Global
    public string LastDatasetPathUsedForValidation { get; protected set; } = "";
    #endregion

    #region constructor
    protected AbstractModel(IModelSample sample, string workingDirectory, string modelName)
    {
        WorkingDirectory = workingDirectory;
        ModelName = modelName;
        Sample = sample;
    }
    #endregion

    public abstract void Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny);
    public abstract CpuTensor<float> Predict(IDataSet dataset);
    public abstract void Save(string workingDirectory, string modelName);
    public List<string> AllFiles()
    {
        var res = Sample.SampleFiles(WorkingDirectory, ModelName);
        res.AddRange(ModelFiles());
        return res;
    }
    public static string DatasetPath(IDataSet dataset, bool addTargetColumnAsFirstColumn, string rootDatasetPath) => Path.Combine(rootDatasetPath, ComputeUniqueDatasetName(dataset, addTargetColumnAsFirstColumn) + ".csv");
    public float ComputeScore(CpuTensor<float> y_true, CpuTensor<float> y_pred)
    {
        using var buffer = new CpuTensor<float>(new[] { y_true.Shape[0] });
        var metricEnum = Sample.GetMetric();
        var lossFunctionEnum = Sample.GetLoss();
        return (float)y_true.ComputeMetric(y_pred, metricEnum, lossFunctionEnum, buffer);
    }
    public bool NewScoreIsBetterTheReferenceScore(float newScore, float referenceScore)
    {
        var metricEnum = Sample.GetMetric();
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
    public abstract string DeviceName();
    public abstract int TotalParams();
    public abstract double GetLearningRate();
    public abstract List<string> ModelFiles();
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
            Log.Error("fail to add line in file:" + Environment.NewLine + line + Environment.NewLine + e);
        }
    }

    public static AbstractModel NewModel(IModelSample sample, string workingDirectory, string modelName)
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

    protected static AbstractModel ValueOfAbstractModel(string workingDirectory, string modelName)
    {
        try { return KFoldModel.ValueOf(workingDirectory, modelName); } catch { }
        try { return Network.ValueOf(workingDirectory, modelName); } catch { }
        try { return LightGBMModel.ValueOf(workingDirectory, modelName); } catch { }
        try { return CatBoostModel.ValueOf(workingDirectory, modelName); } catch { }
        throw new ArgumentException($"cant' load model {modelName} from {workingDirectory}");
    }

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
}
