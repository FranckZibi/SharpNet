using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using SharpNet.CPU;
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
    public string WorkingDirectory { get; }
    public string ModelName { get; }
    public IModelSample ModelSample { get; }
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
    public virtual (DataFrame predictions, string predictionPath) PredictWithPath(IDataSet dataset)
    {
        return (Predict(dataset), "");
    }
    public float ComputeScore(CpuTensor<float> y_true, CpuTensor<float> y_pred)
    {
        if (y_true == null || y_pred == null)
        {
            return float.NaN;
        }
        var metricEnum = ModelSample.GetMetric();
        var lossFunctionEnum = ModelSample.GetLoss();
        using var buffer = new CpuTensor<float>(y_true.ComputeMetricBufferShape(metricEnum));
        return (float)y_true.ComputeMetric(y_pred, metricEnum, lossFunctionEnum, buffer);
    }
    public void AddResumeToCsv(double trainingTimeInSeconds, IScore trainScore, IScore validationScore, string csvPath)
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
                + IScore.ToString(trainScore) + ";"
                + "NaN" + ";"
                + IScore.ToString(validationScore) + ";"
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
    public virtual string RootDatasetPath => Path.Combine(WorkingDirectory, "Dataset");

    public abstract (string train_XDatasetPath, string train_YDatasetPath, string train_XYDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath, string validation_XYDatasetPath) 
        Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny);
    public abstract DataFrame Predict(IDataSet dataset);
    public abstract void Save(string workingDirectory, string modelName);
    public abstract string DeviceName();
    public abstract int TotalParams();
    public abstract List<string> ModelFiles();

    public abstract int GetNumEpochs();
    public abstract double GetLearningRate();
    public abstract void Use_All_Available_Cores();
    
    protected static void LogDebug(string message) { IModel.Log.Debug(message); }
    protected static void LogInfo(string message) { IModel.Log.Info(message); }
    protected static void LogWarn(string message) { IModel.Log.Warn(message); }
    protected static void LogError(string message) { IModel.Log.Error(message); }
}
