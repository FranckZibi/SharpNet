using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
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
    public IScore ComputeLoss(CpuTensor<float> y_true, CpuTensor<float> y_pred)
    {
        if (y_true == null || y_pred == null)
        {
            return null;
        }
        Debug.Assert(y_true.SameShape(y_pred));
        var lossMetric = ModelSample.GetLoss();
        using var buffer = new CpuTensor<float>(y_true.ComputeMetricBufferShape(lossMetric));
        return new Score( (float)y_true.ComputeEvaluationMetric(y_pred, lossMetric, buffer), lossMetric);
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

    protected static bool LoggingForModelShouldBeDebug(string modelName)
    {
        //Log related to embedded KFold training are in debug mode
        return modelName.Contains("_kfold_");
    }

    protected static DataFrame LoadProbaFile(string predictionResultPath, bool hasHeader, bool hasIndex, IDataSet dataset, bool addIdColumnsAtLeft)
    {
        float[][] predictionResultContent = File.ReadAllLines(predictionResultPath)
            .Skip(hasHeader ? 1 : 0)
            .Select(l => l.Split().Skip(hasIndex ? 1 : 0).Select(float.Parse).ToArray()).ToArray();
        int rows = predictionResultContent.Length;
        int columns = predictionResultContent[0].Length;
        var content = new float[rows * columns];
        for (int row = 0; row < rows; row++)
        {
            if (columns != predictionResultContent[row].Length)
            {
                var errorMsg = $"invalid number of predictions in line {row} of file {predictionResultPath}, expecting {columns} ";
                LogError(errorMsg);
                throw new Exception(errorMsg);
            }
            Array.Copy(predictionResultContent[row], 0, content, row * columns, columns);
        }
        var predictionsCpuTensor = new CpuTensor<float>(new[] { rows, columns }, content);

        string[] predictionLabels = dataset.TargetLabels;
        if (predictionLabels.Length != predictionsCpuTensor.Shape[1])
        {
            predictionLabels = Enumerable.Range(0, predictionsCpuTensor.Shape[1]).Select(x => x.ToString()).ToArray();
        }
        var predictionsDf = DataFrame.New(predictionsCpuTensor, predictionLabels);
        if (addIdColumnsAtLeft && dataset.IdColumns.Length != 0)
        {
            predictionsDf = DataFrame.MergeHorizontally(dataset.ExtractIdDataFrame(), predictionsDf);
        }
        if (predictionsDf.Shape[0] != dataset.Count)
        {
            throw new Exception($"Invalid number of predictions, received {predictionsDf.Shape[0]} but expected {dataset.Count}");
        }
        return predictionsDf;
    }


    public virtual string RootDatasetPath => Path.Combine(WorkingDirectory, "Dataset");

    public abstract (string train_XDatasetPath, string train_YDatasetPath, string train_XYDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath, string validation_XYDatasetPath) 
        Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny);
    public abstract DataFrame Predict(IDataSet dataset, bool addIdColumnsAtLeft, bool removeAllTemporaryFilesAtEnd);
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
