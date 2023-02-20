﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Linq;
using log4net;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public abstract class Model: IDisposable
{
    #region private & protected fields
    private static readonly object LockUpdateFileObject = new();
    #endregion

    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(Model));
    public string WorkingDirectory { get; }
    public string ModelName { get; }
    public IModelSample ModelSample { get; }
    #endregion


    #region constructor
    protected Model(IModelSample modelSample, string workingDirectory, string modelName)
    {
        WorkingDirectory = workingDirectory;
        ModelName = modelName;
        ModelSample = modelSample;
    }
    #endregion

    public abstract List<string> AllFiles();


    public virtual (DataFrame trainPredictions_InTargetFormat, IScore trainRankingScore_InTargetFormat,
        DataFrame trainPredictions_InModelFormat, IScore trainLoss_InModelFormat,
        DataFrame validationPredictions_InTargetFormat, IScore validationRankingScore_InTargetFormat,
        DataFrame validationPredictions_InModelFormat, IScore validationLoss_InModelFormat)
        ComputePredictionsAndRankingScore(ITrainingAndTestDataset trainingAndValidation, AbstractDatasetSample datasetSample, bool computeTrainMetrics)
    {
        var validationDataset = trainingAndValidation.Test;
        var trainDataset = trainingAndValidation.Training;

        DataFrame trainPredictions_InTargetFormat = null;
        IScore trainRankingScore_InTargetFormat = null;
        DataFrame trainPredictions_InModelFormat = null;
        IScore trainLoss_InModelFormat = null;
        if (computeTrainMetrics)
        {
            (trainPredictions_InModelFormat, trainLoss_InModelFormat, trainPredictions_InTargetFormat, trainRankingScore_InTargetFormat, _) =
                datasetSample.ComputePredictionsAndRankingScoreV2(trainDataset, this, false);
        }

        DataFrame validationPredictions_InTargetFormat = null;
        IScore validationRankingScore_InTargetFormat = null;
        DataFrame validationPredictions_InModelFormat = null;
        IScore validationLoss_InModelFormat = null;
        if (validationDataset != null)
        {
            (validationPredictions_InModelFormat, validationLoss_InModelFormat, validationPredictions_InTargetFormat, validationRankingScore_InTargetFormat, _) =
                datasetSample.ComputePredictionsAndRankingScoreV2(validationDataset, this, false);
            datasetSample.Validation_XYDatasetPath_InTargetFormat = validationDataset.to_csv_in_directory(DatasetPath, true, true, false);
        }

        return (trainPredictions_InTargetFormat, trainRankingScore_InTargetFormat, trainPredictions_InModelFormat, trainLoss_InModelFormat,
            validationPredictions_InTargetFormat, validationRankingScore_InTargetFormat, validationPredictions_InModelFormat, validationLoss_InModelFormat);
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
            Log.Error("fail to add line in file:" + Environment.NewLine + line + Environment.NewLine + e);
        }
    }

    private static bool LoggingForModelShouldBeDebug(string modelName)
    {
        //Log related to embedded KFold training are in debug mode
        return modelName.Contains("_kfold_");
    }


    protected virtual void LogForModel(string msg)
    {
        if (LoggingForModelShouldBeDebug(ModelName))
        {
            LogDebug(msg);
        }
        else
        {
            LogInfo(msg);
        }
    }

    public static string MetricsToString(IDictionary<EvaluationMetricEnum, double> metrics, string prefix)
    {
        return string.Join(" - ", metrics.OrderBy(x => x.Key).Select(e => prefix + e.Key + ": " + Math.Round(e.Value, 4))).ToLowerInvariant();
    }
    public static string TrainingAndValidationMetricsToString(IDictionary<EvaluationMetricEnum, double> trainingMetrics, IDictionary<EvaluationMetricEnum, double> validationMetrics)
    {
        return MetricsToString(trainingMetrics, "") + " - " + MetricsToString(validationMetrics, "val_");
    }

    public static DataFrame LoadProbaFile(string predictionResultPath, bool hasHeader, bool hasIndex, char ?separator, DataSet dataset)
    {
        var sw = Stopwatch.StartNew();
        Func<string, string[]> split = separator.HasValue ? (s => s.Split(s, separator.Value)) : s=>s.Split();
        var readAllLines = File.ReadAllLines(predictionResultPath);
        float[][] predictionResultContent = readAllLines
            .Skip(hasHeader ? 1 : 0)
            .Select(l => split(l).Skip(hasIndex ? 1 : 0).Select(float.Parse).ToArray()).ToArray();
        int rows = predictionResultContent.Length;
        int columns = predictionResultContent[0].Length;
        var content = new float[rows * columns];
        for (int row = 0; row < rows; row++)
        {
            if (columns != predictionResultContent[row].Length)
            {
                var errorMsg = $"invalid number of predictions in line {row} of file {predictionResultPath}, expecting {columns} but received {predictionResultContent[row].Length}";
                LogError(errorMsg);
                throw new Exception(errorMsg);
            }
            Array.Copy(predictionResultContent[row], 0, content, row * columns, columns);
        }
        var predictionsCpuTensor = new CpuTensor<float>(new[] { rows, columns }, content);

        var predictionsDf = DataFrame.New(predictionsCpuTensor);
        if (dataset != null && predictionsDf.Shape[0] != dataset.Count)
        {
            throw new Exception($"Invalid number of predictions, received {predictionsDf.Shape[0]} but expected {dataset.Count}");
        }
        Log.Debug($"Loading Proba File for {predictionResultPath} took {sw.Elapsed.TotalSeconds}s");
        return predictionsDf;
    }



    public virtual DataFrame ComputeFeatureImportance(AbstractDatasetSample datasetSample, AbstractDatasetSample.DatasetType datasetType)
    {
        return null;
    }


    public virtual string DatasetPath => GetRootPath("Dataset");
    public virtual string TempPath => GetRootPath("Temp");

    private string GetRootPath(string subDirectory)
    {
        if (WorkingDirectory.StartsWith(Utils.ChallengesPath))
        {
            var subDirectoriesAfterChallengesPath = WorkingDirectory.Substring(Utils.ChallengesPath.Length).Split(new[] { '/', '\\' }, StringSplitOptions.RemoveEmptyEntries);
            if (subDirectoriesAfterChallengesPath.Length != 0)
            {
                return Path.Combine(Utils.ChallengesPath, subDirectoriesAfterChallengesPath[0], subDirectory);
            }
        }
        return Path.Combine(WorkingDirectory, subDirectory);
    }

    public abstract (string train_XDatasetPath_InModelFormat, string train_YDatasetPath_InModelFormat, string
        train_XYDatasetPath_InModelFormat, string validation_XDatasetPath_InModelFormat, string
        validation_YDatasetPath_InModelFormat, string validation_XYDatasetPath_InModelFormat, IScore
        trainScoreIfAvailable, IScore validationScoreIfAvailable, IScore trainMetricIfAvailable, IScore
        validationMetricIfAvailable)
        Fit(DataSet trainDataset, DataSet validationDatasetIfAny);

    /// <summary>
    /// do Model inference for dataset 'dataset' and returns the predictions
    /// </summary>
    /// <param name="dataset">the dataset we want to make the inference</param>
    /// <param name="removeAllTemporaryFilesAtEnd">
    ///     if true:
    ///     all temporary files needed by the model for inference will be deleted</param>
    /// <returns>
    /// predictions: the model inferences
    /// datasetPath:
    ///     if the model has needed to store the dataset into a file (ex: LightGBM, CatBoost)
    ///         the associated path
    ///     else
    ///         an empty string
    /// </returns>
    public virtual DataFrame Predict(DataSet dataset, bool removeAllTemporaryFilesAtEnd)
    {
        return PredictWithPath(dataset, removeAllTemporaryFilesAtEnd).predictions;
    }


    public IScore ComputeLoss(DataSet dataset, bool removeAllTemporaryFilesAtEnd)
    {
        var y_pred = Predict(dataset, removeAllTemporaryFilesAtEnd);
        var y_true = dataset.Y;
        return ComputeLoss(y_true, y_pred.FloatTensor);
    }


    public abstract (DataFrame predictions, string datasetPath) PredictWithPath(DataSet dataset, bool removeAllTemporaryFilesAtEnd);
    
    public abstract void Save(string workingDirectory, string modelName);
    public virtual string DeviceName() => "";
    public virtual int TotalParams() => -1;
    public virtual int GetNumEpochs() => -1;
    public virtual double GetLearningRate() => double.NaN;

    protected static void LogDebug(string message) { Log.Debug(message); }
    protected static void LogInfo(string message) { Log.Info(message); }
    protected static void LogWarn(string message) { Log.Warn(message); }
    protected static void LogError(string message) { Log.Error(message); }

    public virtual void Dispose()
    {
    }
}
