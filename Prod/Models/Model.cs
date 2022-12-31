using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
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
public abstract class Model
{
    #region private & protected fields
    private static readonly object LockUpdateFileObject = new();
    #endregion

    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(Model));
    public string WorkingDirectory { get; }
    public string ModelName { get; }
    public IModelSample ModelSample { get; }
    public virtual AbstractDatasetSample DatasetSample { get; } = null;
    #endregion


    #region constructor
    protected Model(IModelSample modelSample, string workingDirectory, string modelName)
    {
        WorkingDirectory = workingDirectory;
        ModelName = modelName;
        ModelSample = modelSample;
    }
    public static Model NewModel(IModelSample sample, AbstractDatasetSample datasetSample, string workingDirectory, string modelName)
    {
        if (sample is CatBoostSample catBoostSample)
        {
            return new CatBoostModel(catBoostSample, workingDirectory, modelName);
        }
        if (sample is LightGBMSample lightGBMSample)
        {
            return new LightGBMModel(lightGBMSample, workingDirectory, modelName);
        }
        //if (sample is WeightedModelSample weightedModelSample)
        //{
        //    return new WeightedModel(weightedModelSample, workingDirectory, modelName);
        //}
        if (sample is KFoldSample kFoldSample)
        {
            return new KFoldModel(kFoldSample, datasetSample, workingDirectory, modelName);
        }
        if (sample is NetworkSample networkSample)
        {
            return new Network(networkSample, datasetSample, workingDirectory, modelName, true);
        }
        throw new ArgumentException($"cant' load model {modelName} from {workingDirectory} for sample type {sample.GetType()}");
    }
    //protected static AbstractModel LoadTrainedAbstractModel(string workingDirectory, string modelName)
    //{
    //    //try { return ModelAndDataset.LoadAutoTrainableModel(workingDirectory, modelName); } catch { }
    //    //try { return KFoldModel.LoadTrainedKFoldModel(workingDirectory, modelName); } catch { }
    //    try { return Network.LoadTrainedNetworkModel(workingDirectory, modelName); } catch { }
    //    try { return LightGBMModel.LoadTrainedLightGBMModel(workingDirectory, modelName); } catch { }
    //    try { return CatBoostModel.LoadTrainedCatBoostModel(workingDirectory, modelName); } catch { }
    //    throw new ArgumentException($"can't load model {modelName} from {workingDirectory}");
    #endregion

    public abstract List<string> AllFiles();

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
                var errorMsg = $"invalid number of predictions in line {row} of file {predictionResultPath}, expecting {columns} ";
                LogError(errorMsg);
                throw new Exception(errorMsg);
            }
            Array.Copy(predictionResultContent[row], 0, content, row * columns, columns);
        }
        var predictionsCpuTensor = new CpuTensor<float>(new[] { rows, columns }, content);

        var predictionsDf = DataFrame.New(predictionsCpuTensor);
        if (predictionsDf.Shape[0] != dataset.Count)
        {
            throw new Exception($"Invalid number of predictions, received {predictionsDf.Shape[0]} but expected {dataset.Count}");
        }
        return predictionsDf;
    }



    public virtual DataFrame ComputeFeatureImportance(AbstractDatasetSample datasetSample, AbstractDatasetSample.DatasetType datasetType)
    {
        return null;
    }


    public virtual string RootDatasetPath
    {
        get
        {
            if (WorkingDirectory.StartsWith(Utils.ChallengesPath))
            {
                var subDirectoriesAfterChallengesPath = WorkingDirectory.Substring(Utils.ChallengesPath.Length).Split(new[]{ '/', '\\' }, StringSplitOptions.RemoveEmptyEntries);
                if (subDirectoriesAfterChallengesPath.Length != 0)
                {
                    return Path.Combine(Utils.ChallengesPath, subDirectoriesAfterChallengesPath[0], "Dataset");
                }
            }
            return Path.Combine(WorkingDirectory, "Dataset");
        }
    }

    public abstract (string train_XDatasetPath_InModelFormat, string train_YDatasetPath_InModelFormat, string train_XYDatasetPath_InModelFormat, string validation_XDatasetPath_InModelFormat, string validation_YDatasetPath_InModelFormat, string validation_XYDatasetPath_InModelFormat) 
        Fit(DataSet trainDataset, DataSet validationDatasetIfAny);

    /// <summary>
    /// do Model inference for dataset 'dataset' and returns the predictions
    /// </summary>
    /// <param name="dataset">teh dataset we want to make the inference</param>
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

    public abstract (DataFrame predictions, string datasetPath) PredictWithPath(DataSet dataset, bool removeAllTemporaryFilesAtEnd);
    
    public abstract void Save(string workingDirectory, string modelName);
    public virtual string DeviceName() => "";
    public virtual int TotalParams() => -1;
    public virtual int GetNumEpochs() => -1;
    public virtual double GetLearningRate() => double.NaN;

    public virtual void Use_All_Available_Cores() { }
    
    protected static void LogDebug(string message) { Log.Debug(message); }
    protected static void LogInfo(string message) { Log.Info(message); }
    protected static void LogWarn(string message) { Log.Warn(message); }
    protected static void LogError(string message) { Log.Error(message); }
}
