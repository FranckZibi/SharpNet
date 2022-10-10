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
    string WorkingDirectory { get; }
    string ModelName { get; }
    IModelSample ModelSample { get; }
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
        if (sample is WeightedModelSample weightedModelSample)
        {
            return new WeightedModel(weightedModelSample, workingDirectory, modelName);
        }
        if (sample is KFoldSample kFoldSample)
        {
            return new KFoldModel(kFoldSample, workingDirectory, modelName);
        }
        if (sample is NetworkSample networkSample)
        {
            networkSample.Config.WorkingDirectory = workingDirectory;
            networkSample.Config.ModelName = modelName;
            return new Network(networkSample, workingDirectory, modelName);
        }
        throw new ArgumentException($"cant' load model {modelName} from {workingDirectory} for sample type {sample.GetType()}");
    }
    //protected static IModel LoadTrainedAbstractModel(string workingDirectory, string modelName)
    //{
    //    //try { return ModelAndDataset.LoadAutoTrainableModel(workingDirectory, modelName); } catch { }
    //    //try { return KFoldModel.LoadTrainedKFoldModel(workingDirectory, modelName); } catch { }
    //    try { return Network.LoadTrainedNetworkModel(workingDirectory, modelName); } catch { }
    //    try { return LightGBMModel.LoadTrainedLightGBMModel(workingDirectory, modelName); } catch { }
    //    try { return CatBoostModel.LoadTrainedCatBoostModel(workingDirectory, modelName); } catch { }
    //    throw new ArgumentException($"can't load model {modelName} from {workingDirectory}");
    //}
    #endregion

    (string train_XDatasetPath, string train_YDatasetPath, string train_XYDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath, string validation_XYDatasetPath) 
        Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny);


    /// <summary>
    /// do Model inference for dataset 'dataset' and returns the predictions
    /// </summary>
    /// <param name="dataset">teh dataset we want to make the inference</param>
    /// <param name="addIdColumnsAtLeft">
    /// if true
    ///     the returned Dataframe will contain in the 1st columns the Id Columns
    /// else
    ///     the Id Column will be discarded in the prediction Dataframe</param>
    /// <param name="removeAllTemporaryFilesAtEnd">
    /// if true:
    ///     all temporary files needed by the model for inference will be deleted</param>
    /// <returns>
    /// predictions: the model inferences
    /// datasetPath:
    ///     if the model has needed to store the dataset into a file (ex: LightGBM, CatBoost)
    ///         the associated path
    ///     else
    ///         an empty string
    /// </returns>
    DataFrame Predict(IDataSet dataset, bool addIdColumnsAtLeft, bool removeAllTemporaryFilesAtEnd);
    void Save(string workingDirectory, string modelName);
    IScore ComputeLoss(CpuTensor<float> y_true, CpuTensor<float> y_pred);
    List<string> ModelFiles();
    void AddResumeToCsv(double trainingTimeInSeconds, IScore trainScore, IScore validationScore, string csvPath);
    int GetNumEpochs();
    string DeviceName();
    double GetLearningRate();
    void Use_All_Available_Cores();

    string RootDatasetPath { get; }

    public static string MetricsToString(IDictionary<EvaluationMetricEnum, double> metrics, string prefix)
    {
        return string.Join(" - ", metrics.OrderBy(x => x.Key).Select(e => prefix + e.Key + ": " + Math.Round(e.Value, 4))).ToLowerInvariant();
    }
    public static string TrainingAndValidationMetricsToString(IDictionary<EvaluationMetricEnum, double> trainingMetrics, IDictionary<EvaluationMetricEnum, double> validationMetrics)
    {
        return MetricsToString(trainingMetrics, "") + " - " + MetricsToString(validationMetrics, "val_");
    }
}