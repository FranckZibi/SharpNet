using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.Datasets.AmazonEmployeeAccessChallenge;
using SharpNet.Datasets.Natixis70;
using SharpNet.Models;

namespace SharpNet.HyperParameters;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
[SuppressMessage("ReSharper", "MemberCanBeProtected.Global")]
public abstract class AbstractDatasetSample : AbstractSample
{
    #region private fields
    private static readonly ConcurrentDictionary<string, CpuTensor<float>> LoadPredictionsWithoutIndex_Cache = new();
    #endregion


    #region constructors
    protected AbstractDatasetSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(mandatoryCategoricalHyperParameters)
    {
    }
    public static AbstractDatasetSample ValueOf(string workingDirectory, string modelName)
    {
        try { return Natixis70DatasetSample.ValueOfNatixis70DatasetSample(workingDirectory, modelName); } catch { }
        try { return AmazonEmployeeAccessChallengeDatasetSample.ValueOfAmazonEmployeeAccessChallengeDatasetHyperParameters(workingDirectory, modelName); } catch { }
        throw new ArgumentException($"can't load {nameof(AbstractDatasetSample)} for model {modelName} from {workingDirectory}");
    }
    #endregion

    #region Hyper-Parameters
    public string Train_XDatasetPath = "";
    public string Train_YDatasetPath = "";
    public string Train_PredictionsPath = "";
    public string Validation_XDatasetPath = "";
    public string Validation_YDatasetPath = "";
    public string Validation_PredictionsPath = "";
    public string Test_DatasetPath = "";
    public string Test_PredictionsPath = "";
    #endregion

    public virtual (string train_PredictionsPath, float trainScore, string validation_PredictionsPath, float validationScore, string test_PredictionsPath) Fit(AbstractModel model, bool computeAndSavePredictions, bool computeValidationScore, bool saveTrainedModel)
    {
        using var trainingAndValidation = SplitIntoTrainingAndValidation();
        (Train_XDatasetPath, Train_YDatasetPath, Validation_XDatasetPath, Validation_YDatasetPath) = model.Fit(trainingAndValidation.Training, trainingAndValidation.Test);
        var res = ("", float.NaN, "", float.NaN, "");
        if (computeAndSavePredictions)
        {
            res =  ComputeAndSavePredictions(model, trainingAndValidation.Training, trainingAndValidation.Test);
        }
        else if (computeValidationScore)
        {
            var validationScore = ComputeScoreAndPredictions(model, trainingAndValidation.Test).score;
            res = ("", float.NaN, "", validationScore, "");
        }
        if (saveTrainedModel)
        {
            model.Save(model.WorkingDirectory, model.ModelName);
        }
        return res;
    }
    /// <param name="model"></param>
    /// <returns>the cost associated with the model</returns>
    public virtual (string train_PredictionsPath, float trainScore, string validation_PredictionsPath, float validationScore, string test_PredictionsPath) ComputeAndSavePredictions(AbstractModel model)
    {
        using var trainAndValidation = SplitIntoTrainingAndValidation();
        return ComputeAndSavePredictions(model, trainAndValidation.Training, trainAndValidation.Test);
    }
    public abstract List<string> CategoricalFeatures();
    public abstract IDataSet FullTraining();
    public abstract CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(string dataframe_path);
    public abstract (CpuTensor<float> trainPredictions, CpuTensor<float> validationPredictions, CpuTensor<float> testPredictions) LoadAllPredictions();
    public abstract void ComputeAndSavePredictions(CpuTensor<float> predictionsInModelFormat, string path);

    protected (CpuTensor<float> trainPredictions, CpuTensor<float> validationPredictions, CpuTensor<float> testPredictions) LoadAllPredictions(bool header, bool predictionsContainIndexColumn, char separator)
    {
        return
            (LoadPredictionsWithoutIndex(Train_PredictionsPath, header, predictionsContainIndexColumn, separator),
             LoadPredictionsWithoutIndex(Validation_PredictionsPath, header, predictionsContainIndexColumn, separator),
             LoadPredictionsWithoutIndex(Test_PredictionsPath, header, predictionsContainIndexColumn, separator));

    }
    protected virtual CpuTensor<float> UnnormalizeYIfNeeded(CpuTensor<float> y)
    {
        //by default: no normalization
        return y;
    }
    protected abstract IDataSet TestDataset();
    protected abstract ITrainingAndTestDataSet SplitIntoTrainingAndValidation();
    protected void SaveTrainPredictions(AbstractModel model, CpuTensor<float> trainPredictions, float trainScore)
    {
        ISample.Log.Debug($"Saving Model '{model.ModelName}' predictions for Training Dataset (score={trainScore})");
        Train_PredictionsPath = Path.Combine(model.WorkingDirectory, model.ModelName + "_predict_train_" + Math.Round(trainScore, 5) + ".csv");
        ComputeAndSavePredictions(trainPredictions, Train_PredictionsPath);
    }
    protected void SaveValidationPredictions(AbstractModel model, CpuTensor<float> validationPredictions, float validationScore)
    {
        ISample.Log.Debug($"Saving Model '{model.ModelName}' predictions for Validation Dataset (score={validationScore})");
        Validation_PredictionsPath = Path.Combine(model.WorkingDirectory, model.ModelName + "_predict_valid_" + Math.Round(validationScore, 5) + ".csv");
        ComputeAndSavePredictions(validationPredictions, Validation_PredictionsPath);
    }
    protected void SaveTestPredictions(AbstractModel model, CpuTensor<float> testPredictions)
    {
        ISample.Log.Debug($"Saving Model '{model.ModelName}' predictions for Test Dataset");
        Test_PredictionsPath = Path.Combine(model.WorkingDirectory, model.ModelName + "_predict_test_.csv");
        ComputeAndSavePredictions(testPredictions, Test_PredictionsPath);
    }

    private static CpuTensor<float> LoadPredictionsWithoutIndex(string path, bool header, bool predictionsContainIndexColumn, char separator)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            return null;
        }

        if (LoadPredictionsWithoutIndex_Cache.TryGetValue(path, out var res))
        {
            return res;
        }

        var y_pred = Dataframe.Load(path, header, separator).Tensor;
        if (predictionsContainIndexColumn)
        {
            y_pred = y_pred.DropColumns(new[] { 0 });
        }

        LoadPredictionsWithoutIndex_Cache.TryAdd(path, y_pred);
        return y_pred;
    }
    private (float score, CpuTensor<float> predictions) ComputeScoreAndPredictions(IModel model, IDataSet dataset)
    {
        var predictions = UnnormalizeYIfNeeded(model.Predict(dataset));
        var predictionsScore = model.ComputeScore(UnnormalizeYIfNeeded(dataset.Y), predictions);
        return (predictionsScore, predictions);
    }
    /// <param name="model"></param>
    /// <param name="trainDataset"></param>
    /// <param name="validationDatasetIfAny"></param>
    /// <returns>the cost associated with the model</returns>
    private (string train_PredictionsPath, float trainScore, string  validation_PredictionsPath, float validationScore, string test_PredictionsPath) ComputeAndSavePredictions(AbstractModel model,  IDataSet trainDataset, IDataSet validationDatasetIfAny)
    {
        Train_PredictionsPath = "";
        Validation_PredictionsPath = "";
        Test_PredictionsPath = "";

        ISample.Log.Debug($"Computing Model '{model.ModelName}' predictions and score for Training Dataset");
        var (trainScore, trainPredictions) = ComputeScoreAndPredictions(model, trainDataset);
        ISample.Log.Info($"Model '{model.ModelName}' score on training: {trainScore}");

        SaveTrainPredictions(model, trainPredictions, trainScore);

        float validationScore = float.NaN;
        if (validationDatasetIfAny != null)
        {
            ISample.Log.Debug($"Computing Model '{model.ModelName}' predictions and score for Validation Dataset");
            (validationScore, var validationPredictions) = ComputeScoreAndPredictions(model, validationDatasetIfAny);
            ISample.Log.Info($"Model '{model.ModelName}' score on Validation: {validationScore}");
            if (!float.IsNaN(validationScore))
            {
                SaveValidationPredictions(model, validationPredictions, validationScore);
            }
        }

        var testDatasetIfAny = TestDataset();
        if (testDatasetIfAny != null)
        {
            ISample.Log.Debug($"Computing Model '{model.ModelName}' predictions for Test Dataset");
            (var pred, Test_DatasetPath) = model.PredictWithPath(testDatasetIfAny);
            var testPredictions = UnnormalizeYIfNeeded(pred);
            SaveTestPredictions(model, testPredictions);
            testDatasetIfAny.Dispose();
        }
        return (Train_PredictionsPath, trainScore, Validation_PredictionsPath, validationScore, Test_PredictionsPath);
    }
}