using System;
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
public abstract class AbstractDatasetSample : AbstractSample
{
    protected AbstractDatasetSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(
        mandatoryCategoricalHyperParameters)
    {
    }

    public abstract List<string> CategoricalFeatures();
    public abstract IDataSet FullTraining();
    public abstract CpuTensor<float> ModelPrediction_2_TargetPredictionFormat(string dataframe_path);

    public (string, float, string, float, string) Fit(IModel model, bool computePredictions)
    {
        using var trainingAndValidation = SplitIntoTrainingAndValidation();
        model.Fit(trainingAndValidation.Training, trainingAndValidation.Test);
        if (computePredictions)
        {
            return ComputePredictions(model, trainingAndValidation.Training, trainingAndValidation.Test);
        }
        return ("", float.NaN, "", float.NaN, "");
    }
    /// <param name="model"></param>
    /// <returns>the cost associated with the model</returns>
    public (string, float, string, float, string) ComputePredictions(IModel model)
    {
        using var trainAndValidation = SplitIntoTrainingAndValidation();
        return ComputePredictions(model, trainAndValidation.Training, trainAndValidation.Test);
    }

    protected virtual CpuTensor<float> UnnormalizeYIfNeeded(CpuTensor<float> y)
    {
        //by default: no normalization
        return y;
    }

    protected abstract void SavePredictions(CpuTensor<float> y_lightGBM, string path);
    protected abstract IDataSet TestDataset();
    protected abstract ITrainingAndTestDataSet SplitIntoTrainingAndValidation();

    public abstract ModelDatasets ToModelDatasets();

    /// <param name="model"></param>
    /// <param name="trainDataset"></param>
    /// <param name="validationDatasetIfAny"></param>
    /// <returns>the cost associated with the model</returns>
    private (string, float, string, float, string) ComputePredictions(
        IModel model,
        IDataSet trainDataset,
        IDataSet validationDatasetIfAny
        )
    {
        ISample.Log.Debug($"Computing Model '{model.ModelName}' predictions for Training Dataset");
        var trainPredictions = UnnormalizeYIfNeeded(model.Predict(trainDataset));
        ISample.Log.Debug("Computing Model score on Training");
        var trainScore = model.ComputeScore(UnnormalizeYIfNeeded(trainDataset.Y), trainPredictions);
        ISample.Log.Info($"Model '{model.ModelName}' score on training: {trainScore}");
        var trainPredictionsPath = Path.Combine(model.WorkingDirectory, model.ModelName + "_predict_train_" + Math.Round(trainScore, 5) + ".csv");
        ISample.Log.Info($"Saving Model '{model.ModelName}' predictions for Training Dataset");
        SavePredictions(trainPredictions, trainPredictionsPath);

        var validationPredictionsPath = "";
        float validationScore = float.NaN;
        if (validationDatasetIfAny != null)
        {
            ISample.Log.Debug($"Computing Model '{model.ModelName}' predictions for Validation Dataset");
            var validationPredictions = UnnormalizeYIfNeeded(model.Predict(validationDatasetIfAny));
            ISample.Log.Debug($"Computing Model '{model.ModelName}' score on Validation");
            validationScore = model.ComputeScore(UnnormalizeYIfNeeded(validationDatasetIfAny.Y), validationPredictions);
            ISample.Log.Info($"Model '{model.ModelName}' score on Validation: {validationScore}");
            if (!float.IsNaN(validationScore))
            {
                ISample.Log.Info($"Saving Model '{model.ModelName}' predictions for Validation Dataset");
                validationPredictionsPath = Path.Combine(model.WorkingDirectory, model.ModelName + "_predict_valid_" + Math.Round(validationScore, 5) + ".csv");
                SavePredictions(validationPredictions, validationPredictionsPath);
            }
        }

        var testPredictionsPath = "";
        var testDatasetIfAny = TestDataset();
        if (testDatasetIfAny != null)
        {
            ISample.Log.Debug($"Computing Model '{model.ModelName}' predictions for Test Dataset");
            var testPredictions = UnnormalizeYIfNeeded(model.Predict(testDatasetIfAny));
            ISample.Log.Info("Saving predictions for Test Dataset");
            testPredictionsPath = Path.Combine(model.WorkingDirectory, model.ModelName + "_predict_test_.csv");
            SavePredictions(testPredictions, testPredictionsPath);
            testDatasetIfAny.Dispose();
        }

        return (trainPredictionsPath, trainScore, validationPredictionsPath, validationScore, testPredictionsPath);
    }

    public static AbstractDatasetSample ValueOf(string workingDirectory, string modelName)
    {
        try { return Natixis70DatasetSample.ValueOfNatixis70DatasetSample(workingDirectory, modelName); } catch { }
        try { return AmazonEmployeeAccessChallengeDatasetHyperParameters.ValueOfAmazonEmployeeAccessChallengeDatasetHyperParameters(workingDirectory, modelName); } catch { }
        throw new ArgumentException($"can't load {nameof(AbstractDatasetSample)} for model {modelName} from {workingDirectory}");
    }
}