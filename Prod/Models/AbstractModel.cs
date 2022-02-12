using System;
using System.Collections.Generic;
using System.IO;
using log4net;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

namespace SharpNet.Models;

public abstract class AbstractModel : IModel
{
    #region private & protected fields
    protected ISample Sample { get; }
    #endregion

    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(AbstractModel));
    public abstract string WorkingDirectory { get; }
    public abstract string ModelName { get; }
    #endregion

    #region constructor
    protected AbstractModel(ISample sample)
    {
        Sample = sample;
    }
    #endregion

    #region files related to the model

    private List<string> SampleFiles()
    {
        return Sample.SampleFiles(WorkingDirectory, ModelName);
    }
    public List<string> AllFiles()
    {
        var res = SampleFiles();
        res.AddRange(ModelFiles());
        return res;
    }
    protected abstract List<string> ModelFiles();
    #endregion

    public abstract void Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny);
    public abstract CpuTensor<float> Predict(IDataSet dataset);
    public abstract void Save(string workingDirectory, string modelName);

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
        var scoreTrain = ComputeScore(UnNormalizeYIfNeeded(trainDataset.Y), trainPredictions);
        Log.Debug($"Model '{ModelName}' score on training: {scoreTrain}");


        var validationPredictionsFileName = "";
        float scoreValidation = float.NaN;
        CpuTensor<float> validationPredictions = null;
        if (validationDatasetIfAny != null)
        {
            Log.Debug($"Computing Model '{ModelName}' predictions for Validation Dataset");
            validationPredictions = UnNormalizeYIfNeeded(Predict(validationDatasetIfAny));
            Log.Debug($"Computing Model '{ModelName}' score on Validation");
            scoreValidation = ComputeScore(UnNormalizeYIfNeeded(validationDatasetIfAny.Y), validationPredictions);
            Log.Debug($"Model '{ModelName}' score on Validation: {scoreValidation}");
        }

        var trainPredictionsFileName = ModelName + "_predict_train_" + Math.Round(scoreTrain, 5) + ".csv";
        Log.Info($"Saving Model '{ModelName}' predictions for Training Dataset");
        savePredictions(trainPredictions, Path.Combine(WorkingDirectory, trainPredictionsFileName));

        if (!float.IsNaN(scoreValidation))
        {
            Log.Info($"Saving Model '{ModelName}' predictions for Validation Dataset");
            validationPredictionsFileName = ModelName + "_predict_valid_" + Math.Round(scoreValidation, 5) + ".csv";
            savePredictions(validationPredictions, Path.Combine(WorkingDirectory, validationPredictionsFileName));
        }

        var testPredictionsFileName = "";
        if (testDatasetIfAny != null)
        {
            Log.Debug($"Computing Model '{ModelName}' predictions for Test Dataset");
            var testPredictions = Predict(testDatasetIfAny);
            Log.Info("Saving predictions for Test Dataset");
            testPredictionsFileName = ModelName + "_predict_test_.csv";
            savePredictions(testPredictions, Path.Combine(WorkingDirectory, testPredictionsFileName));
        }

        return (trainPredictionsFileName, scoreTrain, validationPredictionsFileName, scoreValidation, testPredictionsFileName);
    }

    public abstract float ComputeScore(CpuTensor<float> y_true, CpuTensor<float> y_pred);
}