using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using JetBrains.Annotations;
using log4net;
using SharpNet.CPU;
using SharpNet.Models;

namespace SharpNet.HPO;

public class WeightsOptimizer
{
    #region private fields
    // ReSharper disable once NotAccessedField.Local
    [NotNull] private readonly string _workingDirectory;
    private readonly List<CpuTensor<float>> _originalTrainPredictionsIfAny;
    private readonly List<CpuTensor<float>> _originalValidationPredictions;
    private readonly List<CpuTensor<float>> _originalTestPredictionsIfAny;
    private readonly List<TrainedModel> _trainedModels;
    [NotNull] private readonly CpuTensor<float> _perfect_validation_predictions;
    [CanBeNull] private readonly CpuTensor<float> _perfect_train_predictions_if_any;
    private static readonly ILog Log = LogManager.GetLogger(typeof(WeightsOptimizer));
    #endregion

    public WeightsOptimizer([NotNull] string workingDirectory, List<TrainedModel> trainedModels)
    {
        if (!Directory.Exists(workingDirectory))
        {
            Directory.CreateDirectory(workingDirectory);
        }
        Utils.ConfigureGlobalLog4netProperties(workingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(workingDirectory, "log");

        _workingDirectory = workingDirectory;
        _trainedModels = trainedModels;

        _perfect_validation_predictions = trainedModels[0].Perfect_Validation_Predictions_if_any();
        Debug.Assert(_perfect_validation_predictions != null);
        _perfect_train_predictions_if_any = trainedModels[0].Perfect_Train_Predictions_if_any();

        //we load the validation predictions done by the source models
        _originalValidationPredictions = trainedModels.Select(m => m.Predictions.GetValidationPredictions()).Where(t => t != null).ToList();
        Debug.Assert(_originalValidationPredictions.All(t => t != null));
        Debug.Assert(SameShape(_originalValidationPredictions));

        //we load the train predictions done by the source models (if any)
        _originalTrainPredictionsIfAny = trainedModels.Select(m => m.Predictions.GetTrainPredictions()).Where(t => t != null).ToList();
        Debug.Assert(_originalTrainPredictionsIfAny.Count == 0 || _originalTrainPredictionsIfAny.Count == _originalValidationPredictions.Count);
        Debug.Assert(SameShape(_originalTrainPredictionsIfAny));

        //we load the test predictions done by the source models (if any)
        _originalTestPredictionsIfAny = trainedModels.Select(m=>m.Predictions.GetTestPredictions()).Where(t => t != null).ToList();
        Debug.Assert(_originalTestPredictionsIfAny.Count == 0 || _originalTestPredictionsIfAny.Count == _originalValidationPredictions.Count);
        Debug.Assert(SameShape(_originalTestPredictionsIfAny));
    }

    private static bool SameShape(IList<CpuTensor<float>> tensors)
    {
        return tensors.All(t => t.SameShape(tensors[0]));
    }

    private float BestScore = float.NaN;

    public void Run()
    {
        var searchSpace = new Dictionary<string, object>();
        for (int i = 0; i < _originalTrainPredictionsIfAny.Count; ++i)
        {
            searchSpace["w_" + i.ToString("D2")] = AbstractHyperParameterSearchSpace.Range(0.0f, 1.0f);
            Log.Info($"Original validation score of model#{i} : {_trainedModels[0].ComputeLoss(_perfect_validation_predictions, _originalValidationPredictions[i])}");
            if (_originalTrainPredictionsIfAny[i] != null)
            {
                Log.Info($"Original train score of model#{i} : {_trainedModels[0].ComputeLoss(_perfect_train_predictions_if_any, _originalTrainPredictionsIfAny[i])}");
            }
        }

        var sampleEqualWeights = new WeightsOptimizerHyperParameters();
        sampleEqualWeights.SetEqualWeights();
        Log.Info($"Score with same weight for all models : {TrainWithHyperParameters(sampleEqualWeights)}");

        //var hpo = new RandomSearchHPO(searchSpace,  () => new WeightsOptimizerHyperParameters(), _workingDirectory);
        var hpo = new BayesianSearchHPO(searchSpace,  () => new WeightsOptimizerHyperParameters(), _workingDirectory);
        hpo.Process(t => TrainWithHyperParameters((WeightsOptimizerHyperParameters)t));
    }


    private void SaveModelDescription(string path)
    {
        var sb = new StringBuilder();
        sb.Append("WorkingDirectory" + WeightsOptimizerHyperParameters.separator+ "ModelName" + Environment.NewLine);
        foreach (var m in _trainedModels)
        {
            sb.Append(m.WorkingDirectory+ WeightsOptimizerHyperParameters.separator + m.ModelName + Environment.NewLine);
        }
        File.WriteAllText(path, sb.ToString());
    }
   
    private float TrainWithHyperParameters(WeightsOptimizerHyperParameters sample)
    {
        var weightedValidationPrediction = sample.ApplyWeights(_originalValidationPredictions);
        Debug.Assert(_perfect_validation_predictions.SameShape(weightedValidationPrediction));
        float validationRmse = _trainedModels[0].ComputeLoss(_perfect_validation_predictions, weightedValidationPrediction);

        if (float.IsNaN(BestScore) || validationRmse < BestScore)
        {
            var sampleHash = sample.ComputeHash();
            BestScore = validationRmse;
            
            SaveModelDescription(Path.Combine(_workingDirectory, sampleHash + ".txt"));
            sample.Save(_workingDirectory, sampleHash);

            var pathValidationPrediction = Path.Combine(_workingDirectory, sampleHash+"_predict_valid_" + validationRmse.ToString(CultureInfo.InvariantCulture) + ".csv");
            Log.Info($"saving validation prediction to {pathValidationPrediction}");
            _trainedModels[0].Predictions.SavePredictions(weightedValidationPrediction, pathValidationPrediction);

            if (_originalTrainPredictionsIfAny.Count != 0)
            {
                var weightedTrainPrediction = sample.ApplyWeights(_originalTrainPredictionsIfAny);
                float trainRmse = float.NaN;
                if (_perfect_train_predictions_if_any != null)
                {
                    trainRmse = _trainedModels[0].ComputeLoss(_perfect_train_predictions_if_any, weightedTrainPrediction);
                }
                var pathTrainPrediction = Path.Combine(_workingDirectory, sampleHash + "_predict_train_" 
                    + (float.IsNaN(trainRmse)?"": trainRmse.ToString(CultureInfo.InvariantCulture) )
                    + ".csv");
                _trainedModels[0].Predictions.SavePredictions(weightedTrainPrediction, pathTrainPrediction);
            }

            if (_originalTestPredictionsIfAny.Count != 0)
            {
                var weightedTestPrediction = sample.ApplyWeights(_originalTestPredictionsIfAny);
                var pathTestPrediction = Path.Combine(_workingDirectory, sampleHash + "_predict_test_.csv");
                _trainedModels[0].Predictions.SavePredictions(weightedTestPrediction, pathTestPrediction);
            }
        }
        return validationRmse;
    }
}
