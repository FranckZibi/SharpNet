using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using JetBrains.Annotations;
using log4net;
using SharpNet.CPU;

namespace SharpNet.HPO;

public class WeightsOptimizer
{
    #region private fields
    // ReSharper disable once NotAccessedField.Local
    [NotNull] private readonly string _workingDirectory;
    private readonly List<CpuTensor<float>> _originalPredictions;
    private readonly Func<CpuTensor<float>, float> _objectiveFunction;
    private readonly Action<CpuTensor<float>, string, string, WeightsOptimizerHyperParameters> _savePredictions;
    private static readonly ILog Log = LogManager.GetLogger(typeof(WeightsOptimizer));
    #endregion

    public WeightsOptimizer([NotNull] string workingDirectory, 
        List<CpuTensor<float>> originalPredictions, 
        Func<CpuTensor<float>, float> objectiveFunction,
        Action<CpuTensor<float>, string, string, WeightsOptimizerHyperParameters> savePredictions)
    {
        if (!Directory.Exists(workingDirectory))
        {
            Directory.CreateDirectory(workingDirectory);
        }
        _workingDirectory = workingDirectory;
        _originalPredictions = originalPredictions;
        _objectiveFunction = objectiveFunction;
        _savePredictions = savePredictions;
        Utils.ConfigureGlobalLog4netProperties(workingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(workingDirectory, "log");
    }

    private float BestScore = float.NaN;

    public void Run()
    {
        var searchSpace = new Dictionary<string, object>();
        for (int i = 0; i < _originalPredictions.Count; ++i)
        {
            searchSpace["w_" + i.ToString("D2")] = AbstractHyperParameterSearchSpace.Range(0.0f, 1.0f);
            Log.Info($"Original score of model#{i} : {_objectiveFunction(_originalPredictions[i])}");
        }

        var sampleEqualWeights = new WeightsOptimizerHyperParameters();
        sampleEqualWeights.SetEqualWeights();
        Log.Info($"Score with same weight for all models : {TrainWithHyperParameters(sampleEqualWeights)}");

        //var hpo = new RandomSearchHPO<WeightsOptimizerHyperParameters>(searchSpace,
        //    () => new WeightsOptimizerHyperParameters(),
        //    t => t.PostBuild(),
        //    0,  //no time limit
        //    AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING,
        //    _workingDirectory
        // );

        var hpo = new BayesianSearchHPO<WeightsOptimizerHyperParameters>(searchSpace,
            () => new WeightsOptimizerHyperParameters(),
            t => t.PostBuild(),
            0,  //no time limit
            AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING,
            _workingDirectory,
            new HashSet<string>());

        hpo.Process(TrainWithHyperParameters);
    }

    private float TrainWithHyperParameters(WeightsOptimizerHyperParameters sample)
    {
        var weightedPrediction = sample.ApplyWeights(_originalPredictions);
        var validationRmse = _objectiveFunction(weightedPrediction);
        if (float.IsNaN(BestScore) || validationRmse < BestScore)
        {
            BestScore = validationRmse;
            var pathValidationPrediction = Path.Combine(_workingDirectory, "predict_valid_" + validationRmse.ToString(CultureInfo.InvariantCulture) + ".csv");
            Log.Info($"saving validation prediction to {pathValidationPrediction}");
            var pathTestPrediction = Path.Combine(_workingDirectory, "predict_test_with_valid_" + validationRmse.ToString(CultureInfo.InvariantCulture) + ".csv");
            Log.Info($"saving test prediction to {pathTestPrediction}");
            _savePredictions(weightedPrediction, pathValidationPrediction, pathTestPrediction, sample);
        }
        return validationRmse;
    }
}
