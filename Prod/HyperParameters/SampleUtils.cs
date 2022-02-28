using System;
using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.Models;

namespace SharpNet.HyperParameters;

public static class SampleUtils
{
    public static float TrainWithHyperParameters([NotNull] ITrainableSample trainableSample, string workingDirectory, [CanBeNull] string csvPathIfAny, ref float bestScoreSoFar)
    {
        AbstractDatasetSample datasetSample;
        if (trainableSample is TrainableSample modelAndDatasetSample)
        {
            datasetSample = modelAndDatasetSample.DatasetSample;
        }
        else
        {
            throw new ArgumentException($"invalid sample {trainableSample.GetType()}");
        }

        var sw = Stopwatch.StartNew();
        var model = modelAndDatasetSample.NewUntrainedModel(workingDirectory);
        var validationScore = datasetSample.Fit(model, false, true, false).validationScore;
        var trainScore = float.NaN;
        Debug.Assert(!float.IsNaN(validationScore));

        if (float.IsNaN(bestScoreSoFar) || model.NewScoreIsBetterTheReferenceScore(validationScore, bestScoreSoFar))
        {
            AbstractModel.Log.Debug($"Model '{model.ModelName}' has new best score: {validationScore} (was: {bestScoreSoFar})");
            bestScoreSoFar = validationScore;
            trainScore = datasetSample.ComputeAndSavePredictions(model).trainScore;
            trainableSample.Save(model.WorkingDirectory, model.ModelName);
        }
        else
        {
            AbstractModel.Log.Debug($"Removing all model '{model.ModelName}' files because of low score ({validationScore})");
            model.AllFiles().ForEach(path => Utils.TryDelete(path));
        }

        if (!string.IsNullOrEmpty(csvPathIfAny))
        {
            var trainingTimeInSeconds = sw.Elapsed.TotalSeconds;
            model.AddResumeToCsv(trainingTimeInSeconds, trainScore, validationScore, csvPathIfAny);
        }

        return validationScore;
    }


}