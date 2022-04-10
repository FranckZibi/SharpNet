using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.Models;

namespace SharpNet.HyperParameters;

public static class SampleUtils
{
    public static float TrainWithHyperParameters([NotNull] ModelAndDatasetSample modelAndDatasetSample, string workingDirectory, [CanBeNull] string csvPathIfAny, ref float bestScoreSoFar)
    {
        var sw = Stopwatch.StartNew();
        var modelAndDataset = ModelAndDataset.NewUntrainedModelAndDataset(modelAndDatasetSample, workingDirectory);
        var model = modelAndDataset.Model;
        var validationScore = modelAndDataset.Fit(false, true, false).validationScore;
        var trainScore = float.NaN;
        Debug.Assert(!float.IsNaN(validationScore));

        if (float.IsNaN(bestScoreSoFar) || model.NewScoreIsBetterTheReferenceScore(validationScore, bestScoreSoFar))
        {
            IModel.Log.Debug($"Model '{model.ModelName}' has new best score: {validationScore} (was: {bestScoreSoFar})");
            bestScoreSoFar = validationScore;
            trainScore = modelAndDataset.ComputeAndSavePredictionsInTargetFormat().trainScore;
            modelAndDataset.Save(workingDirectory, model.ModelName);
        }
        else
        {
            IModel.Log.Debug($"Removing all model '{model.ModelName}' files because of low score ({validationScore})");
            modelAndDataset.AllFiles().ForEach(path => Utils.TryDelete(path));
        }

        if (!string.IsNullOrEmpty(csvPathIfAny))
        {
            var trainingTimeInSeconds = sw.Elapsed.TotalSeconds;
            model.AddResumeToCsv(trainingTimeInSeconds, trainScore, validationScore, csvPathIfAny);
        }

        return validationScore;
    }


}