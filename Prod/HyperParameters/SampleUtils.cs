using System.Diagnostics;
using JetBrains.Annotations;
using SharpNet.Models;

namespace SharpNet.HyperParameters;

public static class SampleUtils
{
    /// <summary>
    /// Train the model (in modelAndDatasetSample) using the dataset (in modelAndDatasetSample)
    /// If validation score of the trained model is better then 'bestScoreSoFar'
    ///     update 'bestScoreSoFar'
    ///     save the model (and associated predictions) in disk
    /// Else
    ///     remove all files associated with the model
    /// If 'resumeCsvPathIfAny' is not null
    ///     store train statistics of the model in CSV file 'resumeCsvPathIfAny'
    /// </summary>
    /// <param name="modelAndDatasetSample">the 'model sample' and dataset to use for training the model</param>
    /// <param name="workingDirectory">the directory where the 'model sample' and 'dataset description' is located</param>
    /// <param name="resumeCsvPathIfAny">(optional) the CSV file where to store statistics about the trained model</param>
    /// <param name="bestScoreSoFar">the best score associated with the best sample found so far for the model</param>
    /// <returns></returns>
    public static float TrainWithHyperParameters([NotNull] ModelAndDatasetSample modelAndDatasetSample, string workingDirectory, [CanBeNull] string resumeCsvPathIfAny, ref float bestScoreSoFar)
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

        if (!string.IsNullOrEmpty(resumeCsvPathIfAny))
        {
            var trainingTimeInSeconds = sw.Elapsed.TotalSeconds;
            model.AddResumeToCsv(trainingTimeInSeconds, trainScore, validationScore, resumeCsvPathIfAny);
        }
        return validationScore;
    }
}
