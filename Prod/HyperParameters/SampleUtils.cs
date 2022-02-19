using System;
using System.Diagnostics;
using System.IO;
using JetBrains.Annotations;
using SharpNet.Models;

namespace SharpNet.HyperParameters;

public static class SampleUtils
{
    public static float TrainWithHyperParameters([NotNull] ISample sample, [NotNull] string workingDirectory, [CanBeNull] string csvPathIfAny, ref float bestScoreSoFar)
    {
        IModelSample modelSample;
        AbstractDatasetSample datasetSample;
        if (sample is Model_and_Dataset_Sample modelAndDatasetSample)
        {
            modelSample = modelAndDatasetSample.ModelSample;
            datasetSample = modelAndDatasetSample.DatasetSample;
        }
        else
        {
            throw new ArgumentException($"invalid sample {sample.GetType()}");
        }

        var sw = Stopwatch.StartNew();
        var model = AbstractModel.NewModel(modelSample, workingDirectory, sample.ComputeHash());
        var (trainPredictionsPath, trainScore, validationPredictionsPath, validationScore, testPredictionsPath) = datasetSample.Fit(model, true);

        if (float.IsNaN(bestScoreSoFar) || model.NewScoreIsBetterTheReferenceScore(validationScore, bestScoreSoFar))
        {
            AbstractModel.Log.Debug($"Model '{model.ModelName}' has new best score: {validationScore} (was: {bestScoreSoFar})");
            bestScoreSoFar = validationScore;
            sample.Save(model.WorkingDirectory, model.ModelName);
        }
        else
        {
            AbstractModel.Log.Debug($"Removing all model '{model.ModelName}' files because of low score ({validationScore})");
            model.AllFiles().ForEach(File.Delete);
            File.Delete(trainPredictionsPath);
            File.Delete(validationPredictionsPath);
            File.Delete(testPredictionsPath);
        }

        if (!string.IsNullOrEmpty(csvPathIfAny))
        {
            var trainingTimeInSeconds = sw.Elapsed.TotalSeconds;
            model.AddResumeToCsv(trainingTimeInSeconds, trainScore, validationScore, csvPathIfAny);
        }

        return validationScore;
    }


}