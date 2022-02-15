using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using SharpNet.CatBoost;
using SharpNet.HPO;
using SharpNet.Models;

namespace SharpNet.Datasets.AmazonEmployeeAccessChallenge;

public static class AmazonEmployeeAccessChallengeUtils
{

    private static readonly object LockUpdateFileObject = new();



    public static void Launch_CatBoost_HPO()
    {
        //var workingDirectory007 = AmazonEmployeeAccessChallengeDatasetHyperParameters.WorkingDirectory;
        //var trainingDataset007 = BayesianSearchHPO.LoadSurrogateTrainingDataset(@"C:\Users\fzibi\AppData\Local\SharpNet\AmazonEmployeeAccessChallenge\surrogate_22676_dataset.csv");
        //var validationDataset007 = BayesianSearchHPO.LoadSurrogateValidationDataset(@"C:\Users\fzibi\AppData\Local\SharpNet\AmazonEmployeeAccessChallenge\surrogate_22676_predictions.csv");
        //var model007 = BayesianSearchHPO.BuildCatBoostSurrogateModel(workingDirectory007, "surrogate_22676");
        ////var model007 = BayesianSearchHPO.BuildRandomForestSurrogateModel(workingDirectory007, "surrogate_22676", Array.Empty<string>());
        //model007.Fit(trainingDataset007, null);
        //var res007 = model007.Predict(validationDataset007);
        //return;

        var workingDirectory = AmazonEmployeeAccessChallengeDatasetHyperParameters.WorkingDirectory;
        Utils.ConfigureGlobalLog4netProperties(workingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(workingDirectory, "log");

        // ReSharper disable once ConvertToConstant.Local
        var iterations = 1000;
        var searchSpace = new Dictionary<string, object>
        {
            { "iterations", new[] { iterations } },
            { "allow_writing_files",false},
            { "eval_metric","Logloss"},
            { "loss_function","Logloss"},
            { "od_type","Iter"},
            { "od_wait",iterations/10},
            { "logging_level", "Silent"},


            { "depth", AbstractHyperParameterSearchSpace.Range(2, 10) },
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 1.00f)},
            { "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
            { "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
            { "l2_leaf_reg",AbstractHyperParameterSearchSpace.Range(0, 10)},
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => new AmazonEmployeeAccessChallenge_CatBoost_HyperParameters(), workingDirectory);
        float minLossSoFar = float.NaN;
        hpo.Process(t => TrainWithHyperParameters((AmazonEmployeeAccessChallenge_CatBoost_HyperParameters)t, ref minLossSoFar));
    }


    private static float TrainWithHyperParameters(AmazonEmployeeAccessChallenge_CatBoost_HyperParameters sample, ref float minLossSoFar)
    {
        var sw = Stopwatch.StartNew();
        var datasetSample = sample.DatasetHyperParameters;
        var workingDirectory = AmazonEmployeeAccessChallengeDatasetHyperParameters.WorkingDirectory;
        var model = new CatBoostModel(sample.CatBoostSample, workingDirectory, sample.ComputeHash());

        var trainAndValidation = sample.DatasetHyperParameters.FullTrain.SplitIntoTrainingAndValidation(0.75);


        model.Fit(trainAndValidation.Training, trainAndValidation.Test);

        var (trainPredictionsFileName, trainScore, validationPredictionsFileName, validationLoss, testPredictionsFileName) = model.ComputePredictions(
            trainAndValidation.Training,
            trainAndValidation.Test,
            sample.DatasetHyperParameters.Test,
            datasetSample.SavePredictions,
            null);

        if (float.IsNaN(minLossSoFar) || validationLoss < minLossSoFar)
        {
            AbstractModel.Log.Debug($"Model '{model.ModelName}' has new lowest score: {validationLoss} (was: {minLossSoFar})");
            minLossSoFar = validationLoss;
        }
        else
        {
            AbstractModel.Log.Debug($"Removing model '{model.ModelName}' files: {Path.GetFileName(model.ModelPath)} and {Path.GetFileName(model.ModelConfigPath)} because of low score ({validationLoss})");
            model.AllFiles().ForEach(File.Delete);
            File.Delete(trainPredictionsFileName);
            File.Delete(validationPredictionsFileName);
            File.Delete(testPredictionsFileName);
        }

        string line = "";
        try
        {
            var trainDataset = trainAndValidation.Training;
            var trainingTimeInSeconds = sw.Elapsed.TotalSeconds;
            var totalParams = -666;
            int numEpochs = model.CatBoostSample.iterations;
            //We save the results of the net
            line = DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";"
                                                                            + model.ModelName.Replace(';', '_') + ";"
                                                                            + model.CatBoostSample.DeviceName() + ";"
                                                                            + totalParams + ";"
                                                                            + numEpochs + ";"
                                                                            + "-1" + ";"
                                                                            + model.CatBoostSample.learning_rate + ";"
                                                                            + trainingTimeInSeconds + ";"
                                                                            + (trainingTimeInSeconds / numEpochs) + ";"
                                                                            + trainScore + ";"
                                                                            + "NaN" + ";"
                                                                            + validationLoss + ";"
                                                                            + "NaN" + ";"
                                                                            + Environment.NewLine;
            var testsCsv = string.IsNullOrEmpty(trainDataset.Name) ? "Tests.csv" : ("Tests_" + trainDataset.Name + ".csv");
            lock (LockUpdateFileObject)
            {
                File.AppendAllText(Utils.ConcatenatePathWithFileName(workingDirectory, testsCsv), line);
            }
        }
        catch (Exception e)
        {
            AbstractModel.Log.Error("fail to add line in file:" + Environment.NewLine + line + Environment.NewLine + e);
        }

        return validationLoss;
    }
}