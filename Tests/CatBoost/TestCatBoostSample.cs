using System.IO;
using NUnit.Framework;
using SharpNet;
using SharpNet.CatBoost;
using SharpNet.Datasets.AmazonEmployeeAccessChallenge;

namespace SharpNetTests.CatBoost
{
    [TestFixture]
    public class TestCatBoostSample
    {
        [Test, Explicit] 
        public void TestToJson()
        {
            var workingDirectory = AmazonEmployeeAccessChallengeDatasetHyperParameters.WorkingDirectory;
            if (!Directory.Exists(workingDirectory))
            {
                Directory.CreateDirectory(workingDirectory);
            }
            Utils.ConfigureGlobalLog4netProperties(workingDirectory, "log");
            Utils.ConfigureThreadLog4netProperties(workingDirectory, "log");

            const int num_boost_round = 100;
            var sample = new AmazonEmployeeAccessChallenge_CatBoost_HyperParameters();
            sample.CatBoostSample.iterations = num_boost_round;
            sample.CatBoostSample.loss_function = CatBoostSample.loss_function_enum.Logloss;
            sample.CatBoostSample.verbose = num_boost_round / 10;
            sample.CatBoostSample.eval_metric = CatBoostSample.metric_enum.AUC;
            sample.CatBoostSample.random_seed = 1;
            sample.CatBoostSample.allow_writing_files = false;
            //sample.CatBoostSample.set_early_stopping_rounds(200);
            var model = new CatBoostModel(sample.CatBoostSample, workingDirectory, sample.ComputeHash());

            var splitIntoTrainingAndValidation = sample.DatasetHyperParameters.FullTrain.SplitIntoTrainingAndValidation(0.75);
            model.Fit(splitIntoTrainingAndValidation.Training, splitIntoTrainingAndValidation.Test);

            var predictions = model.Predict(splitIntoTrainingAndValidation.Test);
            Assert.IsNotNull(predictions);
            model.ComputePredictions(splitIntoTrainingAndValidation.Training, splitIntoTrainingAndValidation.Test,
                sample.DatasetHyperParameters.Test,
                sample.DatasetHyperParameters.SavePredictions,
                null);


        }
    }
}
