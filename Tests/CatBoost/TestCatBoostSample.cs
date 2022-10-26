using NUnit.Framework;

namespace SharpNetTests.CatBoost
{
    [TestFixture]
    public class TestCatBoostSample
    {
        [Test, Explicit] 
        public void TestToJson()
        {
            //var workingDirectory = AmazonEmployeeAccessChallengeDatasetSample.WorkingDirectory;
            //if (!Directory.Exists(workingDirectory))
            //{
            //    Directory.CreateDirectory(workingDirectory);
            //}
           // Utils.ConfigureGlobalLog4netProperties(workingDirectory, "log");
           // Utils.ConfigureThreadLog4netProperties(workingDirectory, "log");

           // const int num_boost_round = 100;
           // var catBoostSample = new CatBoostSample();
           //var datasetHyperParameters = new AmazonEmployeeAccessChallengeDatasetHyperParameters();
           // catBoostSample.iterations = num_boost_round;
           // catBoostSample.loss_function = CatBoostSample.loss_function_enum.Logloss;
           // catBoostSample.verbose = num_boost_round / 10;
           // catBoostSample.eval_metric = CatBoostSample.metric_enum.AUC;
           // catBoostSample.random_seed = 1;
           // catBoostSample.allow_writing_files = false;
           // //sample.CatBoostSample.set_early_stopping_rounds(200);
           // var model = new CatBoostModel(catBoostSample, workingDirectory, catBoostSample.ComputeHash());
           // datasetHyperParameters.Fit(model, true);
        }
    }
}
