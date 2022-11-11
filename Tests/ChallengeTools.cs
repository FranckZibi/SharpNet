using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using SharpNet;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Models;

namespace SharpNetTests;

[TestFixture]
public class ChallengeTools
{
    /// <summary>
    ///  compute feature importance of a Model
    /// </summary>
    [Test, Explicit]
    public void ComputeAndSaveFeatureImportance()
    {
        const string workingDirectory = @"C:\Projects\Challenges\WasYouStayWorthItsPrice\sub2\";
        const string modelName = "D875D0F56C_KFOLD_FULL";
        SharpNet.Utils.ConfigureGlobalLog4netProperties(workingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        SharpNet.Utils.ConfigureThreadLog4netProperties(workingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        var m = ModelAndDatasetPredictions.Load(workingDirectory, modelName);
        m.ComputeAndSaveFeatureImportance();
    }



    /// <summary>
    /// Stack several trained models together to compute new predictions
    /// (through a new LightGBM model that will be trained to do the stacking)
    /// </summary>
    [TestCase(100,0), Explicit]
    public void StackedEnsemble(int num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        const string workingDirectory = @"C:/Projects/Challenges/WasYouStayWorthItsPrice/submission";
        var modelName = new[]
        {
            "F2CC39BB32_KFOLD", //1D-CNN    F2CC39BB32_KFOLD_FULL_predict_test_0.31976.csv
            "D875D0F56C_KFOLD", //LightGBM  D875D0F56C_KFOLD_FULL_predict_test_0.3204.csv
            "3F2CB236DB_KFOLD", //CatBoost  3F2CB236DB_KFOLD_FULL_predict_test_0.31759.csv

            "0EF01A90D8_KFOLD", //Deep Learning
            "3580990008_KFOLD",
            "395B343296_KFOLD", //LightGBM GBDT
            //"48F31E6543_KFOLD",
            //"56E668E7DB_KFOLD",
            "66B4F3653A_KFOLD",
            "8CF93D9FA0_KFOLD",
            "90840F212D_KFOLD",
            //"90DAFFB8FC_KFOLD",
            "E72AD5B74B_KFOLD"
        };
        const bool use_features_in_secondary = true;
        const int cv = 2;

        var workingDirectoryAndModelNames = modelName.Select(m => Tuple.Create(workingDirectory, m, m + "_FULL")).ToList();
        var datasetSample = StackingCVClassifierDatasetSample.New(workingDirectoryAndModelNames, workingDirectory, use_features_in_secondary, cv);

        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            { "KFold", cv },



            { "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
            { "bagging_freq", new[]{0, 1} },
            { "boosting", new []{"gbdt", "dart"}},
            { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.3f, 1.0f)},
            //{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)}, //very low priority
            { "drop_rate", new[]{0.05, 0.1, 0.2}},                               //specific to dart mode
            { "early_stopping_round", num_iterations/10 },
            { "extra_trees", new[] { true , false } }, //low priority 
            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.2f)},
            //{ "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.03f)},
            { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            { "max_depth", new[]{10, 20, 50, 100, 255} },
            { "max_drop", new[]{40, 50, 60}},                                   //specific to dart mode
            { "min_data_in_bin", new[]{3, 10, 100, 150}  },
            { "min_data_in_leaf", AbstractHyperParameterSearchSpace.Range(10, 200) },
            { "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },
            { "num_iterations", num_iterations },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 60) },
            { "num_threads", 1},
            { "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) }, //low priority
            { "skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},  //specific to dart mode
            { "verbosity", "0" },
        };

        if (!Directory.Exists(workingDirectory))
        {
            Directory.CreateDirectory(workingDirectory);
        }
        var dataDirectory = Path.Combine(workingDirectory, "Data");
        if (!Directory.Exists(dataDirectory))
        {
            Directory.CreateDirectory(dataDirectory);
        }

        var hpoWorkingDirectory = Path.Combine(workingDirectory, "hpo");
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), datasetSample), hpoWorkingDirectory); IScore bestScoreSoFar = null;
        //var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), datasetSample), hpoWorkingDirectory); IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, hpoWorkingDirectory, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    /// <summary>
    /// retrain some models 
    /// </summary>
    [Test, Explicit]
    public void Retrain()
    {

        foreach (var modelName in new[]
                {
                    "3580990008",
                    "395B343296",
                    "48F31E6543",
                    "56E668E7DB",
                    "66B4F3653A",
                    "8CF93D9FA0",
                    "90840F212D",
                    "90DAFFB8FC",
                    "E72AD5B74B"
                })
        {

            const string workingDirectory = @"C:\Projects\Challenges\WasYouStayWorthItsPrice\submission3";

            SharpNet.Utils.ConfigureGlobalLog4netProperties(workingDirectory, $"{nameof(Retrain)}");
            SharpNet.Utils.ConfigureThreadLog4netProperties(workingDirectory, $"{nameof(Retrain)}");

            var m = ModelAndDatasetPredictions.Load(workingDirectory, modelName);
            //var embeddedModel = mKfold.Model;
            var mKfoldModelAndDatasetPredictionsSample = m.ModelAndDatasetPredictionsSample;
            if (m.Model is not KFoldModel)
            {
                var embeddedModel = m.Model;
                var kfoldSample = new KFoldSample(3, workingDirectory, embeddedModel.ModelSample.GetLoss(), mKfoldModelAndDatasetPredictionsSample.DatasetSample.DatasetRowsInModelFormatMustBeMultipleOf());
                var sample = new ModelAndDatasetPredictionsSample(new ISample[]
                {
                    kfoldSample,
                    mKfoldModelAndDatasetPredictionsSample.DatasetSample.CopyWithNewPercentageInTrainingAndKFold(1.0, kfoldSample.n_splits),
                    mKfoldModelAndDatasetPredictionsSample.PredictionsSample
                });
                m = new ModelAndDatasetPredictions(sample, workingDirectory, embeddedModel.ModelName + KFoldModel.SuffixKfoldModel);
            }

            m.Model.Use_All_Available_Cores();
            m.Fit(true, true, true);

            var kfoldModelName = m.Model.ModelName;
            m.Save(workingDirectory, kfoldModelName);
            var modelAndDatasetPredictionsSampleOnFullDataset = mKfoldModelAndDatasetPredictionsSample.CopyWithNewPercentageInTrainingAndKFold(1.0, 1);
            var modelAndDatasetOnFullDataset = new ModelAndDatasetPredictions(modelAndDatasetPredictionsSampleOnFullDataset, workingDirectory, kfoldModelName + "_FULL");
            Model.Log.Info($"Retraining Model '{kfoldModelName}' on full Dataset no KFold (Model on full Dataset name: {modelAndDatasetOnFullDataset.Model.ModelName})");
            modelAndDatasetOnFullDataset.Model.Use_All_Available_Cores();
            modelAndDatasetOnFullDataset.Fit(true, true, true);
        }
    }
}