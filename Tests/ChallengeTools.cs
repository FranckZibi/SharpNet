using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using SharpNet.HPO;
using SharpNet.HyperParameters;
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
        const string workingDirectory = @"C:\Projects\Challenges\MyChallenge\";
        const string modelName = "ABCDEF";
        SharpNet.Utils.ConfigureGlobalLog4netProperties(workingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        SharpNet.Utils.ConfigureThreadLog4netProperties(workingDirectory, $"{nameof(ComputeAndSaveFeatureImportance)}");
        var m = ModelAndDatasetPredictions.Load(workingDirectory, modelName);
        m.ComputeAndSaveFeatureImportance();
    }



    /// <summary>
    /// Stack several trained models together to compute new predictions
    /// (through a new LightGBM model that will be trained to do the stacking)
    /// </summary>
    [Test, Explicit]
    public void StackedEnsemble()
    {
        const string workingDirectory = @"C:\Projects\Challenges\MyChallenge\";
        var modelName = new[]
        {
            "316CF95B15_KFOLD",
            "B63953F624_KFOLD",
            "6034138E35_KFOLD",
            "F867A78F52_KFOLD",


            //"3580990008_KFOLD",
            //"395B343296_KFOLD",
            //"48F31E6543_KFOLD",
            //"56E668E7DB_KFOLD",
            //"66B4F3653A_KFOLD",
            //"8CF93D9FA0_KFOLD",
            //"90840F212D_KFOLD",
            //"90DAFFB8FC_KFOLD",
            //"E72AD5B74B_KFOLD"
        };
        const bool use_features_in_secondary = true;
        const int cv = 2;
        const int min_num_iterations = 100;
        const int maxAllowedSecondsForAllComputation = 0;

        var workingDirectoryAndModelNames = modelName.Select(m => Tuple.Create(workingDirectory, m, m + "_FULL")).ToList();
        var datasetSample = StackingCVClassifierDatasetSample.New(workingDirectoryAndModelNames, workingDirectory, use_features_in_secondary, cv);
        SampleUtils.LaunchLightGBMHPO(datasetSample, Path.Combine(workingDirectory, "hpo"), min_num_iterations, maxAllowedSecondsForAllComputation);
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