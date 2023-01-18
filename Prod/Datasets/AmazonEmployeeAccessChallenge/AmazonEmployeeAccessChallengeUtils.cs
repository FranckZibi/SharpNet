﻿using System.Collections.Generic;
using SharpNet.CatBoost;
using SharpNet.HPO;
using SharpNet.HyperParameters;

namespace SharpNet.Datasets.AmazonEmployeeAccessChallenge;

public static class AmazonEmployeeAccessChallengeUtils
{
    public const string NAME = "AmazonEmployeeAccessChallenge";

    // ReSharper disable once UnusedMember.Global
    public static void Launch_CatBoost_HPO()
    {
        var workingDirectory = AmazonEmployeeAccessChallengeDatasetSample.WorkingDirectory;

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
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new AmazonEmployeeAccessChallengeDatasetSample()), workingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, workingDirectory, true, ref bestScoreSoFar));
    }
}
