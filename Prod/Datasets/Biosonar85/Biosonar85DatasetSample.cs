using log4net;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpNet.Datasets.Biosonar85;

public class Biosonar85DatasetSample : AbstractDatasetSample
{
    #region private fields
    private InMemoryDataSet trainDataset;
    private InMemoryDataSet testDataset;
    #endregion
    
    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(Biosonar85DatasetSample));
    #endregion

    #region HyperParameters
    
    #endregion




    public Biosonar85DatasetSample() : base(new HashSet<string>())
    {
        Utils.ConfigureGlobalLog4netProperties(Biosonar85Utils.WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(Biosonar85Utils.WorkingDirectory, "log");
    }

    public override IScore ExtractRankingScoreFromModelMetricsIfAvailable(params IScore[] modelMetrics)
    {
        return modelMetrics.FirstOrDefault(v => v != null && v.Metric == GetRankingEvaluationMetric());
    }


    public override string[] CategoricalFeatures { get; } = { };
    public override string IdColumn => "id";
    public override string[] TargetLabels { get; } = { "pos_label" };
    public override Objective_enum GetObjective()
    {
        return Objective_enum.Classification;
    }
    //public override IScore MinimumScoreToSaveModel => new Score(0.48f, GetRankingEvaluationMetric());

    public override string[] TargetLabelDistinctValues => Biosonar85Utils.TargetLabelDistinctValues;

    public override DataSet FullTrainingAndValidation()
    {
        return LoadAndEncodeDataset_If_Needed().fullTrainingAndValidation;
    }
    public override DataSet TestDataset()
    {
        return LoadAndEncodeDataset_If_Needed().testDataset;
    }


    
    private (InMemoryDataSet fullTrainingAndValidation, InMemoryDataSet testDataset) LoadAndEncodeDataset_If_Needed()
    {
        if (trainDataset == null)
        {
            var sw = Stopwatch.StartNew();
            Log.Debug($"Starting loading raw files");
            trainDataset = Biosonar85Utils.Load("X_train_23168_101_64_1024_512.bin", "Y_train_23168_1_64_1024_512.bin", "Y_train_ofTdMHi.csv", true);
            //trainDataset = Biosonar85Utils.Load("X_train_small_1000_101_64_1024_512.bin", "Y_train_small_1000_1_64_1024_512.bin", "Y_train_small.csv", true);
            AddToDispose(trainDataset);
            testDataset = Biosonar85Utils.Load("X_test_950_101_64_1024_512.bin", "Y_test_950_1_64_1024_512.bin", "Y_random_Xwjr6aB.csv", false);
            Log.Info($"Loading of raw files took {sw.Elapsed.Seconds}s");
        }

        return (trainDataset, testDataset);
    }

    public override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        return EvaluationMetricEnum.Accuracy;
    }



}