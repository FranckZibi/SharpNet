using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;

namespace SharpNet.Datasets.AmazonEmployeeAccessChallenge;

public class AmazonEmployeeAccessChallengeDatasetSample : AbstractDatasetSample
{
    #region private fields
    private readonly InMemoryDataSet _testDataset;
    private InMemoryDataSet FullTrain { get; }
    //private const string PredictionHeader = "Id,Action";
    #endregion

    [SuppressMessage("ReSharper", "VirtualMemberCallInConstructor")]
    public AmazonEmployeeAccessChallengeDatasetSample() : base(new HashSet<string>())
    {
        var train = DataFrame.read_float_csv(TrainRawFile);
        var x_dataframe = train.Drop(TargetLabels);
        var x_train_full = x_dataframe.FloatCpuTensor();
        var y_train_full = train[TargetLabels].FloatCpuTensor();
        FullTrain = new InMemoryDataSet(
            x_train_full,
            y_train_full,
            "AmazonEmployeeAccessChallenge",
            GetObjective(),
            null,
            columnNames: x_dataframe.Columns,
            categoricalFeatures: CategoricalFeatures,
            useBackgroundThreadToLoadNextMiniBatch: false);

        _testDataset = new InMemoryDataSet(DataFrame.read_float_csv(GetXTestDatasetPath()).Drop("id").FloatCpuTensor(),
            null,
            nameof(AmazonEmployeeAccessChallengeDatasetSample),
            GetObjective(),
            null,
            columnNames: x_dataframe.Columns,
            categoricalFeatures: CategoricalFeatures,
            useBackgroundThreadToLoadNextMiniBatch: false);

    }

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, AmazonEmployeeAccessChallengeUtils.NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    public override DataSet FullTrainingAndValidation()
    {
        return FullTrain;
    }

    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat)
    {
        throw new NotImplementedException();
    }

    private static string GetXTestDatasetPath()
    {
        return Path.Combine(DataDirectory, "test.csv");
    }

    public override Objective_enum GetObjective() => Objective_enum.Classification;

    public override string[] CategoricalFeatures => new[] { "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE" };
    public override string[] IdColumns => new[] { "id" };
    public override string[] TargetLabels => new[] { "ACTION" };

    public override DataSet TestDataset()
    {
        return _testDataset;
    }

    public override EvaluationMetricEnum GetRankingEvaluationMetric() => throw new NotImplementedException();
    private static string TrainRawFile => Path.Combine(DataDirectory, "train.csv");
}
