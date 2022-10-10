using SharpNet.HyperParameters;
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
        var train = DataFrameT<float>.Load(TrainRawFile, true, float.Parse, DataFrame.Float2String);
        var targetLabels = new List<string> { "ACTION" };
        var x_dataframe = train.Drop(targetLabels);
        var x_train_full = x_dataframe.Tensor;
        var y_train_full = train.Keep(targetLabels).Tensor;
        FullTrain = new InMemoryDataSet(
            x_train_full,
            y_train_full,
            "AmazonEmployeeAccessChallenge",
            GetObjective(),
            null,
            featureNames: x_dataframe.ColumnNames,
            categoricalFeatures: CategoricalFeatures().ToArray(),
            useBackgroundThreadToLoadNextMiniBatch: false);

        _testDataset = new InMemoryDataSet(DataFrame.LoadFloatDataFrame(GetXTestDatasetPath(), true).Drop(new[] { "id" }).Tensor,
            null,
            nameof(AmazonEmployeeAccessChallengeDatasetSample),
            GetObjective(),
            null,
            featureNames: x_dataframe.ColumnNames,
            categoricalFeatures: CategoricalFeatures().ToArray(),
            useBackgroundThreadToLoadNextMiniBatch: false);

    }

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, AmazonEmployeeAccessChallengeUtils.NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    //public override IDataSet FullTraining()
    //{
    //    return FullTrain;
    //}
   
    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat_with_IdColumns)
    {
        throw new NotImplementedException();
    }

    private static string GetXTestDatasetPath()
    {
        return Path.Combine(DataDirectory, "test.csv");
    }

    public override Objective_enum GetObjective() => Objective_enum.Classification;

    public override List<string> CategoricalFeatures()
    {
        return new List<string> { "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE" };
    }

    public override List<string> IdColumns()
    {
        return new List<string> { "id" };
    }

    public override List<string> TargetLabels()
    {
        return new List<string> { "ACTION" };
    }

    public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
    {
        throw new NotImplementedException();
    }
    public override IDataSet TestDataset()
    {
        return _testDataset;
    }

    protected override EvaluationMetricEnum GetRankingEvaluationMetric() => throw new NotImplementedException();
    private static string TrainRawFile => Path.Combine(DataDirectory, "train.csv");
}
