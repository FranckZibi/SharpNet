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
        var train = DataFrameT<float>.Load(TrainRawFile, true, float.Parse, CategoricalFeatures(), DataFrame.Float2String);
        var targetFeatures = new List<string> { "ACTION" };
        var x_dataframe = train.Drop(targetFeatures);
        var x_train_full = x_dataframe.Tensor;
        var y_train_full = train.Keep(targetFeatures).Tensor;
        FullTrain = new InMemoryDataSet(
            x_train_full,
            y_train_full,
            "AmazonEmployeeAccessChallenge",
            Objective_enum.Classification,
            null,
            featureNames: x_dataframe.FeatureNames,
            categoricalFeatures: CategoricalFeatures().ToArray(),
            useBackgroundThreadToLoadNextMiniBatch: false);

        _testDataset = new InMemoryDataSet(
            LoadNumericalDataFrame(GetXTestDatasetPath(), true).Drop(new[] { "id" }).Tensor,
            null,
            nameof(AmazonEmployeeAccessChallengeDatasetSample),
            Objective_enum.Classification,
            null,
            featureNames: x_dataframe.FeatureNames,
            categoricalFeatures: CategoricalFeatures().ToArray(),
            useBackgroundThreadToLoadNextMiniBatch: false);

    }

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, AmazonEmployeeAccessChallengeUtils.NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    //public override IDataSet FullTraining()
    //{
    //    return FullTrain;
    //}
   
    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat)
    {
        throw new NotImplementedException();
    }

    private static string GetXTestDatasetPath()
    {
        return Path.Combine(DataDirectory, "test.csv");
    }

    public override List<string> CategoricalFeatures()
    {
        return new List<string> { "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE" };
    }

    public override List<string> IdFeatures()
    {
        return new List<string> { "id" };
    }

    public override List<string> TargetFeatures()
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

    private static string TrainRawFile => Path.Combine(DataDirectory, "train.csv");
}
