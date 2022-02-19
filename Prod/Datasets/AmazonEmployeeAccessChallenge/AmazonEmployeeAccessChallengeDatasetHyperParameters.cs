using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Text;
using SharpNet.CPU;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.Datasets.AmazonEmployeeAccessChallenge;

public class AmazonEmployeeAccessChallengeDatasetHyperParameters : AbstractDatasetSample
{
    #region private fields
    private readonly InMemoryDataSet _testDataset;
    private InMemoryDataSet FullTrain { get; }
    private const string PredictionHeader = "Id,Action";
    #endregion

    [SuppressMessage("ReSharper", "VirtualMemberCallInConstructor")]
    public AmazonEmployeeAccessChallengeDatasetHyperParameters() : base(new HashSet<string>())
    {
        var train = Dataframe.Load(TrainRawFile, true, ',');
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
            new[] { "NONE" },
            x_dataframe.FeatureNames,
            CategoricalFeatures().ToArray(),
            false);

        _testDataset = new InMemoryDataSet(
            Dataframe.Load(TestRawFile, true, ',').Drop(new []{"id"}).Tensor,
            null,
            nameof(AmazonEmployeeAccessChallengeDatasetHyperParameters),
            Objective_enum.Classification,
            null,
            new[] { "NONE" },
            x_dataframe.FeatureNames,
            CategoricalFeatures().ToArray(),
            false);

    }

    public static string WorkingDirectory => Path.Combine(Utils.LocalApplicationFolderPath, "SharpNet", "AmazonEmployeeAccessChallenge");
    public static AmazonEmployeeAccessChallengeDatasetHyperParameters ValueOfAmazonEmployeeAccessChallengeDatasetHyperParameters(string workingDirectory, string modelName)
    {
        return (AmazonEmployeeAccessChallengeDatasetHyperParameters)ISample.LoadConfigIntoSample(() => new AmazonEmployeeAccessChallengeDatasetHyperParameters(), workingDirectory, modelName);
    }
    public override IDataSet FullTraining()
    {
        return FullTrain;
    }
    public override CpuTensor<float> ModelPrediction_2_TargetPredictionFormat(string dataframe_path)
    {
        throw new NotImplementedException(); //TODO
    }
    public override List<string> CategoricalFeatures()
    {
        return new List<string> { "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE" };
    }
    public override ModelDatasets ToModelDatasets()
    {
        throw new NotImplementedException();
    }

    protected override void SavePredictions(CpuTensor<float> y_pred, string path)
    {
        var sb = new StringBuilder();
        sb.Append(PredictionHeader + Environment.NewLine);

        var ySpan = y_pred.AsReadonlyFloatCpuContent;
        for (int i = 0; i < ySpan.Length; ++i)
        {
            sb.Append((i + 1) + "," + ySpan[i].ToString(CultureInfo.InvariantCulture) + Environment.NewLine);
        }
        File.WriteAllText(path, sb.ToString().Trim());
    }
    protected override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
    {
        throw new NotImplementedException();
    }
    protected override IDataSet TestDataset()
    {
        return _testDataset;
    }

    private static string TrainRawFile => Path.Combine(WorkingDirectory, "Data", "train.csv");
    private static string TestRawFile => Path.Combine(WorkingDirectory, "Data", "test.csv");
}
