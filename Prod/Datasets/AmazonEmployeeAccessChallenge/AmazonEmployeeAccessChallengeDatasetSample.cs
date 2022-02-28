using SharpNet.CPU;
using SharpNet.HyperParameters;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using System.Text;

namespace SharpNet.Datasets.AmazonEmployeeAccessChallenge;

public class AmazonEmployeeAccessChallengeDatasetSample : AbstractDatasetSample
{
    #region private fields
    private readonly InMemoryDataSet _testDataset;
    private InMemoryDataSet FullTrain { get; }
    private const string PredictionHeader = "Id,Action";
    #endregion

    [SuppressMessage("ReSharper", "VirtualMemberCallInConstructor")]
    public AmazonEmployeeAccessChallengeDatasetSample() : base(new HashSet<string>())
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
            Dataframe.Load(GetXTestDatasetPath(), true, ',').Drop(new[] { "id" }).Tensor,
            null,
            nameof(AmazonEmployeeAccessChallengeDatasetSample),
            Objective_enum.Classification,
            null,
            new[] { "NONE" },
            x_dataframe.FeatureNames,
            CategoricalFeatures().ToArray(),
            false);

    }

    public static string WorkingDirectory => Path.Combine(Utils.LocalApplicationFolderPath, "SharpNet", "AmazonEmployeeAccessChallenge");
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    public static AmazonEmployeeAccessChallengeDatasetSample ValueOfAmazonEmployeeAccessChallengeDatasetHyperParameters(string workingDirectory, string modelName)
    {
        return (AmazonEmployeeAccessChallengeDatasetSample)ISample.LoadConfigIntoSample(() => new AmazonEmployeeAccessChallengeDatasetSample(), workingDirectory, modelName);
    }
    public override IDataSet FullTraining()
    {
        return FullTrain;
    }
    public override CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(string dataframe_path)
    {
        throw new NotImplementedException(); //TODO
    }

    public override (CpuTensor<float> trainPredictions, CpuTensor<float> validationPredictions, CpuTensor<float> testPredictions) LoadAllPredictions()
    {
        return LoadAllPredictions(true, false, ',');
    }

    private static string GetXTestDatasetPath()
    {
        return Path.Combine(DataDirectory, "test.csv");
    }

    public override List<string> CategoricalFeatures()
    {
        return new List<string> { "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE" };
    }

    public override void ComputeAndSavePredictions(CpuTensor<float> predictionsInModelFormat, string path)
    {
        var sb = new StringBuilder();
        sb.Append(PredictionHeader + Environment.NewLine);

        var ySpan = predictionsInModelFormat.AsReadonlyFloatCpuContent;
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

    private static string TrainRawFile => Path.Combine(DataDirectory, "train.csv");
}
