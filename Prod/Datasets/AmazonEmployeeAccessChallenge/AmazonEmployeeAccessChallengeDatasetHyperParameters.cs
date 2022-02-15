using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using SharpNet.CPU;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;

namespace SharpNet.Datasets.AmazonEmployeeAccessChallenge;

public class AmazonEmployeeAccessChallengeDatasetHyperParameters : AbstractSample
{
    public AmazonEmployeeAccessChallengeDatasetHyperParameters() : base(new HashSet<string>())
    {
        var train = Dataframe.Load(TrainRawFile, true, ',');
        var x_dataframe = train.Drop(TargetFeatures());
        var x_train_full = x_dataframe.Tensor;
        var y_train_full = train.Keep(TargetFeatures()).Tensor;
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

        Test = new InMemoryDataSet(
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

    public InMemoryDataSet FullTrain { get; }
    public InMemoryDataSet Test { get; }

    public static string WorkingDirectory => Path.Combine(Utils.LocalApplicationFolderPath, "SharpNet", "AmazonEmployeeAccessChallenge");
    public static string TrainRawFile => Path.Combine(WorkingDirectory, "Data", "train.csv");
    public static string TestRawFile => Path.Combine(WorkingDirectory, "Data", "test.csv");

    public List<string> CategoricalFeatures()
    {
        return new List<string> { "RESOURCE", "MGR_ID", "ROLE_ROLLUP_1", "ROLE_ROLLUP_2", "ROLE_DEPTNAME", "ROLE_TITLE", "ROLE_FAMILY_DESC", "ROLE_FAMILY", "ROLE_CODE" };
    }

    public static AmazonEmployeeAccessChallengeDatasetHyperParameters ValueOf(string workingDirectory, string modelName)
    {
        return (AmazonEmployeeAccessChallengeDatasetHyperParameters)ISample.LoadConfigIntoSample(() => new AmazonEmployeeAccessChallengeDatasetHyperParameters(), workingDirectory, modelName);
    }

    private const string PredictionHeader = "Id,Action";

    public void SavePredictions(CpuTensor<float> y_pred, string path)
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

    public List<string> TargetFeatures()
    {
        return new List<string> { "ACTION" };
    }
}