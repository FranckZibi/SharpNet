using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Networks;

namespace SharpNet.Datasets.QRT72;

[SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
public class QRT72DatasetSample : AbstractDatasetSample
{
    #region constructors
    public QRT72DatasetSample(QRT72NetworkSample networkSample) : base(new HashSet<string>())
    {
        _networkSample = networkSample;
        PercentageInTraining = _networkSample.PercentageInTraining;
    }
    #endregion

    #region Hyper-parameters
    private readonly QRT72NetworkSample _networkSample;
    #endregion


    public override bool FixErrors()
    {
        return true;
    }
    //only numerical features

    public override DataSet TestDataset()
    {
        return null;
    }
    public override DataSet FullTrainingAndValidation()
    {
        return NewDataSet(XTrainRawFile);
    }

    public override string[] CategoricalFeatures => Array.Empty<string>();
    public override string[] IdColumns => new[] { "" };
    public override string[] TargetLabels => new[] { "0" };
    public override int DatasetRowsInModelFormatMustBeMultipleOf() => Tensor.CosineSimilarity504_TimeSeries_Length;
    public override EvaluationMetricEnum GetRankingEvaluationMetric() => EvaluationMetricEnum.CosineSimilarity504;
    public override Objective_enum GetObjective() => Objective_enum.Regression;


    private DataSet NewDataSet([JetBrains.Annotations.NotNull] string xFileInTargetFormat)
    {
        var (xTensor, yTensor) = Load_XY(xFileInTargetFormat);
        var columnNames = Enumerable.Range(0, xTensor.Shape[1]).Select(x => x.ToString()).ToArray();
        return new InMemoryDataSet(
            xTensor,
            yTensor,
            QRT72Utils.NAME,
            GetObjective(),
            null,
            columnNames: columnNames,
            categoricalFeatures: CategoricalFeatures,
            useBackgroundThreadToLoadNextMiniBatch: false,
            separator: GetSeparator());
    }
    /// <summary>
    /// path to the test dataset in LightGBM compatible format
    /// </summary>
    /// <returns></returns>
    private (CpuTensor<float>, CpuTensor<float>) Load_XY(string xFileInTargetFormat)
    {
        var x = new List<float[]>();
        var y = new List<float[]>();

        foreach (var l in File.ReadAllLines(xFileInTargetFormat).Skip(1))
        {
            var content = l.Split(',').Skip(1).Select(float.Parse).ToList();
            for (int yIndex = QRT72Utils.D; yIndex < content.Count; ++yIndex)
            {
                x.Add(content.GetRange(yIndex- QRT72Utils.D, QRT72Utils.D).ToArray());
                y.Add(content.GetRange(yIndex, 1).ToArray());
            }
        }

        var xTensor = CpuTensor<float>.NewCpuTensor(x);
        var yTensor = CpuTensor<float>.NewCpuTensor(y);
        return (xTensor, yTensor);
    }
    private const string FILE_SUFFIX = "";
    //private const string FILE_SUFFIX = "_small";
    private static string XTrainRawFile => Path.Combine(QRT72Utils.DataDirectory, "X_train_YG7NZSq" + FILE_SUFFIX + ".csv");

}