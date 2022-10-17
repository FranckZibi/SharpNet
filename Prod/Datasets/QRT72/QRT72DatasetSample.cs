﻿using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.HyperParameters;
using SharpNet.Networks;

namespace SharpNet.Datasets.QRT72;

[SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
public class QRT72DatasetSample : AbstractDatasetSample
{
    #region constructors
    public QRT72DatasetSample(QRT72HyperParameters hyperParameters) : base(new HashSet<string>())
    {
        _hyperParameters = hyperParameters;
        PercentageInTraining = _hyperParameters.PercentageInTraining;
    }
    #endregion

    #region Hyper-parameters
    private readonly QRT72HyperParameters _hyperParameters;
    #endregion


    public override bool FixErrors()
    {
        return true;
    }
    //only numerical features
    public override string[] CategoricalFeatures => Array.Empty<string>();
    public override string[] IdColumns => throw new NotImplementedException();
    public override string[] TargetLabels => throw new NotImplementedException();
    
    protected override List<string> PredictionInTargetFormatHeader()
    {
        return new List<string>{"", "0"};
    }

    public override DataSet TestDataset()
    {
        return null;
    }

    public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
    {
        var percentageInTraining = PercentageInTraining;
   
        using var trainingAndValidationDataset = NewDataSet(XTrainRawFile);
        int rowsForTraining = (int)(percentageInTraining * trainingAndValidationDataset.Count + 0.1);
        rowsForTraining -= rowsForTraining % DatasetRowsInModelFormatMustBeMultipleOf();
        return trainingAndValidationDataset.IntSplitIntoTrainingAndValidation(rowsForTraining);
    }

    public override int DatasetRowsInModelFormatMustBeMultipleOf()
    {
        return Tensor.CosineSimilarity504_TimeSeries_Length;
    }

    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat_with_IdColumns)
    {
        return DataFrame.New(predictionsInModelFormat_with_IdColumns.FloatCpuTensor(), PredictionInTargetFormatHeader());
    }

    protected override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        return EvaluationMetricEnum.CosineSimilarity504;
    }

    //public override EvaluationMetricEnum GetLossFunction()
    //{
    //    return EvaluationMetricEnum.CosineSimilarity504;
    //}

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