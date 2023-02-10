using System;
using log4net;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace SharpNet.Datasets.QRT97;

public class QRT97DatasetSample : AbstractDatasetSample
{
    #region private static fields
    private static readonly DataFrame x_training_raw;
    private static readonly DataFrame y_training_raw;
    private static readonly DataFrame x_test_raw;
    #endregion
    
    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(QRT97DatasetSample));
    #endregion

    #region HyperParameters
   
    #endregion



    static QRT97DatasetSample()
    {
        Utils.ConfigureGlobalLog4netProperties(QRT97Utils.WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(QRT97Utils.WorkingDirectory, "log");
        var sw = Stopwatch.StartNew();
        Log.Debug($"Starting loading raw files");
        x_training_raw = DataFrame.read_csv_normalized(QRT97Utils.XTrainPath, ',', true, QRT97Utils.ColumnNameToType);
        y_training_raw = DataFrame.read_csv_normalized(QRT97Utils.YTrainPath, ',', true, QRT97Utils.ColumnNameToType);
        x_test_raw = DataFrame.read_csv_normalized(QRT97Utils.XTestPath, ',', true, QRT97Utils.ColumnNameToType);
        Log.Debug($"Loading of raw files took {sw.Elapsed.Seconds}s");
    }

    public QRT97DatasetSample() : base(new HashSet<string>())
    {
    }

    public override IScore ExtractRankingScoreFromModelMetricsIfAvailable(params IScore[] modelMetrics)
    {
        return modelMetrics.FirstOrDefault(v => v != null && v.Metric == GetRankingEvaluationMetric());
    }


    public override string[] CategoricalFeatures { get; } = {  "DAY_ID", "COUNTRY" };
    public override string[] IdColumns { get; } = { "ID" };
    public override string[] TargetLabels { get; } = { "TARGET" };
    public override Objective_enum GetObjective()
    {
        return Objective_enum.Regression;
    }
    //public override IScore MinimumScoreToSaveModel => new Score(0.48f, GetRankingEvaluationMetric());



    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat)
    {
        AssertNoIdColumns(predictionsInModelFormat);
        return DataFrame.New(predictionsInModelFormat.FloatCpuTensor(), TargetLabels);
    }

    public override DataSet TestDataset()
    {
        return LoadAndEncodeDataset_If_Needed().testDataset;
    }

    public override DataFrameDataSet FullTrainingAndValidation()
    {
        return LoadAndEncodeDataset_If_Needed().fullTrainingAndValidation;
    }


    public override ITrainingAndTestDataset SplitIntoTrainingAndValidation()
    {
        var fullTrain = FullTrainingAndValidation();
        if (PercentageInTraining >= 0.999)
        {
            return new TrainingAndTestDataset(fullTrain, null, Name);
        }

        var days = x_training_raw.IntColumnContent("DAY_ID");
        Debug.Assert(fullTrain.Count == days.Length);

        var daysSorted = (int[])days.Clone();
        Array.Sort(daysSorted);
        int indexDayForTraining = (int)(PercentageInTraining * daysSorted.Length + 0.1);
        indexDayForTraining -= indexDayForTraining % DatasetRowsInModelFormatMustBeMultipleOf();
        var maxDayForTraining = daysSorted[indexDayForTraining];
        Log.Debug($"All days up to {maxDayForTraining} will be in training, and the rest in validation");

        var training = fullTrain.SubDataSet(id => days[id]<= maxDayForTraining+0.1);
        var test = fullTrain.SubDataSet(id => days[id] > maxDayForTraining + 0.1);
        return new TrainingAndTestDataset(training, test, Name);
    }

    private (DataFrameDataSet fullTrainingAndValidation, DataFrameDataSet testDataset) LoadAndEncodeDataset_If_Needed()
    {
        var sw = Stopwatch.StartNew();
        var key = ComputeHash();
        Log.Debug($"In {nameof(LoadAndEncodeDataset_If_Needed)} for key {key}");

        DatasetEncoder = new DatasetEncoder(this, StandardizeDoubleValues, true);

        var xTrain = UpdateFeatures(x_training_raw);
        var xtest = UpdateFeatures(x_test_raw);
        DatasetEncoder.Fit(xTrain);
        DatasetEncoder.Fit(y_training_raw);
        DatasetEncoder.Fit(xtest);

        var xTrain_Encoded = DatasetEncoder.Transform(xTrain);
        var yTrain_Encoded = DatasetEncoder.Transform(y_training_raw[TargetLabels]);
        var xtest_Encoded = DatasetEncoder.Transform(xtest);

        var fullTrainingAndValidation = new DataFrameDataSet(this, xTrain_Encoded, yTrain_Encoded, false);
        var testDataset = new DataFrameDataSet(this, xtest_Encoded, null, false);

        Log.Debug($"{nameof(LoadAndEncodeDataset_If_Needed)} for key {key} took {sw.Elapsed.Seconds} s");
        return (fullTrainingAndValidation, testDataset);
    }

    private DataFrame UpdateFeatures(DataFrame x)
    {
        return x;
    }


    public override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        return EvaluationMetricEnum.Mae;
    }



}