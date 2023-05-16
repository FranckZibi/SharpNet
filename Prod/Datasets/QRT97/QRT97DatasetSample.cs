using System;
using log4net;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.MathTools;

namespace SharpNet.Datasets.QRT97;

public class QRT97DatasetSample : AbstractDatasetSample
{
    #region private static fields
    private static readonly DataFrame x_training_raw;
    private static readonly DataFrame y_training_raw;
    private static readonly DataFrame x_test_raw;
    private static readonly DataFrame y_test_random_raw;
    private static readonly Dictionary<string, Dictionary<string, DoubleAccumulator>> IdToFeatureToStats = new();
    #endregion
    #region private fields & properties
    private static readonly ILog Log = LogManager.GetLogger(typeof(QRT97DatasetSample));
    #endregion

    #region HyperParameters
    public bool use_DAY_ID = true;
    public bool use_COUNTRY = true;
    public bool fillna_with_0 = false;
    #endregion




    public static float NormalizeReturn(float r, string featureName)
    {
        if (float.IsNaN(r))
        {
            return float.NaN;
        }

        var featureNameToStats = IdToFeatureToStats["*_*"];
        if (featureNameToStats.TryGetValue(featureName, out var stats))
        {
            var mean = stats.Average;
            var rAdjusted = r - mean;
            var minAdjusted = stats.Min - mean;
            var maxAdjusted = stats.Max - mean;
            var divider = Math.Max(Math.Abs(minAdjusted), Math.Abs(maxAdjusted));
            return (float)(rAdjusted / Math.Max(divider, 1));
        }
        return r;
    }

    static QRT97DatasetSample()
    {
        Utils.ConfigureGlobalLog4netProperties(QRT97Utils.WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(QRT97Utils.WorkingDirectory, "log");
        var sw = Stopwatch.StartNew();
        Log.Debug($"Starting loading raw files");
        x_training_raw = DataFrame.read_csv_normalized(QRT97Utils.XTrainPath, ',', true, QRT97Utils.ColumnNameToType);
        y_training_raw = DataFrame.read_csv_normalized(QRT97Utils.YTrainPath, ',', true, QRT97Utils.ColumnNameToType);
        x_test_raw = DataFrame.read_csv_normalized(QRT97Utils.XTestPath, ',', true, QRT97Utils.ColumnNameToType);
        y_test_random_raw = DataFrame.read_csv_normalized(QRT97Utils.YTestRandomPath, ',', true, QRT97Utils.ColumnNameToType);
        Log.Debug($"Loading of raw files took {sw.Elapsed.Seconds}s");


        var idToProcessedDays = new Dictionary<string, HashSet<int>>();
        
        for (var index = 0; index < 2; index++)
        {
            var df = new[] { x_training_raw, x_test_raw }[index];
            var datasetName = index==0?"TRAIN":"TEST";
            int rows = df.FloatTensor.Shape[0];
            int colsInDf = df.Shape[1]; 
            int colsInFloatTensor = df.FloatTensor.Shape[1]; 
            var dfSpan = df.FloatTensor.SpanContent;

            var countries = df.StringColumnContent("COUNTRY");
            var days = df.IntColumnContent("DAY_ID");

            foreach (var datasetFilter in new[] { "TRAIN", "TEST", "*" })
            {
                foreach (var countryFilter in new[] { "FR", "DE", "*" })
                {
                    var id = datasetFilter + "_" + countryFilter;
                    if (!idToProcessedDays.TryGetValue(id, out var processedDays))
                    {
                        processedDays = new HashSet<int>();
                        idToProcessedDays[id] = processedDays;
                    }
                    for (int row = 0; row < rows; ++row)
                    {
                        var countryName = countries[row];
                        var day = days[row];
                        if (datasetFilter != "*" && datasetFilter != datasetName)
                        {
                            continue;
                        }
                        if (countryFilter != "*" && countryFilter != countryName)
                        {
                            continue;
                        }
                        if (!processedDays.Add(day))
                        {
                            continue;
                        }
                        for (int col=0;col< colsInDf; ++col)
                        {
                            if (df.ColumnsDesc[col].Item2 != DataFrame.FLOAT_TYPE_IDX)
                            {
                                continue;
                            }

                            int colInTensor = df.ColumnsDesc[col].Item3;

                            var featureValue = dfSpan[colInTensor + row* colsInFloatTensor];
                            if (float.IsNaN(featureValue))
                            {
                                continue;
                            }
                            var featureName = df.ColumnsDesc[col].Item1;

                        
                            if (!IdToFeatureToStats.TryGetValue(id, out var featureToStats))
                            {
                                featureToStats = new Dictionary<string, DoubleAccumulator>();
                                IdToFeatureToStats[id] = featureToStats;
                            }
                            if (!IdToFeatureToStats[id].TryGetValue(featureName, out var stats))
                            {
                                stats = new DoubleAccumulator();
                                IdToFeatureToStats[id][featureName] = stats;
                            }
                        
                            stats.Add(featureValue);
                        }
                    }
                }
            }
        }
    }

    public QRT97DatasetSample() : base(new HashSet<string>())
    {
    }

    public override IScore ExtractRankingScoreFromModelMetricsIfAvailable(params IScore[] modelMetrics)
    {
        return modelMetrics.FirstOrDefault(v => v != null && v.Metric == GetRankingEvaluationMetric());
    }


    public override string[] CategoricalFeatures { get; } = {  "DAY_ID", "COUNTRY" };
    public override string IdColumn => "ID";
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

        var fullTrainingAndValidation = new DataFrameDataSet(this, xTrain_Encoded, yTrain_Encoded, y_training_raw.StringColumnContent(IdColumn));
        var testDataset = new DataFrameDataSet(this, xtest_Encoded, null, y_test_random_raw.StringColumnContent(IdColumn));

        Log.Debug($"{nameof(LoadAndEncodeDataset_If_Needed)} for key {key} took {sw.Elapsed.Seconds} s");
        return (fullTrainingAndValidation, testDataset);
    }

    private DataFrame UpdateFeatures(DataFrame x)
    {
        var toDrop = new List<string>();
        if (!use_DAY_ID) { toDrop.Add("DAY_ID"); }
        if (!use_COUNTRY) { toDrop.Add("COUNTRY"); }
        x = x.DropIgnoreErrors(toDrop.ToArray()).Clone();
        if (fillna_with_0)
        {
            x.fillna_inplace(0);
        }
        return x;
    }


    public override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        return EvaluationMetricEnum.SpearmanCorrelation;
    }



}