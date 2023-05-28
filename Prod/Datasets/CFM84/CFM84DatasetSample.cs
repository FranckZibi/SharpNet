using System;
using log4net;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using SharpNet.MathTools;
// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable UnassignedField.Global
// ReSharper disable FieldCanBeMadeReadOnly.Global

namespace SharpNet.Datasets.CFM84;

public class CFM84DatasetSample : AbstractDatasetSample
{
    #region private static fields
    private static readonly DataFrame x_training_raw;
    private static readonly DataFrame y_training_raw;
    private static readonly DataFrame x_test_raw;
    private static readonly DataFrame y_test_random_raw;
    private static readonly DataFrame stats_raw;
    //private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, Tuple<DataFrameDataSet, DataFrameDataSet, DatasetEncoder>> CacheDataset = new();
    #endregion
    
    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(CFM84DatasetSample));
    #endregion

    #region HyperParameters
    public bool use_r_day_equity;
    public bool use_vol_r_day_equity;
    public bool use_r_day_market;
    public bool use_vol_r_day_market;
    public bool use_feature_101 = false;

    public bool use_market_correl_r_day_equity;
    public bool use_r_dataset_equity;
    public bool use_vol_r_dataset_equity;
    public bool use_r_dataset = false; //not used
    public bool use_vol_r_dataset = false; //not used
    public int rr_count = 0;
    public bool use_ccat = false;
    public bool use_dcat = false;
    public bool use_ecat = false;
    public bool use_fcat = false; //must be false
    public bool use_wd = false;

    public bool fillna_with_0 = false;
    #endregion



    static CFM84DatasetSample()
    {
        Utils.ConfigureGlobalLog4netProperties(CFM84Utils.WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(CFM84Utils.WorkingDirectory, "log");
        var sw = Stopwatch.StartNew();
        Log.Debug($"Starting loading raw files");
        x_training_raw = DataFrame.read_csv_normalized(CFM84Utils.XTrainPath, ',', true, CFM84Utils.ColumnNameToType);
        y_training_raw = DataFrame.read_csv_normalized(CFM84Utils.YTrainPath, ',', true, CFM84Utils.ColumnNameToType);
        x_test_raw = DataFrame.read_csv_normalized(CFM84Utils.XTestPath, ',', true, CFM84Utils.ColumnNameToType);
        y_test_random_raw = DataFrame.read_csv_normalized(CFM84Utils.YTestRandomPath, ',', true, CFM84Utils.ColumnNameToType);
        stats_raw = DataFrame.read_csv_normalized(CFM84Utils.StatPath, ',', true, CFM84Utils.ColumnNameToType);
        Log.Debug($"Loading of raw files took {sw.Elapsed.Seconds}s");
    }

    public CFM84DatasetSample() : base(new HashSet<string>())
    {
    }

    public override string[] CategoricalFeatures { get; } = {  "equity", "reod", "wd" };
    public override string IdColumn => "ID";
    public override string[] TargetLabels { get; } = { "reod" };
    public override Objective_enum GetObjective()
    {
        return Objective_enum.Classification;
    }
    //public override IScore MinimumScoreToSaveModel => new Score(0.48f, GetRankingEvaluationMetric());

    public override int NumClass => TargetLabelDistinctValues.Length;
    public override string[] TargetLabelDistinctValues => CFM84Utils.TargetLabelDistinctValues;

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

        var days = x_training_raw.IntColumnContent("day");
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
        //if (CacheDataset.TryGetValue(key, out var result))
        //{
        //    Log.Debug($"key {key} already exists in {nameof(LoadAndEncodeDataset_If_Needed)} ");
        //    DatasetEncoder = result.Item3;
        //    return (result.Item1, result.Item2);
        //}


        DatasetEncoder = new DatasetEncoder(this, StandardizeDoubleValues, true);

        var xTrain = UpdateFeatures(x_training_raw);
        var xtest = UpdateFeatures(x_test_raw);
        DatasetEncoder.Fit(xTrain);
        DatasetEncoder.Fit(y_training_raw);
        DatasetEncoder.Fit(xtest);

        var xTrain_Encoded = DatasetEncoder.Transform(xTrain);
        var yTrain_Encoded_Sparse = DatasetEncoder.Transform(y_training_raw[TargetLabels]);
        var yTrain_Encoded_OneHot = yTrain_Encoded_Sparse.FromSparseToOneHotEncoding(NumClass);
        var xtest_Encoded = DatasetEncoder.Transform(xtest);

        var fullTrainingAndValidation = new DataFrameDataSet(this, xTrain_Encoded, yTrain_Encoded_OneHot, y_training_raw.StringColumnContent(IdColumn));
        var testDataset = new DataFrameDataSet(this, xtest_Encoded, null, y_test_random_raw.StringColumnContent(IdColumn));

        //CacheDataset.TryAdd(key, Tuple.Create(fullTrainingAndValidation, testDataset, DatasetEncoder));
        Log.Debug($"{nameof(LoadAndEncodeDataset_If_Needed)} for key {key} took {sw.Elapsed.Seconds} s");
        return (fullTrainingAndValidation, testDataset);
    }

    private DataFrame UpdateFeatures(DataFrame x)
    {
        var ids =  new HashSet<string>(x.StringColumnContent("ID"));
        var equityToReturnsAccumulator = new Dictionary<string, DoubleAccumulator>();
        for (int row = 0; row < stats_raw.Shape[0]; ++row)
        {
            var stringStatContent = stats_raw.StringTensor.RowSpanSlice(row, 1);
            var statId = stringStatContent[0];
            if (!ids.Contains(statId))
            {
                continue;
            }
            var equity = stringStatContent[1];
            if (!equityToReturnsAccumulator.TryGetValue(equity, out var acc))
            {
                acc = new DoubleAccumulator();
                equityToReturnsAccumulator[equity] = acc;
            }
            var floatStatContent = stats_raw.FloatTensor.RowSpanSlice(row, 1);
            var r_day_equity = floatStatContent[1];
            acc.Add(r_day_equity);
        }
        var dataFrameReturnsAccumulator = DoubleAccumulator.Sum(equityToReturnsAccumulator.Values.ToArray());
        var r_dataset = (float)dataFrameReturnsAccumulator.Average;
        var vol_r_dataset = (float)dataFrameReturnsAccumulator.Volatility;
        var statsForDataFrameColumnName = new[]
        {
            "r_dataset_equity",         // mean return of the equity in the DataFrame
            "vol_r_dataset_equity",     // volatility of the return of the equity in the DataFrame
            "r_dataset",                // mean return of the DataFrame
            "vol_r_dataset"             // volatility of the return of the DataFrame
        };
        var statsForDataSet = DataFrame.New(new float[stats_raw.Shape[0] * statsForDataFrameColumnName.Length], statsForDataFrameColumnName);
        var allStats = DataFrame.MergeHorizontally(stats_raw, statsForDataSet);
        for (int row = 0; row < allStats.Shape[0]; ++row)
        {
            var stringStatContent = allStats.StringTensor.RowSpanSlice(row, 1);
            var statId = stringStatContent[0];
            if (!ids.Contains(statId))
            {
                continue;
            }
            var equity = stringStatContent[1];
            var floatStatContent = allStats.FloatTensor.RowSpanSlice(row, 1);
            var r_dataset_equity = (float)equityToReturnsAccumulator[equity].Average;
            var vol_r_dataset_equity = (float)equityToReturnsAccumulator[equity].Volatility;
            floatStatContent[^4] = r_dataset_equity;
            floatStatContent[^3] = vol_r_dataset_equity;
            floatStatContent[^2] = r_dataset;
            floatStatContent[^1] = vol_r_dataset;
        }


        //var allStats = stats_raw;

        allStats = allStats.DropIgnoreErrors("day", "equity").Clone();

        var xWithStats = x.LeftJoinWithoutDuplicates(allStats, new []{"ID"});


        var toDrop = new List<string>{ "day", "equity"};
        if (!use_r_day_equity) { toDrop.Add("r_day_equity");}
        if (!use_vol_r_day_equity) { toDrop.Add("vol_r_day_equity"); }
        if (!use_r_day_market) { toDrop.Add("r_day_market"); }
        if (!use_vol_r_day_market) { toDrop.Add("vol_r_day_market"); }
        if (!use_feature_101) { toDrop.Add("feature_101"); }
        if (!use_market_correl_r_day_equity) { toDrop.Add("market_correl_r_day_equity"); }
        if (!use_r_dataset_equity) { toDrop.Add("r_dataset_equity"); }
        if (!use_vol_r_dataset_equity) { toDrop.Add("vol_r_dataset_equity"); }
        if (!use_r_dataset) { toDrop.Add("r_dataset"); }
        if (!use_vol_r_dataset) { toDrop.Add("vol_r_dataset"); }
        if (!use_ccat) { toDrop.AddRange(xWithStats.Columns.Where(c=>c.StartsWith("ccat"))); }
        if (!use_dcat) { toDrop.AddRange(xWithStats.Columns.Where(c=>c.StartsWith("dcat"))); }
        if (!use_ecat) { toDrop.AddRange(xWithStats.Columns.Where(c=>c.StartsWith("ecat"))); }
        if (!use_fcat) { toDrop.AddRange(xWithStats.Columns.Where(c=>c.StartsWith("fcat"))); }
        if (!use_wd) { toDrop.AddRange(xWithStats.Columns.Where(c=>c.StartsWith("wd"))); }

        toDrop.AddRange(Enumerable.Range(rr_count, 10).Select(i => "rr"+i));
        xWithStats = xWithStats.DropIgnoreErrors(toDrop.ToArray()).Clone();

        if (fillna_with_0)
        {
            xWithStats.fillna_inplace(0);
        }

        return xWithStats;
    }
}