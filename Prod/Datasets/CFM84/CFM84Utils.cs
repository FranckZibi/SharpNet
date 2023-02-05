using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using log4net;
using SharpNet.CatBoost;
using SharpNet.Data;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.MathTools;

namespace SharpNet.Datasets.CFM84;

public static class CFM84Utils
{
    public const string NAME = "CFM84";


    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(CFM84Utils));
    #endregion

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    // ReSharper disable once MemberCanBePrivate.Global

    public static string XTrainPath => Path.Combine(DataDirectory, "input_training.csv");
    public static string YTrainPath => Path.Combine(DataDirectory, "output_training_gmEd6Zt.csv");
    public static string XTestPath => Path.Combine(DataDirectory, "input_test.csv");
    public static string StatPath => Path.Combine(DataDirectory, "cfm84_stats.csv");
    public static string OutputTestRandomPath => Path.Combine(DataDirectory, "output_test_random.csv");
    public static readonly string[] TargetLabelDistinctValues = new[] { "0", "-1", "1" };




    public static void AverageTestPredictions(params string[] predictionsPaths)
    {
        var outputTestRandom_df = DataFrame.read_float_csv(OutputTestRandomPath);
        var id_df = outputTestRandom_df["ID"].Clone();
        var avg = DataFrame.Average(predictionsPaths.Select(p => DataFrame.read_float_csv(p)).ToArray());
        Debug.Assert(id_df.Shape[0] == avg.Shape[0]);
        var floatTensorArgMax = avg.FloatTensor.ArgMax().AsFloatCpuSpan.ToArray().Select(f=>Utils.NearestInt(f));
        var strContent = floatTensorArgMax.Select(i => TargetLabelDistinctValues[i]).ToArray();
        var reod_df = DataFrame.New(strContent, new List<string>() { "reod" });
        var pred_df = DataFrame.MergeHorizontally(id_df, reod_df);
        var avgFileName = "avg_"+string.Join("_", predictionsPaths.Select(Path.GetFileNameWithoutExtension))+".csv";
        var avgPath = Path.Combine(Path.GetDirectoryName(predictionsPaths[0])??"", avgFileName);
        pred_df.to_csv(avgPath);
        Log.Info($"Average predictions stored in {avgPath}");
    }

    public static float NormalizeReturn(float r)
    {
        if (float.IsNaN(r) || Math.Abs(r)>10000)
        {
            return float.NaN;
        }

        return r/10000;
    }

    public static void BuildStatNormalizeFile()
    {
        var train = DataFrame.read_csv(XTrainPath, true, ColumnNameToType, true);
        var return_columns = train.Columns.Where(c => c.StartsWith("r")).ToArray();
        train.UpdateColumnsInPlace(NormalizeReturn, return_columns);

        train = train.sort_values("day");
        train.to_csv(XTrainPath.Replace(".csv","")+"_normalized.csv");

        //we re order the y train dataset so it has the same order as the x train dataset
        var y_train = DataFrame.read_string_csv(YTrainPath, true, true);
        var y_train_dico = new Dictionary<string, string>();
        var y_train_ID = y_train.StringColumnContent("ID");
        var y_train_reod = y_train.StringColumnContent("reod");
        for (int i = 0; i < y_train_ID.Length; ++i)
        {
            y_train_dico.Add(y_train_ID[i], y_train_reod[i]);
        }
        var y_train_ordered = new List<string>();
        foreach(var sorted_id in train.StringColumnContent("ID"))
        {
            y_train_ordered.Add(sorted_id);
            y_train_ordered.Add(y_train_dico[sorted_id]);
        }
        DataFrame.New(y_train_ordered.ToArray(), y_train.Columns).to_csv(YTrainPath.Replace(".csv", "") + "_normalized.csv");
        var test = DataFrame.read_csv(XTestPath, true, ColumnNameToType, true);
        test.UpdateColumnsInPlace(NormalizeReturn, return_columns);
        test.to_csv(XTestPath.Replace(".csv", "") + "_normalized.csv");
    }

    [SuppressMessage("ReSharper", "UnusedVariable")]
    public static void BuildStatFile()
    {
        var train = DataFrame.read_csv(XTrainPath, true, ColumnNameToType, true);
        var test = DataFrame.read_csv(XTestPath, true, ColumnNameToType, true);
        var df = DataFrame.MergeVertically(train, test);


        var indexBySubDay = new List<int[]>()
        {
            //new [] { 0, 10 },
            //new [] { 11, 20 },
            //new [] { 21, 31 },
            //new [] { 32, 41 },
            //new [] { 42, 52 }
        };
        int nb_splits_by_day = indexBySubDay.Count;
        const int rr_count = 10;

        var return_columns = train.Columns.Where(c => c.StartsWith("r")).ToArray();
        var equityToDayToReturn = new Dictionary<string, Dictionary<int, float>>();
        var equityToDayToReturnBySubDay = new Dictionary<string, Dictionary<int, float[]>>();
        var equityToDayToVol = new Dictionary<string, Dictionary<int, float>>();
        var equityToDayToRow = new Dictionary<string, Dictionary<int, int>>();
        var equityToDayToVolBySubDay = new Dictionary<string, Dictionary<int, float[]>>();
        var dayToReturns = new Dictionary<int, DoubleAccumulator>();
        var dayToReturnsBySubDay = new Dictionary<int, DoubleAccumulator[]>();
        var dayToEachReturn = new Dictionary<int, List<DoubleAccumulator>>();

        var rows = df.Shape[0];
        
        for (int row = 0; row < rows; ++row)
        {
            var stringContent = df.StringTensor.RowSpanSlice(row,1);
            var floatContent = df.FloatTensor.RowSpanSlice(row, 1);
            var intContent = df.IntTensor.RowSpanSlice(row, 1);
            Debug.Assert(floatContent.Length == return_columns.Length);
            var ID = stringContent[0];
            var equity = stringContent[1];
            var day = intContent[0];

            if (!equityToDayToReturn.ContainsKey(equity))
            {
                equityToDayToReturn[equity] = new Dictionary<int, float>();
                equityToDayToReturnBySubDay[equity] = new Dictionary<int, float[]>();
                equityToDayToVol[equity] = new Dictionary<int, float>();
                equityToDayToVolBySubDay[equity] = new Dictionary<int, float[]>();
                equityToDayToRow[equity] = new Dictionary<int, int>();
            }
            equityToDayToRow[equity][day] = row;

            if (!equityToDayToReturnBySubDay[equity].ContainsKey(day))
            {
                equityToDayToReturnBySubDay[equity][day] = new float[nb_splits_by_day];
                equityToDayToVolBySubDay[equity][day] = new float[nb_splits_by_day];
            }

            if (!dayToReturns.ContainsKey(day))
            {
                dayToReturns[day] = new DoubleAccumulator();
                dayToEachReturn[day] = new List<DoubleAccumulator>();
                while (dayToEachReturn[day].Count < return_columns.Length)
                {
                    dayToEachReturn[day].Add(new DoubleAccumulator());
                }

                dayToReturnsBySubDay[day] = new DoubleAccumulator[nb_splits_by_day];
                for (int i = 0; i < nb_splits_by_day; ++i)
                {
                    dayToReturnsBySubDay[day][i] = new DoubleAccumulator();
                }
            }

            var acc = new DoubleAccumulator();
            var total_return = 1f;
            for (int col = 0; col < floatContent.Length; ++col)
            {
                var r = floatContent[col];
                if (float.IsNaN(r))
                {
                    r = 0;
                }
                total_return *= 1 + r;
                acc.Add(r, 1);
                dayToEachReturn[day][col].Add(r, 1);
            }
            equityToDayToReturn[equity][day] = total_return - 1;
            equityToDayToVol[equity][day] = (float)acc.Volatility;
            dayToReturns[day].Add(total_return - 1, 1);

            for (int i = 0; i < nb_splits_by_day; ++i)
            {
                var startIdx = indexBySubDay[i][0];
                var endIdx = indexBySubDay[i][1];
                acc = new DoubleAccumulator();
                total_return = 1f;
                
                for (int col = startIdx; col < endIdx; ++col)
                {
                    var r = floatContent[col];
                    if (float.IsNaN(r))
                    {
                        r = 0;
                    }
                    total_return *= 1 + r;
                    acc.Add(r, 1);
                }
                equityToDayToReturnBySubDay[equity][day][i] = total_return - 1;
                equityToDayToVolBySubDay[equity][day][i] = (float)acc.Volatility;
                dayToReturnsBySubDay[day][i].Add(total_return - 1, 1);
            }
        }

        var ids = df["ID", "day", "equity"].Clone();
        var computedColumns = new List<string>();

        for (int i = 0; i < nb_splits_by_day; ++i)
        {
            var startIdx = indexBySubDay[i][0];
            var endIdx = indexBySubDay[i][1];
            var rName = "r" + startIdx.ToString("00") + endIdx.ToString("00");
            computedColumns.Add($"{rName}_day_equity");
            computedColumns.Add($"vol_{rName}_day_equity");
            computedColumns.Add($"{rName}_day_market");
            computedColumns.Add($"vol_{rName}_day_market");
        }
        computedColumns.AddRange(new []
        { 
            "r_day_equity",     // return for equity 'equity' and day 'day' (0.01 => increase by 1% )
            "vol_r_day_equity", // volatility of the return for equity 'equity' for day 'day'
            "r_day_market",     // average return of the market for day 'day'
            "vol_r_day_market", // volatility of the return of all market equities for day 'day'
            "market_correl_r_day_equity" //correlation between the return of 'equity' and the return of the market for day 'day' 
        });


        for (int i = 0; i < rr_count; ++i)
        {
            computedColumns.Add($"rr{i}");
        }

        var data = DataFrame.New(new float[rows*computedColumns.Count], computedColumns);
        var results = DataFrame.MergeHorizontally(ids, data);
        void ProcessResultRow(int row)
        {
            var equity = results.StringTensor.RowSpanSlice(row, 1)[1];
            var day = df.IntTensor.RowSpanSlice(row, 1)[0];
            var floatContent = results.FloatTensor.RowSpanSlice(row, 1);

            int col = floatContent.Length - nb_splits_by_day * 4 - 5 - rr_count;
            for (int i = 0; i < nb_splits_by_day; ++i)
            {
                floatContent[col++] = equityToDayToReturnBySubDay[equity][day][i];           //rXXYY_day_equity
                floatContent[col++] = equityToDayToVolBySubDay[equity][day][i];            //vol_rXXYY_day_equity
                floatContent[col++] = (float)dayToReturnsBySubDay[day][i].Average;       //rXXYY_day_market
                floatContent[col++] = (float)dayToReturnsBySubDay[day][i].Volatility;    //vol_rXXYY_day_market
            }

            floatContent[col++] = equityToDayToReturn[equity][day];    //r_day_equity
            floatContent[col++] = equityToDayToVol[equity][day];       //vol_r_day_equity
            floatContent[col++] = (float)dayToReturns[day].Average;    //r_day_market
            floatContent[col++] = (float)dayToReturns[day].Volatility; //vol_r_day_market

            // we compute the correl of the equity return with the market for this day
            var floatReturnContent = df.FloatTensor.RowSpanSlice(row, 1);
            var lr = new LinearRegression();
            for (int r_index = 0; r_index < return_columns.Length; ++r_index)
            {
                var r = floatReturnContent[r_index];
                if (!float.IsNaN(r))
                {
                    lr.Add(r, dayToEachReturn[day][r_index].Average);
                }
            }
            var market_correl_r_day_equity = (float)lr.PearsonCorrelationCoefficient;
            if (float.IsNaN(market_correl_r_day_equity))
            {
                market_correl_r_day_equity = 0;
            }
            floatContent[col++] = market_correl_r_day_equity;

            if (rr_count > 0)
            {
                int rrRow = -1;
                for (int rrDay = day+1; rrDay < day+100; ++rrDay)
                {
                    if (equityToDayToRow[equity].ContainsKey(rrDay))
                    {
                        rrRow = equityToDayToRow[equity][rrDay];
                        break;
                    }
                }

                if (rrRow >= 0)
                {
                    var floatContentRRRow = df.FloatTensor.RowSpanSlice(rrRow, 1);
                    for (int i = 0; i < rr_count; ++i)
                    {
                        floatContent[col++] = floatContentRRRow[i];
                    }
                }
                else
                {
                    for (int i = 0; i < rr_count; ++i)
                    {
                        floatContent[col++] = 0;
                    }
                }




            }
        }

        Parallel.For(0, rows, ProcessResultRow);

        results.to_csv(Path.Combine(DataDirectory, StatPath));
    }

    public static void Run()
    {
        //BuildStatNormalizeFile();
        //BuildStatFile();

        //using var m = ModelAndDatasetPredictions.Load(@"C:\Projects\Challenges\CFM84\dump\", "FB801BE40C", true);
        //m.EstimateLossContribution(computeAlsoRankingScore: true, maxGroupSize: 5000);


        //AverageTestPredictions(@"C:\Projects\Challenges\CFM84\submit\9DE295AB09_modelformat_predict_test_.csv",  @"C:\Projects\Challenges\CFM84\submit\CF37BAB198_modelformat_predict_test_.csv");
        //LaunchLightGBMHPO(10);
        LaunchCatBoostHPO(2000);
        //FitDistribution();
        //FixDistribution();
    }


    public static Type ColumnNameToType(string columnName)
    {
        if (columnName.Equals("ID") || columnName.Equals("equity") || columnName.Equals("reod"))
        {
            return typeof(string);
        }
        if (columnName.Equals("day"))
        {
            return typeof(int);
        }
        return typeof(float);
    }

    public static void LaunchCatBoostHPO(int iterations = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        // ReSharper disable once ConvertToConstant.Local
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            //uncomment appropriate one
            {"loss_function", "MultiClass"},  //for multi class classification
            
            {"use_r_day_equity", new []{true , false}},                 //0.48218 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_day_market", new []{true , false}},             //0.48194 Valid Accuracy (False) vs 0.47576 Valid Accuracy (True) , but much better result in training
            {"use_r_dataset_equity", new []{true, false}},              //0.48294 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_day_equity", new []{true, false}},              //0.48194 Valid Accuracy (False) vs 0.47949 Valid Accuracy (True) 
            {"use_r_day_market", new []{true /*, false*/}},             //0.47531 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_market_correl_r_day_equity", new []{true, false}},    //0.48109 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_dataset_equity", new []{true, false}},          //0.48194 Valid Accuracy (False) vs 0.48116 Valid Accuracy (True)
            {"rr_count", AbstractHyperParameterSearchSpace.Range(0, 5)},


            {"use_r_dataset", new []{/*true ,*/ false}}, //must be false
            {"use_vol_r_dataset", new []{/*true,*/ false}}, //must be false

            //{"grow_policy", new []{ "SymmetricTree", "Depthwise" /*, "Lossguide"*/}},

            { "logging_level", nameof(CatBoostSample.logging_level_enum.Verbose)},
            { "allow_writing_files",false},
            { "thread_count",1},
            { "iterations", iterations },
            { "od_type", "Iter"},
            { "od_wait",iterations/10},
            { "depth", AbstractHyperParameterSearchSpace.Range(4, 10) },
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.10f)},
            { "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
            { "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
            { "l2_leaf_reg",AbstractHyperParameterSearchSpace.Range(0, 10)},
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new CFM84DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = true;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }
    
    public static (ISample bestSample, IScore bestScore) LaunchLightGBMHPO(int num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled


            {"use_r_day_equity", new []{true , false}},                 //0.48218 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_day_market", new []{true , false}},             //0.48194 Valid Accuracy (False) vs 0.47576 Valid Accuracy (True) , but much better result in training
            {"use_r_dataset_equity", new []{true, false}},              //0.48294 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_day_equity", new []{true, false}},              //0.48194 Valid Accuracy (False) vs 0.47949 Valid Accuracy (True) 
            {"use_r_day_market", new []{true /*, false*/}},             //0.47531 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_market_correl_r_day_equity", new []{true, false}},    //0.48109 Valid Accuracy (False) vs 0.48194 Valid Accuracy (True)
            {"use_vol_r_dataset_equity", new []{true, false}},          //0.48194 Valid Accuracy (False) vs 0.48116 Valid Accuracy (True)
            {"rr_count", AbstractHyperParameterSearchSpace.Range(0, 3)},

            {"use_r_dataset", new []{/*true ,*/ false}}, //must be false
            {"use_vol_r_dataset", new []{/*true,*/ false}}, //must be false


            { "num_threads", -1},
            { "verbosity", "0" },
            { "early_stopping_round", num_iterations/10 },
            { "num_iterations", num_iterations },

            {"objective", "multiclass"},
            {"num_class", TargetLabelDistinctValues.Length},


            { "colsample_bytree",0.9},

            //{ "bagging_fraction", 0.8},
            { "bagging_fraction", new[]{0.8f, 1.0f} },
            { "bagging_freq", new[]{/*0,*/ 10} },

            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},

            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.2f)},
            { "max_depth", new[]{10, 20, 50} },
            { "min_data_in_leaf", new[]{20, 100 , 200} },


            //{ "boosting", new []{"gbdt", "dart"}},
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 50) },

/*
            //high priority
            { "bagging_fraction", 0.8},
            //{ "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
            { "bagging_freq", new[]{0, 10} },
            
            
            //{ "boosting", new []{"gbdt", "dart"}},
            { "boosting", "dart"},
            { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.6f, 1.0f)},

            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            //{ "learning_rate",AbstractHyperParameterSearchSpace.Range(0.005f, 0.2f)},
            
            { "learning_rate", new[]{0.001, 0.01, 0.1}},

            { "max_depth", new[]{10, 20, 50} },

            { "min_data_in_leaf", new[]{20, 50 ,100} },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 50) },
            //{ "num_leaves", new[]{3,15,50} },

            ////medium priority
            //{ "drop_rate", new[]{0.05, 0.1, 0.2}},                               //specific to dart mode
            { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            //{ "min_data_in_bin", new[]{3, 10, 100, 150}  },
            //{ "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            
            //{ "max_drop", new[]{40, 50, 60}},                                   //specific to dart mode

            //{ "skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},  //specific to dart mode

            ////low priority
            //{ "extra_trees", new[] { true , false } }, //low priority 
            ////{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)}, //very low priority
            //{ "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) }, //low priority
            //{ "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) }
*/
        };

        

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), new CFM84DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
        return (hpo.BestSampleFoundSoFar, hpo.ScoreOfBestSampleFoundSoFar);
    }


    private static (double accuracy, int numCorrect) PrintAccuracy(int[] y_true, int[] y_predicted, string name)
    {
        var (accuracy, numCorrect) = ComputeAccuracy(y_true, y_predicted);
        Console.WriteLine($"{name} Accuracy: {accuracy} ({numCorrect}/{y_true.Length})");
        return (accuracy, numCorrect);
    }
    private static (int[] class2Count, float[] class2Frequency) ToClassCountAndFrequency(int numClass, int[] y_true)
    {
        var class2Count = new int[numClass];
        foreach (var c in y_true)
        {
            ++class2Count[c];
        }

        var class2Frequency = new float[numClass];
        for (int c = 0; c < numClass; ++c)
        {
            class2Frequency[c] = (float)class2Count[c] / y_true.Length;
        }

        return (class2Count, class2Frequency);
    }
    private static (double accuracy, int numCorrect) ComputeAccuracy(int[] y_true, int[] y_predicted)
    {
        int numCorrect = 0;
        for (int i = 0; i < y_true.Length; ++i)
        {
            if (y_true[i] == y_predicted[i])
            {
                ++numCorrect;
            }
        }
        var accuracy = Math.Round(numCorrect / (double)y_true.Length, 5);
        return (accuracy, numCorrect);
    }


}