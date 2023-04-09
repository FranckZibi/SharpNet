using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.Networks;
// ReSharper disable UnusedMember.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets.CFM60;

public class CFM60DatasetSample : DatasetSampleForTimeSeries
{
    private static readonly TimeSeriesSinglePoint[] EntriesTrain;
    private static readonly TimeSeriesSinglePoint[] EntriesTest;

    static CFM60DatasetSample()
    {
        EntriesTrain = CFM60Entry.Load(
            Path.Combine(NetworkSample.DefaultDataDirectory, "CFM60", "input_training.csv"),
            Path.Combine(NetworkSample.DefaultDataDirectory, "CFM60", "output_training_IxKGwDV.csv"),
            _ => { });
        EntriesTest = CFM60Entry.Load(
            Path.Combine(NetworkSample.DefaultDataDirectory, "CFM60", "input_test.csv"),
            null,
            _ => { });

    }


    public override bool PredictionsMustBeOrderedByIdColumn => true;

    protected override int CountOfDistinctCategoricalValues(string columnName)
    {
        switch (columnName)
        {
            case "pid": return CFM60Entry.DISTINCT_PID_COUNT;
            default:
                throw new ArgumentException($"unknown categorical feature {columnName}");
        }
    }
    public override int LoadEntry(TimeSeriesSinglePoint entry0, float prev_Y, Span<float> xElementId, int idx, EncoderDecoder_NetworkSample networkSample, bool isEncoder)
    {
        var entry = (CFM60Entry)entry0;
        int start_idx = idx;

        if (isEncoder)
        {
            xElementId[idx++] = entry.ID;
        }

        //pid
        if (networkSample.Pid_EmbeddingDim >= 1)
        {
            //pids are in range  [0, 899]
            //EmbeddingLayer is expecting them in range [1,900] that's why we add +1
            xElementId[idx++] = entry.pid + 1;
        }

        //day/year
        if (Use_day)
        {
            xElementId[idx++] = entry.day / Use_day_Divider;
        }

        if (Use_fraction_of_year)
        {
            xElementId[idx++] = DayToFractionOfYear(entry.day);
        }

        if (Use_year_Cyclical_Encoding)
        {
            xElementId[idx++] = (float)Math.Sin(2 * Math.PI * DayToFractionOfYear(entry.day));
            xElementId[idx++] = (float)Math.Cos(2 * Math.PI * DayToFractionOfYear(entry.day));
        }

        if (Use_EndOfYear_flag)
        {
            xElementId[idx++] = EndOfYear.Contains(entry.day) ? 1 : 0;
        }

        if (Use_Christmas_flag)
        {
            xElementId[idx++] = Christmas.Contains(entry.day) ? 1 : 0;
        }

        if (Use_EndOfTrimester_flag)
        {
            xElementId[idx++] = EndOfTrimester.Contains(entry.day) ? 1 : 0;
        }

        //abs_ret
        if (Use_abs_ret)
        {
            //entry.abs_ret.AsSpan().CopyTo(xDest.Slice(idx, entry.abs_ret.Length));
            //idx += entry.abs_ret.Length;
            for (int i = 0; i < entry.abs_ret.Length; ++i)
            {
                xElementId[idx++] = entry.abs_ret[i];
            }
        }

        //rel_vol
        if (Use_rel_vol)
        {

            //asSpan.CopyTo(xDest.Slice(idx, entry.rel_vol.Length));
            //idx += entry.rel_vol.Length;
            for (int i = 0; i < entry.rel_vol.Length; ++i)
            {
                xElementId[idx++] = entry.rel_vol[i];
            }
        }

        //LS
        if (Use_LS)
        {
            xElementId[idx++] = entry.LS;
        }

        //NLV
        if (Use_NLV)
        {
            xElementId[idx++] = entry.NLV;
        }

        //y estimate
        if (Use_prev_Y && isEncoder)
        {
            xElementId[idx++] = prev_Y;
        }

        return idx- start_idx;
    }


    public override int[] GetInputShapeOfSingleElement()
    {
        return new[] { encoderDecoder_NetworkSample.Encoder_TimeSteps, GetInputSize(true) };
    }


    public const string NAME = "CFM60";
    #region load of datasets
    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(NetworkSample.DefaultDataDirectory, NAME);
    // ReSharper disable once MemberCanBePrivate.Global
    #endregion




    public CFM60DatasetSample()
    {
    }

    #region Hyper-Parameters
    //pid embedding
    public int Pid_EmbeddingDim = 4;  //validated on 16-jan-2021: -0.0236

    //y estimate
    public bool Use_y_LinearRegressionEstimate = true; //validated on 19-jan-2021: -0.0501 (with other changes)
    /// <summary>
    /// should we use the average observed 'y' outcome of the company (computed in the training dataSet) in the input tensor
    /// </summary>
    public bool Use_mean_pid_y = false; //discarded on 19-jan-2021: +0.0501 (with other changes)
    public bool Use_volatility_pid_y = true; //validated on 17-jan-2021: -0.0053
    public bool Use_variance_pid_y = false;
    

    //rel_vol fields
    public bool Use_rel_vol = true;
    public bool Use_rel_vol_start_and_end_only = false; //use only the first 12 and last 12 elements of rel_vol
    public bool Use_volatility_rel_vol = false; //'true' discarded on 22-jan-2020: +0.0065

    //abs_ret
    public bool Use_abs_ret = true;  //validated on 16-jan-2021: -0.0515

    //LS
    public bool Use_LS = true; //validated on 16-jan-2021: -0.0164
    /// <summary>
    /// normalize LS value between in [0,1] interval
    /// </summary>

    //NLV
    public bool Use_NLV = true; //validated on 5-july-2021: -0.0763

    // year/day field
    /// <summary>
    /// add the fraction of the year of the current day as an input
    /// it is a value between 1/250f (1-jan) to 1f (31-dec)
    /// </summary>
    public bool Use_fraction_of_year = false;

    /// <summary>
    /// add a cyclical encoding of the current year into 2 new features:
    ///     year_sin: sin(2*pi*fraction_of_year)
    ///and 
    ///     year_cos: cos (2*pi*fraction_of_year)
    ///where
    ///     fraction_of_year: a real between 0 (beginning of the year(, and 1 (end of the year)
    /// see: https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
    /// </summary>
    public bool Use_year_Cyclical_Encoding = false; //discarded on 18-sep-2021: +0.01

    /// <summary>
    /// when 'Use_day' is true, we add the following feature value:   day / Use_day_Divider 
    /// </summary>
    //public float Use_day_Divider = 1151f;
    public float Use_day_Divider = 650f; //validated on 18-sept-2021 -0.002384 vs 1151f

    public bool Use_day = false; //discarded on 19-jan-2021: +0.0501 (with other changes)
    public bool Use_EndOfYear_flag = true;  //validated on 19-jan-2021: -0.0501 (with other changes)
    public bool Use_Christmas_flag = true;  //validated on 19-jan-2021: -0.0501 (with other changes)
    public bool Use_EndOfTrimester_flag = true;  //validated on 19-jan-2021: -0.0501 (with other changes)
    #endregion



    /// <summary>
    /// day is the end of year
    /// </summary>
    public static readonly int[] SortedEndOfYear = { 19, 269, 519, 770, 1021 };
    public static readonly HashSet<int> EndOfYear = new(SortedEndOfYear);
    /// <summary>
    /// day is the end of teh trimester
    /// </summary>
    public static readonly HashSet<int> EndOfTrimester = new(new[]
    {
        77, 327, 577, 828,1084, //march
        141, 391, 641, 892,1147, //june
        201, 451, 702, 953, //sept
        19, 269, 519, 770,1021 // dec //TODO: check without line
    });

    /// <summary>
    /// return the day as a fraction of the day in year in ]0,1] range
    /// 1-jan   => 1/250f
    /// 31-dec  => 1
    /// </summary>
    /// <param name="day"></param>
    /// <returns></returns>
    public static float DayToFractionOfYear(int day)
    {
        for (int i = SortedEndOfYear.Length - 1; i >= 0; --i)
        {
            if (day > SortedEndOfYear[i])
            {
                float daysInYear = (i == SortedEndOfYear.Length - 1) ? 250f : (SortedEndOfYear[i + 1] - SortedEndOfYear[i]);
                return ((day - SortedEndOfYear[i]) / daysInYear);
            }
        }
        return (day + (250f - SortedEndOfYear[0])) / 250f;
    }


    /// <summary>
    /// day is just before the end of year (Christmas ?)
    /// </summary>
    public static readonly HashSet<int> Christmas = new(new[] { 14, 264, 514, 765, 1016 });

    public override string[] CategoricalFeatures => new[] { "pid" };
    public override string IdColumn => "ID";
    public override string[] TargetLabels => new[] { "target" };
    public override Objective_enum GetObjective()
    {
        return Objective_enum.Regression;
    }

    private List<string> Encoder_FeatureNames;
    private List<string> Decoder_FeatureNames;





    public override string[] GetColumnNames()
    {
        GetInputSize(true);
        return Encoder_FeatureNames.ToArray();
    }
    public override int GetInputSize(bool isEncoderInputSize)
    {
        if (isEncoderInputSize && Encoder_FeatureNames != null)
        {
            return Encoder_FeatureNames.Count;
        }
        if (!isEncoderInputSize && Decoder_FeatureNames != null)
        {
            return Decoder_FeatureNames.Count;
        }

        var featureNames = new List<string>();

        int result = 0;

        if (isEncoderInputSize)
        {
            ++result;
            featureNames.Add(nameof(CFM60Entry.ID));
        }

        //pid embedding
        if (Pid_EmbeddingDim >= 1) { ++result; featureNames.Add(nameof(CFM60Entry.pid)); }

        //y estimate
        if (Use_prev_Y && isEncoderInputSize) { ++result; featureNames.Add("prev_y"); }

        //day/year
        if (Use_day) { ++result; featureNames.Add("day/" + (int)Use_day_Divider); }
        if (Use_fraction_of_year) { ++result; featureNames.Add("fraction_of_year"); }
        if (Use_year_Cyclical_Encoding)
        {
            ++result;
            featureNames.Add("sin_year");
            ++result;
            featureNames.Add("cos_year");
        }
        if (Use_EndOfYear_flag) { ++result; featureNames.Add("EndOfYear_flag"); }
        if (Use_Christmas_flag) { ++result; featureNames.Add("Christmas_flag"); }
        if (Use_EndOfTrimester_flag) { ++result; featureNames.Add("EndOfTrimester_flag"); }

        //abs_ret
        if (Use_abs_ret)
        {
            result += CFM60Entry.POINTS_BY_DAY;
            for (int i = 0; i < CFM60Entry.POINTS_BY_DAY; ++i)
            {
                featureNames.Add(VectorFeatureName("abs_ret", i));
            }
        }
        //rel_vol
        if (Use_rel_vol)
        {
            result += CFM60Entry.POINTS_BY_DAY;
            for (int i = 0; i < CFM60Entry.POINTS_BY_DAY; ++i)
            {
                featureNames.Add(VectorFeatureName("rel_vol", i));
            }
        }

        //LS
        if (Use_LS)
        {
            ++result;
            featureNames.Add("LS");
        }

        //NLV
        if (Use_NLV)
        {
            ++result;
            featureNames.Add("NLV");
        }
        if (result != featureNames.Count)
        {
            throw new Exception($"invalid count {result} != {featureNames.Count}");
        }

        if (isEncoderInputSize)
        {
            Encoder_FeatureNames = featureNames;
            _cacheColumns = featureNames.ToArray();
        }
        else
        {
            Decoder_FeatureNames = featureNames;
        }

        return result;
    }


    private TimeSeriesDataset _TestDataset;
    private TimeSeriesDataset _FullTrainingAndValidation;

    public override TimeSeriesDataset TestDataset()
    {
        return LoadAndEncodeDataset_If_Needed().testDataset;
    }

    public override TimeSeriesDataset FullTrainingAndValidation()
    {
        return LoadAndEncodeDataset_If_Needed().fullTrainingAndValidation;
    }



    
    private (TimeSeriesDataset fullTrainingAndValidation, TimeSeriesDataset testDataset) LoadAndEncodeDataset_If_Needed()
    {
        if (_TestDataset != null && _FullTrainingAndValidation != null)
        {
            return (_FullTrainingAndValidation, _TestDataset);
        }

        _FullTrainingAndValidation = new TimeSeriesDataset(
            NAME,
            EntriesTrain,
            encoderDecoder_NetworkSample,
            this,
            null);

        _TestDataset = new TimeSeriesDataset(
            NAME,
            EntriesTest,
            encoderDecoder_NetworkSample,
            this,
            _FullTrainingAndValidation);

        return (_FullTrainingAndValidation, _TestDataset);

    }

    public override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        return EvaluationMetricEnum.Mse;
    }



    public static int DayThreshold(IList<CFM60Entry> entries, double percentageInTrainingSet)
    {
        var sortedDays = entries.Select(e => e.day).OrderBy(x => x).ToArray();
        var countInTraining = (int)(percentageInTrainingSet * entries.Count);
        var dayThreshold = sortedDays[countInTraining];
        return dayThreshold;
    }

    public static IDictionary<int, double> LoadPredictions(string datasetPath)
    {
        var res = new Dictionary<int, double>();
        foreach (var l in File.ReadAllLines(datasetPath).Skip(1))
        {
            var splitted = l.Split(new[] { ';', ',' }, StringSplitOptions.RemoveEmptyEntries);
            var ID = int.Parse(splitted[0]);
            var Y = double.Parse(splitted[1], CultureInfo.InvariantCulture);
            res[ID] = Y;
        }
        return res;
    }



    public static string VectorFeatureName(string vectorName, int indexInVector)
    {
        return vectorName + "[" + indexInVector.ToString("D2") + "]";
    }



    public static void TrainNetwork(int numEpochs = 1, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {

            //related to Dataset 
            //{"KFold", 2},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            
            
            //Related to model
            {"RandomizeOrder", true},
            {"CompatibilityMode", "TensorFlow"},
            {"BatchSize", 2048},
            {"NumEpochs", numEpochs},
            {"InitialLearningRate", 0.002},
            
            {"Use_day", true},
            {"Use_LS", false},
            {"Pid_EmbeddingDim", 8},
            {"DenseUnits", 50},
            {"ClipValueForGradients", 1000},
            {"HiddenSize", 64},
            {"DropoutRate_Between_Encoder_And_Decoder", 0.2},
            {"DropoutRate_After_EncoderDecoder", 0.2},
            {"DisableReduceLROnPlateau", true},
            {"LearningRateSchedulerType", "OneCycle"},
            {"OneCycle_DividerForMinLearningRate", 20},
            {"OneCycle_PercentInAnnealing", 0.1},

            {"OptimizerType", "Adam"},
            {"lambdaL2Regularization", 0.00005},
            
            {"Encoder_NumLayers", 1},
            {"Encoder_TimeSteps", 60},
            {"Encoder_DropoutRate", 0.0},

            {"Decoder_NumLayers", 1},
            {"Decoder_TimeSteps", 1},
            {"Decoder_DropoutRate", 0.0},

            {"Conv1DKernelWidth", 3},
            {"Conv1DPaddingType", "SAME"},
            {"ActivationFunctionAfterFirstDense", "CUDNN_ACTIVATION_CLIPPED_RELU"},

            //run on GPU
            {"EncoderDecoder_NetworkSample_UseGPU", true},
            
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new EncoderDecoder_NetworkSample(), new CFM60DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, true, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

}
