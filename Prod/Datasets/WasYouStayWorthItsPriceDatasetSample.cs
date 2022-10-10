using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;

namespace SharpNet.Datasets;

public class WasYouStayWorthItsPriceDatasetSample : AbstractDatasetSample
{
    #region private fields
    private const string NAME = "WasYouStayWorthItsPrice";


    private static readonly object LockObject = new();
    private static DatasetEncoder _trainTestEncoder;
    private static IDataSet _fullTrainDatasetEncoded;
    private static IDataSet _fullTestDatasetEncoded;

    //private readonly DatasetEncoder _reviewsEncoder;
    //private readonly DataFrameT<float> _reviewEncoded;
    private static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    private static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    #endregion

    private WasYouStayWorthItsPriceDatasetSample() : base(new HashSet<string>())
    {
        lock (LockObject)
        {
            if (_trainTestEncoder == null)
            {
                _trainTestEncoder = new DatasetEncoder(this);
                _fullTrainDatasetEncoded = _trainTestEncoder.NumericalEncoding(XYTrainRawFile);
                _fullTestDatasetEncoded = _trainTestEncoder.NumericalEncoding(XTestRawFile, null);
            }
        }

        //_fullTrainDatasetEncoded.to_csv(@"C:\Projects\Challenges\WasYouStayWorthItsPrice\yx_train.csv", GetSeparator(), true, true);
        //_fullTestDatasetEncoded.to_csv(@"C:\Projects\Challenges\WasYouStayWorthItsPrice\x_test.csv", GetSeparator(), false, true);

        //_reviewsEncoder = new DatasetEncoder(new List<string>{"id", "listing_id", "renters_comments"}, new List<string>{"id"}, new List<string>());
        //_reviewEncoded = _reviewsEncoder.NumericalEncoding(ReviewsRawFile);
        //_xTrainEncoded.T
    }

    private int NumClasses() => _trainTestEncoder.NumClasses();
    public static void LaunchCatBoostHPO()
    {
        // ReSharper disable once ConvertToConstant.Local
        var iterations = 100;
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            //{"KFold", new[]{1,3}},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled

            //related to CatBoost model
            { "loss_function", "MultiClass"},
            { "logging_level", "Silent"},
            { "allow_writing_files",false},
            { "thread_count",1},
            { "iterations", iterations },
            { "od_type", "Iter"},
            { "od_wait",iterations/10},
            { "depth", AbstractHyperParameterSearchSpace.Range(2, 5) },
            //{ "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.10f)},
            { "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
            //{ "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
            //{ "l2_leaf_reg",AbstractHyperParameterSearchSpace.Range(0f, 10f)},
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample(), new WasYouStayWorthItsPriceDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        var csvPath = Path.Combine(DataDirectory, "Tests_" + NAME + ".csv");
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, csvPath, ref bestScoreSoFar));
    }

    public static void LaunchLightGBMHPO()
    {
        // ReSharper disable once ConvertToConstant.Local
        var min_num_iterations = 50;
        var numClasses = new WasYouStayWorthItsPriceDatasetSample().NumClasses();

        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            //{"KFold", new[]{1,3,5}},
            {"KFold", new[]{3}},

            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            //{"KFold", 1},

            //{"FromProbaDistributionToPredictedCategory_Method", new[]{0, 1, 2}},
            
            /*
            //{"boosting", "dart"},
            //dart mode
            //{"drop_rate", new[]{0.05, 0.1, 0.2}},
            {"max_drop", new[]{40, 50, 60}},
            {"skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},
            //{"xgboost_dart_mode", new[]{false, true}},
            //{"uniform_drop", new[]{false, true}},
            //{"drop_seed", 4},
            */

            //related to LightGBM model
            { "metric", "multi_logloss" },
            { "objective", "multiclass" },
            //{ "num_iterations", AbstractHyperParameterSearchSpace.Range(min_num_iterations, 3*min_num_iterations) },
            { "num_iterations", min_num_iterations },
            { "num_class", numClasses },
            { "verbosity", "0" },
            { "num_threads", 1},
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.2f)},
            { "extra_trees", false },
            { "early_stopping_round", min_num_iterations/10 },
            { "bagging_fraction", new[]{0.9f, 1.0f} },
            { "bagging_freq", new[]{0, 1} },
            { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
            //{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            { "max_depth", new[]{10, 20, 50, 100, 255} },
            //{ "min_data_in_bin", AbstractHyperParameterSearchSpace.Range(3, 100) },
            { "min_data_in_bin", new[]{2, 3, 10, 100, 150}  },
            //{ "min_data_in_leaf", AbstractHyperParameterSearchSpace.Range(20, 200) },
            //{ "min_data_in_leaf", new[]{10, 20, 30} },
            //{ "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(5, 60) },
            //{ "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) },
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample(), new WasYouStayWorthItsPriceDatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        var csvPath = Path.Combine(DataDirectory, "Tests_" + NAME + ".csv");
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, csvPath, ref bestScoreSoFar), 100);
    }

    public override Objective_enum GetObjective() => Objective_enum.Classification;

    //public override EvaluationMetricEnum GetLossFunction()
    //{
    //    return EvaluationMetricEnum.CategoricalCrossentropy;
    //}

    protected override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        return EvaluationMetricEnum.F1Micro;
    }



    public override List<string> CategoricalFeatures()
    {
        return new List<string> { "host_2", "host_3", "host_4", "host_5", "property_10", "property_15", "property_4", "property_5", "property_7"};
    }

    public override List<string> IdColumns()
    {
        return new List<string> { "id" };
    }

    public override List<string> TargetLabels()
    {
        return new List<string> { "max_rating_class" }; ;
    }

    public override IDataSet TestDataset()
    {
        return _fullTestDatasetEncoded;
    }

    public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
    {
        int rowsForTraining = (int)(PercentageInTraining * _fullTrainDatasetEncoded.Count + 0.1);
        rowsForTraining -= rowsForTraining % DatasetRowsInModelFormatMustBeMultipleOf();
        return _fullTrainDatasetEncoded.IntSplitIntoTrainingAndValidation(rowsForTraining);

    }

    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat_with_IdColumns)
    {
        Debug.Assert(TargetLabels().Count == 1);
        var (predictionsInModelFormat_without_Ids, idDataFrame) = predictionsInModelFormat_with_IdColumns.Split(IdColumns());
        var cpuTensor = FromProbaDistributionToPredictedCategory(predictionsInModelFormat_without_Ids.FloatCpuTensor());
        var probaDataFrame_InTargetFormat = DataFrame.New(cpuTensor, TargetLabels());
        var predictionsInTargetFormat_encoded = DataFrame.MergeHorizontally(idDataFrame, probaDataFrame_InTargetFormat);
        return predictionsInTargetFormat_encoded;
    }


    private readonly int FromProbaDistributionToPredictedCategory_Method = 0;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="predictionsInModelFormat_without_Ids">a tensor of shape (batchSize, num_classes) </param>
    /// <returns>a tensor of shape (batchSize, 1)</returns>
    private CpuTensor<float> FromProbaDistributionToPredictedCategory(CpuTensor<float> predictionsInModelFormat_without_Ids)
    {
        Debug.Assert(predictionsInModelFormat_without_Ids.Shape.Length == 2);
        var rows = predictionsInModelFormat_without_Ids.Shape[0];
        var columns = predictionsInModelFormat_without_Ids.Shape[1];
        Debug.Assert(_trainTestEncoder.NumClasses() == columns);
        if (FromProbaDistributionToPredictedCategory_Method == 0)
        {
            return CpuTensor<float>.ArgMax(predictionsInModelFormat_without_Ids);
        }

        var count = _trainTestEncoder.GetDistinctCategoricalCount(TargetLabels()[0]);
        Debug.Assert(count.Count == columns);
        float sumCount = count.Sum();
        List<Tuple<int, float>> percentageInEachClass = new();
        for (var index = 0; index < count.Count; index++)
        {
            percentageInEachClass.Add(Tuple.Create(index, count[index]/sumCount));
        }

        float[] content = Enumerable.Repeat(float.NaN, rows).ToArray();
        percentageInEachClass = (FromProbaDistributionToPredictedCategory_Method == 1)
            ?percentageInEachClass.OrderByDescending(t => t.Item2).ToList()
            :percentageInEachClass.OrderBy(t => t.Item2).ToList();

        var observedValue = new List<float>(rows);
        var probaContent = predictionsInModelFormat_without_Ids.ReadonlyContent;
        for (int iClass = 0; iClass < percentageInEachClass.Count - 1; ++iClass)
        {
            observedValue.Clear();
            int classIndex = percentageInEachClass[iClass].Item1;
            float classPercentage = percentageInEachClass[iClass].Item2;

            //we compute the proba threshold for class 'classIndex'
            //all non classified element with an observed proba greater than this threshold will be predicted as 'classIndex'
            for (int row = 0; row < rows; ++row)
            {
                if (float.IsNaN(content[row]))
                {
                    observedValue.Add(probaContent[row * columns + classIndex]);
                }
            }
            if (observedValue.Count == 0)
            {
                continue;
            }
            observedValue.Sort();
            observedValue.Reverse();
            int idxInSortedObservedValue = (int)(classPercentage * rows);
            idxInSortedObservedValue = Math.Min(idxInSortedObservedValue, observedValue.Count - 1);
            var probaThreshold = observedValue[idxInSortedObservedValue];

            for (int row = 0; row < rows; ++row)
            {
                if (float.IsNaN(content[row]) && probaContent[row * columns + classIndex]>= probaThreshold)
                {
                    content[row] = classIndex;
                }
            }
        }

        for (int row = 0; row < rows; ++row)
        {
            if (float.IsNaN(content[row]))
            {
                content[row] = percentageInEachClass.Last().Item1;
            }
        }

        //predictionsInModelFormat_without_Ids.Transpose();



        return new CpuTensor<float>(new int[]{rows, 1}, content);
    }

    public override void SavePredictionsInTargetFormat(DataFrame predictionsInTargetFormat, string path)
    {
        var csvContent = _trainTestEncoder.NumericalDecoding(predictionsInTargetFormat, GetSeparator(), "");
        File.WriteAllText(path, csvContent);
    }



    private static string XYTrainRawFile => Path.Combine(DataDirectory, "train.csv");
    private static string XTestRawFile => Path.Combine(DataDirectory, "test.csv");
    //private static string ReviewsRawFile => Path.Combine(DataDirectory, "reviews.csv");
}
