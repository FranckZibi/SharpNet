using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using SharpNet.CPU;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;

namespace SharpNet.Datasets;

public class WasYouStayWorthItsPriceDatasetSample : AbstractDatasetSample
{
    #region private fields
    private const string NAME = "WasYouStayWorthItsPrice";
    private readonly DatasetEncoder _trainTestEncoder;
    private readonly InMemoryDataSet _fullTrainDatasetEncoded;
    private readonly InMemoryDataSet _fullTestDatasetEncoded;

    //private readonly DatasetEncoder _reviewsEncoder;
    //private readonly DataFrameT<float> _reviewEncoded;
    private static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    private static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    #endregion

    private WasYouStayWorthItsPriceDatasetSample() : base(new HashSet<string>())
    {
        _trainTestEncoder = new DatasetEncoder(this);

        _fullTrainDatasetEncoded = _trainTestEncoder.NumericalEncoding(XYTrainRawFile);
        _fullTestDatasetEncoded = _trainTestEncoder.NumericalEncoding(XTestRawFile, null);

        //_fullTrainDatasetEncoded.to_csv(@"C:\Projects\Challenges\WasYouStayWorthItsPrice\yx_train.csv", GetSeparator(), true, true);
        //_fullTestDatasetEncoded.to_csv(@"C:\Projects\Challenges\WasYouStayWorthItsPrice\x_test.csv", GetSeparator(), false, true);

        //_reviewsEncoder = new DatasetEncoder(new List<string>{"id", "listing_id", "renters_comments"}, new List<string>{"id"}, new List<string>());
        //_reviewEncoded = _reviewsEncoder.NumericalEncoding(ReviewsRawFile);
        //_xTrainEncoded.T
    }

    public static void LaunchLightGBMHPO()
    {
        var datasetSample = new WasYouStayWorthItsPriceDatasetSample();

        // ReSharper disable once ConvertToConstant.Local
        var min_num_iterations = 300;


        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 

            //related to LightGBM model
            { "metric", "multi_logloss" },
            { "objective", "multiclass" },
            //{ "num_iterations", AbstractHyperParameterSearchSpace.Range(min_num_iterations, 3*min_num_iterations) },
            { "num_iterations", min_num_iterations },
            { "num_class", datasetSample._trainTestEncoder.NumClasses() },
            { "verbosity", "0" },
            { "num_threads", 1},
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 0.1f)},
            { "extra_trees", new[] { true , false} },
            { "early_stopping_round", min_num_iterations/5 },
            { "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
            { "bagging_freq", new[]{0, 1} },
            { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
            { "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)},
            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 1f)},
            { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 1f)},
            { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            { "max_depth", new[]{10, 20, 50, 100, 255} },
            { "min_data_in_bin", AbstractHyperParameterSearchSpace.Range(3, 100) },
            { "min_data_in_leaf", AbstractHyperParameterSearchSpace.Range(20, 200) },
            { "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(5, 60) },
            { "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) },
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => { return ModelAndDatasetPredictionsSample.New(new LightGBMSample(), datasetSample);
        }, WorkingDirectory);
        IScore bestScoreSoFar = null;
        var csvPath = Path.Combine(DataDirectory, "Tests_" + NAME + ".csv");
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, csvPath, ref bestScoreSoFar)
        );
    }

    public override void SavePredictionsInTargetFormat(DataFrame predictionsInTargetFormat, string path)
    {
        var csvContent = _trainTestEncoder.NumericalDecoding(predictionsInTargetFormat, GetSeparator(), "");
        File.WriteAllText(path, csvContent);
    }

    public override Objective_enum GetObjective() => Objective_enum.Classification;

    public override LossFunctionEnum GetLoss()
    {
        return LossFunctionEnum.CategoricalCrossentropy;
    }
    public override MetricEnum GetMetric()
    {
        return MetricEnum.F1Micro;
    }



    public override List<string> CategoricalFeatures()
    {
        return new List<string> { "id", "host_2", "host_3", "host_4", "host_5", "property_10", "property_15", "property_4", "property_5", "property_7"};
    }

    public override List<string> IdFeatures()
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

    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat)
    {
        Debug.Assert(TargetLabels().Count == 1);
        var (predictionsInModelFormat_without_Ids, idDataFrame) = predictionsInModelFormat.Split(IdFeatures());
        var cpuTensor = CpuTensor<float>.ArgMax(predictionsInModelFormat_without_Ids.FloatCpuTensor());
        var probaDataFrame_InTargetFormat = DataFrame.New(cpuTensor, TargetLabels(), Array.Empty<string>());
        var predictionsInTargetFormat_encoded = DataFrame.MergeHorizontally(idDataFrame, probaDataFrame_InTargetFormat);
        return predictionsInTargetFormat_encoded;
    }

    private static string XYTrainRawFile => Path.Combine(DataDirectory, "train.csv");
    private static string XTestRawFile => Path.Combine(DataDirectory, "test.csv");
    //private static string ReviewsRawFile => Path.Combine(DataDirectory, "reviews.csv");
}
