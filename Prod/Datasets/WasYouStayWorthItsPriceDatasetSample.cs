using System;
using System.Collections.Generic;
using System.IO;
using SharpNet.HyperParameters;

namespace SharpNet.Datasets;

public class WasYouStayWorthItsPriceDatasetSample : AbstractDatasetSample
{
    #region private fields
    private const string NAME = "WasYouStayWorthItsPrice";
    private readonly DatasetEncoder _trainTestEncoder;
    private readonly DataFrameT<float> _trainEncoded;
    private readonly DataFrameT<float> _testEncoded;
    //private readonly DatasetEncoder _reviewsEncoder;
    //private readonly DataFrame<float> _reviewEncoded;
    private static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    private static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    #endregion


    public WasYouStayWorthItsPriceDatasetSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(mandatoryCategoricalHyperParameters)
    {
        _trainTestEncoder = new DatasetEncoder(CategoricalFeatures(), IdFeatures(), TargetFeatures());
        _trainEncoded = _trainTestEncoder.NumericalEncoding(XYTrainRawFile);
        _testEncoded = _trainTestEncoder.NumericalEncoding(XTestRawFile);
        //_reviewsEncoder = new DatasetEncoder(new List<string>{"id", "listing_id", "renters_comments"}, new List<string>{"id"}, new List<string>());
        //_reviewEncoded = _reviewsEncoder.NumericalEncoding(ReviewsRawFile);
    }

    public override List<string> CategoricalFeatures()
    {
        return new List<string> { "id", "host_2", "host_3", "host_4", "host_5", "property_10", "property_15", "property_4", "property_5", "property_7", "max_rating_class" };
    }

    public override List<string> IdFeatures()
    {
        return new List<string> { "id" };
    }

    public override List<string> TargetFeatures()
    {
        return new List<string> { "max_rating_class" }; ;
    }

    public override IDataSet TestDataset()
    {
        throw new NotImplementedException();
    }

    public override ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
    {
        throw new NotImplementedException();
    }

    public override DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat)
    {
        throw new NotImplementedException();
    }

    private static string XYTrainRawFile => Path.Combine(DataDirectory, "train.csv");
    private static string XTestRawFile => Path.Combine(DataDirectory, "test.csv");
    //private static string ReviewsRawFile => Path.Combine(DataDirectory, "reviews.csv");
}
