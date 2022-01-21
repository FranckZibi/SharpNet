using System;
using System.Diagnostics;

namespace SharpNet.HPO;

public class IntRangeHyperParameterSearchSpace : AbstractRangeHyperParameterSearchSpace
{
    #region private fields
    private readonly int _min;
    private readonly int _max;
    #endregion

    public IntRangeHyperParameterSearchSpace(int min, int max, range_type rangeType) : base(rangeType)
    {
        Debug.Assert(max>=min);
        _min = min;
        _max = max;
    }

    public override float Next_BayesianSearchFloatValue(Random rand, RANDOM_SEARCH_OPTION randomSearchOption)
    {
        var floatValue = Next_BayesianSearchFloatValue(_min, _max, rand, _rangeType, randomSearchOption, StatsByBucket);
        return Utils.NearestInt(floatValue);
    }
    public override void RegisterCost(object sampleValue, float cost, double elapsedTimeInSeconds)
    {
        int bucketIndex = SampleValueToBucketIndex(SampleValueToFloat(sampleValue), _min, _max, _rangeType);
        StatsByBucket[bucketIndex].RegisterCost(cost, elapsedTimeInSeconds);
    }
    public override string SampleStringValue_at_Index_For_GridSearch(int index)
    {
        var lowerValue = BucketIndexToBucketLowerValue(index, _min, _max, _rangeType);
        var upperValue = BucketIndexToBucketUpperValue(index, _min, _max, _rangeType);
        var sampleValue = Utils.NearestInt((lowerValue + upperValue) / 2);
        return sampleValue.ToString();
    }

    public override bool IsConstant => _min == _max;
}
