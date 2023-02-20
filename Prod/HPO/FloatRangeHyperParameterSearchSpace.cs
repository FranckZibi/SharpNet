using System;
using System.Diagnostics;
using System.Globalization;

namespace SharpNet.HPO;

public class FloatRangeHyperParameterSearchSpace : AbstractRangeHyperParameterSearchSpace
{
    #region private fields
    private readonly float _min;
    private readonly float _max;
    #endregion

    public FloatRangeHyperParameterSearchSpace(float min, float max, range_type rangeType) : base(rangeType)
    {
        Debug.Assert(max>=min);
        _min = min;
        _max = max;
    }

    public override float Next_BayesianSearchFloatValue(Random rand, RANDOM_SEARCH_OPTION randomSearchOption)
    {
        return Next_BayesianSearchFloatValue(_min, _max, rand, _rangeType, randomSearchOption, StatsByBucket);
    }
    public override void RegisterScore(object sampleValue, IScore score, double elapsedTimeInSeconds)
    {
        int bucketIndex = SampleValueToBucketIndex(SampleValueToFloat(sampleValue), _min, _max, _rangeType);
        StatsByBucket[bucketIndex].RegisterScore(score, elapsedTimeInSeconds);
    }
    public override string SampleStringValue_at_Index_For_GridSearch(int index)
    {
        return ((LowerValueForBucket(index) + UpperValueForBucket(index)) / 2).ToString(CultureInfo.InvariantCulture);
    }

    public override bool IsConstant => MathF.Abs(_min - _max)<1e-6f;
    protected override float LowerValueForBucket(int bucketIndex)
    {
        return BucketIndexToBucketLowerValue(bucketIndex, _min, _max, _rangeType);
    }
    protected override float UpperValueForBucket(int bucketIndex)
    {
        return BucketIndexToBucketUpperValue(bucketIndex, _min, _max, _rangeType);
    }
}
