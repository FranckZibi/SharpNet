﻿using System;
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
    public override void RegisterCost(object sampleValue, float cost, double elapsedTimeInSeconds)
    {
        int bucketIndex = SampleValueToBucketIndex(SampleValueToFloat(sampleValue), _min, _max, _rangeType);
        StatsByBucket[bucketIndex].RegisterCost(cost, elapsedTimeInSeconds);
    }
    public override string SampleStringValue_at_Index_For_GridSearch(int index)
    {
        var lowerValue  = BucketIndexToBucketLowerValue(index, _min, _max, _rangeType);
        var upperValue  = BucketIndexToBucketUpperValue(index, _min, _max, _rangeType);
        return ((lowerValue + upperValue) / 2).ToString(CultureInfo.InvariantCulture);
    }
    public override bool IsConstant => Math.Abs(_min - _max)<1e-6;
}