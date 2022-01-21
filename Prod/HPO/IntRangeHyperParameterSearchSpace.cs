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
        var floatValue = Next_BayesianSearchFloatValue(_min, _max, rand, _rangeType, randomSearchOption);
        return Utils.NearestInt(floatValue);
    }

    public override bool IsConstant => _min == _max;
    public override int Length => _max-_min+1;
}
