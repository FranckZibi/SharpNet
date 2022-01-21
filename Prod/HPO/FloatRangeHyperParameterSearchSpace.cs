using System;
using System.Diagnostics;

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
        return Next_BayesianSearchFloatValue(_min, _max, rand, _rangeType, randomSearchOption);
    }

    public override bool IsConstant => Math.Abs(_min - _max)<1e-6;
    public override int Length => throw new NotImplementedException();
}
