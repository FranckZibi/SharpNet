using System;
using System.Diagnostics;
using System.Globalization;

namespace SharpNet.HPO;

public abstract class AbstractRangeHyperParameterSearchSpace : AbstractHyperParameterSearchSpace
{
    #region private fields
    protected readonly range_type _rangeType;

    protected readonly SingleHyperParameterValueStatistics _singleHyperParameterValueStatistics =  new SingleHyperParameterValueStatistics();


    #endregion

    protected AbstractRangeHyperParameterSearchSpace(range_type rangeType)
    {
        _rangeType = rangeType;
    }
   
    public override string BayesianSearchFloatValue_to_SampleStringValue(float f)
    {
        return f.ToString(CultureInfo.InvariantCulture);
    }
    public override string SampleStringValue_at_Index(int index)
    {
        throw new NotImplementedException();
    }

    protected static float Next_BayesianSearchFloatValue(float min, float max, Random rand, range_type rangeType, RANDOM_SEARCH_OPTION randomSearchOption)
    {
        if (Math.Abs(max - min) < 1e-6)
        {
            return max;
        }

        if (randomSearchOption != RANDOM_SEARCH_OPTION.FULLY_RANDOM)
        {
            throw new NotImplementedException($"randomSearchOption {randomSearchOption} is not supported");
        }
        if (rangeType == range_type.uniform)
        {
            return min + (max - min) * rand.NextSingle();
        }
        if (rangeType == range_type.normal)
        {
            return NormalValue(min, max, rand.NextDouble());
        }
        throw new NotImplementedException($"rangeType {rangeType} is not supported");
    }

    private static float NormalValue(float min, float max, double randomUniformBetween_0_and_1)
    {
        Debug.Assert(max>=min);
        Debug.Assert(randomUniformBetween_0_and_1>=0);
        Debug.Assert(randomUniformBetween_0_and_1<=1);
        const double epsilon = 1e-6;
        var logSampleValue = Math.Log(epsilon) + (Math.Log(max - min + epsilon) - Math.Log(epsilon)) * randomUniformBetween_0_and_1;
        return (float)(Math.Exp(logSampleValue) - epsilon);
    }

    public override void RegisterCost(object sampleValue, float cost, double elapsedTimeInSeconds)
    {
        //TODO
        _singleHyperParameterValueStatistics.RegisterCost(cost, elapsedTimeInSeconds);
    }

    public override string ToString()
    {
        //TODO
        return "N/A"+Environment.NewLine;
    }

    public override bool IsCategoricalHyperParameter => false;
}