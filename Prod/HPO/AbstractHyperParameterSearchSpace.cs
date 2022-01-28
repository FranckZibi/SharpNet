using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.MathTools;

namespace SharpNet.HPO;

public abstract class AbstractHyperParameterSearchSpace
{
    public enum RANDOM_SEARCH_OPTION { FULLY_RANDOM, PREFER_MORE_PROMISING };
    public enum range_type { uniform, normal };

    protected AbstractHyperParameterSearchSpace(bool isCategoricalHyperParameter)
    {
        IsCategoricalHyperParameter = isCategoricalHyperParameter;
    }

    public bool IsCategoricalHyperParameter { get; }
    public static AbstractHyperParameterSearchSpace ValueOf([CanBeNull] object hyperParameterSearchSpace, bool isCategoricalHyperParameter)
    {
        if (hyperParameterSearchSpace is AbstractHyperParameterSearchSpace)
        {
            return (AbstractHyperParameterSearchSpace)hyperParameterSearchSpace;
        }
        return new DiscreteHyperParameterSearchSpace(hyperParameterSearchSpace, isCategoricalHyperParameter);
    }
    public static AbstractHyperParameterSearchSpace Range(float min, float max, range_type rangeType = range_type.uniform)
    {
        return new FloatRangeHyperParameterSearchSpace(min, max, rangeType);
    }
    public static AbstractHyperParameterSearchSpace Range(int min, int max, range_type rangeType = range_type.uniform)
    {
        return new IntRangeHyperParameterSearchSpace(min, max, rangeType);
    }
    public string Next_SampleStringValue(Random rand, RANDOM_SEARCH_OPTION randomSearchOption)
    {
        return BayesianSearchFloatValue_to_SampleStringValue(Next_BayesianSearchFloatValue(rand, randomSearchOption));
    }
    public abstract float Next_BayesianSearchFloatValue(Random rand, RANDOM_SEARCH_OPTION randomSearchOption);
    public abstract string BayesianSearchFloatValue_to_SampleStringValue(float f);
    public abstract void RegisterCost(object sampleValue, float cost, double elapsedTimeInSeconds);
    public abstract bool IsConstant { get; }
    public abstract int LengthForGridSearch { get; }
    public abstract string SampleStringValue_at_Index_For_GridSearch(int index);

    protected static double[] TargetCpuInvestmentTime(IEnumerable<SingleHyperParameterValueStatistics> t)
    {
        Tuple<double, double, int> ToTuple(DoubleAccumulator acc)
        {
            return Tuple.Create(acc.Average, acc.Volatility, acc.Count);
        }
        var allUseCases = t.Select(a => ToTuple(a.Cost)).ToList();
        return Utils.TargetCpuInvestmentTime(allUseCases);
    }


}
