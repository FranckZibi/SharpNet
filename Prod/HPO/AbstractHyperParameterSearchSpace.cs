using System;
using JetBrains.Annotations;

namespace SharpNet.HPO;

public abstract class AbstractHyperParameterSearchSpace
{
    public enum RANDOM_SEARCH_OPTION { FULLY_RANDOM, PREFER_MORE_PROMISING };
    public enum range_type { uniform, normal };

    public static AbstractHyperParameterSearchSpace ValueOf([CanBeNull] object hyperParameterSearchSpace)
    {
        if (hyperParameterSearchSpace is AbstractHyperParameterSearchSpace)
        {
            return (AbstractHyperParameterSearchSpace)hyperParameterSearchSpace;
        }
        return new DiscreteHyperParameterSearchSpace(hyperParameterSearchSpace);
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
    public abstract bool IsCategoricalHyperParameter { get; }
    public abstract bool IsConstant { get; }
    public abstract int Length { get; }
    public abstract string SampleStringValue_at_Index(int index);
}
