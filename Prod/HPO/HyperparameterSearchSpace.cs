using System;
using System.Collections.Generic;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.MathTools;

namespace SharpNet.HPO;

public abstract class HyperparameterSearchSpace
{
    public enum RANDOM_SEARCH_OPTION { FULLY_RANDOM, PREFER_MORE_PROMISING };
    public enum range_type { uniform, normal };

    protected HyperparameterSearchSpace(bool isCategoricalHyperparameter)
    {
        IsCategoricalHyperparameter = isCategoricalHyperparameter;
    }

    public bool IsCategoricalHyperparameter { get; }
    public static HyperparameterSearchSpace ValueOf([CanBeNull] object HyperparameterSearchSpace, bool isCategoricalHyperparameter)
    {
        if (HyperparameterSearchSpace is HyperparameterSearchSpace)
        {
            return (HyperparameterSearchSpace)HyperparameterSearchSpace;
        }
        return new DiscreteHyperparameterSearchSpace(HyperparameterSearchSpace, isCategoricalHyperparameter);
    }
    public static HyperparameterSearchSpace Range(float min, float max, range_type rangeType = range_type.uniform)
    {
        return new FloatRangeHyperparameterSearchSpace(min, max, rangeType);
    }
    public static HyperparameterSearchSpace Range(int min, int max, range_type rangeType = range_type.uniform)
    {
        return new IntRangeHyperparameterSearchSpace(min, max, rangeType);
    }
    public string Next_SampleStringValue(Random rand, RANDOM_SEARCH_OPTION randomSearchOption)
    {
        return BayesianSearchFloatValue_to_SampleStringValue(Next_BayesianSearchFloatValue(rand, randomSearchOption));
    }
    public abstract float Next_BayesianSearchFloatValue(Random rand, RANDOM_SEARCH_OPTION randomSearchOption);
    public abstract string BayesianSearchFloatValue_to_SampleStringValue(float f);
    public abstract void RegisterScore(object sampleValue, IScore score, double elapsedTimeInSeconds);
    public abstract bool IsConstant { get; }
    public abstract int LengthForGridSearch { get; }
    public abstract string SampleStringValue_at_Index_For_GridSearch(int index);

    protected static double[] TargetCpuInvestmentTime(IEnumerable<SingleHyperparameterValueStatistics> t)
    {
        Tuple<double, double, long> ToTuple(DoubleAccumulator acc)
        {
            return Tuple.Create(acc.Average, acc.Volatility, acc.Count);
        }
        var allUseCases = t.Select(a => ToTuple(a.CostToDecrease)).ToList();
        return Utils.TargetCpuInvestmentTime(allUseCases);
    }


}
