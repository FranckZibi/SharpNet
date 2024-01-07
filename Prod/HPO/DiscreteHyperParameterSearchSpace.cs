using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Linq;

namespace SharpNet.HPO;

public class DiscreteHyperparameterSearchSpace : HyperparameterSearchSpace
{
    #region private fields
    private readonly string[] _allHyperparameterValuesAsString;
    private readonly IDictionary<string, SingleHyperparameterValueStatistics> _statistics = new Dictionary<string, SingleHyperparameterValueStatistics>();
    #endregion

    public DiscreteHyperparameterSearchSpace(object HyperparameterSearchSpace, bool isCategoricalHyperparameter) : base(isCategoricalHyperparameter)
    {
        _allHyperparameterValuesAsString = ToObjectArray(HyperparameterSearchSpace);
        foreach (var e in _allHyperparameterValuesAsString)
        {
            _statistics[e] = new SingleHyperparameterValueStatistics();
        }
    }

    public override bool IsConstant => _allHyperparameterValuesAsString.Length <= 1;

    public override string ToString()
    {
        var res = "";
        var targetInvestmentTime = TargetCpuInvestmentTime();
        foreach (var e in _statistics.OrderBy(e=>(e.Value.CostToDecrease.Count==0)?double.MaxValue:e.Value.CostToDecrease.Average))
        {
            res += " "+e.Key + ":" + e.Value;

            int index = Array.IndexOf(_allHyperparameterValuesAsString, e.Key);
            Debug.Assert(index >= 0);
            res += " (target Time: " + Math.Round(100 * targetInvestmentTime[index], 1) + "%)";

            res += Environment.NewLine;
        }
        return res;
    }

    public override float Next_BayesianSearchFloatValue(Random rand, RANDOM_SEARCH_OPTION randomSearchOption)
    {
        int randomIndex = -1;
        if (randomSearchOption == RANDOM_SEARCH_OPTION.FULLY_RANDOM)
        {
            randomIndex = rand.Next(_allHyperparameterValuesAsString.Length);
        }
        else if (randomSearchOption == RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING)
        {
            var targetInvestmentTime = TargetCpuInvestmentTime();
            randomIndex = Utils.RandomIndexBasedOnWeights(targetInvestmentTime, rand);
        }
        if (randomIndex == -1)
        {
            throw new ArgumentException($"invalid argument {randomSearchOption}");
        }

        if (IsCategoricalHyperparameter)
        {
            return randomIndex;
        }
        return float.Parse(_allHyperparameterValuesAsString[randomIndex], CultureInfo.InvariantCulture);
    }

    public override string BayesianSearchFloatValue_to_SampleStringValue(float f)
    {
        if (IsCategoricalHyperparameter)
        {
            // f is an index
            int index = Utils.NearestInt(f);
            return _allHyperparameterValuesAsString[index];
        }
        return f.ToString(CultureInfo.InvariantCulture);
    }


    public override int LengthForGridSearch => _allHyperparameterValuesAsString.Length;
    public override void RegisterScore(object sampleValue, IScore score, double elapsedTimeInSeconds)
    {
        var parameterValueAsString = Utils.FieldValueToString(sampleValue);
        if (_statistics.TryGetValue(parameterValueAsString, out var key))
        {
            key.RegisterScore(score, elapsedTimeInSeconds);
            //throw new Exception($"invalid value {parameterValueAsString} : can not be found among {string.Join(' ', _statistics.Keys)}");
        }
    }
    public override string SampleStringValue_at_Index_For_GridSearch(int index)
    {
        return _allHyperparameterValuesAsString[index];
    }

    /// <summary>
    /// for each possible value of the Hyper-Parameter '_HyperparameterName'
    /// the % of time (between 0 and 1.0) we are willing to invest on search this specif value.
    ///  => a value close to 1 means we want to invest most of our time on this value (because it seems very promising
    ///  => a value close to 0 means we want to invest very little CPU time on this value (because it doesn't seem use full
    /// </summary>
    /// <returns></returns>
    private double[] TargetCpuInvestmentTime()
    {
        var t = _allHyperparameterValuesAsString.Select(HyperparameterName => _statistics[HyperparameterName]).ToArray();
        return TargetCpuInvestmentTime(t);
    }
    
    private static string[] ToObjectArray(object parameterValues)
    {
        string[] ToStringObjects<T>(IEnumerable<T> values)
        {
            return values.Select(t => Utils.FieldValueToString(t)).ToArray();
        }

        if (parameterValues is bool[])
        {
            return ToStringObjects((bool[])parameterValues);
        }
        if (parameterValues is int[])
        {
            return ToStringObjects((int[])parameterValues);
        }
        if (parameterValues is float[])
        {
            return ToStringObjects((float[])parameterValues);
        }
        if (parameterValues is double[])
        {
            return ToStringObjects((double[])parameterValues);
        }
        if (parameterValues is string[])
        {
            return ToStringObjects((string[])parameterValues);
        }
        if (parameterValues is bool || parameterValues is int || parameterValues is float || parameterValues is double || parameterValues is string)
        {
            return new[] { Utils.FieldValueToString(parameterValues) };
        }
        throw new InvalidEnumArgumentException($"can not process {parameterValues}");
    }
}
