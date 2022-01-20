using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using SharpNet.MathTools;

namespace SharpNet.HPO;



public class HyperParameterSearchSpace
{
    #region private fields
    private readonly string _hyperParameterName;
    private readonly string[] _allHyperParameterValuesAsString;
    private readonly IDictionary<string, SingleHyperParameterValueStatistics> _statistics = new Dictionary<string, SingleHyperParameterValueStatistics>();
    #endregion

    public enum RANDOM_SEARCH_OPTION { FULLY_RANDOM, PREFER_MORE_PROMISING };


    public HyperParameterSearchSpace(string hyperParameterName, object hyperParameterSearchSpace)
    {
        _hyperParameterName = hyperParameterName;
        _allHyperParameterValuesAsString = ToObjectArray(hyperParameterSearchSpace);
        foreach (var e in _allHyperParameterValuesAsString)
        {
            _statistics[e] = new SingleHyperParameterValueStatistics();
        }
    }

    public override string ToString()
    {
        var res = "";
        var targetInvestmentTime = TargetCpuInvestmentTime();
        foreach (var e in _statistics.OrderBy(e=>(e.Value.Cost.Count==0)?double.MaxValue:e.Value.Cost.Average))
        {
            res += " "+e.Key + ":" + e.Value;

            int index = Array.IndexOf(_allHyperParameterValuesAsString, e.Key);
            Debug.Assert(index >= 0);
            res += " (target Time: " + Math.Round(100 * targetInvestmentTime[index], 1) + "%)";

            res += Environment.NewLine;
        }
        return res;
    }
    public string GetRandomSearchSpaceHyperParameterStringValue(Random rand, RANDOM_SEARCH_OPTION randomSearchOption)
    {
        if (randomSearchOption == RANDOM_SEARCH_OPTION.FULLY_RANDOM)
        {
            int randomIndex = rand.Next(Length);
            return _allHyperParameterValuesAsString[randomIndex];
        }
        if (randomSearchOption == RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING)
        {
            var targetInvestmentTime = TargetCpuInvestmentTime();
            int randomIndex = Utils.RandomIndexBasedOnWeights(targetInvestmentTime, rand);
            return _allHyperParameterValuesAsString[randomIndex];
        }

        throw new ArgumentException($"invalid argument {randomSearchOption}");
    }
    public int Length => _allHyperParameterValuesAsString.Length;
    public void RegisterCost(object parameterValue, double cost, double elapsedTimeInSeconds)
    {
        HyperParameterValueToStatistics(parameterValue).RegisterCost(cost, elapsedTimeInSeconds);
    }
    public string HyperParameterStringValueAtIndex(int index)
    {
        return _allHyperParameterValuesAsString[index];
    }
    private SingleHyperParameterValueStatistics HyperParameterValueToStatistics(object parameterValue)
    {
        var parameterValueAsString = ClassFieldSetter.FieldValueToString(parameterValue);
        if (!_statistics.ContainsKey(parameterValueAsString))
        {
            throw new Exception($"invalid value {parameterValueAsString} : can not be found among {string.Join(' ', _statistics.Keys)} for {_hyperParameterName}");
        }
        return _statistics[parameterValueAsString];
    }

    /// <summary>
    /// for each possible value of the Hyper-Parameter '_hyperParameterName'
    /// the % of time (between 0 and 1.0) we are willing to invest on search this specif value.
    ///  => a value close to 1 means we want to invest most of our time on this value (because it seems very promising
    ///  => a value close to 0 means we want to invest very little CPU time on this value (because it doesn't seem use full
    /// </summary>
    /// <returns></returns>
    private double[] TargetCpuInvestmentTime()
    {
        Tuple<double, double, int> ToTuple(DoubleAccumulator acc)
        {
            return Tuple.Create(acc.Average, acc.Volatility, acc.Count);
        }
        var allUseCases = _allHyperParameterValuesAsString.Select(n => ToTuple(_statistics[n].Cost)).ToList();
        return Utils.TargetCpuInvestmentTime(allUseCases);
    }
    private static string[] ToObjectArray(object parameterValues)
    {
        string[] ToStringObjects<T>(IEnumerable<T> values)
        {
            return values.Select(t => ClassFieldSetter.FieldValueToString(t)).ToArray();
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
            return new[] { ClassFieldSetter.FieldValueToString(parameterValues) };
        }
        throw new InvalidEnumArgumentException($"can not process {parameterValues}");
    }
}
