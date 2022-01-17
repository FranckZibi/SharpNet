using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace SharpNet.HPO;

public class HyperParameterSearchSpace
{
    private readonly string _hyperParameterName;
    private readonly string[] _allValuesAsString;
    private readonly IDictionary<string, SingleHyperParameterValueStatistics> _statistics = new Dictionary<string, SingleHyperParameterValueStatistics>();


    public override string ToString()
    {
        var res = "";
        foreach (var e in _statistics.OrderBy(e=>(e.Value.Errors.Count==0)?double.MaxValue:e.Value.Errors.Average))
        {
            res += " "+e.Key + ":" + e.Value + Environment.NewLine;
        }
        return res;
    }

    public HyperParameterSearchSpace(string hyperParameterName, object hyperParameterSearchSpace)
    {
        _hyperParameterName = hyperParameterName;
        _allValuesAsString = ToObjectArray(hyperParameterSearchSpace);
        foreach (var e in _allValuesAsString)
        {
            _statistics[e] = new SingleHyperParameterValueStatistics();
        }
    }

    public int Length => _allValuesAsString.Length;


    public void RegisterError(object parameterValue, double error)
    {
        ParameterValueToStatistics(parameterValue).RegisterError(error);
    }

    private SingleHyperParameterValueStatistics ParameterValueToStatistics(object parameterValue)
    {
        var parameterValueAsString = ClassFieldSetter.FieldValueToString(parameterValue);
        if (!_statistics.ContainsKey(parameterValueAsString))
        {
            throw new Exception($"invalid value {parameterValueAsString} : can not be found among {string.Join(' ', _statistics.Keys)} for {_hyperParameterName}");
        }
        return _statistics[parameterValueAsString];
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
            return new[]{ ClassFieldSetter.FieldValueToString(parameterValues) };
        }
        throw new InvalidEnumArgumentException($"can not process {parameterValues}");
    }

    public string ExtractParameterValueForIndex(int index)
    {
        return _allValuesAsString[index];
    }
}