using System;
using System.Globalization;

namespace SharpNet;

public interface IScore
{
    float Value { get; }
    MetricEnum Metric { get; }

    bool IsBetterThan(IScore other);
    public static string ToString(IScore score, int digitsForRounding = -1)
    {
        if (score == null)
        {
            return "";
        }
        var val = score.Value;
        if (digitsForRounding >= 0)
        {
            val = (float)Math.Round(val, digitsForRounding);
        }
        return val.ToString(CultureInfo.InvariantCulture);
    }
}



public class Score : IScore
{
    public Score(float score, MetricEnum metricEnum)
    {
        Value = score;
        Metric = metricEnum;
    }
    public float Value { get; }
    public MetricEnum Metric { get; }

    public bool IsBetterThan(IScore other)
    {
        if (other == null)
        {
            return true;
        }
        if (Metric != other.Metric)
        {
            throw new ArgumentException($"can't compare score between {Metric} and {other.Metric}");
        }
        return Utils.IsBetterScore(Value, other.Value, Metric);
    }

    public override string ToString()
    {
        return $"{Value.ToString(CultureInfo.InvariantCulture)} (metric={Metric})";
    }
}
