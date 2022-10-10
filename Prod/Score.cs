using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace SharpNet;

public interface IScore
{
    float Value { get; }
    EvaluationMetricEnum Metric { get; }
    bool HigherIsBetter { get; }

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

    public static IScore Average(IList<IScore> scores)
    {
        if (scores == null)
        {
            return null;
        }
        var avgScore = scores.Select(s => s.Value).Average();
        return new Score(avgScore, scores.First().Metric);
    }
}



public class Score : IScore
{
    public Score(float score, EvaluationMetricEnum metricEnum)
    {
        Value = score;
        Metric = metricEnum;
    }
    public float Value { get; }
    public EvaluationMetricEnum Metric { get; }

    public bool HigherIsBetter => Utils.HigherScoreIsBetter(Metric);

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
        return $"{Value.ToString(CultureInfo.InvariantCulture)} ({Metric})";
    }
}
