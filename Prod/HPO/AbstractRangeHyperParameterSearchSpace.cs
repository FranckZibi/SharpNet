using System;
using System.Diagnostics;
using System.Globalization;
using System.Linq;

namespace SharpNet.HPO;

public abstract class AbstractRangeHyperParameterSearchSpace : AbstractHyperParameterSearchSpace
{
    #region private fields
    private const int BucketCount = 5;
    protected readonly range_type _rangeType;
    protected readonly SingleHyperParameterValueStatistics[] StatsByBucket;
    #endregion
   
    public override string BayesianSearchFloatValue_to_SampleStringValue(float f)
    {
        return f.ToString(CultureInfo.InvariantCulture);
    }
    public override string ToString()
    {
        var res = "";
        var targetInvestmentTime = TargetCpuInvestmentTime(StatsByBucket);

        var stats = StatsByBucket.Select((t, i) => Tuple.Create(t, i)).ToList();
        foreach (var e in stats.OrderBy(e => (e.Item1.CostToDecrease.Count == 0) ? double.MaxValue : e.Item1.CostToDecrease.Average))
        {
            int bucketIndex = e.Item2;
            res += BucketDescription(bucketIndex) + ":" + e.Item1;
            res += " (target Time: " + Math.Round(100 * targetInvestmentTime[bucketIndex], 1) + "%)";
            res += Environment.NewLine;
        }
        return res;
    }
    public override int LengthForGridSearch => BucketCount;

    protected AbstractRangeHyperParameterSearchSpace(range_type rangeType) : base(false)
    {
        _rangeType = rangeType;
        StatsByBucket = new SingleHyperParameterValueStatistics[BucketCount];
        for (int i = 0; i < StatsByBucket.Length; ++i)
        {
            StatsByBucket[i] = new SingleHyperParameterValueStatistics();
        }
    }
    protected static float Next_BayesianSearchFloatValue(float min, float max, Random rand, range_type rangeType, RANDOM_SEARCH_OPTION randomSearchOption, SingleHyperParameterValueStatistics[] statsByBucket)
    {
        Debug.Assert(max>=min);
        if (Math.Abs(max - min) < 1e-6)
        {
            return max;
        }
        if (randomSearchOption == RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING)
        {
            var targetInvestmentTime = TargetCpuInvestmentTime(statsByBucket);
            int randomIndex = Utils.RandomIndexBasedOnWeights(targetInvestmentTime, rand);
            var minValueInBucket = BucketIndexToBucketLowerValue(randomIndex, min, max, rangeType);
            var maxValueInBucket = BucketIndexToBucketUpperValue(randomIndex, min, max, rangeType);
            Debug.Assert(maxValueInBucket >= minValueInBucket);
            return Next_BayesianSearchFloatValue(minValueInBucket, maxValueInBucket, rand, rangeType, RANDOM_SEARCH_OPTION.FULLY_RANDOM, null);
        }
        var result = RandomValue(min, max, rangeType, rand.NextSingle());
        Debug.Assert(result>=(min-1e-6));
        Debug.Assert(result<=(max+1e-6));
        return result;
    }
    protected static float SampleValueToFloat(object sampleValue)
    {
        if (sampleValue is double)
        {
            return (float)(double)sampleValue;
        }
        if (sampleValue is float)
        {
            return (float)sampleValue;
        }
        if (sampleValue is int)
        {
            return (int)sampleValue;
        }
        if (sampleValue is string)
        {
            return float.Parse((string)sampleValue);
        }
        throw new ArgumentException($"can't transform {sampleValue} to float");
    }
    protected static int SampleValueToBucketIndex(float sampleValue, float min, float max, range_type rangeType)
    {
        //float sampleValueAsFloat = SampleValueToFloat(sampleValue)
        for (int bucketIndex = 0; bucketIndex < BucketCount - 1; ++bucketIndex)
        {
            if (sampleValue <= BucketIndexToBucketUpperValue(bucketIndex, min, max, rangeType))
            {
                return bucketIndex;
            }
        }

        return BucketCount - 1;
    }

    private static float RandomValue(float min, float max, range_type rangeType, float randomUniformBetween_0_and_1)
    {
        if (rangeType == range_type.uniform)
        {
            return UniformValue(min, max, randomUniformBetween_0_and_1);
        }
        if (rangeType == range_type.normal)
        {
            return NormalValue(min, max, randomUniformBetween_0_and_1);
        }
        throw new NotImplementedException($"rangeType {rangeType} is not supported");
    }
    private static float NormalValue(float min, float max, float randomUniformBetween_0_and_1)
    {
        Debug.Assert(max>=min);
        Debug.Assert(randomUniformBetween_0_and_1>=0);
        Debug.Assert(randomUniformBetween_0_and_1<=1);
        const double epsilon = 1e-3; //TODO tune this parameter
        var logSampleValue = Math.Log(epsilon) + (Math.Log(max - min + epsilon) - Math.Log(epsilon)) * randomUniformBetween_0_and_1;
        return (float)(min + Math.Exp(logSampleValue) - epsilon);
    }
    private static float UniformValue(float min, float max, float randomUniformBetween_0_and_1)
    {
        Debug.Assert(max >= min);
        Debug.Assert(randomUniformBetween_0_and_1 >= 0);
        Debug.Assert(randomUniformBetween_0_and_1 <= 1);
        return min + (max - min) * randomUniformBetween_0_and_1;
    }


    protected abstract float LowerValueForBucket(int bucketIndex);
    protected abstract float UpperValueForBucket(int bucketIndex);

    protected static float BucketIndexToBucketLowerValue(int bucketIndex, float min, float max, range_type rangeType)
    {
        if (bucketIndex == 0)
        {
            return min;
        }
        return BucketIndexToBucketUpperValue(bucketIndex-1, min, max, rangeType);
    }
    protected static float BucketIndexToBucketUpperValue(int bucketIndex, float min, float max, range_type rangeType)
    {
        if (bucketIndex == BucketCount - 1)
        {
            return max;
        }
        return RandomValue(min, max, rangeType, (bucketIndex + 1f) / BucketCount);
    }

    private string BucketDescription(int bucketIndex)
    {
        string FloatToString(float f)
        {
            return Math.Round(f,4).ToString(CultureInfo.InvariantCulture);
        }
        return "[" + FloatToString(LowerValueForBucket(bucketIndex)) + "," + FloatToString(UpperValueForBucket(bucketIndex)) + "]";
    }
}
