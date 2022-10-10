using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using SharpNet.MathTools;

namespace SharpNet.Datasets;

public class FeatureStats
{
    #region private fields
    // true if the '_subSum' field is outdated and needs to be recomputed
    private readonly Dictionary<string, int> _categoricalFeatureToCount = new();
    private readonly List<string> _distinctCategoricalValues = new();
    private readonly Dictionary<string, int> _categoricalFeatureToIndex = new();
    private readonly DoubleAccumulator _numericalValues = new();
    /// <summary>
    /// true if the feature is categorical (non numerical)
    /// </summary>
    #endregion

    #region public properties
    public bool IsCategoricalFeature { get; }
    /// <summary>
    /// true if the feature is the target of the dataset
    /// </summary>
    public bool IsTargetLabel { get; }
    /// <summary>
    /// true if the feature is (among) the ids needed to identify a unique row
    /// </summary>
    public bool IsId { get; }
    /// <summary>
    /// number of elements of the given features
    /// </summary>
    public int Count { get; private set; }
    /// <summary>
    /// number of empty elements in the DataSet for this feature
    /// </summary>
    public int CountEmptyFeatures { get; private set; }

    public IList<string> GetDistinctCategoricalValues() => _distinctCategoricalValues;

    public IList<int> GetDistinctCategoricalCount()
    {
        var result = new List<int>();
        foreach (var c in GetDistinctCategoricalValues())
        {
            result.Add(_categoricalFeatureToCount.ContainsKey(c)? _categoricalFeatureToCount[c]:0);
        }
        return result;
    }

    #endregion


    #region constructors
    public FeatureStats(bool isCategoricalFeature, bool isTargetLabel, bool isId)
    {
        IsCategoricalFeature = isCategoricalFeature;
        IsTargetLabel = isTargetLabel;
        IsId = isId;
    }
    #endregion


    public double NumericalEncoding(string featureValue)
    {
        ++Count;
        if (IsCategoricalFeature)
        {
            var categoricalFeatures = NormalizeCategoricalFeatureValue(featureValue);
            if (categoricalFeatures.Length == 0)
            {
                ++CountEmptyFeatures;
                return -1;
            }
            //it is a categorical feature, we add it to the dictionary
            if (!_categoricalFeatureToCount.ContainsKey(categoricalFeatures))
            {
                _categoricalFeatureToCount[categoricalFeatures] = 1;
                _distinctCategoricalValues.Add(categoricalFeatures);
                _categoricalFeatureToIndex[categoricalFeatures] = _distinctCategoricalValues.Count - 1;
            }
            else
            {
                ++_categoricalFeatureToCount[categoricalFeatures];
            }
            return _categoricalFeatureToIndex[categoricalFeatures];
        }
        else
        {
            //it is a numerical field
            var doubleValue = ExtractDouble(featureValue);
            if (double.IsNaN(doubleValue))
            {
                ++CountEmptyFeatures;
                return double.NaN; //missing numerical value
            }
            _numericalValues.Add(doubleValue, 1);
            return doubleValue;
        }
    }


    public string NumericalDecoding(double numericalEncodedFeatureValue, string missingNumberValue)
    {
        if (IsCategoricalFeature)
        {
            var categoricalFeatureIndex = (int)Math.Round(numericalEncodedFeatureValue);
            return categoricalFeatureIndex<0?"": _distinctCategoricalValues[categoricalFeatureIndex];
        }

        if (double.IsNaN(numericalEncodedFeatureValue))
        {
            return missingNumberValue;
        }
        return numericalEncodedFeatureValue.ToString(CultureInfo.InvariantCulture);
    }

    public static double ExtractDouble(string featureValue)
    {
        bool isNegative = false;
        bool insideNumber = false;
        bool isLeftNumber = true;
        double result = 0.0;
        double divider = 10.0;

        for (var index = 0; index < featureValue.Length; index++)
        {
            var c = featureValue[index];
            if (insideNumber)
            {
                if (isLeftNumber)
                {
                    if (char.IsWhiteSpace(c) || c == ',')
                    {
                        continue;
                    }

                    if (c == '.')
                    {
                        isLeftNumber = false;
                        continue;
                    }

                    if (char.IsDigit(c))
                    {
                        result = 10 * result + c - '0';
                        continue;
                    }

                    break;
                }
                else
                {
                    if (char.IsDigit(c))
                    {
                        result += (c - '0') / divider;
                        divider *= 10;
                        continue;
                    }

                    break;
                }
            }
            else
            {
                if (c == '-')
                {
                    isNegative = true;
                    insideNumber = true;
                    continue;
                }

                if (c == '+')
                {
                    insideNumber = true;
                    continue;
                }

                if (char.IsDigit(c))
                {
                    result = c - '0';
                    insideNumber = true;
                    continue;
                }
            }
        }

        if (!insideNumber)
        {
            return double.NaN;
        }
        return isNegative ? -result : result;

    }

    private static string NormalizeCategoricalFeatureValue(string value)
    {
        value = value.Trim();
        if (!value.Any(CharToBeRemovedInStartOrEnd))
        {
            return value;
        }
        int nbToRemoveStart = 0;
        foreach (var c in value)
        {
            if (!CharToBeRemovedInStartOrEnd(c))
            {
                break;
            }
            ++nbToRemoveStart;
        }

        if (nbToRemoveStart != 0)
        {
            value = value.Substring(nbToRemoveStart);
        }

        int nbToRemoveEnd = 0;
        for (int i = value.Length - 1; i >= 0; --i)
        {
            if (!CharToBeRemovedInStartOrEnd(value[i]))
            {
                break;
            }
            ++nbToRemoveEnd;
        }
        if (nbToRemoveEnd != 0)
        {
            value = value.Substring(0, value.Length - nbToRemoveEnd);
        }

        if (value.Any(CharToBeRemovedInStartOrEnd))
        {
            var sb = new StringBuilder();
            foreach (var c in value)
            {
                sb.Append(CharToBeRemovedInStartOrEnd(c)?' ':c);
            }
            value = sb.ToString();
        }
        return value;
    }

    private static bool CharToBeRemovedInStartOrEnd(char c)
    {
        return char.IsWhiteSpace(c) || c == '\'' || c == '\"' || c == '\n' || c == '\r' || c == ';' || c == ',';
    }

}