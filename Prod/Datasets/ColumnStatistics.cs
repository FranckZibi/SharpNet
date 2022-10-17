using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using SharpNet.MathTools;

namespace SharpNet.Datasets;

public class ColumnStatistics
{
    #region private fields
    private readonly Dictionary<string, int> _distinctCategoricalValueToCount = new();
    private readonly List<string> _distinctCategoricalValues = new();
    private readonly Dictionary<string, int> _distinctCategoricalValueToIndex = new();
    private readonly DoubleAccumulator _numericalValues = new();
    #endregion

    #region public properties
    /// <summary>
    /// true if the column is categorical (non numerical)
    /// </summary>
    public bool IsCategorical { get; }
    /// <summary>
    /// true if the column contains the target of the dataset
    /// </summary>
    public bool IsTargetLabel { get; }
    /// <summary>
    /// true if the column is (among) the ids needed to identify a unique row
    /// </summary>
    public bool IsId { get; }
    /// <summary>
    /// number of elements of the given features
    /// </summary>
    public int Count { get; private set; }
    /// <summary>
    /// number of empty elements in the DataSet for this column
    /// </summary>
    public int CountEmptyElements { get; private set; }
    public IList<string> GetDistinctCategoricalValues() => _distinctCategoricalValues;
    #endregion


    #region constructors
    public ColumnStatistics(bool isCategorical, bool isTargetLabel, bool isId)
    {
        IsCategorical = isCategorical;
        IsTargetLabel = isTargetLabel;
        IsId = isId;
    }
    #endregion

    public double NumericalEncoding(string elementValue)
    {
        ++Count;
        if (IsCategorical)
        {
            var categoricalElementValue = NormalizeCategoricalFeatureValue(elementValue);
            if (categoricalElementValue.Length == 0)
            {
                ++CountEmptyElements;
                return -1;
            }
            //it is a categorical column, we add it to the dictionary
            if (!_distinctCategoricalValueToCount.ContainsKey(categoricalElementValue))
            {
                _distinctCategoricalValueToCount[categoricalElementValue] = 1;
                _distinctCategoricalValues.Add(categoricalElementValue);
                _distinctCategoricalValueToIndex[categoricalElementValue] = _distinctCategoricalValues.Count - 1;
            }
            else
            {
                ++_distinctCategoricalValueToCount[categoricalElementValue];
            }
            return _distinctCategoricalValueToIndex[categoricalElementValue];
        }
        else
        {
            //it is a numerical field
            var doubleValue = ExtractDouble(elementValue);
            if (double.IsNaN(doubleValue))
            {
                ++CountEmptyElements;
                return double.NaN; //missing numerical value
            }
            _numericalValues.Add(doubleValue, 1);
            return doubleValue;
        }
    }
    public string NumericalDecoding(double numericalEncodedFeatureValue, string missingNumberValue)
    {
        if (IsCategorical)
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