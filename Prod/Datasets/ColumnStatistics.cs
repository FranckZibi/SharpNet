using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using SharpNet.MathTools;

namespace SharpNet.Datasets;

public class ColumnStatistics
{
    private readonly bool _standardizeDoubleValues;
    private readonly bool _allDataFrameAreAlreadyNormalized;

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

    /// <summary>
    /// 
    /// </summary>
    /// <param name="isCategorical"></param>
    /// <param name="isTargetLabel"></param>
    /// <param name="isId"></param>
    /// <param name="standardizeDoubleValues">
    /// true if we should standardize double values (mean 0 and volatility 1)
    /// false if we should not transform double values
    /// </param>
    /// <param name="allDataFrameAreAlreadyNormalized"></param>
    public ColumnStatistics(bool isCategorical, bool isTargetLabel, bool isId, bool standardizeDoubleValues, bool allDataFrameAreAlreadyNormalized)
    {
        _standardizeDoubleValues = standardizeDoubleValues;
        _allDataFrameAreAlreadyNormalized = allDataFrameAreAlreadyNormalized;
        IsCategorical = isCategorical;
        IsTargetLabel = isTargetLabel;
        IsId = isId;
    }
    #endregion

    public void Fit(string val_before_encoding)
    {
        ++Count;
        if (IsCategorical)
        {
            if (!_allDataFrameAreAlreadyNormalized)
            {
                val_before_encoding = Utils.NormalizeCategoricalFeatureValue(val_before_encoding);
            }
            if (val_before_encoding.Length == 0)
            {
                ++CountEmptyElements;
                return;
            }
            //it is a categorical column, we add it to the dictionary
            if (!_distinctCategoricalValueToCount.ContainsKey(val_before_encoding))
            {
                _distinctCategoricalValueToCount[val_before_encoding] = 1;
                _distinctCategoricalValues.Add(val_before_encoding);
                _distinctCategoricalValueToIndex[val_before_encoding] = _distinctCategoricalValues.Count - 1;
            }
            else
            {
                ++_distinctCategoricalValueToCount[val_before_encoding];
            }
            return;
        }

        //it is a numerical field
        var numeric_val_before_encoding = ExtractDouble(val_before_encoding);
        if (double.IsNaN(numeric_val_before_encoding))
        {
            ++CountEmptyElements;
            return; //missing numerical value
        }
        _numericalValues.Add(numeric_val_before_encoding);
    }

    public void Fit(float val_before_encoding)
    {
        ++Count;

        if (IsCategorical)
        {
            var str_val_before_encoding = val_before_encoding.ToString(CultureInfo.InvariantCulture);
            //it is a categorical column, we add it to the dictionary
            if (!_distinctCategoricalValueToCount.ContainsKey(str_val_before_encoding))
            {
                _distinctCategoricalValueToCount[str_val_before_encoding] = 1;
                _distinctCategoricalValues.Add(str_val_before_encoding);
                _distinctCategoricalValueToIndex[str_val_before_encoding] = Utils.NearestInt(val_before_encoding);
                Debug.Assert(MathF.Abs(Utils.NearestInt(val_before_encoding) - val_before_encoding) < 1e-5f);
            }
            else
            {
                ++_distinctCategoricalValueToCount[str_val_before_encoding];
            }
            return;
        }

        //it is a numerical field
        if (float.IsNaN(val_before_encoding))
        {
            ++CountEmptyElements;
            return; //missing numerical value
        }
        _numericalValues.Add(val_before_encoding);
    }

    public void Fit(int val_before_encoding)
    {
        Fit((float)val_before_encoding);
    }


    public double Transform(string val_before_encoding)
    {
        if (IsCategorical)
        {
            if (!_allDataFrameAreAlreadyNormalized)
            {
                val_before_encoding = Utils.NormalizeCategoricalFeatureValue(val_before_encoding);
            }
            if (val_before_encoding.Length == 0 || !_distinctCategoricalValueToCount.ContainsKey(val_before_encoding))
            {
                return -1;
            }
            return _distinctCategoricalValueToIndex[val_before_encoding];
        }

        //it is a numerical field
        var val_after_encoding = ExtractDouble(val_before_encoding);
        if (double.IsNaN(val_after_encoding))
        {
            return double.NaN; //missing numerical value
        }
        //We do the standardization for the double value
        var volatility = _numericalValues.Volatility;
        if (volatility > 0 && _standardizeDoubleValues && !IsTargetLabel)
        {
            val_after_encoding = (val_after_encoding - _numericalValues.Average) / volatility;
        }
        return val_after_encoding;
    }

    public double Transform(float val_before_encoding)
    {
        if (IsCategorical)
        {
            return val_before_encoding;
        }

        //it is a numerical field
        double val_after_encoding = val_before_encoding;
        if (double.IsNaN(val_after_encoding))
        {
            return double.NaN; //missing numerical value
        }
        //We do the standardization for the double value
        var volatility = _numericalValues.Volatility;
        if (volatility > 0 && _standardizeDoubleValues && !IsTargetLabel)
        {
            val_after_encoding = (val_after_encoding - _numericalValues.Average) / volatility;
        }
        return val_after_encoding;
    }

    public string Inverse_Transform(double val_after_encoding, string missingNumberValue)
    {
        if (IsCategorical)
        {
            var categoricalFeatureIndex = (int)Math.Round(val_after_encoding);
            if (categoricalFeatureIndex < 0 || categoricalFeatureIndex>= _distinctCategoricalValues.Count)
            {
                return "";
            }
            return _distinctCategoricalValues[categoricalFeatureIndex];
        }

        if (double.IsNaN(val_after_encoding))
        {
            return missingNumberValue;
        }

        var val_before_encoding = val_after_encoding;
        var volatility = _numericalValues.Volatility;
        if (volatility > 0 && _standardizeDoubleValues && !IsTargetLabel)
        {
            val_before_encoding = volatility * val_before_encoding + _numericalValues.Average;
        }
        return val_before_encoding.ToString(CultureInfo.InvariantCulture);
    }

    public float Inverse_Transform_float(float val_after_encoding)
    {
        if (IsCategorical || double.IsNaN(val_after_encoding))
        {
            return val_after_encoding;
        }
        var volatility = _numericalValues.Volatility;
        if (volatility > 0 && _standardizeDoubleValues && !IsTargetLabel)
        {
            return (float) (volatility * val_after_encoding + _numericalValues.Average);
        }
        return val_after_encoding;
    }
    public static double ExtractDouble(string featureValue)
    {
        bool isNegative = false;
        bool insideNumber = false;
        bool isLeftNumber = true;
        double result = 0.0;
        double divider = 10.0;

        foreach (var c in featureValue)
        {
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
   
}