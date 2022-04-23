using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.CPU;
using SharpNet.HyperParameters;

// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.HPO;

public class WeightsOptimizerSample : AbstractSample, IModelSample
{
    #region private fields
    private readonly List<Tuple<string, string>> _workingDirectoryAndModelNames = new();
    #endregion

    #region constructors
    public WeightsOptimizerSample() : base(new HashSet<string>())
    {
    }
    public WeightsOptimizerSample(string workingDirectoryAndModelNames, MetricEnum metric, LossFunctionEnum loss) : base(new HashSet<string>())
    {
        WorkingDirectoryAndModelNames = workingDirectoryAndModelNames;
        Metric = metric;
        Loss = loss;
        _workingDirectoryAndModelNames = Split(workingDirectoryAndModelNames);
        SetZeroWeights();
    }
    public WeightsOptimizerSample(List<Tuple<string,string>> workingDirectoryAndModelNames, MetricEnum metric, LossFunctionEnum loss) 
        : this(string.Join(";",workingDirectoryAndModelNames.Select(t=>t.Item1+";"+t.Item2)), metric, loss)
    {

    }
    #endregion

    #region Hyper-parameters
    public float w_00;
    public float w_01;
    public float w_02;
    public float w_03;
    public float w_04;
    public float w_05;
    public float w_06;
    public float w_07;
    public float w_08;
    public float w_09;
    public float w_10;
    public float w_11;
    public float w_12;
    public float w_13;
    public float w_14;
    public float w_15;
    public float w_16;
    public float w_17;
    public float w_18;
    public float w_19;
    // ReSharper disable once FieldCanBeMadeReadOnly.Global
    public string WorkingDirectoryAndModelNames;
    public MetricEnum Metric;
    public LossFunctionEnum Loss;
    #endregion

    public List<Tuple<string, string>> GetWorkingDirectoryAndModelNames()
    {
        return _workingDirectoryAndModelNames;
    }
    public override bool FixErrors()
    {
        var weights = GetWeights();
        var sum = weights.Sum();
        if (sum <= 0)
        {
            return false;
        }
        NormalizeWeightsToSum_One(weights);
        SetWeights(weights);
        return true;
    }
    public void SetEqualWeights()
    {
        SetZeroWeights();
        var weights = GetWeights();
        for (int i = 0; i < _workingDirectoryAndModelNames.Count; ++i)
        {
            weights[i] = 1.0f / _workingDirectoryAndModelNames.Count;
        }
        SetWeights(weights);
    }
    public CpuTensor<float> ApplyWeights(List<CpuTensor<float>> t)
    {
        if (t == null || t.Count == 0)
        {
            return null;
        }
        var weights = GetWeights();
        if (weights.Length < t.Count)
        {
            throw new ArgumentException($"too many {t.Count} tensors, max={weights.Length}");
        }
        var sumWeights = weights.Take(t.Count).Sum();
        var res = new CpuTensor<float>(t[0].Shape);
        res.SetValue(0.0f);
        for (int i = 0; i < t.Count; ++i)
        {
            res.Update_Adding_Alpha_X(weights[i] / sumWeights, t[i]);
        }
        return res;
    }
    public MetricEnum GetMetric()
    {
        return Metric;
    }
    public LossFunctionEnum GetLoss()
    {
        return Loss;
    }
    private float[] GetWeights()
    {

        return new[] { w_00, w_01, w_02, w_03, w_04, w_05, w_06, w_07, w_08, w_09, w_10, w_11, w_12, w_13, w_14, w_15, w_16, w_17, w_18, w_19 };
    }
    private void SetWeights(float[] weights)
    {
        w_00 = weights[0];
        w_01 = weights[1];
        w_02 = weights[2];
        w_03 = weights[3];
        w_04 = weights[4];
        w_05 = weights[5];
        w_06 = weights[6];
        w_07 = weights[7];
        w_08 = weights[8];
        w_09 = weights[9];
        w_10 = weights[10];
        w_11 = weights[11];
        w_12 = weights[12];
        w_13 = weights[13];
        w_14 = weights[14];
        w_15 = weights[15];
        w_16 = weights[16];
        w_17 = weights[17];
        w_18 = weights[18];
        w_19 = weights[19];
    }
    private void SetZeroWeights()
    {
        var weights = GetWeights();
        for (int i = 0; i < weights.Length; ++i)
        {
            weights[i] = 0;
        }
        SetWeights(weights);
    }
    private static void NormalizeWeightsToSum_One(float[] weights)
    {
        var sum = weights.Sum();
        for (int i = 0; i < weights.Length; ++i)
        {
            weights[i] *= 1.0f / sum;
        }
    }
    private static List<Tuple<string, string>> Split(string workingDirectoryAndModelNames)
    {
        List<Tuple<string, string>> result = new();
        var splitted = workingDirectoryAndModelNames.Split(';').ToList();
        if (splitted.Count % 2 != 0)
        {
            throw new ArgumentException($" invalid number of elements in {workingDirectoryAndModelNames}: must be even");
        }
        for (int i = 0; i < splitted.Count; i += 2)
        {
            result.Add(Tuple.Create(splitted[i], splitted[i + 1]));
        }

        return result;
    }
}