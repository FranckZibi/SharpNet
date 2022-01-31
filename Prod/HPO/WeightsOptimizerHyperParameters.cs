using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.CPU;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.HPO;

public class WeightsOptimizerHyperParameters
{
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

    public WeightsOptimizerHyperParameters()
    {
        SetZeroWeights();
    }

    public float[] GetWeights()
    {

        return new[] { w_00, w_01, w_02, w_03, w_04, w_05, w_06, w_07, w_08, w_09 };
    }

    public void SetWeights(float[] weights)
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
    }

    public void SetEqualWeights()
    {
        var weights = GetWeights();
        for (int i = 0; i < weights.Length; ++i)
        {
            weights[i] = 1.0f / weights.Length;
        }
        SetWeights(weights);
    }
    public void SetZeroWeights()
    {
        var weights = GetWeights();
        for (int i = 0; i < weights.Length; ++i)
        {
            weights[i] = 0;
        }
        SetWeights(weights);
    }

    public bool PostBuild()
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

    private static void NormalizeWeightsToSum_One(float[] weights)
    {
        var sum = weights.Sum();
        for (int i = 0; i < weights.Length; ++i)
        {
            weights[i] *= 1.0f / sum;
        }
    }

    public CpuTensor<float> ApplyWeights(List<CpuTensor<float>> t)
    {
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

}