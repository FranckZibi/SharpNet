using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.HyperParameters;
using SharpNet.Models;
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.HPO;

public class WeightsOptimizerHyperParameters : AbstractSample
{
    #region constructors
    public WeightsOptimizerHyperParameters() : base(new HashSet<string>())
    {
        SetZeroWeights();
    }
    public static WeightsOptimizerHyperParameters ValueOf(string workingDirectory, string modelName)
    {
        return (WeightsOptimizerHyperParameters)ISample.LoadConfigIntoSample(() => new WeightsOptimizerHyperParameters(), workingDirectory, modelName);
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
    #endregion

    #region ISample methods
    public override bool PostBuild()
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
    #endregion

    public const char separator = ',';
    public List<TrainedModel> LoadModelDescription(string workingDirectory)
    {
        var path = Path.Combine(workingDirectory, ComputeHash());
        List<TrainedModel> result = new();
        foreach (var l in File.ReadLines(path).Skip(1))
        {
            var splitted = l.Split(separator);
            result.Add(TrainedModel.ValueOf(splitted[0], splitted[1]));
        }
        return result;
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

    private float[] GetWeights()
    {

        return new[] { w_00, w_01, w_02, w_03, w_04, w_05, w_06, w_07, w_08, w_09 };
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

}