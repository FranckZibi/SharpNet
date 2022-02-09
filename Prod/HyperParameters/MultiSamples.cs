using System;
using System.Collections.Generic;
using System.Linq;
using SharpNet.CPU;

namespace SharpNet.HyperParameters;

public abstract class MultiSamples : ISample
{
    #region protected fields
    protected ISample[] Samples { get; }
    #endregion

    #region constructor
    protected MultiSamples(ISample[] samples)
    {
        Samples = samples;
    }
    #endregion

    #region ISample methods
    public void Save(string workingDirectory, string modelName)
    {
        for (int sampleIndex = 0; sampleIndex < Samples.Length; ++sampleIndex)
        {
            Samples[sampleIndex].Save(workingDirectory, modelName+"_"+sampleIndex);
        }
    }
    public void Set(IDictionary<string, object> dico)
    {
        foreach (var (key, value) in dico)
        {
            Set(key, value);
        }
    }
    public void Set(string hyperParameterName, object fieldValue)
    {
        FirstSampleWithField(hyperParameterName).Set(hyperParameterName, fieldValue);
    }
    public object Get(string hyperParameterName)
    {
        return FirstSampleWithField(hyperParameterName).Get(hyperParameterName);
    }
    public virtual bool PostBuild()
    {
        foreach (var s in Samples)
        {
            if (!s.PostBuild())
            {
                return false;
            }
        }
        return true;
    }
    public bool HasHyperParameter(string hyperParameterName)
    {
        return FirstSampleWithField(hyperParameterName) != null;
    }
    public string ComputeHash()
    {
        var allHash = string.Join("_", Samples.Select(s => s.ComputeHash()));
        return Utils.ComputeHash(allHash, 10);
    }
    public abstract CpuTensor<float> Y_Train_dataset_to_Perfect_Predictions(string y_train_dataset);
    public Type GetFieldType(string hyperParameterName)
    {
        return FirstSampleWithField(hyperParameterName).GetFieldType(hyperParameterName);
    }
    public bool IsCategoricalHyperParameter(string hyperParameterName)
    {
        return FirstSampleWithField(hyperParameterName).IsCategoricalHyperParameter(hyperParameterName);
    }
    #endregion

    private ISample FirstSampleWithField(string hyperParameterName)
    {
        return Samples.FirstOrDefault(s => s.HasHyperParameter(hyperParameterName));
    }
}
