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
            Samples[sampleIndex].Save(workingDirectory, modelName + ((sampleIndex==0)?"":("_"+sampleIndex)) );
        }
    }
    public List<string> SampleFiles(string workingDirectory, string modelName)
    {
        HashSet<string> res = new();
        for (var sampleIndex = 0; sampleIndex < Samples.Length; sampleIndex++)
        {
            var sample = Samples[sampleIndex];
            res.UnionWith(sample.SampleFiles(workingDirectory, modelName + "_" + sampleIndex));
        }
        return res.ToList();
    }
    public HashSet<string> HyperParameterNames()
    {
        HashSet<string> res = new();
        foreach (var sample in Samples)
        {
            res.UnionWith(sample.HyperParameterNames());
        }
        return res;
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
    public string ComputeHash()
    {
        var allHash = string.Join("_", Samples.Select(s => s.ComputeHash()));
        return Utils.ComputeHash(allHash, 10);
    }
    public virtual CpuTensor<float> Y_Train_dataset_to_Perfect_Predictions(string y_train_dataset)
    {
        throw new NotImplementedException();
    }
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
        return Samples.FirstOrDefault(s => s.HyperParameterNames().Contains(hyperParameterName));
    }
}
