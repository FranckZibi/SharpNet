using System;
using System.Collections.Generic;
using System.Linq;

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
        if (samples.Any(s => s == null))
        {
            throw new ArgumentException($"some null samples found in {nameof(MultiSamples)}");
        }
    }
    #endregion


    public bool MustUseGPU => Samples.Any(s => s.MustUseGPU);

    public virtual void SetTaskId(int taskId)
    {
        Array.ForEach(Samples, s => s.SetTaskId(taskId));
    }

    public virtual void FillSearchSpaceWithDefaultValues(IDictionary<string, object> hyperParameterSearchSpace)
    {
        Array.ForEach(Samples, s => s.FillSearchSpaceWithDefaultValues(hyperParameterSearchSpace));
    }

    public virtual List<string> Save(string workingDirectory, string modelName)
    {
        List<string> res = new();
        for (int sampleIndex = 0; sampleIndex < Samples.Length; ++sampleIndex)
        {
            var newFiles = Samples[sampleIndex].Save(workingDirectory, SampleIndexToSampleName(modelName, sampleIndex));
            res.AddRange(newFiles);
        }
        return res;
    }
    public virtual List<string> SampleFiles(string workingDirectory, string modelName)
    {
        HashSet<string> res = new();
        for (var sampleIndex = 0; sampleIndex < Samples.Length; sampleIndex++)
        {
            var sample = Samples[sampleIndex];
            res.UnionWith(sample.SampleFiles(workingDirectory, SampleIndexToSampleName(modelName, sampleIndex)));
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
    public virtual bool FixErrors()
    {
        bool ok = true;
        foreach (var s in Samples)
        {
            if (!s.FixErrors())
            {
                ok = false;
            }
        }
        return ok;
    }
    public string ComputeHash()
    {
        var allHash = string.Join("_", Samples.Select(s => s.ComputeHash()));
        return Utils.ComputeHash(allHash, 10);
    }
    public virtual ISample Clone()
    {
        var clonedSamples = Samples.Select(s => s.Clone()).ToArray();
        var ctor = GetType().GetConstructor(new []{clonedSamples.GetType()});
        var clonedInstance = (ISample)ctor?.Invoke(new object[] {clonedSamples});
        return clonedInstance;
    }
    public Type GetFieldType(string hyperParameterName)
    {
        return FirstSampleWithField(hyperParameterName).GetFieldType(hyperParameterName);
    }
    public bool IsCategoricalHyperParameter(string hyperParameterName)
    {
        var firstSampleWithField = FirstSampleWithField(hyperParameterName);
        if (firstSampleWithField == null)
        {
            throw new ArgumentException($"hyper parameter {hyperParameterName} not found");
        }
        return firstSampleWithField.IsCategoricalHyperParameter(hyperParameterName);
    }

    protected virtual string SampleIndexToSampleName(string modelName, int sampleIndex)
    {
        return ISample.SampleName(modelName, sampleIndex);
    }

    private ISample FirstSampleWithField(string hyperParameterName)
    {
        return Samples.FirstOrDefault(s => s.HyperParameterNames().Contains(hyperParameterName));
    }
}
