using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpNet.CPU;

namespace SharpNet.HyperParameters;

public abstract class AbstractSample : ISample
{
    #region private fields
    private readonly HashSet<string> _mandatoryCategoricalHyperParameters;
    #endregion


    #region constructors
    protected AbstractSample(HashSet<string> mandatoryCategoricalHyperParameters)
    {
        _mandatoryCategoricalHyperParameters = mandatoryCategoricalHyperParameters;
    }
    #endregion

    #region ISample methods

    public HashSet<string> HyperParameterNames()
    {
        var type = GetType();
        return new HashSet<string>(ClassFieldSetter.FieldNames(type));
    }

    public string ComputeHash()
    {
        return Utils.ComputeHash(ToConfigContent(), 10);
    }
    public CpuTensor<float> Y_Train_dataset_to_Perfect_Predictions(string y_train_dataset)
    {
        throw new NotImplementedException();
    }
    public virtual void Save(string workingDirectory, string modelName)
    {
        var configFile = ISample.ToPath(workingDirectory, modelName);
        Save(configFile);
    }

    public virtual List<string> SampleFiles(string workingDirectory, string modelName)
    {
        return new List<string> { ISample.ToPath(workingDirectory, modelName) };
    }


    public void Save(string path)
    {
        var configContent = ToConfigContent();
        File.WriteAllText(path, configContent);
    }
    public virtual void Set(string fieldName, object fieldValue)
    {
        ClassFieldSetter.Set(this, fieldName, fieldValue);
    }
    public virtual object Get(string fieldName)
    {
        return ClassFieldSetter.Get(this, fieldName);
    }
    public abstract bool PostBuild();
    public Type GetFieldType(string hyperParameterName)
    {
        return ClassFieldSetter.GetFieldType(GetType(), hyperParameterName);
    }
    public virtual void Set(IDictionary<string, object> dico)
    {
        foreach (var (key, value) in dico)
        {
            Set(key, value);
        }
    }
    public virtual bool IsCategoricalHyperParameter(string hyperParameterName)
    {
        if (_mandatoryCategoricalHyperParameters.Contains(hyperParameterName))
        {
            return true;
        }
        var hyperParameterType = GetFieldType(hyperParameterName);
        if (hyperParameterType == typeof(double) || hyperParameterType == typeof(float) || hyperParameterType == typeof(int))
        {
            return false;
        }
        if (hyperParameterType == typeof(string) || hyperParameterType == typeof(bool) || hyperParameterType.IsEnum)
        {
            return true;
        }
        throw new ArgumentException($"can't determine if {hyperParameterName} ({hyperParameterType}) field of class {GetType()} is categorical");
    }
    #endregion

    private string ToConfigContent()
    {
        var result = new List<string>();
        foreach (var parameterName in HyperParameterNames().OrderBy(f => f))
        {
            result.Add($"{parameterName} = {Utils.FieldValueToString(Get(parameterName))}");
        }
        return string.Join(Environment.NewLine, result) + Environment.NewLine;
    }
}
