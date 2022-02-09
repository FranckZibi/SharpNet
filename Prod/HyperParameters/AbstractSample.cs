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
    public bool HasHyperParameter(string hyperParameterName)
    {
        return ClassFieldSetter.HasField(GetType(), hyperParameterName);
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
        Save(Path.Combine(workingDirectory, modelName + ".conf"));
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
        return ClassFieldSetter.GetFieldInfo(GetType(), hyperParameterName).FieldType;
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
        var type = GetType();
        var result = new List<string>();
        foreach (var (parameterName, _) in ClassFieldSetter.GetFieldName2FieldInfo(type).OrderBy(f => f.Key))
        {
            result.Add($"{parameterName} = {Utils.FieldValueToString(Get(parameterName))}");
        }
        return string.Join(Environment.NewLine, result) + Environment.NewLine;
    }
}
