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

    public const int DEFAULT_VALUE = -6666;
    protected const string DEFAULT_VALUE_STR = nameof(DEFAULT_VALUE);

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
        return Utils.ComputeHash(ToConfigContent(DefaultAcceptForConfigContent), 10);
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
    public virtual void Save(string path)
    {
        var configContent = ToConfigContent(DefaultAcceptForConfigContent);
        File.WriteAllText(path, configContent);
    }
    public virtual List<string> SampleFiles(string workingDirectory, string modelName)
    {
        return new List<string> { ISample.ToPath(workingDirectory, modelName) };
    }
    public virtual void Set(string fieldName, object fieldValue)
    {
        ClassFieldSetter.Set(this, fieldName, fieldValue);
    }
    public virtual object Get(string fieldName)
    {
        return ClassFieldSetter.Get(this, fieldName);
    }
    public virtual bool PostBuild()
    {
        return true;
    }
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
    //TODO add tests
    /// <summary>
    /// to determine if an Hyper-Parameter is categorical, we'll user the following rules:
    ///     1/ if the Hyper-Parameter name is in '_mandatoryCategoricalHyperParameters'
    ///             => it is categorical
    ///     2/ if the Hype-Parameter is a int/float/double
    ///             => it is numerical
    ///     3/ if the Hype-Parameter is a bool/string/enum
    ///             => it is categorical
    ///     4/ in all other cases
    ///             => an exception is thrown (can't determine if the Hyper-Parameter is categorical)
    /// </summary>
    /// <param name="hyperParameterName"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
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

    protected static bool DefaultAcceptForConfigContent(string fieldName, object fieldValue)
    {
        return !IsDefaultValue(fieldValue);
    }
    protected string ToJsonConfigContent(Func<string,object, bool> accept)
    {
        var result = new List<string>();
        foreach (var parameterName in HyperParameterNames().OrderBy(f => f))
        {
            var fieldValue = Get(parameterName);
            if (accept == null || accept(parameterName, fieldValue))
            {
                var fieldValueAsJsonString = Utils.FieldValueToJsonString(fieldValue);
                result.Add($"\t\"{parameterName}\": {fieldValueAsJsonString}");
            }
        }
        return "{"+Environment.NewLine+string.Join(","+Environment.NewLine, result) + Environment.NewLine+"}";
    }

    private string ToConfigContent(Func<string, object, bool> accept)
    {
        var result = new List<string>();
        foreach (var parameterName in HyperParameterNames().OrderBy(f => f))
        {
            var fieldValue = Get(parameterName);
            if (accept == null || accept(parameterName, fieldValue))
            {
                result.Add($"{parameterName} = {Utils.FieldValueToString(fieldValue)}");
            }
        }
        return string.Join(Environment.NewLine, result) + Environment.NewLine;
    }
    /// <summary>
    /// TODO : add tests
    /// </summary>
    /// <param name="fieldValue"></param>
    /// <returns></returns>
    private static bool IsDefaultValue(object fieldValue)
    {
        if (fieldValue == null)
        {
            return true;
        }
        if ((fieldValue is string fieldValueStr && Equals(fieldValueStr, DEFAULT_VALUE_STR))
            || (fieldValue.GetType().IsEnum && Equals(fieldValue.ToString(), DEFAULT_VALUE_STR))
            || (fieldValue is int fieldValueInt && (fieldValueInt == DEFAULT_VALUE))
            || (fieldValue is float fieldValueFloat && Math.Abs(fieldValueFloat - DEFAULT_VALUE) < 1e-6)
            || (fieldValue is double fieldValueDouble && Math.Abs(fieldValueDouble - DEFAULT_VALUE) < 1e-6)
           )
        {
            return true;
        }
        return false;
    }
}
