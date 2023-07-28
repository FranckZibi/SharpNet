using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SharpNet.HyperParameters;

public abstract class AbstractSample : ISample
{
    #region private fields
    private readonly HashSet<string> _mandatoryCategoricalHyperParameters;
    #endregion


    public virtual string ToPath(string workingDirectory, string sampleName)
    {
        return Path.Combine(workingDirectory, sampleName + "."+GetType().Name+".conf");
    }
    public const int DEFAULT_VALUE = -6666;

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
        var fieldsToDiscardInComputeHash = FieldsToDiscardInComputeHash();
        bool Accept(string fieldName, object fieldValue)
        {
            return DefaultAcceptForConfigContent(fieldName, fieldValue) && !fieldsToDiscardInComputeHash.Contains(fieldName);
        }
        return Utils.ComputeHash(ToConfigContent(Accept), 10);
    }
    public virtual ISample Clone()
    {
        var clonedInstance = (ISample)Activator.CreateInstance(GetType(), true);
        clonedInstance?.Set(ToDictionaryConfigContent(DefaultAcceptForConfigContent));
        return clonedInstance;
    }
    public virtual List<string> Save(string workingDirectory, string modelName)
    {
        var path = ToPath(workingDirectory, modelName);
        Save(path);
        return new List<string> { path };
    }
    public virtual bool MustUseGPU => false;

    public void Save(string path)
    {
        var content = ToConfigContent(DefaultAcceptForConfigContent);
        File.WriteAllText(path, content);
    }
    public virtual List<string> SampleFiles(string workingDirectory, string modelName)
    {
        return new List<string> { ToPath(workingDirectory, modelName) };
    }
    public virtual void Set(string fieldName, object fieldValue)
    {
        ClassFieldSetter.Set(this, fieldName, fieldValue);
    }
    public virtual object Get(string fieldName)
    {
        return ClassFieldSetter.Get(this, fieldName);
    }
    public virtual bool FixErrors()
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
        if (hyperParameterType == typeof(string) || hyperParameterType == typeof(bool) || hyperParameterType.IsEnum || hyperParameterType.IsClass)
        {
            return true;       
        }
        throw new ArgumentException($"can't determine if {hyperParameterName} ({hyperParameterType}) field of class {GetType()} is categorical");
    }
    #endregion

    public virtual HashSet<string> FieldsToDiscardInComputeHash()
    {
        return new HashSet<string>();
    }

    protected virtual string ToConfigContent(Func<string, object, bool> accept)
    {
        var result = new List<string>();
        foreach (var (parameterName,fieldValue) in ToDictionaryConfigContent(accept).OrderBy(f => f.Key))
        {
            var fieldValueToString = Utils.FieldValueToString(fieldValue);
            result.Add($"{parameterName} = {fieldValueToString}");
        }
        return string.Join(Environment.NewLine, result) + Environment.NewLine;
    }
    protected IDictionary<string,object> ToDictionaryConfigContent(Func<string, object, bool> accept)
    {
        var result = new Dictionary<string, object>();
        foreach (var parameterName in HyperParameterNames())
        {
            var fieldValue = Get(parameterName);
            if (accept == null || accept(parameterName, fieldValue))
            {
                result[parameterName] = fieldValue;
            }
        }
        return result;
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
        if (   (fieldValue.GetType().IsEnum && ((int)fieldValue ==  DEFAULT_VALUE) )
            || (fieldValue is int fieldValueInt && (fieldValueInt == DEFAULT_VALUE))
            || (fieldValue is float fieldValueFloat && MathF.Abs(fieldValueFloat - DEFAULT_VALUE) < 1e-6f)
            || (fieldValue is double fieldValueDouble && Math.Abs(fieldValueDouble - DEFAULT_VALUE) < 1e-6)
           )
        {
            return true;
        }
        return false;
    }

    public virtual void SetTaskId(int taskId)
    {
    }

    public virtual void FillSearchSpaceWithDefaultValues(IDictionary<string, object> hyperParameterSearchSpace)
    {
    }

    private static bool DefaultAcceptForConfigContent(string fieldName, object fieldValue)
    {
        return !IsDefaultValue(fieldValue);
    }


}
