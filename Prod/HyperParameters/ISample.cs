﻿using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Datasets.Natixis70;
using SharpNet.HPO;
using SharpNet.LightGBM;

namespace SharpNet.HyperParameters;

public interface IMetricFunction
{
    MetricEnum GetMetric();
    LossFunctionEnum GetLoss();
}

public interface ISample
{
    #region private fields
    private static readonly string WrongPath;
    private static readonly string ValidPath;
    #endregion

    #region constructor
    static ISample()
    {
        var localApplicationFolderPath = Utils.LocalApplicationFolderPath;
        WrongPath = localApplicationFolderPath.Contains("fzibi") ? localApplicationFolderPath.Replace("fzibi", "Franck") : localApplicationFolderPath.Replace("Franck", "fzibi");
        ValidPath = localApplicationFolderPath;
    }
    #endregion

    /// <summary>
    /// method to be called ofter building a new sample
    /// it will update it using any needed standardization/ normalization /etc...
    /// return true if everything working OK
    /// return false if the sample is invalid and should be discarded
    /// </summary>
    bool PostBuild();
    void Set(IDictionary<string, object> dico);
    void Set(string fieldName, object fieldValue);
    object Get(string fieldName);
    Type GetFieldType(string hyperParameterName);
    bool IsCategoricalHyperParameter(string hyperParameterName);
    void Save(string workingDirectory, string modelName);

    /// <summary>
    /// all Hyper-Parameters file associated with the Sample
    /// </summary>
    /// <param name="workingDirectory"></param>
    /// <param name="modelName"></param>
    /// <returns></returns>
    List<string> SampleFiles(string workingDirectory, string modelName);

    HashSet<string> HyperParameterNames();
    string ComputeHash();
    CpuTensor<float> Y_Train_dataset_to_Perfect_Predictions(string y_train_dataset);



    public static string ToPath(string workingDirectory, string modelName)
    {
        return Path.Combine(workingDirectory, modelName + ".conf");
    }
    public static string ToJsonPath(string workingDirectory, string modelName)
    {
        return Path.Combine(workingDirectory, modelName + "_conf.json");
    }

    public static IDictionary<string, string> LoadConfig(string workingDirectory, string modelName)
    {
        var textPath = ToPath(workingDirectory, modelName);
        var jsonPath = ToJsonPath(workingDirectory, modelName);

        if (File.Exists(textPath) && File.Exists(jsonPath))
        {
            throw new ArgumentException($"both files {textPath} and {jsonPath} exist");
        }
        return File.Exists(textPath) ? LoadTextConfig(textPath) : LoadJsonConfig(jsonPath);
    }

    private static IDictionary<string, string> LoadTextConfig(string path)
    {
        if (!File.Exists(path))
        {
            throw new ArgumentException($"invalid file path {path}");
        }
        var res = new Dictionary<string, string>();
        foreach (var e in File.ReadAllLines(path))
        {
            if (string.IsNullOrEmpty(e.Trim()))
            {
                continue;
            }
            var splitted = e.Split(" = ");
            if (splitted.Length != 2)
            {
                throw new ArgumentException($"invalid line {e}");
            }
            res[splitted[0].Trim()] = splitted[1].Trim();
        }
        NormalizePath(res);
        return res;
    }


    private static IDictionary<string, string> LoadJsonConfig(string path)
    {
        if (!File.Exists(path))
        {
            throw new ArgumentException($"invalid file path {path}");
        }
        var res = new Dictionary<string, string>();
        foreach (var e in File.ReadAllLines(path))
        {
            if (!e.Contains(":"))
            {
                continue;
            }
            var splitted = e.Split(":").Select(s=>s.Trim()).ToList();
            if (splitted.Count != 2)
            {
                throw new ArgumentException($"invalid line {e}");
            }

            var strName = splitted[0];
            if (!strName.StartsWith('"') || !strName.EndsWith('"'))
            {
                throw new ArgumentException($"invalid line {e}");
            }

            strName = strName.Substring(1, strName.Length - 2);
            var strValue = splitted[1].TrimEnd(' ', ',', '\t');
            if (strValue.StartsWith('"') && strValue.EndsWith('"'))
            {
                strValue = strValue.Substring(1, strValue.Length - 2);
            }
            if (strValue.StartsWith('[') && strValue.EndsWith(']'))
            {
                strValue = strValue.Substring(1, strValue.Length - 2);
            }
            res[strName] = strValue;
        }
        NormalizePath(res);
        return res;
    }

    /// <summary>
    /// the computation are done in 2 different computers (using different path)
    /// Depending on the computer we are currently using, we'll normalize the path for the current computer
    /// </summary>
    /// <param name="config"></param>
    private static void NormalizePath(IDictionary<string, string> config)
    {
        foreach (var key in config.Keys.ToList())
        {
            config[key] = NormalizePath(config[key]);
        }
    }
    public static string NormalizePath(string field)
    {
        return field.Replace(WrongPath, ValidPath);
    }


    public static ISample LoadConfigIntoSample(Func<ISample> createDefaultSample, string workingDirectory, string modelName)
    {
        var sample = createDefaultSample();
        var content = LoadConfig(workingDirectory, modelName);
        sample.Set(Utils.FromString2String_to_String2Object(content));
        return sample;
    }

    [SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
    public static ISample ValueOf(string workingDirectory, string modelName)
    {
        try { return Natixis70_LightGBM_HyperParameters.ValueOf(workingDirectory, modelName); } catch {}
        try { return Natixis70_CatBoost_HyperParameters.ValueOf(workingDirectory, modelName); } catch {}
        try { return Natixis70DatasetHyperParameters.ValueOf(workingDirectory, modelName); } catch {}
        try { return LightGBMSample.ValueOf(workingDirectory, modelName); } catch {}
        try { return WeightsOptimizerHyperParameters.ValueOf(workingDirectory, modelName); } catch {}
        throw new Exception($"can't load sample from model {modelName} in directory {workingDirectory}");
    }

}