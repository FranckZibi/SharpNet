using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using log4net;

namespace SharpNet.HyperParameters;

public interface ISample
{
    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(ISample));
    #endregion

    #region constructor
    static ISample()
    {
    }
    #endregion

    /// <summary>
    /// Try to fix any errors found in the current sample.
    /// return true if the sample has been fixed successfully (so that it can be used for training)
    /// return false if we failed to fix the sample: it should be discarded and never used for training
    /// </summary>
    bool FixErrors();
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
    /// <summary>
    /// names of all the Hyper-Parameters associated with the sample
    /// </summary>
    /// <returns></returns>
    HashSet<string> HyperParameterNames();
    string ComputeHash();
    ISample Clone();

    /// <summary>
    /// true if the sample runs on GPU
    /// false if it runs on CPU
    /// </summary>
    // ReSharper disable once UnusedMember.Global
    bool UseGPU { get; }
    /// <summary>
    /// When training several models in parallel, 'taskId' is the index of the model being trained
    /// (in range [0, number_of_models_trained_in_parallel[ )
    /// This is only used for GPU training to make sure each model is trained in a different GPU
    /// </summary>
    /// <param name="taskId"></param>
    void SetTaskId(int taskId);

    public static string ToPath(string workingDirectory, string sampleName)
    {
        return Path.Combine(workingDirectory, sampleName + ".conf");
    }
    public static string ToJsonPath(string workingDirectory, string sampleName)
    {
        return Path.Combine(workingDirectory, sampleName + "_conf.json");
    }
    public static string SampleName(string modelName, int sampleIndex)
    {
        if (sampleIndex == 0)
        {
            return modelName;
        }
        return modelName + "_" + sampleIndex;
    }
    public static IDictionary<string, string> LoadConfig(string workingDirectory, string sampleName)
    {
        var textPath = ToPath(workingDirectory, sampleName);
        var jsonPath = ToJsonPath(workingDirectory, sampleName);

        if (File.Exists(textPath) && File.Exists(jsonPath))
        {
            throw new ArgumentException($"both files {textPath} and {jsonPath} exist");
        }
        return File.Exists(textPath) ? LoadTextConfig(textPath) : LoadJsonConfig(jsonPath);
    }
    public static T LoadSample<T>(string workingDirectory, string sampleName) where T:ISample
    {
        var sample = (T)Activator.CreateInstance(typeof(T), true);
        var content = LoadConfig(workingDirectory, sampleName);
        sample.Set(Utils.FromString2String_to_String2Object(content));
        return sample;
    }

    [SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
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
            var splitted = e.Split(":").Select(s => s.Trim()).ToList();
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
        return res;
    }
}