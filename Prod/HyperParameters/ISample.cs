using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using log4net;

namespace SharpNet.Hyperparameters;

public interface ISample
{
    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(ISample));
    #endregion
    #region private fields & properties
    private static readonly Dictionary<string, Type> sampleClassName_to_type = new ();
    #endregion

    #region constructor
    static ISample()
    {
        var sampleTypes = AppDomain.CurrentDomain.GetAssemblies()
            .SelectMany(s => s.GetTypes())
            .Where(p => typeof(ISample).IsAssignableFrom(p));
        foreach (var t in sampleTypes)
        {
            sampleClassName_to_type[t.Name] = t;
        }
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
    Type GetFieldType(string HyperparameterName);
    bool IsCategoricalHyperparameter(string HyperparameterName);
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
    HashSet<string> HyperparameterNames();
    string ComputeHash();
    ISample Clone();

    /// <summary>
    /// true if the sample must be run on GPU
    /// false if it should run on CPU
    /// </summary>
    // ReSharper disable once UnusedMember.Global
    bool MustUseGPU { get; }
    /// <summary>
    /// When training several models in parallel, 'taskId' is the index of the model being trained
    /// (in range [0, number_of_models_trained_in_parallel[ )
    /// This is only used for GPU training to make sure each model is trained in a different GPU
    /// </summary>
    /// <param name="taskId"></param>
    void SetTaskId(int taskId);

   
    public static string SampleName(string modelName, int sampleIndex)
    {
        if (sampleIndex == 0)
        {
            return modelName;
        }
        return modelName + "_" + sampleIndex;
    }

    #region serialization and deserialization
    public static ISample Load(string workingDirectory, string sampleName, Action<IDictionary<string,string>> contentUpdater = null)
    {
        var allMatchingFiles = new DirectoryInfo(workingDirectory).GetFileSystemInfos(sampleName + ".*.conf").Union(new DirectoryInfo(workingDirectory).GetFileSystemInfos(sampleName + "_conf.*.json")).Select(t=>t.Name).OrderBy(x=>x).ToList();
        if (allMatchingFiles.Count >= 2)
        {
            Log.Warn($"found several files in directory {workingDirectory} for sample {sampleName} : {string.Join(" ", allMatchingFiles)}, keeping only the first one {allMatchingFiles[0]}");
        }
        if (allMatchingFiles.Count == 0)
        {
            string errorMsg = $"No file in directory {workingDirectory} for sample {sampleName}";
            Log.Error(errorMsg);
            throw new Exception(errorMsg);
        }
        var fileName = allMatchingFiles[0];
        var filePath = Path.Combine(workingDirectory, fileName);
        var className = fileName.Split('.')[^2];
        var classType = sampleClassName_to_type[className];
        var sample = (ISample)Activator.CreateInstance(classType, true);
        var content = fileName.ToLower().EndsWith("json") ? LoadJsonConfig(filePath) : LoadTextConfig(filePath);
        if (contentUpdater != null)
        {
            contentUpdater(content);
        }
        sample?.Set(Utils.FromString2String_to_String2Object(content));
        return sample;
    }
    List<string> Save(string workingDirectory, string modelName);
    private static IDictionary<string, string> LoadTextConfig(string path)
    {
        if (!File.Exists(path))
        {
            throw new ArgumentException($"invalid file path {path}");
        }
        var res = new Dictionary<string, string>();
        foreach (var e in File.ReadAllLines(path))
        {
            if (string.IsNullOrEmpty(e.Trim()) || e.StartsWith("#"))
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
    #endregion

    void FillSearchSpaceWithDefaultValues(IDictionary<string, object> HyperparameterSearchSpace);
}