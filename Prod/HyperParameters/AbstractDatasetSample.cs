using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.Datasets.AmazonEmployeeAccessChallenge;
using SharpNet.Datasets.Natixis70;

namespace SharpNet.HyperParameters;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public abstract class AbstractDatasetSample : AbstractSample
{
    #region private fields
    private static readonly ConcurrentDictionary<string, CpuTensor<float>> LoadPredictionsInTargetFormatWithoutIndex_Cache = new();
    #endregion


    #region constructors
    protected AbstractDatasetSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(mandatoryCategoricalHyperParameters)
    {
    }
    public static AbstractDatasetSample ValueOf(string workingDirectory, string sampleName)
    {
        try { return Natixis70DatasetSample.ValueOfNatixis70DatasetSample(workingDirectory, sampleName); } catch { }
        try { return AmazonEmployeeAccessChallengeDatasetSample.ValueOfAmazonEmployeeAccessChallengeDatasetSample(workingDirectory, sampleName); } catch { }
        throw new ArgumentException($"can't load {nameof(AbstractDatasetSample)} for sample {sampleName} from {workingDirectory}");
    }
    #endregion

    #region Hyper-Parameters
    public string Train_XDatasetPath = "";
    public string Train_YDatasetPath = "";
    public string Train_PredictionsPath = "";

    public string Validation_XDatasetPath = "";
    public string Validation_YDatasetPath = "";
    public string Validation_PredictionsPath = "";

    public string Test_DatasetPath = "";
    public string Test_PredictionsPath = "";
    #endregion

 
    public abstract List<string> CategoricalFeatures();
    public abstract void SavePredictionsInTargetFormat(CpuTensor<float> predictionsInTargetFormat, string path);
    public abstract IDataSet TestDataset();
    public abstract ITrainingAndTestDataSet SplitIntoTrainingAndValidation();
    //public abstract IDataSet FullTraining();
    //public abstract CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(string dataframe_path);
    //public abstract (CpuTensor<float> trainPredictionsInTargetFormatWithoutIndex, CpuTensor<float> validationPredictionsInTargetFormatWithoutIndex, CpuTensor<float> testPredictionsInTargetFormatWithoutIndex) LoadAllPredictionsInTargetFormatWithoutIndex();

    /// <summary>
    /// list of indexes of columns in the prediction file that are indexes, not actual predicted values
    /// </summary>
    /// <returns></returns>
    public virtual IList<int> IndexColumnsInPredictionsInTargetFormat()
    {
        //by default, we'll consider that the first column of the predicted file contains an index
        return new[] { 0 };
    }

    protected (CpuTensor<float> trainPredictionsInTargetFormatWithoutIndex, CpuTensor<float> validationPredictionsInTargetFormatWithoutIndex, CpuTensor<float> testPredictionsInTargetFormatWithoutIndex) 
        LoadAllPredictionsInTargetFormatWithoutIndex(bool header, char separator)
    {
        return
            (LoadPredictionsInTargetFormatWithoutIndex(Train_PredictionsPath, header, separator),
             LoadPredictionsInTargetFormatWithoutIndex(Validation_PredictionsPath, header, separator),
             LoadPredictionsInTargetFormatWithoutIndex(Test_PredictionsPath, header, separator));
    }



    public abstract CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(CpuTensor<float> predictionsInModelFormat);
    public abstract CpuTensor<float> PredictionsInTargetFormat_2_PredictionsInModelFormat(CpuTensor<float> predictionsInTargetFormat);

    private CpuTensor<float> LoadPredictionsInTargetFormatWithoutIndex(string path, bool header, char separator)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            return null;
        }

        if (LoadPredictionsInTargetFormatWithoutIndex_Cache.TryGetValue(path, out var res))
        {
            return res;
        }

        var y_pred = Dataframe.Load(path, header, separator).Tensor;
        if (IndexColumnsInPredictionsInTargetFormat().Count != 0)
        {
            y_pred = y_pred.DropColumns(IndexColumnsInPredictionsInTargetFormat());
        }

        LoadPredictionsInTargetFormatWithoutIndex_Cache.TryAdd(path, y_pred);
        return y_pred;
    }
}