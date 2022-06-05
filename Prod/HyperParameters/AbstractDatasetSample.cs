using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
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
    private static readonly ConcurrentDictionary<string, CpuTensor<float>> LoadPredictionsInTargetFormat_Cache = new();
    #endregion


    #region constructors
    protected AbstractDatasetSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(mandatoryCategoricalHyperParameters)
    {
    }
    public static AbstractDatasetSample ValueOf(string workingDirectory, string sampleName)
    {
        try { return ISample.LoadSample<Natixis70DatasetSample>(workingDirectory, sampleName); } catch { }
        try { return ISample.LoadSample<AmazonEmployeeAccessChallengeDatasetSample>(workingDirectory, sampleName); } catch { }
        throw new ArgumentException($"can't load a {nameof(AbstractDatasetSample)} with name {sampleName} from directory {workingDirectory}");
    }
    #endregion

    #region Hyper-Parameters
    public double PercentageInTraining = 0.8;

    public string Train_XDatasetPath;
    public string Train_YDatasetPath;
    public string Train_XYDatasetPath;

    public string Validation_XDatasetPath;
    public string Validation_YDatasetPath;
    public string Validation_XYDatasetPath;

    public string Test_XDatasetPath;
    public string Test_YDatasetPath;
    public string Test_XYDatasetPath;
    #endregion

    public AbstractDatasetSample CopyWithNewPercentageInTraining(double newPercentageInTraining)
    {
        var cloned = (AbstractDatasetSample)Clone();
        cloned.PercentageInTraining = newPercentageInTraining;
        cloned.Train_XDatasetPath = cloned.Train_YDatasetPath = null;
        cloned.Validation_XDatasetPath = cloned.Validation_YDatasetPath = null;
        return cloned;
    }


    protected virtual MetricEnum GetMetric()
    {
        throw new NotImplementedException();
    }

    protected virtual LossFunctionEnum GetLoss()
    {
        throw new NotImplementedException();
    }

    protected override HashSet<string> FieldsToDiscardInComputeHash()
    {
        return new HashSet<string>{nameof(Train_XDatasetPath), nameof(Train_YDatasetPath), nameof(Train_XYDatasetPath), nameof(Validation_XDatasetPath), nameof(Validation_YDatasetPath), nameof(Validation_XYDatasetPath), nameof(Test_XDatasetPath), nameof(Test_YDatasetPath), nameof(Test_XYDatasetPath) };
    }

    public float ComputeScore(CpuTensor<float> true_predictions_true_in_target_format, CpuTensor<float> predictions_in_target_format)
    {
        if (true_predictions_true_in_target_format == null || predictions_in_target_format == null)
        {
            return float.NaN;
        }
        Debug.Assert(true_predictions_true_in_target_format.SameShape(predictions_in_target_format));
        var y_true = true_predictions_true_in_target_format.DropColumns(IndexColumnsInPredictionsInTargetFormat());
        var y_pred = predictions_in_target_format.DropColumns(IndexColumnsInPredictionsInTargetFormat());

        using var buffer = new CpuTensor<float>(new[] { y_true.Shape[0] });
        return (float)y_true.ComputeMetric(y_pred, GetMetric(), GetLoss(), buffer);
    }


    /// <summary>
    /// in some cases, the Dataset (in Model Format) must have a number of rows that is a multiple of some constant
    /// </summary>
    /// <returns></returns>
    public virtual int DatasetRowsInModelFormatMustBeMultipleOf()
    {
        return 1;
    }
    public abstract List<string> CategoricalFeatures();
    public virtual char GetSeparator() { return ',';}
    /// <summary>
    /// true if predictions files have header
    /// </summary>
    /// <returns></returns>
    // ReSharper disable once MemberCanBeProtected.Global
    // ReSharper disable once VirtualMemberNeverOverridden.Global
    public virtual bool HeaderInPredictionFile() { return true;}
    public abstract void SavePredictionsInTargetFormat(CpuTensor<float> predictionsInTargetFormat, string path);
    public abstract IDataSet TestDataset();
    
    /// <summary>
    /// returns the train and validation dataset
    /// </summary>
    /// <returns></returns>
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

    public abstract CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(CpuTensor<float> predictionsInModelFormat);
    public virtual CpuTensor<float> PredictionsInModelFormat_2_PredictionsInTargetFormat(string dataframe_path)
    {
        throw new NotImplementedException();
    }

    public CpuTensor<float> LoadPredictionsInTargetFormat(string path)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            return null;
        }

        if (LoadPredictionsInTargetFormat_Cache.TryGetValue(path, out var res))
        {
            return res;
        }

        var y_pred = Dataframe.Load(path, HeaderInPredictionFile(), GetSeparator()).Tensor;
        LoadPredictionsInTargetFormat_Cache.TryAdd(path, y_pred);
        return y_pred;
    }

}