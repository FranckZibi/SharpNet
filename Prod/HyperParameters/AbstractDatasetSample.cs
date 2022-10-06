using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.Datasets.AmazonEmployeeAccessChallenge;
using SharpNet.Datasets.Natixis70;

namespace SharpNet.HyperParameters;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public abstract class AbstractDatasetSample : AbstractSample
{
    #region private fields
    private static readonly ConcurrentDictionary<string, DataFrame> LoadPredictionsInTargetFormat_Cache = new();
    #endregion


    #region constructors
    protected AbstractDatasetSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(mandatoryCategoricalHyperParameters)
    {
        Name = GetType().Name.Replace("DatasetSample", "");
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

    public string Name { get; }

    public virtual MetricEnum GetMetric()
    {
        throw new NotImplementedException();
    }

    public virtual LossFunctionEnum GetLoss()
    {
        throw new NotImplementedException();
    }

    public abstract Objective_enum GetObjective();

    public bool IsRegressionProblem => GetObjective() == Objective_enum.Regression;
    public bool IsClassificationProblem => GetObjective() == Objective_enum.Classification;

    protected override HashSet<string> FieldsToDiscardInComputeHash()
    {
        return new HashSet<string>{nameof(Train_XDatasetPath), nameof(Train_YDatasetPath), nameof(Train_XYDatasetPath), nameof(Validation_XDatasetPath), nameof(Validation_YDatasetPath), nameof(Validation_XYDatasetPath), nameof(Test_XDatasetPath), nameof(Test_YDatasetPath), nameof(Test_XYDatasetPath) };
    }

    public IScore ComputeMetricScore(DataFrame y_true_in_target_format, DataFrame y_pred_in_target_format)
    {
        if (y_true_in_target_format == null || y_pred_in_target_format == null)
        {
            return null;
        }
        Debug.Assert(y_true_in_target_format.Shape.SequenceEqual(y_pred_in_target_format.Shape));
        var y_true = y_true_in_target_format.FloatCpuTensor().DropColumns(IndexColumnsInPredictionsInTargetFormat());
        var y_pred = y_pred_in_target_format.FloatCpuTensor().DropColumns(IndexColumnsInPredictionsInTargetFormat());

        using var buffer = new CpuTensor<float>(y_true.ComputeMetricBufferShape(GetMetric()));
        return new Score ( (float)y_true.ComputeMetric(y_pred, GetMetric(), GetLoss(), buffer) , GetMetric());
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

    /// <summary>
    /// features used to identify a row in the dataset
    /// such features should be ignored during the training
    /// </summary>
    /// <returns>list of id features </returns>
    public abstract List<string> IdFeatures();

    /// <summary>
    /// list of target feature names (usually a single element)
    /// </summary>
    /// <returns>list of target feature names </returns>
    public abstract List<string> TargetLabels();

    
    /// <summary>
    /// by default, the prediction file starts first with the ids columns, then with the target columns
    /// </summary>
    /// <returns></returns>
    public virtual List<string> PredictionInTargetFormatFeatures()
    {
        var res = IdFeatures().ToList();
        res.AddRange(TargetLabels());
        return res;
    }


    public virtual char GetSeparator() { return ',';}
    /// <summary>
    /// true if predictions files have header
    /// </summary>
    /// <returns></returns>
    // ReSharper disable once MemberCanBeProtected.Global
    // ReSharper disable once VirtualMemberNeverOverridden.Global
    public virtual bool HeaderInPredictionFile() { return true;}
    public virtual void SavePredictionsInTargetFormat(DataFrame predictionsInTargetFormat, string path)
    {
        //var start = Stopwatch.StartNew();
        predictionsInTargetFormat.to_csv(path);
        //ISample.Log.Debug($"SavePredictionsInTargetFormat in {path} took {start.Elapsed.TotalSeconds}s");
    }




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

    public abstract DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat);
    public virtual DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(string dataframe_path)
    {
        throw new NotImplementedException();
    }

    public DataFrame LoadPredictionsInTargetFormat(string path)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            return null;
        }

        if (LoadPredictionsInTargetFormat_Cache.TryGetValue(path, out var res))
        {
            return res;
        }

        var y_pred = DataFrame.LoadFloatDataFrame(path, HeaderInPredictionFile());
        LoadPredictionsInTargetFormat_Cache.TryAdd(path, y_pred);
        return y_pred;
    }
}