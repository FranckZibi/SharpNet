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

    /// <summary>
    /// number of splits for KFold
    /// a value less or equal 1 will disable KFold
    /// a value of 2 or more will enable KFold
    ///  1/ the dataset will be splitted into 'KFold' part:
    ///  2/ we'll perform 'KFold' distinct training
    ///     in each training we'll use a distinct 'Kfold' part for validation,
    ///     and all the remaining 'Kfold-1' parts for training
    /// </summary>
    public int KFold = 1;

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

    public AbstractDatasetSample CopyWithNewPercentageInTrainingAndKFold(double newPercentageInTraining, int newKFold)
    {
        var cloned = (AbstractDatasetSample)Clone();
        cloned.PercentageInTraining = newPercentageInTraining;
        cloned.KFold = newKFold;
        cloned.Train_XDatasetPath = cloned.Train_YDatasetPath = null;
        cloned.Validation_XDatasetPath = cloned.Validation_YDatasetPath = null;
        return cloned;
    }

    public string Name { get; }

    /// <summary>
    /// the evaluation metric used to rank the final submission
    /// depending on the evaluation metric, higher (ex: Accuracy) or lower (ex: Rmse) may be better
    /// </summary>
    /// <returns></returns>
    protected abstract EvaluationMetricEnum GetRankingEvaluationMetric();

    public abstract Objective_enum GetObjective();

    public bool IsRegressionProblem => GetObjective() == Objective_enum.Regression;
    public bool IsClassificationProblem => GetObjective() == Objective_enum.Classification;

    protected override HashSet<string> FieldsToDiscardInComputeHash()
    {
        return new HashSet<string>{nameof(Train_XDatasetPath), nameof(Train_YDatasetPath), nameof(Train_XYDatasetPath), nameof(Validation_XDatasetPath), nameof(Validation_YDatasetPath), nameof(Validation_XYDatasetPath), nameof(Test_XDatasetPath), nameof(Test_YDatasetPath), nameof(Test_XYDatasetPath) };
    }

    public IScore ComputeRankingEvaluationMetric(DataFrame y_true_in_target_format, DataFrame y_pred_in_target_format)
    {
        if (y_true_in_target_format == null || y_pred_in_target_format == null)
        {
            return null;
        }
        Debug.Assert(y_true_in_target_format.Shape.SequenceEqual(y_pred_in_target_format.Shape));
        var idxOfIdColumns = y_true_in_target_format.ColumnNamesToIndexes(IdColumns());
        var y_true = y_true_in_target_format.FloatCpuTensor().DropColumns(idxOfIdColumns);
        var y_pred = y_pred_in_target_format.FloatCpuTensor().DropColumns(idxOfIdColumns);

        var rankingEvaluationMetric = GetRankingEvaluationMetric();
        using var buffer = new CpuTensor<float>(y_true.ComputeMetricBufferShape(rankingEvaluationMetric));
        var evaluationMetric = y_true.ComputeEvaluationMetric(y_pred, rankingEvaluationMetric, buffer);
        return new Score ( (float)evaluationMetric , rankingEvaluationMetric);
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
    public abstract List<string> IdColumns();

    /// <summary>
    /// list of target feature names (usually a single element)
    /// </summary>
    /// <returns>list of target feature names </returns>
    public abstract List<string> TargetLabels();

    
    /// <summary>
    /// by default, the prediction file starts first with the ids columns, then with the target columns
    /// </summary>
    /// <returns></returns>
    protected virtual List<string> PredictionInTargetFormatHeader()
    {
        var res = IdColumns().ToList();
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
        predictionsInTargetFormat.to_csv(path, GetSeparator().ToString(), true);
    }
    public virtual void SavePredictionsInModelFormat(DataFrame predictionsInTargetFormat, string path)
    {
        predictionsInTargetFormat.to_csv(path, GetSeparator().ToString(), true);
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
    /// transform a DataFrame of prediction in model format (with the Id Columns at left)
    /// to a DataFrame in Challenge expected format (also with the Id Column at left)
    /// </summary>
    /// <param name="predictionsInModelFormat_with_IdColumns"></param>
    /// <returns></returns>
    public abstract DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat_with_IdColumns);
    public virtual DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(string dataframe_path)
    {
        throw new NotImplementedException();
    }

    public override bool FixErrors()
    {
        if (!base.FixErrors())
        {
            return false;
        }
        if (KFold>=2 && PercentageInTraining != 1.0)
        {
            PercentageInTraining = 1.0;
        }
        return true;
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