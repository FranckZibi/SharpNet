using System;
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
    #region constructors
    protected AbstractDatasetSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(mandatoryCategoricalHyperParameters)
    {
        Name = GetType().Name.Replace("DatasetSample", "");
    }
    public static AbstractDatasetSample ValueOf(string workingDirectory, string sampleName)
    {
        try { return ISample.LoadSample<Natixis70DatasetSample>(workingDirectory, sampleName); } catch { }
        try { return ISample.LoadSample<AmazonEmployeeAccessChallengeDatasetSample>(workingDirectory, sampleName); } catch { }
        try { return ISample.LoadSample<WasYouStayWorthItsPriceDatasetSample>(workingDirectory, sampleName); } catch { }
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


    public ITrainingAndTestDataSet LoadTrainingAndValidationDataset_Encoded_InTargetFormat()
    {
        var training = LoadTrainingAndTestDataSet(Train_XDatasetPath, Train_YDatasetPath, Train_XYDatasetPath);
        var validation = LoadTrainingAndTestDataSet(Validation_XDatasetPath, Validation_YDatasetPath, Validation_XYDatasetPath);
        return new TrainingAndTestDataset(training, validation, Name);
    }
    public ITrainingAndTestDataSet LoadTestDataset_Encoded_InTargetFormat()
    {
        var test = LoadTrainingAndTestDataSet(Test_XDatasetPath, Test_YDatasetPath, Test_XYDatasetPath);
        return new TrainingAndTestDataset(test, null, Name);
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

    public IScore ComputeRankingEvaluationMetric(DataFrame y_true_InTargetFormat, DataFrame y_pred_InTargetFormat)
    {
        if (y_true_InTargetFormat == null || y_pred_InTargetFormat == null)
        {
            return null;
        }
        var y_true = DropIdColumnsIfFound(y_true_InTargetFormat).FloatCpuTensor();
        var y_pred = DropIdColumnsIfFound(y_pred_InTargetFormat).FloatCpuTensor();
        Debug.Assert(y_true.Shape.SequenceEqual(y_pred.Shape));
        var rankingEvaluationMetric = GetRankingEvaluationMetric();
        using var buffer = new CpuTensor<float>(y_true.ComputeMetricBufferShape(rankingEvaluationMetric));
        var evaluationMetric = y_true.ComputeEvaluationMetric(y_pred, rankingEvaluationMetric, buffer);
        return new Score ( (float)evaluationMetric , rankingEvaluationMetric);
    }


    private IDataSet LoadTrainingAndTestDataSet(string XDatasetPath, string YDatasetPath,  string XYDatasetPath)
    {
        DataFrame x = null;
        DataFrame y = null;
        if (!string.IsNullOrEmpty(XYDatasetPath))
        {
            Debug.Assert(string.IsNullOrEmpty(XDatasetPath));
            Debug.Assert(string.IsNullOrEmpty(YDatasetPath));
            var xy = DataFrame.LoadFloatDataFrame(XYDatasetPath, true);
            x = xy.Keep(Utils.Without(xy.ColumnNames, TargetLabels));
            y = xy.Keep(TargetLabels);
        }
        else
        {
            if (!string.IsNullOrEmpty(XDatasetPath))
            {
                Debug.Assert(string.IsNullOrEmpty(XYDatasetPath));
                x = DataFrame.LoadFloatDataFrame(XDatasetPath, true);
            }
            if (!string.IsNullOrEmpty(YDatasetPath))
            {
                y = DropIdColumnsIfFound(DataFrame.LoadFloatDataFrame(YDatasetPath, true));
            }
        }
        if (x == null)
        {
            Debug.Assert(y == null);
            return null;
        }
        return new InMemoryDataSet(x.FloatCpuTensor(), y?.FloatCpuTensor(), Name, GetObjective(), null, x.ColumnNames, CategoricalFeatures, IdColumns, TargetLabels, false, GetSeparator());
    }

    private DataFrame DropIdColumnsIfFound(DataFrame df)
    {
        if (IdColumns.Length == 0)
        {
            return df;
        }
        var intersection = Utils.Intersect(df.ColumnNames, IdColumns);
        if (intersection.Count == 0)
        {
            return df;
        }

        if (intersection.Count != IdColumns.Length)
        {
            throw new Exception($"found only a part {string.Join(' ', intersection)} of Id Columns ({string.Join(' ', IdColumns)})");
        }
        return df.Drop(IdColumns);
    }

    //private DataFrame ExtractIdColumnsIfFound(DataFrame df)
    //{
    //    if (IdColumns.Length == 0)
    //    {
    //        return null;
    //    }
    //    var intersection = Utils.Intersect(df.ColumnNames, IdColumns);
    //    if (intersection.Count == 0)
    //    {
    //        return null;
    //    }

    //    if (intersection.Count != IdColumns.Length)
    //    {
    //        throw new Exception($"found only a part {string.Join(' ', intersection)} of Id Columns ({string.Join(' ', IdColumns)})");
    //    }
    //    return df.Keep(IdColumns);
    //}



    /// <summary>
    /// in some cases, the Dataset (in Model Format) must have a number of rows that is a multiple of some constant
    /// </summary>
    /// <returns></returns>
    public virtual int DatasetRowsInModelFormatMustBeMultipleOf()
    {
        return 1;
    }
    public abstract string[] CategoricalFeatures { get; }

    /// <summary>
    /// features used to identify a row in the dataset
    /// such features should be ignored during the training
    /// </summary>
    /// <returns>list of id features </returns>
    public abstract string[] IdColumns { get; }

    /// <summary>
    /// list of target feature names (usually a single element)
    /// </summary>
    /// <returns>list of target feature names </returns>
    public abstract string[] TargetLabels { get; }


    /// <summary>
    /// by default, the prediction file starts first with the ids columns, then with the target columns
    /// </summary>
    /// <returns></returns>
    protected virtual List<string> PredictionInTargetFormatHeader()
    {
        var res = IdColumns.ToList();
        res.AddRange(TargetLabels);
        return res;
    }


    public virtual char GetSeparator() { return ',';}
    public virtual void SavePredictionsInTargetFormat(DataFrame encodedPredictionsInTargetFormat, string path)
    {
        encodedPredictionsInTargetFormat.to_csv(path, GetSeparator().ToString(), true);
    }
    public virtual void SavePredictionsInModelFormat(DataFrame predictionsInTargetFormat, string path)
    {
        predictionsInTargetFormat.to_csv(path, GetSeparator().ToString(), true);
    }
    public virtual DataFrame LoadPredictionsInModelFormat(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }
        return DataFrame.LoadFloatDataFrame(path, true);
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
}
