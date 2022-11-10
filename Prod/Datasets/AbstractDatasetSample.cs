using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;

namespace SharpNet.Datasets;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public abstract class AbstractDatasetSample : AbstractSample
{
    #region public properties
    public string Name { get; }
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
    #endregion

    #region constructors
    protected AbstractDatasetSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(mandatoryCategoricalHyperParameters)
    {
        Name = GetType().Name.Replace("DatasetSample", "");
    }
    public static AbstractDatasetSample ValueOf(string workingDirectory, string sampleName)
    {
        //try { return ISample.LoadSample<Natixis70DatasetSample>(workingDirectory, sampleName); } catch { }
        //try { return ISample.LoadSample<AmazonEmployeeAccessChallengeDatasetSample>(workingDirectory, sampleName); } catch { }
        try { return ISample.LoadSample<WasYouStayWorthItsPriceDatasetSample>(workingDirectory, sampleName); } catch { }
        throw new ArgumentException($"can't load a {nameof(AbstractDatasetSample)} with name {sampleName} from directory {workingDirectory}");
    }
    #endregion




    public EvaluationMetricEnum DefaultLossFunction
    {
        get
        {
            if (GetObjective() == Objective_enum.Regression)
            {
                return EvaluationMetricEnum.Rmse;
            }
            if (NumClass == 1)
            {
                return EvaluationMetricEnum.BinaryCrossentropy;
            }
            return EvaluationMetricEnum.CategoricalCrossentropy;
        }
    }

    public cudnnActivationMode_t ActivationForLastLayer
    {
        get
        {
            if (GetObjective() == Objective_enum.Regression)
            {
                return cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID;
            }
            if (NumClass == 1)
            {
                return cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID;
            }
            return cudnnActivationMode_t.CUDNN_ACTIVATION_SOFTMAX;
        }
    }


    public enum DatasetType
    {
        Train,      //The Dataset is used for Training (optimization of the Weights of the Model)
        Validation, //The Dataset is used for Validation (optimization of the Hyper-Parameters of the Model)
        Test        //The Dataset is used for Test (final and 'official' Ranking of the Model)
    };

    public string ExtractDatasetPath_InModelFormat(DatasetType datasetType)
    {
        switch (datasetType)
        {
            case DatasetType.Train:
                return string.IsNullOrEmpty(Train_XYDatasetPath_InModelFormat)
                    ? Train_XDatasetPath_InModelFormat
                    : Train_XYDatasetPath_InModelFormat;
            case DatasetType.Validation:
                return string.IsNullOrEmpty(Validation_XYDatasetPath_InModelFormat)
                    ? Validation_XDatasetPath_InModelFormat
                    : Validation_XYDatasetPath_InModelFormat;
            case DatasetType.Test:
                return string.IsNullOrEmpty(Test_XYDatasetPath_InModelFormat)
                    ? Test_XDatasetPath_InModelFormat
                    : Test_XYDatasetPath_InModelFormat;
            default:
                throw new NotSupportedException($"invalid {nameof(DatasetType)}: {datasetType}");
        }
    }


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
    public string Train_XDatasetPath_InTargetFormat;
    public string Train_YDatasetPath_InTargetFormat;
    public string Train_XYDatasetPath_InTargetFormat;
    public string Train_XDatasetPath_InModelFormat;
    public string Train_YDatasetPath_InModelFormat;
    public string Train_XYDatasetPath_InModelFormat;

    public string Validation_XDatasetPath_InTargetFormat;
    public string Validation_YDatasetPath_InTargetFormat;
    public string Validation_XYDatasetPath_InTargetFormat;
    public string Validation_XDatasetPath_InModelFormat;
    public string Validation_YDatasetPath_InModelFormat;
    public string Validation_XYDatasetPath_InModelFormat;

    public string Test_XDatasetPath_InTargetFormat;
    public string Test_YDatasetPath_InTargetFormat;
    public string Test_XYDatasetPath_InTargetFormat;
    public string Test_XDatasetPath_InModelFormat;
    public string Test_YDatasetPath_InModelFormat;
    public string Test_XYDatasetPath_InModelFormat;
    #endregion

    public AbstractDatasetSample CopyWithNewPercentageInTrainingAndKFold(double newPercentageInTraining, int newKFold)
    {
        var cloned = (AbstractDatasetSample)Clone();
        cloned.PercentageInTraining = newPercentageInTraining;
        cloned.KFold = newKFold;
        cloned.SetAll_Train_XDatasetPath(null);
        cloned.SetAll_Validation_XDatasetPath(null);
        return cloned;
    }

    private void SetAll_Train_XDatasetPath(string newValue)
    {
        Train_XDatasetPath_InTargetFormat = Train_YDatasetPath_InTargetFormat = Train_XYDatasetPath_InTargetFormat = newValue;
        Train_XDatasetPath_InModelFormat = Train_YDatasetPath_InModelFormat = Train_XYDatasetPath_InModelFormat = newValue;
    }
    private void SetAll_Validation_XDatasetPath(string newValue)
    {
        Validation_XDatasetPath_InTargetFormat = Validation_YDatasetPath_InTargetFormat = Validation_XYDatasetPath_InTargetFormat = newValue;
        Validation_XDatasetPath_InModelFormat = Validation_YDatasetPath_InModelFormat = Validation_XYDatasetPath_InModelFormat = newValue;
    }


    public InMemoryDataSetV2 LoadTrainDataset() => LoadDataSet(Train_XDatasetPath_InTargetFormat, Train_YDatasetPath_InTargetFormat, Train_XYDatasetPath_InTargetFormat);
    public InMemoryDataSetV2 LoadValidationDataset() => LoadDataSet(Validation_XDatasetPath_InTargetFormat, Validation_YDatasetPath_InTargetFormat, Validation_XYDatasetPath_InTargetFormat);

    ///// <summary>
    ///// return a DataSet with all labeled Data (all data found both in training and validation dataset)
    ///// </summary>
    ///// <returns></returns>
    //public InMemoryDataSet LoadTrainAndValidationDataset() => InMemoryDataSet.MergeVertically(LoadTrainDataset(), LoadValidationDataset());

    public InMemoryDataSetV2 LoadTestDataset() => LoadDataSet(Test_XDatasetPath_InTargetFormat, Test_YDatasetPath_InTargetFormat, Test_XYDatasetPath_InTargetFormat);
    
    //// ReSharper disable once MemberCanBeMadeStatic.Global
    public DataFrame LoadPredictionsInModelFormat(string directory, string fileName)
    {
        if (string.IsNullOrEmpty(fileName))
        {
            return null;
        }
        var path = Path.Combine(directory, fileName);
        if (!File.Exists(path))
        {
            return null;
        }
        return DataFrame.read_float_csv(path);
    }

    public void FillWithDefaultLightGBMHyperParameterValues(IDictionary<string, object> existingHyperParameterValues)
    {
        var objectiveKeyName = nameof(LightGBMSample.objective);
        if (!existingHyperParameterValues.ContainsKey(objectiveKeyName))
        {
            existingHyperParameterValues[objectiveKeyName] = GetDefaultHyperParameterValueForLightGBM(objectiveKeyName);
        }
        var numClassKeyName = nameof(LightGBMSample.num_class);
        if (!existingHyperParameterValues.ContainsKey(numClassKeyName) && GetObjective() == Objective_enum.Classification)
        {
            existingHyperParameterValues[numClassKeyName] = GetDefaultHyperParameterValueForLightGBM(numClassKeyName);
        }
    }

    private object GetDefaultHyperParameterValueForLightGBM(string hyperParameterName)
    {
        switch (hyperParameterName)
        {
            case nameof(LightGBMSample.objective):
                if (GetObjective() == Objective_enum.Regression)
                {
                    return nameof(LightGBMSample.objective_enum.regression);
                }
                if (GetObjective() == Objective_enum.Classification)
                {
                    if (NumClass >= 2)
                    {
                        return nameof(LightGBMSample.objective_enum.multiclass);
                    }
                    if (NumClass == 1)
                    {
                        return nameof(LightGBMSample.objective_enum.binary);
                    }
                }
                break;
            case nameof(LightGBMSample.num_class):
                return NumClass;
        }
        var errorMsg = $"do not know default value for Hyper Parameter {hyperParameterName} for model {typeof(LightGBMModel)}";
        ISample.Log.Error(errorMsg);
        throw new ArgumentException(errorMsg);
    }

    // ReSharper disable once UnusedMember.Local
    private object GetDefaultHyperParameterValueForCatBoost(string hyperParameterName)
    {
        switch (hyperParameterName)
        {
        }
        var errorMsg = $"do not know default value for Hyper Parameter {hyperParameterName} for model {typeof(CatBoostModel)}";
        ISample.Log.Error(errorMsg);
        throw new ArgumentException(errorMsg);
    }

    public virtual int NumClass
    {
        get
        {
            if (GetObjective() == Objective_enum.Regression)
            {
                return 1;
            }
            var errorMsg = $"the method {nameof(NumClass)} must be override for classification problem";
            ISample.Log.Error(errorMsg);
            throw new NotImplementedException(errorMsg);
        }
    }

    public virtual string[] TargetLabelDistinctValues
    {
        get
        {
            if (GetObjective() == Objective_enum.Regression)
            {
                return Array.Empty<string>();
            }

            if (DatasetEncoder != null)
            {
                return DatasetEncoder.TargetLabelDistinctValues;
            }
            var errorMsg = $"the method {nameof(TargetLabelDistinctValues)} must be overriden for problem";
            ISample.Log.Error(errorMsg);
            throw new NotImplementedException(errorMsg);
        }
    }

    public abstract Objective_enum GetObjective();
    public IScore ComputeRankingEvaluationMetric(DataFrame y_true_InTargetFormat, DataFrame y_pred_InTargetFormat)
    {
        if (y_true_InTargetFormat == null || y_pred_InTargetFormat == null)
        {
            return null;
        }
        AssertNoIdColumns(y_true_InTargetFormat);
        AssertNoIdColumns(y_pred_InTargetFormat);

        var y_true = y_true_InTargetFormat.FloatCpuTensor();
        var y_pred = y_pred_InTargetFormat.FloatCpuTensor();
        Debug.Assert(y_true.Shape.SequenceEqual(y_pred.Shape));
        var rankingEvaluationMetric = GetRankingEvaluationMetric();
        using var buffer = new CpuTensor<float>(y_true.ComputeMetricBufferShape(rankingEvaluationMetric));
        var evaluationMetric = y_true.ComputeEvaluationMetric(y_pred, rankingEvaluationMetric, buffer);
        return new Score((float)evaluationMetric, rankingEvaluationMetric);
    }

    /// <summary>
    /// in some cases, the Dataset (in Model Format) must have a number of rows that is a multiple of some constant
    /// </summary>
    /// <returns></returns>
    public virtual int DatasetRowsInModelFormatMustBeMultipleOf()
    {
        return 1;
    }
    public virtual char GetSeparator() { return ',';}
    /// save the predictions in the Challenge target format, adding an Id at left if needed
    public virtual void SavePredictionsInTargetFormat(DataFrame encodedPredictionsInTargetFormat, DataSet xDataset, string path)
    {
        if (encodedPredictionsInTargetFormat == null)
        {
            return;
        }
        AssertNoIdColumns(encodedPredictionsInTargetFormat);
        encodedPredictionsInTargetFormat = xDataset.AddIdColumnsAtLeftIfNeeded(encodedPredictionsInTargetFormat);
        if (DatasetEncoder != null)
        {
            encodedPredictionsInTargetFormat = DatasetEncoder.Inverse_Transform(encodedPredictionsInTargetFormat);
        }
        encodedPredictionsInTargetFormat.to_csv(path, GetSeparator());
    }

    public virtual DatasetEncoder DatasetEncoder => null;

    public virtual void SavePredictionsInModelFormat(DataFrame predictionsInModelFormat, string path)
    {
        AssertNoIdColumns(predictionsInModelFormat);
        predictionsInModelFormat?.to_csv(path, GetSeparator());
    }
   
    public virtual IScore MinimumScoreToSaveModel => null;

    public abstract DataSet TestDataset();
    /// <summary>
    /// returns the full train and validation dataset
    /// </summary>
    /// <returns></returns>
    public abstract DataSet FullTrainingAndValidation();
    public virtual ITrainingAndTestDataSet SplitIntoTrainingAndValidation()
    {
        var fullTrain = FullTrainingAndValidation();
        int rowsForTraining = (int)(PercentageInTraining * fullTrain.Count + 0.1);
        rowsForTraining -= rowsForTraining % DatasetRowsInModelFormatMustBeMultipleOf();
        return fullTrain.IntSplitIntoTrainingAndValidation(rowsForTraining);
    }

    /// <summary>
    /// transform a DataFrame of prediction in model format to a DataFrame in Challenge expected format
    /// Those DataFrame are not allowed to contain Id Columns
    /// </summary>
    /// <param name="predictionsInModelFormat"></param>
    /// <returns></returns>
    public virtual DataFrame PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame predictionsInModelFormat)
    {
        AssertNoIdColumns(predictionsInModelFormat);
        if (GetObjective() == Objective_enum.Regression)
        {
            return predictionsInModelFormat;
        }
        if (GetObjective() == Objective_enum.Classification && NumClass >= 2)
        {
            return DataFrame.New(predictionsInModelFormat.FloatCpuTensor().ArgMax(), TargetLabels);
        }
        throw new NotImplementedException("Can't manage binary classification");
    }


    /// <summary>
    /// ensure that the DataFrame 'df' has no Id Columns
    /// throw an exception if it contains Id Columns
    /// </summary>
    /// <param name="df"></param>
    /// <exception cref="Exception"></exception>
    private void AssertNoIdColumns(DataFrame df)
    {
        if (df == null)
        {
            return;
        }
        var idCol = Utils.Intersect(df.Columns, IdColumns);
        if (idCol.Count != 0)
        {
            throw new Exception($"The DataFrame is not allowed to contain Id Columns : {string.Join(' ', idCol)}");
        }

    }


    /// <summary>
    /// ensure that the DataFrame 'df' has all Id Columns
    /// throw an exception if it contains Id Columns
    /// </summary>
    /// <param name="df"></param>
    /// <exception cref="Exception"></exception>
    private void AssertAllIdColumns(DataFrame df)
    {
        if (df == null || IdColumns.Length == 0)
        {
            return;
        }
        var idCol = Utils.Intersect(df.Columns, IdColumns);
        if (idCol.Count != IdColumns.Length)
        {
            var missingIdColumn = Utils.Without(IdColumns, idCol);
            throw new Exception($"The DataFrame has {missingIdColumn.Count} missing Id Columns : {string.Join(' ', missingIdColumn)}");
        }
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

    public override HashSet<string> FieldsToDiscardInComputeHash()
    {
        return new HashSet<string>
        {
            nameof(Train_XDatasetPath_InTargetFormat), nameof(Train_YDatasetPath_InTargetFormat), nameof(Train_XYDatasetPath_InTargetFormat), 
            nameof(Validation_XDatasetPath_InTargetFormat), nameof(Validation_YDatasetPath_InTargetFormat), nameof(Validation_XYDatasetPath_InTargetFormat), 
            nameof(Test_XDatasetPath_InTargetFormat), nameof(Test_YDatasetPath_InTargetFormat), nameof(Test_XYDatasetPath_InTargetFormat),
            nameof(Train_XDatasetPath_InModelFormat), nameof(Train_YDatasetPath_InModelFormat), nameof(Train_XYDatasetPath_InModelFormat),
            nameof(Validation_XDatasetPath_InModelFormat), nameof(Validation_YDatasetPath_InModelFormat), nameof(Validation_XYDatasetPath_InModelFormat),
            nameof(Test_XDatasetPath_InModelFormat), nameof(Test_YDatasetPath_InModelFormat), nameof(Test_XYDatasetPath_InModelFormat)
        };
    }
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
    /// <summary>
    /// the evaluation metric used to rank the final submission
    /// depending on the evaluation metric, higher (ex: Accuracy) or lower (ex: Rmse) may be better
    /// </summary>
    /// <returns></returns>
    public abstract EvaluationMetricEnum GetRankingEvaluationMetric();

    private InMemoryDataSetV2 LoadDataSet(string XDatasetPath, string YDatasetPath, string XYDatasetPath)
    {
        DataFrame x = null;
        DataFrame y = null;
        if (!string.IsNullOrEmpty(XYDatasetPath))
        {
            Debug.Assert(string.IsNullOrEmpty(XDatasetPath));
            Debug.Assert(string.IsNullOrEmpty(YDatasetPath));
            var xy = DataFrame.read_float_csv(XYDatasetPath);
            x = xy[Utils.Without(xy.Columns, TargetLabels).ToArray()];
            y = xy[TargetLabels];
        }
        else
        {
            if (!string.IsNullOrEmpty(XDatasetPath))
            {
                Debug.Assert(string.IsNullOrEmpty(XYDatasetPath));
                x = DataFrame.read_float_csv(XDatasetPath);
            }
            if (!string.IsNullOrEmpty(YDatasetPath))
            {
                y = DataFrame.read_float_csv(YDatasetPath);
            }
        }
        if (x == null)
        {
            Debug.Assert(y == null);
            return null;
        }
        AssertAllIdColumns(x);
        AssertNoIdColumns(y);

        if (y != null && NumClass >= 2)
        {
            var yTensor = CpuTensor<float>.FromClassIndexToProba(y.FloatTensor, NumClass);
            y = DataFrame.New(yTensor);
        }
        return new InMemoryDataSetV2(this, x, y, false);
    }
    //private DataFrame DropIdColumnsIfFound(DataFrame df)
    //{
    //    if (df == null || IdColumns.Length == 0)
    //    {
    //        return df;
    //    }
    //    var intersection = Utils.Intersect(df.Columns, IdColumns);
    //    if (intersection.Count == 0)
    //    {
    //        return df;
    //    }

    //    if (intersection.Count != IdColumns.Length)
    //    {
    //        throw new Exception($"found only a part {string.Join(' ', intersection)} of Id Columns ({string.Join(' ', IdColumns)})");
    //    }
    //    return df.Drop(IdColumns);
    //}
}
