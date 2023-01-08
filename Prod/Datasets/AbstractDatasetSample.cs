using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.GPU;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.Datasets;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public abstract class AbstractDatasetSample : AbstractSample
{
    #region public properties

    public string Name { get; }
    public abstract string[] CategoricalFeatures { get; }


    public virtual bool PredictionsMustBeOrderedByIdColumn => false;
    /// <summary>
    /// features used to identify a row in the dataset
    /// such features should be ignored during the training
    /// </summary>
    /// <returns>list of id features </returns>
    public abstract string[] IdColumns { get; }

    // ReSharper disable once AutoPropertyCanBeMadeGetOnly.Local
    protected DatasetEncoder DatasetEncoder { get; set; }


    /// <summary>
    /// list of target feature names (usually a single element)
    /// </summary>
    /// <returns>list of target feature names </returns>
    public abstract string[] TargetLabels { get; }

    private int[] _cacheInputShape_CHW = null;
    protected string[] _cacheColumns = null;
    private readonly object lockInputShape_CHW = new();


    public (DataFrame predictionsInTargetFormat, DataFrame predictionsInModelFormat, IScore rankingScore, string path_pred_InModelFormat)
        ComputePredictionsAndRankingScore(DataSet dataset, Model model)
    {
        Debug.Assert(dataset != null);
        var start = Stopwatch.StartNew();
        var (y_pred_InModelFormat, path_pred_InModelFormat) = model.PredictWithPath(dataset, false);
        var y_pred_InTargetFormat = PredictionsInModelFormat_2_PredictionsInTargetFormat(y_pred_InModelFormat);
        IScore rankingScore = null;
        if (dataset.Y != null)
        {
            var y_true_InTargetFormat = PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame.New(dataset.Y));
            rankingScore = ComputeRankingEvaluationMetric(y_true_InTargetFormat, y_pred_InTargetFormat);
        }
        ISample.Log.Debug($"{nameof(ComputePredictionsAndRankingScore)} took {start.Elapsed.TotalSeconds}s");
        return (y_pred_InTargetFormat, y_pred_InModelFormat, rankingScore, path_pred_InModelFormat);
    }

    public virtual int[] GetInputShapeOfSingleElement()
    {
        if (_cacheInputShape_CHW != null)
        {
            return _cacheInputShape_CHW;
        }
        lock(lockInputShape_CHW)
        {
            if (_cacheInputShape_CHW != null)
            {
                return _cacheInputShape_CHW;
            }

            var fullTrainingAndValidation = FullTrainingAndValidation();
            var full = fullTrainingAndValidation as DataSetV2;
            if (full == null)
            {
                throw new ArgumentException($"can't compute shape for dataset type {fullTrainingAndValidation.GetType()}");
            }
            _cacheInputShape_CHW = full.XDataFrame.Shape.Skip(1).ToArray();
            _cacheColumns = full.ColumnNames.ToArray();
        }
        return _cacheInputShape_CHW;
    }

    public virtual string[] GetColumnNames()
    {
        if (_cacheColumns == null)
        {
            GetInputShapeOfSingleElement(); //we compute the columns
        }
        return _cacheColumns;
    }

    #endregion

    #region constructors
    protected AbstractDatasetSample(HashSet<string> mandatoryCategoricalHyperParameters) : base(mandatoryCategoricalHyperParameters)
    {
        Name = GetType().Name.Replace("DatasetSample", "");
    }
    #endregion


    public virtual int EmbeddingForColumn(string columnName, int defaultEmbeddingSize)
    {
        return defaultEmbeddingSize;
    }

    public virtual int CountOfDistinctCategoricalValues(string columnName)
    {
        var columnStats = DatasetEncoder[columnName];
        return columnStats.GetDistinctCategoricalValues().Count;
    }

    /// <summary>
    /// VocabularySizes : for each categorical feature, the number of distinct value it has
    /// EmbeddingDims: the default embedding for each categorical feature
    /// IndexesInLastDimensionToUse: for each categorical feature, the associated index of this categorical feature in the input
    /// </summary>
    /// <param name="defaultEmbeddingSize"></param>
    /// <returns></returns>
    public (int[] vocabularySizes, int[] embeddingDims, int[] indexesInLastDimensionToUse) EmbeddingDescription(int defaultEmbeddingSize)
    {
        List<int> vocabularySizes = new();
        List<int> embeddingDims = new();
        List<int> indexesInLastDimensionToUse = new();

        string[] columnNames = GetColumnNames();

        for (var i = 0; i < columnNames.Length; i++)
        {
            var column = columnNames[i];

            if (Array.IndexOf(IdColumns, column) >= 0)
            {
                //we'll discard Id columns
                indexesInLastDimensionToUse.Add(i);
                embeddingDims.Add(0); //0 embedding dim :  the feature will be discarded
                vocabularySizes.Add(1);
                continue;
            }

            if (Array.IndexOf(CategoricalFeatures, column) < 0)
            {
                continue;
            }
            indexesInLastDimensionToUse.Add(i);
            embeddingDims.Add(EmbeddingForColumn(column, defaultEmbeddingSize));
            vocabularySizes.Add(1 + CountOfDistinctCategoricalValues(column));
        }
        return (vocabularySizes.ToArray(), embeddingDims.ToArray(), indexesInLastDimensionToUse.ToArray());
    }
    public EvaluationMetricEnum DefaultLossFunction
    {
        get
        {
            if (GetObjective() == Objective_enum.Regression)
            {
                //!D TODO return EvaluationMetricEnum.Rmse;
                return EvaluationMetricEnum.Mse;
                //return EvaluationMetricEnum.Mae;
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
    //For numerical features:
    // should we standardize them (with mean=0 & volatility=1) before sending them to the model ?
    public bool StandardizeDoubleValues = false;
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

    public DataSetV2 LoadTrainDataset() => LoadDataSet(Train_XDatasetPath_InTargetFormat, Train_YDatasetPath_InTargetFormat, Train_XYDatasetPath_InTargetFormat);
    public DataSetV2 LoadValidationDataset() => LoadDataSet(Validation_XDatasetPath_InTargetFormat, Validation_YDatasetPath_InTargetFormat, Validation_XYDatasetPath_InTargetFormat);
    public DataSetV2 LoadTestDataset() => LoadDataSet(Test_XDatasetPath_InTargetFormat, Test_YDatasetPath_InTargetFormat, Test_XYDatasetPath_InTargetFormat);

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
    public virtual void SavePredictionsInTargetFormat(DataFrame y_pred_Encoded_InTargetFormat, DataSet xDataset, string path)
    {
        if (y_pred_Encoded_InTargetFormat == null)
        {
            return;
        }
        AssertNoIdColumns(y_pred_Encoded_InTargetFormat);
        y_pred_Encoded_InTargetFormat = xDataset.AddIdColumnsAtLeftIfNeeded(y_pred_Encoded_InTargetFormat);
        if (DatasetEncoder != null)
        {
            y_pred_Encoded_InTargetFormat = DatasetEncoder.Inverse_Transform(y_pred_Encoded_InTargetFormat);
        }

        if (PredictionsMustBeOrderedByIdColumn && IdColumns.Length != 0)
        {
            y_pred_Encoded_InTargetFormat = y_pred_Encoded_InTargetFormat.sort_values(IdColumns[0]);
        }

        y_pred_Encoded_InTargetFormat.to_csv(path, GetSeparator());
    }

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

        return predictionsInModelFormat;
        //throw new NotImplementedException("Can't manage binary classification");
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


    ///// <summary>
    ///// ensure that the DataFrame 'df' has all Id Columns
    ///// throw an exception if it contains Id Columns
    ///// </summary>
    ///// <param name="df"></param>
    ///// <exception cref="Exception"></exception>
    //private void AssertAllIdColumns(DataFrame df)
    //{
    //    if (df == null || IdColumns.Length == 0)
    //    {
    //        return;
    //    }
    //    var idCol = Utils.Intersect(df.Columns, IdColumns);
    //    if (idCol.Count != IdColumns.Length)
    //    {
    //        var missingIdColumn = Utils.Without(IdColumns, idCol);
    //        throw new Exception($"The DataFrame has {missingIdColumn.Count} missing Id Columns : {string.Join(' ', missingIdColumn)}");
    //    }
    //}
    

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
    /// the evaluation metric used to rank the final submission
    /// depending on the evaluation metric, higher (ex: Accuracy) or lower (ex: Rmse) may be better
    /// </summary>
    /// <returns></returns>
    public abstract EvaluationMetricEnum GetRankingEvaluationMetric();

    private DataSetV2 LoadDataSet(string XDatasetPath, string YDatasetPath, string XYDatasetPath)
    {
        DataFrame x = null;
        DataFrame y = null;
        if (!string.IsNullOrEmpty(XYDatasetPath))
        {
            Debug.Assert(string.IsNullOrEmpty(XDatasetPath));
            Debug.Assert(string.IsNullOrEmpty(YDatasetPath));
            var xy = DataFrame.read_float_csv(XYDatasetPath);
            //only the first column contains the label
            var targetLabels = xy.Columns.Take(1).ToArray();
            x = xy[Utils.Without(xy.Columns, targetLabels).ToArray()];
            y = xy[targetLabels];
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
        //!D AssertAllIdColumns(x);
        AssertNoIdColumns(y);

        if (y != null && NumClass >= 2)
        {
            var yTensor = CpuTensor<float>.FromClassIndexToProba(y.FloatTensor, NumClass);
            y = DataFrame.New(yTensor);
        }
        return new DataSetV2(this, x, y, false);
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
