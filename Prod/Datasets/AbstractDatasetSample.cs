using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.GPU;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.Datasets;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public abstract class AbstractDatasetSample : AbstractSample, IDisposable
{
    private readonly List<IDisposable> _toDispose = new();

    public void AddToDispose(IDisposable disposable)
    {
        _toDispose.Add(disposable);
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



    #region public properties

    public string Name { get; }
    public abstract string[] CategoricalFeatures { get; }


    public virtual bool PredictionsMustBeOrderedByIdColumn => false;
    /// <summary>
    /// feature used to identify a row in the dataset
    /// such features should be ignored during the training
    /// </summary>
    public abstract string IdColumn { get; }

    // ReSharper disable once AutoPropertyCanBeMadeGetOnly.Local
    protected DatasetEncoder DatasetEncoder { get; set; }

    public virtual List<EvaluationMetricEnum> GetMetrics()
    {
        return  new List<EvaluationMetricEnum>() { DefaultLossFunction, GetRankingEvaluationMetric() };
    }

    /// <summary>
    /// list of target feature names (usually a single element)
    /// </summary>
    /// <returns>list of target feature names </returns>
    public abstract string[] TargetLabels { get; }

    private int[] _cacheInputShape_CHW = null;
    protected string[] _cacheColumns = null;
    private readonly object lockInputShape_CHW = new();


    public (DataFrame predictionsInModelFormat, IScore modelLossScore, DataFrame predictionsInTargetFormat, IScore targetRankingScore, string path_pred_InModelFormat)
        ComputePredictionsAndRankingScoreV2(DataSet dataset, Model model, bool removeAllTemporaryFilesAtEnd, bool computeAlsoRankingScore = true)
    {
        Debug.Assert(dataset != null);
        var start = Stopwatch.StartNew();
        var (y_pred_InModelFormat, path_pred_InModelFormat) = model.PredictWithPath(dataset, removeAllTemporaryFilesAtEnd);
        var modelLossScore = model.ComputeLoss(dataset.Y, y_pred_InModelFormat.FloatTensor);
        if (!computeAlsoRankingScore)
        {
            return (y_pred_InModelFormat, modelLossScore, null, null, path_pred_InModelFormat);
        }
        var y_pred_InTargetFormat = PredictionsInModelFormat_2_PredictionsInTargetFormat(y_pred_InModelFormat);
        IScore targetRankingScore = null;
        if (dataset.Y != null)
        {
            var y_true_InTargetFormat = PredictionsInModelFormat_2_PredictionsInTargetFormat(DataFrame.New(dataset.Y));
            targetRankingScore = ComputeRankingEvaluationMetric(y_true_InTargetFormat, y_pred_InTargetFormat);
        }
        ISample.Log.Debug($"{nameof(ComputePredictionsAndRankingScoreV2)} took {start.Elapsed.TotalSeconds}s");
        return (y_pred_InModelFormat, modelLossScore, y_pred_InTargetFormat, targetRankingScore, path_pred_InModelFormat);
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
            var full = fullTrainingAndValidation as DataFrameDataSet;
            if (full == null)
            {
                throw new ArgumentException($"can't compute shape for dataset type {fullTrainingAndValidation.GetType()}");
            }
            _cacheInputShape_CHW = full.XDataFrame.Shape.Skip(1).ToArray();
            _cacheColumns = full.ColumnNames.ToArray();
        }
        return _cacheInputShape_CHW;
    }

    public int FeatureByElement()
    {
        return Utils.Product(GetInputShapeOfSingleElement());

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


    protected virtual int EmbeddingForColumn(string columnName, int defaultEmbeddingDim)
    {
        return defaultEmbeddingDim;
    }

    protected virtual int CountOfDistinctCategoricalValues(string columnName)
    {
        var columnStats = DatasetEncoder[columnName];
        return columnStats.GetDistinctCategoricalValues().Count;
    }

    public bool IsIdColumn(string columnName)
    {
        return !string.IsNullOrEmpty(IdColumn) && Equals(columnName, IdColumn);
    }

    /// <summary>
    /// VocabularySizes : for each categorical feature, the number of distinct value it has
    /// EmbeddingDims: the default embedding for each categorical feature
    /// IndexesInLastDimensionToUse: for each categorical feature, the associated index of this categorical feature in the input
    /// </summary>
    /// <param name="defaultEmbeddingDim"></param>
    /// <returns></returns>
    public (int[] vocabularySizes, int[] embeddingDims, int[] indexesInLastDimensionToUse) EmbeddingDescription(int defaultEmbeddingDim)
    {
        List<int> vocabularySizes = new();
        List<int> embeddingDims = new();
        List<int> indexesInLastDimensionToUse = new();

        string[] columnNames = GetColumnNames();

        var columnNamesLength = columnNames?.Length??0;
        for (var i = 0; i < columnNamesLength; i++)
        {
            var column = columnNames[i];

            if (IsIdColumn(column))
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
            embeddingDims.Add(EmbeddingForColumn(column, defaultEmbeddingDim));
            vocabularySizes.Add(1 + CountOfDistinctCategoricalValues(column));
        }
        return (vocabularySizes.ToArray(), embeddingDims.ToArray(), indexesInLastDimensionToUse.ToArray());
    }
    public virtual EvaluationMetricEnum DefaultLossFunction
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

    public DataFrameDataSet LoadTrainDataset() => LoadDataSet(Train_XDatasetPath_InTargetFormat, Train_YDatasetPath_InTargetFormat, Train_XYDatasetPath_InTargetFormat);
    public DataFrameDataSet LoadValidationDataset() => LoadDataSet(Validation_XDatasetPath_InTargetFormat, Validation_YDatasetPath_InTargetFormat, Validation_XYDatasetPath_InTargetFormat);
    public DataFrameDataSet LoadTestDataset() => LoadDataSet(Test_XDatasetPath_InTargetFormat, Test_YDatasetPath_InTargetFormat, Test_XYDatasetPath_InTargetFormat);

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
            return TargetLabelDistinctValues.Length;
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
            const string errorMsg = $"the method {nameof(TargetLabelDistinctValues)} must be overriden for classification problem";
            ISample.Log.Error(errorMsg);
            throw new NotImplementedException(errorMsg);
        }
    }


    private bool IsRegressionProblem => GetObjective() == Objective_enum.Regression;

    public abstract Objective_enum GetObjective();
    public IScore ComputeRankingEvaluationMetric(DataFrame y_true_InTargetFormat, DataFrame y_pred_InTargetFormat)
    {
        AssertNoIdColumns(y_true_InTargetFormat);
        AssertNoIdColumns(y_pred_InTargetFormat);
        return DataFrame.ComputeEvaluationMetric(y_true_InTargetFormat, y_pred_InTargetFormat, GetRankingEvaluationMetric());
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

        var y_pred_Decoded_InTargetFormat = (DatasetEncoder != null)
                ?DatasetEncoder.Inverse_Transform(y_pred_Encoded_InTargetFormat)
                : y_pred_Encoded_InTargetFormat;
        y_pred_Decoded_InTargetFormat = xDataset.AddIdColumnsAtLeftIfNeeded(y_pred_Decoded_InTargetFormat);
    
        if (PredictionsMustBeOrderedByIdColumn && !string.IsNullOrEmpty(IdColumn))
        {
            y_pred_Decoded_InTargetFormat = y_pred_Decoded_InTargetFormat.sort_values(IdColumn);
        }

        y_pred_Decoded_InTargetFormat.to_csv(path, GetSeparator());
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
    public virtual ITrainingAndTestDataset SplitIntoTrainingAndValidation()
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
    protected void AssertNoIdColumns(DataFrame df)
    {
        if (df == null || string.IsNullOrEmpty(IdColumn))
        {
            return;
        }
        var indexIdCol = Array.IndexOf(df.Columns, IdColumn);
        if (indexIdCol >= 0)
        {
            throw new Exception($"The DataFrame is not allowed to contain Id Columns : {string.Join(' ', indexIdCol)}");
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
    /// the evaluation metric used to rank the final submission
    /// depending on the evaluation metric, higher (ex: Accuracy) or lower (ex: Rmse) may be better
    /// </summary>
    /// <returns></returns>
    public abstract EvaluationMetricEnum GetRankingEvaluationMetric();


    /// <summary>
    /// try to use one of the metric computed when training the model as a ranking score
    /// </summary>
    /// <param name="modelMetrics">the metrics computed when training the model</param>
    /// <returns></returns>
    public virtual IScore ExtractRankingScoreFromModelMetricsIfAvailable(params IScore[] modelMetrics)
    {
        return null;
    }

    private DataFrameDataSet LoadDataSet(string XDatasetPath, string YDatasetPath, string XYDatasetPath)
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
        return new DataFrameDataSet(this, x, y, null, false);
    }

    #region Dispose pattern
    protected bool disposed = false;
    protected virtual void Dispose(bool disposing)
    {
        _toDispose.ForEach(d=>d?.Dispose());
        _toDispose.Clear();

        if (disposed)
        {
            return;
        }
        disposed = true;
        //Release Unmanaged Resources
        if (disposing)
        {
            //Release Managed Resources
            DatasetEncoder = null;
        }
    }
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }
    ~AbstractDatasetSample()
    {
        Dispose(false);
    }
    #endregion
}
