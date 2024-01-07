using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.CPU;
using SharpNet.Hyperparameters;
using SharpNet.Models;
// ReSharper disable MemberCanBeProtected.Global
// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets;

[SuppressMessage("ReSharper", "EmptyGeneralCatchClause")]
public abstract class AbstractDatasetSample : AbstractSample, IDisposable
{
    #region Hyper-Parameters
    //For numerical features:
    // should we standardize them (with mean=0 & volatility=1) before sending them to the model ?
    // ReSharper disable once FieldCanBeMadeReadOnly.Global
    public bool StandardizeDoubleValues = false;
    /// <summary>
    /// should we shuffle the dataset before splitting it into training / validation ?
    /// </summary>
    // ReSharper disable once FieldCanBeMadeReadOnly.Global
    public bool ShuffleDatasetBeforeSplit = false;
    /// <summary>
    /// in classification task :
    ///     when doing the split, if we should make sure that split has the same percentage of each category as in the original dataset
    /// in regression task :
    ///     it has no effect (must be false)
    /// </summary>
    // ReSharper disable once ConvertToConstant.Global
    public bool StratifiedDatasetBeforeSplit = false;
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

    #region private and protected fields
    // ReSharper disable once AutoPropertyCanBeMadeGetOnly.Local
    protected DatasetEncoder DatasetEncoder { get; set; }
    private int[] _cacheInputShape_1CHW = null;
    protected string[] _cacheColumns = null;
    private readonly object lockInputShape_CHW = new();
    #endregion

    #region public properties
    public string Name { get; }
    public virtual bool PredictionsMustBeOrderedByIdColumn => false;
    #endregion


    #region constructors
    protected AbstractDatasetSample()
    {
        Name = GetType().Name.Replace("DatasetSample", "");
    }
    #endregion


    #region abstract methods that must be implemented
    /// <summary>
    /// feature used to identify a row in the dataset
    /// such features should be ignored during the training
    /// </summary>
    public abstract string IdColumn { get; }
    /// <summary>
    /// </summary>
    /// <returns>list of target feature names </returns>
    public abstract string[] TargetLabels { get; }
    public abstract bool IsCategoricalColumn(string columnName);
    public abstract int[] Y_Shape(int batchSize);
    public abstract int NumClass { get; }
    public abstract Objective_enum GetObjective();
    public abstract DataSet TestDataset();
    /// <summary>
    /// returns the full train and validation dataset
    /// </summary>
    /// <returns></returns>
    public abstract DataSet FullTrainingAndValidation();
    #endregion

    public (DataFrame predictions_InModelFormat, IScore modelLossScore, DataFrame predictions_InTargetFormat, IScore targetRankingScore, string path_pred_InModelFormat)
        ComputePredictionsAndRankingScoreV2(DataSet dataset, Model model, bool removeAllTemporaryFilesAtEnd, bool computeAlsoRankingScore = true)
    {
        Debug.Assert(dataset != null);
        var start = Stopwatch.StartNew();
        var (y_pred_InModelFormat, path_pred_InModelFormat) = model.PredictWithPath(dataset, removeAllTemporaryFilesAtEnd);
        var y = dataset.LoadFullY();
        var modelLossScore = model.ComputeLoss(y, y_pred_InModelFormat.FloatTensor);
        if (!computeAlsoRankingScore)
        {
            return (y_pred_InModelFormat, modelLossScore, null, null, path_pred_InModelFormat);
        }
        var y_pred_InTargetFormat = Predictions_InModelFormat_2_Predictions_InTargetFormat(y_pred_InModelFormat, model.ModelSample.GetObjective());
        IScore targetRankingScore = null;
        if (y != null)
        {
            var y_true_InTargetFormat = Predictions_InModelFormat_2_Predictions_InTargetFormat(DataFrame.New(y), model.ModelSample.GetObjective());
            targetRankingScore = ComputeRankingEvaluationMetric(y_true_InTargetFormat, y_pred_InTargetFormat, model.ModelSample);
        }
        ISample.Log.Debug($"{nameof(ComputePredictionsAndRankingScoreV2)} took {start.Elapsed.TotalSeconds}s");
        return (y_pred_InModelFormat, modelLossScore, y_pred_InTargetFormat, targetRankingScore, path_pred_InModelFormat);
    }
    public virtual int[] X_Shape(int batchSize)
    {
        if (_cacheInputShape_1CHW == null)
        {
            lock (lockInputShape_CHW)
            {
                var fullTrainingAndValidation = FullTrainingAndValidation();
                if (fullTrainingAndValidation is DataFrameDataSet full)
                {
                    _cacheInputShape_1CHW = full.XDataFrame.Shape.Skip(1).ToArray();
                    _cacheColumns = full.ColumnNames.ToArray();
                }
                else if (fullTrainingAndValidation is InMemoryDataSet inMemoryDataSet)
                {
                    _cacheInputShape_1CHW = inMemoryDataSet.X.Shape.Skip(1).ToArray();
                }
                else
                {
                    throw new ArgumentException($"can't compute shape for dataset type {fullTrainingAndValidation.GetType()}, you must override method {nameof(X_Shape)} for class {GetType()}");
                }
            }
        }
        var res = (int[])_cacheInputShape_1CHW.Clone();
        res[0] = batchSize;
        return res;
    }
    public virtual string[] GetColumnNames()
    {
        if (_cacheColumns == null)
        {
            X_Shape(1); //we compute the columns
        }
        return _cacheColumns;
    }
    public bool IsIdColumn(string columnName)
    {
        return !string.IsNullOrEmpty(IdColumn) && Equals(columnName, IdColumn);
    }


    #region Embedding methods
    /// <summary>
    /// VocabularySizes : for each categorical feature, the number of distinct value it has
    /// EmbeddingDims: the default embedding for each categorical feature
    /// IndexesInLastDimensionToUse: for each categorical feature, the associated index of this categorical feature in the input
    /// </summary>
    /// <param name="defaultEmbeddingDim"></param>
    /// <returns></returns>
    public (int[] vocabularySizes, int[] embeddingDims, int[] indexesInLastDimensionToUse, int[] embeddingTensorIndex) EmbeddingDescription(int defaultEmbeddingDim)
    {
        List<int> vocabularySizes = new();
        List<int> embeddingDims = new();
        List<int> indexesInLastDimensionToUse = new();
        List<int> embeddingTensorIndex = new();

        string[] columnNames = GetColumnNames();

        var groupIndex_to_EmbeddingTensorIndex =new Dictionary<int, int>();
        var columnName_to_GroupIndex = GetCategoricalFeature_to_IndexInGetCategoricalColumnSharingSameValues();

        var columnNamesLength = columnNames?.Length??0;
        int embeddingTensorCount = 0;
        for (var i = 0; i < columnNamesLength; i++)
        {
            // ReSharper disable once PossibleNullReferenceException
            var column = columnNames[i];

            if (IsIdColumn(column))
            {
                //we'll discard Id columns
                vocabularySizes.Add(1);
                embeddingDims.Add(0); //0 embedding dim :  the feature will be discarded
                indexesInLastDimensionToUse.Add(i);
                embeddingTensorIndex.Add(embeddingTensorCount++);
                continue;
            }

            if (!IsCategoricalColumn(column))
            {
                continue;
            }
            
            vocabularySizes.Add(1 + CountOfDistinctCategoricalValues(column));
            embeddingDims.Add(EmbeddingForColumn(column, defaultEmbeddingDim));
            indexesInLastDimensionToUse.Add(i);


            if (columnName_to_GroupIndex.TryGetValue(column, out var groupIndex))
            {
                if (!groupIndex_to_EmbeddingTensorIndex.ContainsKey(groupIndex))
                {
                    groupIndex_to_EmbeddingTensorIndex[groupIndex] = embeddingTensorCount++;
                }
                embeddingTensorIndex.Add(groupIndex_to_EmbeddingTensorIndex[groupIndex]);
            }
            else
            {
                embeddingTensorIndex.Add(embeddingTensorCount++);
            }
        }
        return (vocabularySizes.ToArray(), embeddingDims.ToArray(), indexesInLastDimensionToUse.ToArray(), embeddingTensorIndex.ToArray());
    }
    public virtual List<HashSet<string>> GetCategoricalColumnSharingSameValues()
    {
        return new List<HashSet<string>>();
    }
    public Dictionary<string, HashSet<string>> GetCategoricalFeature_to_CategoricalColumnSharingSameValues()
    {
        Dictionary<string, HashSet<string>> res = new();
        var groups = GetCategoricalColumnSharingSameValues();
        foreach (var t in groups)
        {
            foreach (var feature in t)
            {
                res[feature] = t;
            }
        }
        return res;
    }
    // ReSharper disable once VirtualMemberNeverOverridden.Global
    // ReSharper disable once UnusedParameter.Global
    protected virtual int EmbeddingForColumn(string columnName, int defaultEmbeddingDim)
    {
        return defaultEmbeddingDim;
    }
    protected virtual int CountOfDistinctCategoricalValues(string columnName)
    {
        var columnStats = DatasetEncoder[columnName];
        return columnStats.GetDistinctCategoricalValues().Count;
    }
    private Dictionary<string, int> GetCategoricalFeature_to_IndexInGetCategoricalColumnSharingSameValues()
    {
        Dictionary<string, int> res = new();
        var groups = GetCategoricalColumnSharingSameValues();
        for (int i = 0; i < groups.Count; ++i)
        {
            foreach (var feature in groups[i])
            {
                res[feature] = i;
            }
        }

        return res;
    }
    #endregion

    #region Dispose pattern
    protected bool disposed = false;
    protected virtual void Dispose(bool disposing)
    {
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
        cloned.SetAll_Train_Validation_DatasetPath_to_null();
        return cloned;
    }
    //// ReSharper disable once MemberCanBeMadeStatic.Global
    public DataFrame LoadPredictions_InModelFormat(string directory, string fileName)
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
    public IScore ComputeRankingEvaluationMetric(DataFrame y_true_InTargetFormat, DataFrame y_pred_InTargetFormat, IMetricConfig metricConfig)
    {
        AssertNoIdColumns(y_true_InTargetFormat);
        AssertNoIdColumns(y_pred_InTargetFormat);
        if (y_true_InTargetFormat == null || y_pred_InTargetFormat == null)
        {
            return null;
        }
        var y_true_tensor = y_true_InTargetFormat.FloatCpuTensor();
        var y_pred_tensor = y_pred_InTargetFormat.FloatCpuTensor();
        var rankingEvaluationMetric = metricConfig.GetRankingEvaluationMetric();
        using var buffer = new CpuTensor<float>(y_true_tensor.ComputeMetricBufferShape(rankingEvaluationMetric));
        var evaluationMetric = buffer.ComputeEvaluationMetric(y_true_tensor, y_pred_tensor, rankingEvaluationMetric, metricConfig);
        return new Score((float)evaluationMetric, rankingEvaluationMetric);
    }
    /// <summary>
    /// in some cases, the Dataset (in Model Format) must have a number of rows that is a multiple of some constant
    /// </summary>
    /// <returns></returns>
    public virtual int DatasetRows_InModelFormat_MustBeMultipleOf()
    {
        return 1;
    }
    public virtual char GetSeparator() { return ',';}
    /// save the predictions in the Challenge target format, adding an Id at left if needed
    public virtual void SavePredictions_InTargetFormat(DataFrame y_pred_Encoded_InTargetFormat, DataSet xDataset, string path)
    {
        if (y_pred_Encoded_InTargetFormat == null)
        {
            return;
        }
        AssertNoIdColumns(y_pred_Encoded_InTargetFormat);

        var y_pred_Decoded_InTargetFormat = (DatasetEncoder != null)
                ?DatasetEncoder.Inverse_Transform(y_pred_Encoded_InTargetFormat)
                : y_pred_Encoded_InTargetFormat;

        //we add the Y_ID column
        if (!string.IsNullOrEmpty(xDataset.IdColumn) && !y_pred_Decoded_InTargetFormat.Columns.Contains(xDataset.IdColumn))
        {
            y_pred_Decoded_InTargetFormat = DataFrame.MergeHorizontally(xDataset.ExtractIdDataFrame(y_pred_Decoded_InTargetFormat.Shape[0]), y_pred_Decoded_InTargetFormat);
        }
        if (PredictionsMustBeOrderedByIdColumn && !string.IsNullOrEmpty(IdColumn))
        {
            y_pred_Decoded_InTargetFormat = y_pred_Decoded_InTargetFormat.sort_values(IdColumn);
        }

        y_pred_Decoded_InTargetFormat.to_csv(path, GetSeparator());
    }
    public virtual void SavePredictions_InModelFormat(DataFrame predictions_InModelFormat, string path)
    {
        AssertNoIdColumns(predictions_InModelFormat);
        predictions_InModelFormat?.to_csv(path, GetSeparator());
    }
    public virtual ITrainingAndTestDataset SplitIntoTrainingAndValidation()
    {
        var fullTrain = FullTrainingAndValidation();
        int rowsForTraining = (int)(PercentageInTraining * fullTrain.Count + 0.1);
        rowsForTraining -= rowsForTraining % DatasetRows_InModelFormat_MustBeMultipleOf();
        return fullTrain.IntSplitIntoTrainingAndValidation(rowsForTraining, ShuffleDatasetBeforeSplit, StratifiedDatasetBeforeSplit);
    }
    /// <summary>
    /// transform a DataFrame of prediction in model format to a DataFrame in Challenge expected format
    /// Those DataFrame are not allowed to contain Id Columns
    /// </summary>
    /// <param name="predictions_InModelFormat"></param>
    /// <param name="objective"></param>
    /// <returns></returns>
    public virtual DataFrame Predictions_InModelFormat_2_Predictions_InTargetFormat(DataFrame predictions_InModelFormat, Objective_enum objective)
    {
        AssertNoIdColumns(predictions_InModelFormat);
        if (objective == Objective_enum.Regression)
        {
            return predictions_InModelFormat;
        }
        if (objective == Objective_enum.Classification && NumClass >= 2)
        {
            return DataFrame.New(predictions_InModelFormat.FloatCpuTensor().ArgMax(), TargetLabels);
        }

        return predictions_InModelFormat;
        //throw new NotImplementedException("Can't manage binary classification");
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
    public DataFrameDataSet LoadDataSet(string XDatasetPath, string YDatasetPath, string XYDatasetPath)
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
        return new DataFrameDataSet(this, x, y, null);
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
    protected bool DefaultIsCategoricalColumn(string columnName, string[] categoricalFeatures = null)
    {
        if (IsIdColumn(columnName))
        {
            return true;
        }
        return categoricalFeatures != null && Array.IndexOf(categoricalFeatures, columnName) != -1;
    }
    private void SetAll_Train_Validation_DatasetPath_to_null()
    {
        Train_XDatasetPath_InTargetFormat = Train_YDatasetPath_InTargetFormat = Train_XYDatasetPath_InTargetFormat = null;
        Train_XDatasetPath_InModelFormat = Train_YDatasetPath_InModelFormat = Train_XYDatasetPath_InModelFormat = null;
        Validation_XDatasetPath_InTargetFormat = Validation_YDatasetPath_InTargetFormat = Validation_XYDatasetPath_InTargetFormat = null;
        Validation_XDatasetPath_InModelFormat = Validation_YDatasetPath_InModelFormat = Validation_XYDatasetPath_InModelFormat = null;
    }

  
}
