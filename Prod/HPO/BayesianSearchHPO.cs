using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.Hyperparameters;
using SharpNet.LightGBM;
using SharpNet.Models;

namespace SharpNet.HPO;

public class BayesianSearchHPO : AbstractHpo
{
    #region private fields
    private readonly Random _rand = new();
    private readonly HashSet<string> _processedSpaces = new();
    private readonly HyperparameterSearchSpace.RANDOM_SEARCH_OPTION _randomSearchOption;
    /// <summary>
    /// the model use to predict the objective function score
    /// </summary>
    private readonly Model  _surrogateModel;
    private bool? _higherScoreIsBetter = null;
    /// <summary>
    /// Item1:  the sample
    /// Item2:  the sample as a float vector array
    /// Item3:  the score predicted by the surrogate model
    /// Item4:  the actual score of the sample (or NaN if it has not been computed yet)
    /// Item5:  the sample id
    /// Item6:  the sample description
    /// </summary>
    private readonly List< Tuple<ISample, float[], float, float, int, string>> _samplesWithScoreIfAvailable = new ();
    private List<Tuple<ISample, float[], float, float, int, string>> SamplesWithScore =>  _samplesWithScoreIfAvailable.Where(e => !float.IsNaN(e.Item4)).ToList();
    /// <summary>
    /// next samples to use for the objective function
    /// if this list is empty, it means that we'll need to call the surrogate model to retrieve new samples to test
    /// Item1:  the sample
    /// Item2:  the sample as a float vector array
    /// Item3:  the score predicted by the surrogate model
    /// Item3:  the sample description
    /// </summary>
    private readonly List<Tuple<ISample, float[], float, string>> _nextSamplesToCompute = new();
    private readonly object _lockObject = new ();
    /// <summary>
    /// Number of samples that have been used to train the model '_surrogateModel'
    /// 0 means the model has not been trained so far
    /// The value is in the range [0, SamplesWithScore.Length] 
    /// </summary>
    private int _samplesUsedForModelTraining = 0;
    private readonly DateTime _searchStartTime;
    #endregion

    public BayesianSearchHPO(IDictionary<string, object> searchSpace,
        Func<ISample> createDefaultSample,
        [NotNull] string workingDirectory,
        HyperparameterSearchSpace.RANDOM_SEARCH_OPTION randomSearchOption = HyperparameterSearchSpace.RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING) :
        base(searchSpace, createDefaultSample, workingDirectory)
    {
        _randomSearchOption = randomSearchOption;
        _searchStartTime = DateTime.Now;
        var surrogateModelName = "surrogate_" + System.Diagnostics.Process.GetCurrentProcess().Id;
        // _surrogateModel = BuildRandomForestSurrogateModel(_workingDirectory, surrogateModelName, SurrogateModelCategoricalFeature());
        _surrogateModel = BuildCatBoostSurrogateModel(_workingDirectory, surrogateModelName);
    }

    //public static InMemoryDataSet LoadSurrogateTrainingDataset(string dataFramePath, Func<string,bool> isCategoricalColumn = null)
    //{
    //    var df = DataFrame.read_float_csv(dataFramePath);
    //    var x_df = df.Drop("y");
    //    var x = x_df.FloatCpuTensor();
    //    var y_df = df["y"];
    //    var y = y_df.FloatCpuTensor();
    //    return new InMemoryDataSet(x, y, "", Objective_enum.Regression, null, columnNames: x_df.Columns, isCategoricalColumn);
    //}
    //public static InMemoryDataSet LoadSurrogateValidationDataset(string dataFramePath, Func<string, bool> isCategoricalColumn = null)
    //{
    //    var df = DataFrame.read_float_csv(dataFramePath);
    //    var x = df.FloatCpuTensor();
    //    return new InMemoryDataSet(x, null, "", Objective_enum.Regression, null, columnNames: df.Columns, isCategoricalColumn);
    //}

    protected override (ISample, int, string) Next
    {
        get
        {
            lock (_lockObject)
            {
                if (_nextSamplesToCompute.Count == 0)
                {
                    Log.Debug("No more samples available, computing new ones");
                    _nextSamplesToCompute.AddRange(NextSamplesForObjectiveFunction());
                }

                if (_nextSamplesToCompute.Count == 0)
                {
                    return (null, -1, "");
                }

                var sample = _nextSamplesToCompute[0].Item1;
                var sampleAsFloatVector = _nextSamplesToCompute[0].Item2;
                var surrogateScoreEstimate = _nextSamplesToCompute[0].Item3;
                var sampleDescription = _nextSamplesToCompute[0].Item4;
                int sampleId = _samplesWithScoreIfAvailable.Count;
                _samplesWithScoreIfAvailable.Add(Tuple.Create(sample, sampleAsFloatVector, surrogateScoreEstimate, float.NaN, sampleId, sampleDescription));
                _nextSamplesToCompute.RemoveAt(0);
                return (sample, sampleId, sampleDescription);
            }
        }
    }
    protected override void RegisterSampleScore(ISample sample, int sampleId, [NotNull] IScore actualScore, double elapsedTimeInSeconds)
    {
        lock (_lockObject)
        {
            if (!_higherScoreIsBetter.HasValue)
            {
                _higherScoreIsBetter = Utils.HigherScoreIsBetter(actualScore.Metric);
                Log.Info($"Higher Score Is Better = {_higherScoreIsBetter}");
            }
            _allActualScores.Add(actualScore.Value);
            var sampleTuple = _samplesWithScoreIfAvailable[sampleId];
            var sampleAsFloatVector = sampleTuple.Item2;

            //we update the values used to train the surrogate model: they may have been modified during the training
            var surrogateModelFeatureNames = SurrogateModelFeatureNames();
            for (int i = 0; i < surrogateModelFeatureNames.Length; i++)
            {
                var fieldValue = sample.Get(surrogateModelFeatureNames[i]);
                if (fieldValue is float floatValue && !float.IsNaN(floatValue)) { sampleAsFloatVector[i] = floatValue; }
                else if (fieldValue is double doubleValue &&!double.IsNaN(doubleValue)) { sampleAsFloatVector[i] = (float)doubleValue; }
                else if (fieldValue is int intValue) { sampleAsFloatVector[i] = intValue; }
            }

            var surrogateScorePrediction = sampleTuple.Item3;
            var sampleDescription = sampleTuple.Item6;
            if (float.IsNaN(surrogateScorePrediction))
            {
                throw new Exception($"surrogate score prediction is missing from sampleId {sampleId}");
            }
            if (!float.IsNaN(sampleTuple.Item4))
            {
                throw new Exception($"score has been already registered for sampleId {sampleId}");
            }
            if (sampleTuple.Item5 != sampleId)
            {
                throw new Exception($"invalid sampleId {sampleTuple.Item5} , should be {sampleId}");
            }
            if (actualScore == null)
            {
                throw new Exception($"score can not be NaN for sampleId {sampleId}");
            }

            Log.Debug($"Registering actual score ({actualScore}) of sample {sampleId} (surrogate score prediction: {surrogateScorePrediction})");
            _samplesWithScoreIfAvailable[sampleId] = Tuple.Create(sampleTuple.Item1, sampleAsFloatVector, surrogateScorePrediction, actualScore.Value, sampleId, sampleDescription);
            RegisterSampleScore(SearchSpace, sample, actualScore, elapsedTimeInSeconds);

            // For the first 'samplesUsingRandomSearch' samples, we'll use a random search
            // Then we'll use the surrogate Model to select the samples to use
            const int samplesUsingRandomSearch = 50;
            //if (SamplesWithScore.Count >= Math.Max(samplesUsingRandomSearch, Math.Sqrt(2) * (_samplesUsedForModelTraining)))
            if (SamplesWithScore.Count >= Math.Max(samplesUsingRandomSearch, 2*_samplesUsedForModelTraining))
            {
                _samplesUsedForModelTraining = TrainSurrogateModel();
            }
        }
    }

    // ReSharper disable once UnusedMember.Local
    private static Model BuildRandomForestSurrogateModel(string workingDirectory, string modelName, string[] surrogateModelCategoricalFeature)
    {
        surrogateModelCategoricalFeature ??= Array.Empty<string>();
        // the surrogate model will be trained with a LightGBM using random forests (boosting=rf)
        var surrogateModelSample = new LightGBMSample();
        var categoricalFeaturesFieldValue = (surrogateModelCategoricalFeature.Length >= 1) ? ("name:" + string.Join(',', surrogateModelCategoricalFeature)) : "";
        surrogateModelSample.Set(new Dictionary<string, object> {
            { "bagging_fraction", 0.5 },
            { "bagging_freq", 1 },
            { "boosting", "rf" },
            { "categorical_feature", categoricalFeaturesFieldValue },
            { "colsample_bytree", 0.9 },
            { "colsample_bynode", 0.9 },
            { "device_type", "cpu" },
            { "early_stopping_round", 0 },
            { "lambda_l1", 0.15f },
            { "lambda_l2", 0.15f },
            { "learning_rate", 0.05f },
            { "max_bin", 255 },
            { "max_depth", 128},
            { "min_data_in_leaf", 3 },
            { "min_data_in_bin", 20 },
            { "num_iterations", 100 },
            { "num_leaves", 50 },
            { "num_threads", 2},
            { "objective", "regression"},
            { "path_smooth", 1.0f},
            { "verbosity", -1 },
        });
        return new LightGBMModel(surrogateModelSample, workingDirectory, modelName);
    }
    // ReSharper disable once UnusedMember.Local
    private static Model BuildCatBoostSurrogateModel(string workingDirectory, string modelName)
    {
        // the surrogate model will be trained with CatBoost
        CatBoostSample surrogateModelSample = new()
        {
            iterations = 100,
            loss_function = CatBoostSample.loss_function_enum.RMSE,
            eval_metric = CatBoostSample.metric_enum.RMSE,
            allow_writing_files = false,
            thread_count = 2,
            logging_level = CatBoostSample.logging_level_enum.Silent
        };

        surrogateModelSample.set_early_stopping_rounds(surrogateModelSample.iterations / 10);
        return new CatBoostModel(surrogateModelSample, workingDirectory, modelName);
    }
    private int CountNextSamplesForObjectiveFunction()
    {
        const int min_count = 5;
        const int max_count = 1000;
        int nbProcessedSamples = SamplesWithScore.Count;
        if (nbProcessedSamples <= min_count)
        {
            return min_count;
        }
        var elapsedSeconds = (DateTime.Now - _searchStartTime).TotalSeconds;
        double averageSpeed = elapsedSeconds / nbProcessedSamples;
        int count = (int)(30 / averageSpeed);
        if (nbProcessedSamples < 100 && count>50)
        {
            count = 50;
        }
        count = Math.Max(count, min_count);
        count = Math.Min(count, max_count);
        Log.Debug($"Computing {count} random samples for next training (observed average speed : {Math.Round(averageSpeed,4)}s/sample training)");
        return count;
    }
    /// <summary>
    /// retrieve 'CountNextSamplesForObjectiveFunction()' promising samples based on their estimate associated score
    /// (computed with the surrogate model)
    /// </summary>
    /// <returns></returns>
    private IEnumerable<Tuple<ISample, float[], float, string>> NextSamplesForObjectiveFunction()
    {
        int count = CountNextSamplesForObjectiveFunction();

        var result = new List<Tuple<ISample, float[], float, string>>();

        if (SearchSpace.All(t => t.Value.IsConstant))
        {
            var sampleAsFloatArray = new float[0];
            var (randomSample, randomSampleDescription) = FromFloatVectorToSampleAndDescription(sampleAsFloatArray);
            if (randomSample != null)
            {
                result.Add(Tuple.Create(randomSample, sampleAsFloatArray, 0f, randomSampleDescription));
            }
            return result;
        }

        // we retrieve 100x more random samples then needed to keep only the top 1%
        using var x = RandomSamplesForPrediction(count*100);
        //using var dataset = new InMemoryDataSet(x, null, "", Objective_enum.Regression, null, featureNames: SurrogateModelFeatureNames(), categoricalFeatures: SurrogateModelCategoricalFeature());
        using var dataset = new InMemoryDataSet(x, null, "", Objective_enum.Regression, null, SurrogateModelFeatureNames(), IsCategoricalColumn);
        
        // we compute the estimate score associated with each random sample (using the surrogate model)
        var y = _samplesUsedForModelTraining == 0 
                
                // the model has not been trained so far, we can not use it for now
                ? DataFrame.New(new CpuTensor<float>(new [] { x.Shape[0], 1 }), new List<string>{"y"})

                // the model has been already trained, we can use it
                : _surrogateModel.Predict(dataset, true); //no id columns for surrogate model
        var ySpan = y.FloatCpuTensor().AsFloatCpuSpan;
        var estimateRandomSampleScoreAndIndex = new List<Tuple<float, int>>();
        for (var index = 0; index < ySpan.Length; index++)
        {
            estimateRandomSampleScoreAndIndex.Add(Tuple.Create(ySpan[index], index));
        }

        //we sort all random samples from more promising to less interesting
        var orderedSamples = (_higherScoreIsBetter.HasValue && _higherScoreIsBetter.Value)
            //higher score is better : the most promising sample is the one with the highest score
            ? estimateRandomSampleScoreAndIndex.OrderByDescending(e => e.Item1)
            //lower score is better : the most promising sample is the one with the lowest score
            : estimateRandomSampleScoreAndIndex.OrderBy(e => e.Item1);
        foreach (var (estimateRandomSampleScore, index) in orderedSamples)
        {
            var sampleAsFloatArray = x.RowSlice(index, 1).ContentAsFloatArray();
            var (randomSample,randomSampleDescription) = FromFloatVectorToSampleAndDescription(sampleAsFloatArray);
            if (randomSample == null)
            {
                continue;
            }
            result.Add(Tuple.Create(randomSample, sampleAsFloatArray, estimateRandomSampleScore, randomSampleDescription));
            if (result.Count >= count)
            {
                break;
            }
        }
        return result;
    }
    private (ISample, string) FromFloatVectorToSampleAndDescription(float[] sampleAsFloatVector)
    {
        var searchSpaceHyperparameters = new Dictionary<string, string>();
        int idx = 0;
        foreach (var (parameterName, parameterSearchSpace) in SearchSpace.OrderBy(l => l.Key))
        {
            if (parameterSearchSpace.IsConstant)
            {
                searchSpaceHyperparameters[parameterName] = parameterSearchSpace.Next_SampleStringValue(_rand, HyperparameterSearchSpace.RANDOM_SEARCH_OPTION.FULLY_RANDOM);
            }
            else
            {
                searchSpaceHyperparameters[parameterName] = parameterSearchSpace.BayesianSearchFloatValue_to_SampleStringValue(sampleAsFloatVector[idx++]);
            }
        }
        Debug.Assert(idx == sampleAsFloatVector.Length);
        var sample = CreateDefaultSample();
        sample.Set(Utils.FromString2String_to_String2Object(searchSpaceHyperparameters));
        //we try to fix inconsistencies in the sample
        if (!sample.FixErrors())
        {
            return (null, ""); //we failed to fix the inconsistencies in the sample : we have to discard it 
        }
        //we ensure that we have not already processed this search space
        lock (_processedSpaces)
        {
            if (!_processedSpaces.Add(sample.ComputeHash()))
            {
                return (null, ""); //already processed before
            }
        }
        return (sample, ToSampleDescription(searchSpaceHyperparameters, sample));
    }
    /// <summary>
    /// return 'count' random samples
    /// </summary>
    /// <param name="count">the number of random samples to return (number of rows in returned tensor)</param>
    /// <returns></returns>
    private CpuTensor<float> RandomSamplesForPrediction(int count)
    {
        var orderedValues = SearchSpace.OrderBy(t => t.Key).Where(t=>!t.Value.IsConstant).Select(t=>t.Value).ToList();

        float[] NextRandomSamplesForPrediction()
        {
            var res = new float[orderedValues.Count];
            int idx = 0;
            foreach (var v in orderedValues)
            {
                res[idx++] = v.Next_BayesianSearchFloatValue(_rand, _randomSearchOption);
            }
            return res;
        }

        var rows = new List<float[]>();
        while (rows.Count < count)
        {
            rows.Add(NextRandomSamplesForPrediction());
        }
        return CpuTensor<float>.NewCpuTensor(rows);
    }
    /// <summary>
    /// train the model with all samples with an associated score
    /// </summary>
    /// <returns>the number of samples used to train the model</returns>
    private int TrainSurrogateModel()
    {
        var samplesWithScore = SamplesWithScore;
        var xRows = samplesWithScore.Select(t => t.Item2).ToList();

        if (xRows.Count == 0)
        {
            Log.Info($"No samples to train the surrogate model");
            return 0;
        }

        // we train the surrogate models with all samples with a computed score
        using var x = CpuTensor<float>.NewCpuTensor(xRows);
        var yData = samplesWithScore.Select(t => t.Item4).ToArray();
        using var y_true = new CpuTensor<float>(new[] { x.Shape[0], 1 }, yData);

        //var (df1, df2) = DataFrame.LoadFloatDataFrame("C:/Projects/Challenges/WasYouStayWorthItsPrice/Dataset/7C84EFF429.csv", true).Split(new[] { "y" });
        //var x = df1.FloatCpuTensor();
        //var y_true = df2.FloatCpuTensor();
        
        using var trainingDataset = new InMemoryDataSet(x, y_true, "", Objective_enum.Regression, null, SurrogateModelFeatureNames(), IsCategoricalColumn);
        Log.Info($"Training surrogate model with {x.Shape[0]} samples");
        Utils.TryDelete(_surrogateTrainedFiles.train_XDatasetPath);
        Utils.TryDelete(_surrogateTrainedFiles.train_YDatasetPath);
        Utils.TryDelete(_surrogateTrainedFiles.train_XYDatasetPath);
        Utils.TryDelete(_surrogateTrainedFiles.validation_XDatasetPath);
        Utils.TryDelete(_surrogateTrainedFiles.validation_YDatasetPath);
        Utils.TryDelete(_surrogateTrainedFiles.validation_XYDatasetPath);
        
        //AdjustSurrogateModelSampleForTrainingDatasetCount(trainingDataset.Count);

        DataSet resizedTrainingDataset = trainingDataset;
        const int minimumRowsForTraining = 1;
        if (trainingDataset.Count < minimumRowsForTraining)
        {
            resizedTrainingDataset = trainingDataset.Resize(minimumRowsForTraining, false);
        }
        AdjustSurrogateModelSampleForTrainingDatasetCount(resizedTrainingDataset.Count);
        _surrogateTrainedFiles = _surrogateModel.Fit(resizedTrainingDataset, null);

        // we compute the score of the surrogate model on the training dataset
        var y_pred = _surrogateModel.Predict(trainingDataset, true); // no id columns for surrogate model
        var surrogateModelTrainingLoss = _surrogateModel.ComputeLoss(y_true, y_pred.FloatCpuTensor());
        Log.Info($"Surrogate model Training Loss: {surrogateModelTrainingLoss} (trained on {x.Shape[0]} samples)");
        y_pred.FloatCpuTensor().Dispose();
        return xRows.Count;
    }
    private (string train_XDatasetPath, string train_YDatasetPath, string train_XYDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath, string validation_XYDatasetPath, IScore trainLossIfAvailable, IScore validationLossIfAvailable, IScore trainRankingMetricIfAvailable, IScore validationRankingMetricIfAvailable) _surrogateTrainedFiles = (null, null, null, null, null, null, null, null, null, null);
    private void AdjustSurrogateModelSampleForTrainingDatasetCount(int trainingDatasetCount)
    {
        if (_surrogateModel is LightGBMModel lightGbmModel)
        {
            var adjusted_min_data_in_bin = Math.Max(1, trainingDatasetCount / 5);
            if (adjusted_min_data_in_bin < lightGbmModel.LightGbmSample.min_data_in_bin)
            {
                lightGbmModel.LightGbmSample.min_data_in_bin = adjusted_min_data_in_bin;
            }
        }
    }
    private string[] SurrogateModelFeatureNames()
    {
        return SearchSpace.OrderBy(t => t.Key).Where(t=>!t.Value.IsConstant).Select(t=>t.Key).ToArray();
    }
    private bool IsCategoricalColumn(string columnName)
    {
        if (!SearchSpace.ContainsKey(columnName))
        {
            return false;
        }
        var val = SearchSpace[columnName];
        return !val.IsConstant && val.IsCategoricalHyperparameter;
    }
}
