using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Models;

namespace SharpNet.HPO;

public class BayesianSearchHPO : AbstractHpo
{
    #region private fields
    private readonly Random _rand = new();
    private readonly HashSet<string> _processedSpaces = new();
    private readonly AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION _randomSearchOption;
    /// <summary>
    /// the model use to predict the objective function score
    /// </summary>
    private readonly AbstractModel  _surrogateModel;
    /// <summary>
    /// Item1:  the sample
    /// Item2:  the sample as a float vector array
    /// Item3:  the cost predicted by the surrogate model
    /// Item4:  the actual cost of the sample (or NaN if it has not been computed yet)
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
    private readonly object _lockObject = new object();
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
        AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION randomSearchOption = AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION.PREFER_MORE_PROMISING) :
        base(searchSpace, createDefaultSample, workingDirectory)
    {
        _randomSearchOption = randomSearchOption;
        _searchStartTime = DateTime.Now;
        var surrogateModelName = "surrogate_" + System.Diagnostics.Process.GetCurrentProcess().Id;
        _surrogateModel = BuildRandomForestSurrogateModel(_workingDirectory, surrogateModelName, SurrogateModelCategoricalFeature());
        //_surrogateModel = BuildCatBoostSurrogateModel(_workingDirectory, surrogateModelName);
    }

    private static AbstractModel BuildRandomForestSurrogateModel(string workingDirectory, string modelName, string[] surrogateModelCategoricalFeature)
    {
        surrogateModelCategoricalFeature ??= Array.Empty<string>();
        // the surrogate model will be trained with a LightGBM using random forests (boosting=rf)
        var surrogateModelSample = new LightGBMSample();
        var categoricalFeaturesFieldValue = (surrogateModelCategoricalFeature.Length>= 1) ? ("name:" + string.Join(',', surrogateModelCategoricalFeature)) : "";
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
            { "num_boost_round", 100 },
            { "num_leaves", 50 },
            { "num_threads", 2},
            { "objective", "regression"},
            { "path_smooth", 1.0f},
            { "verbosity", -1 },
        });
        return new LightGBMModel(surrogateModelSample, workingDirectory, modelName);
    }

    private static AbstractModel BuildCatBoostSurrogateModel(string workingDirectory, string modelName)
    {
        // the surrogate model will be trained with CatBoost
        CatBoostSample surrogateModelSample = new ()
        {
            iterations = 100,
            loss_function = CatBoostSample.loss_function_enum.RMSE,
            eval_metric = CatBoostSample.metric_enum.RMSE,
            allow_writing_files = false,
            thread_count = 2,
            logging_level = CatBoostSample.logging_level_enum.Silent
        };

        surrogateModelSample.set_early_stopping_rounds(surrogateModelSample.iterations/10);
        return new CatBoostModel(surrogateModelSample, workingDirectory, modelName);
    }
    public static InMemoryDataSet LoadSurrogateTrainingDataset(string dataFramePath, string[] categoricalFeature = null)
    {
        var df = Dataframe.Load(dataFramePath, true, ',');
        var x_df = df.Drop(new[] {"y"});
        var x = x_df.Tensor;
        var y_df = df.Keep(new[] {"y"});
        var y = y_df.Tensor;
        return new InMemoryDataSet(x, y, "", Objective_enum.Regression, null, new[] { "NONE" }, x_df.FeatureNames, categoricalFeature??Array.Empty<string>(), false);
    }
    public static InMemoryDataSet LoadSurrogateValidationDataset(string dataFramePath, string[] categoricalFeature = null)
    {
        var df = Dataframe.Load(dataFramePath, true, ',');
        var x = df.Tensor;
        return new InMemoryDataSet(x, null, "", Objective_enum.Regression, null, new[] { "NONE" }, df.FeatureNames, categoricalFeature ?? Array.Empty<string>(), false);
    }

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
                var surrogateCostEstimate = _nextSamplesToCompute[0].Item3;
                var sampleDescription = _nextSamplesToCompute[0].Item4;
                int sampleId = _samplesWithScoreIfAvailable.Count;
                _samplesWithScoreIfAvailable.Add(Tuple.Create(sample, sampleAsFloatVector, surrogateCostEstimate, float.NaN, sampleId, sampleDescription));
                _nextSamplesToCompute.RemoveAt(0);
                return (sample, sampleId, sampleDescription);
            }
        }
    }
    protected override void RegisterSampleCost(ISample sample, int sampleId, float cost, double elapsedTimeInSeconds)
    {
        lock (_lockObject)
        {
            _allCost.Add(cost, 1);
            var sampleTuple = _samplesWithScoreIfAvailable[sampleId];
            var surrogateCostEstimate = sampleTuple.Item3;
            var sampleDescription = sampleTuple.Item6;
            if (float.IsNaN(surrogateCostEstimate))
            {
                throw new Exception($"surrogate cost is missing from sampleId {sampleId}");
            }
            if (!float.IsNaN(sampleTuple.Item4))
            {
                throw new Exception($"cost has been already registered for sampleId {sampleId}");
            }
            if (sampleTuple.Item5 != sampleId)
            {
                throw new Exception($"invalid sampleId {sampleTuple.Item5} , should be {sampleId}");
            }
            if (float.IsNaN(cost))
            {
                throw new Exception($"cost can not be NaN for sampleId {sampleId}");
            }

            Log.Debug($"Registering actual cost ({cost}) of sample {sampleId} (surrogate cost estimate: {surrogateCostEstimate})");
            _samplesWithScoreIfAvailable[sampleId] = Tuple.Create(sampleTuple.Item1, sampleTuple.Item2, surrogateCostEstimate, cost, sampleId, sampleDescription);
            RegisterSampleCost(SearchSpace, sample, cost, elapsedTimeInSeconds);

            if (SamplesWithScore.Count >= Math.Max(10, 1.2 * (_samplesUsedForModelTraining)))
            {
                _samplesUsedForModelTraining = TrainSurrogateModel();
            }
        }
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
    /// retrieve 'count' promising samples based on their estimate associated cost (computed with the surrogate model)
    /// </summary>
    /// <returns></returns>
    private IEnumerable<Tuple<ISample, float[], float, string>> NextSamplesForObjectiveFunction()
    {
        int count = CountNextSamplesForObjectiveFunction();

        var result = new List<Tuple<ISample, float[], float, string>>();

        // we retrieve 100x more random samples then needed to keep only the top 1%
        using var x = RandomSamplesForPrediction(count*100);
        using var dataset = new InMemoryDataSet(x, null, "", Objective_enum.Regression, null, new[] { "NONE" }, SurrogateModelFeatureNames(), SurrogateModelCategoricalFeature(), false);

        // we compute the estimate cost associated with each random sample (using the surrogate model)
        if (File.Exists(_surrogateModel.LastDatasetPathUsedForPrediction))
        {
            File.Delete(_surrogateModel.LastDatasetPathUsedForPrediction);
        }

        var y = _samplesUsedForModelTraining == 0 
                
                // the model has not been trained so far, we can not use it for now
                ? new CpuTensor<float>(new [] { x.Shape[0], 1 }) 

                // the model has been already trained, we can use it
                : _surrogateModel.Predict(dataset);
        var ySpan = y.AsFloatCpuSpan;
        var estimateRandomSampleCostAndIndex = new List<Tuple<float, int>>();
        for (var index = 0; index < ySpan.Length; index++)
        {
            estimateRandomSampleCostAndIndex.Add(Tuple.Create(ySpan[index], index));
        }

        //we sort all random samples from more promising (lowest cost) to less interesting (highest cost)
        foreach (var (estimateRandomSampleCost, index) in estimateRandomSampleCostAndIndex.OrderBy(e => e.Item1))
        {
            var sampleAsFloatArray = x.RowSlice(index, 1).ContentAsFloatArray();
            var (randomSample,randomSampleDescription) = FromFloatVectorToSampleAndDescription(sampleAsFloatArray);
            if (randomSample == null)
            {
                continue;
            }
            result.Add(Tuple.Create(randomSample, sampleAsFloatArray, estimateRandomSampleCost, randomSampleDescription));
            if (result.Count >= count)
            {
                break;
            }
        }
        return result;
    }
    private (ISample, string) FromFloatVectorToSampleAndDescription(float[] sampleAsFloatVector)
    {
        var searchSpaceHyperParameters = new Dictionary<string, string>();
        int idx = 0;
        foreach (var (parameterName, parameterSearchSpace) in SearchSpace.OrderBy(l => l.Key))
        {
            if (parameterSearchSpace.IsConstant)
            {
                searchSpaceHyperParameters[parameterName] = parameterSearchSpace.Next_SampleStringValue(_rand, AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION.FULLY_RANDOM);
            }
            else
            {
                searchSpaceHyperParameters[parameterName] = parameterSearchSpace.BayesianSearchFloatValue_to_SampleStringValue(sampleAsFloatVector[idx++]);
            }
        }
        Debug.Assert(idx == sampleAsFloatVector.Length);
        var sample = CreateDefaultSample();
        sample.Set(Utils.FromString2String_to_String2Object(searchSpaceHyperParameters));
        //we ensure that we have not already processed this search space
        lock (_processedSpaces)
        {
            if (!_processedSpaces.Add(sample.ComputeHash()))
            {
                return (null, ""); //already processed before
            }
        }
        if (!sample.PostBuild())
        {
            return (null, "");
        }
        return (sample, ToSampleDescription(searchSpaceHyperParameters, sample));
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
        return NewCpuTensor(rows);
    }
    private static CpuTensor<float> NewCpuTensor(IList<float[]> rows)
    {
        var x = new CpuTensor<float>(new[] { rows.Count, rows[0].Length });
        var xSpan = x.AsFloatCpuSpan;
        int xSpanIndex = 0;
        foreach(var row in rows)
        {
            Debug.Assert(row.Length == x.Shape[1]);
            foreach (var t in row)
            {
                xSpan[xSpanIndex++] = t;
            }
        }
        Debug.Assert(xSpanIndex == x.Count);
        return x;
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
        using var x = NewCpuTensor(xRows);
        var yData = samplesWithScore.Select(t => t.Item4).ToArray();
        using var y_true = new CpuTensor<float>(new[] { x.Shape[0], 1 }, yData);
        using var trainingDataset = new InMemoryDataSet(x, y_true, "", Objective_enum.Regression, null, null, SurrogateModelFeatureNames(), SurrogateModelCategoricalFeature(), false);
        Log.Info($"Training surrogate model with {x.Shape[0]} samples");
        if (File.Exists(_surrogateModel.LastDatasetPathUsedForTraining))
        {
            File.Delete(_surrogateModel.LastDatasetPathUsedForTraining);
        }
        AdjustSurrogateModelSampleForTrainingDatasetCount(trainingDataset.Count);
        _surrogateModel.Fit(trainingDataset, null);

        // we compute the RMSE of the surrogate model on the training dataset
        if (File.Exists(_surrogateModel.LastDatasetPathUsedForPrediction))
        {
            File.Delete(_surrogateModel.LastDatasetPathUsedForPrediction);
        }
        using var y_pred = _surrogateModel.Predict(trainingDataset);
        double surrogateModelTrainingRmse = _surrogateModel.ComputeScore(y_true, y_pred);
        Log.Info($"Surrogate model Training RMSE: {surrogateModelTrainingRmse} (trained on {x.Shape[0]} samples)");

        return xRows.Count;
    }

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
    private string[] SurrogateModelCategoricalFeature()
    {
        return SearchSpace.OrderBy(t => t.Key).Where(t=> !t.Value.IsConstant&&t.Value.IsCategoricalHyperParameter).Select(t=>t.Key).ToArray();
    }
}
