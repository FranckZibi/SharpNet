using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.LightGBM;

namespace SharpNet.HPO;

public class BayesianSearchHPO<T> : AbstractHpo<T> where T : class, new()
{

    #region private fields
    private readonly Random _rand = new();
    private readonly HashSet<string> _processedSpaces = new();
    private readonly AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION _randomSearchOption;
    /// <summary>
    /// the model use to predict the objective function score
    /// </summary>
    private readonly LightGBMModel _surrogateModel;

    /// <summary>
    /// Item1:  the sample
    /// Item2:  the sample as a float vector array
    /// Item3:  the cost predicted by the surrogate model
    /// Item4:  the actual cost of the sample (or NaN if it has not been computed yet)
    /// Item5:  the sample id
    /// Item6:  the sample description
    /// </summary>
    private readonly List< Tuple<T, float[], float, float, int, string>> _samplesWithScoreIfAvailable = new ();

    private List<Tuple<T, float[], float, float, int, string>> SamplesWithScore =>  _samplesWithScoreIfAvailable.Where(e => !float.IsNaN(e.Item4)).ToList();

    /// <summary>
    /// next samples to use for the objective function
    /// if this list is empty, it means that we'll need to call the surrogate model to retrieve new samples to test
    /// Item1:  the sample
    /// Item2:  the sample as a float vector array
    /// Item3:  the score predicted by the surrogate model
    /// Item3:  the sample description
    /// </summary>
    private readonly List<Tuple<T, float[], float, string>> _nextSamplesToCompute = new();
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
        Func<T> createDefaultSample,
        Func<T, bool> postBuild,
        int maxAllowedSecondsForAllComputation,
        AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION randomSearchOption,
        [NotNull] string workingDirectory,
        HashSet<string> mandatoryCategoricalHyperParameters) :
        base(searchSpace, createDefaultSample, postBuild, maxAllowedSecondsForAllComputation, workingDirectory, mandatoryCategoricalHyperParameters)
    {
        _randomSearchOption = randomSearchOption;
        _searchStartTime = DateTime.Now;

        var surrogateModelParameters = new Parameters();
        var categoricalFeaturesFieldValue = (SurrogateModelCategoricalFeature().Length>= 1) ? ("name:" + string.Join(',', SurrogateModelCategoricalFeature())) : "";
        ClassFieldSetter.Set(surrogateModelParameters, new Dictionary<string, object> {
            { "bagging_fraction", 0.5 },
            { "bagging_freq", 1 },
            { "boosting", "rf" },
            { "categorical_feature", categoricalFeaturesFieldValue },
            { "colsample_bytree", 0.9 },
            { "colsample_bynode", 0.9 },
            { "device_type", "cpu" },
            { "early_stopping_round", 0 },
            { "extra_trees", false },
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
            { "path_smooth", 1.0f},
            { "verbosity", -1 },
        });
        SurrogatePrefix = "surrogate_" + System.Diagnostics.Process.GetCurrentProcess().Id;
        _surrogateModel = new LightGBMModel(surrogateModelParameters, workingDirectory, SurrogatePrefix);
    }

    private string SurrogatePrefix { get; }
    private string SurrogateModelTrainingDatasetPath => Path.Combine(_workingDirectory, SurrogatePrefix+"_dataset.csv");
    private string SurrogateModelPredictDatasetPath => Path.Combine(_workingDirectory, SurrogatePrefix+"_predictions.csv");

    protected override (T,int,string) Next
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
    private IEnumerable<Tuple<T, float[], float, string>> NextSamplesForObjectiveFunction()
    {
        int count = CountNextSamplesForObjectiveFunction();

        var result = new List<Tuple<T, float[], float, string>>();

        // we retrieve 100x more random samples then needed to keep only the top 1%
        using var x = RandomSamplesForPrediction(count*100);
        using var dataset = new InMemoryDataSet(x, null, "", Objective_enum.Regression, null, new[] { "NONE" }, SurrogateModelFeatureNames(), false);

        // we compute the estimate cost associated with each random sample (using the surrogate model)
        LightGBMModel.Save(dataset, SurrogateModelPredictDatasetPath, Parameters.task_enum.predict, true);
        var y = _samplesUsedForModelTraining == 0 
                
                // the model has not been trained so far, we can not use it for now
                ? new CpuTensor<float>(new [] { x.Shape[0], 1 }) 

                // the model has been already trained, we can use it
                : _surrogateModel.Predict(SurrogateModelPredictDatasetPath);
        File.Delete(SurrogateModelPredictDatasetPath);
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


    private (T,string) FromFloatVectorToSampleAndDescription(float[] sampleAsFloatVector)
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
        //we ensure that we have not already processed this search space
        var searchSpaceHash = ComputeHash(searchSpaceHyperParameters);
        lock (_processedSpaces)
        {
            if (!_processedSpaces.Add(searchSpaceHash))
            {
                return (null, ""); //already processed before
            }
        }
        var sample = CreateDefaultSample();
        ClassFieldSetter.Set(sample, FromString2String_to_String2Object(searchSpaceHyperParameters));
        if (!PostBuild(sample))
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

    protected override void RegisterSampleCost(object sample, int sampleId, float cost, double elapsedTimeInSeconds)
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

            if (SamplesWithScore.Count >= Math.Max(10, 1.2*(_samplesUsedForModelTraining)))
            {
                _samplesUsedForModelTraining = TrainSurrogateModel();
            }
        }
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
        using var x = NewCpuTensor(xRows);
        var yData = samplesWithScore.Select(t => t.Item4).ToArray();
        // ReSharper disable once InconsistentNaming
        using var y_true = new CpuTensor<float>(new[] { x.Shape[0], 1 }, yData);
        using var dataset = new InMemoryDataSet(x, y_true, "", Objective_enum.Regression, null, null, SurrogateModelFeatureNames(), false);
        Log.Info($"Training surrogate model with {x.Shape[0]} samples");

        LightGBMModel.Save(dataset, SurrogateModelTrainingDatasetPath, Parameters.task_enum.train, true);
        _surrogateModel.Train(SurrogateModelTrainingDatasetPath);
        //File.Delete(SurrogateModelTrainingDatasetPath);

        LightGBMModel.Save(dataset, SurrogateModelPredictDatasetPath, Parameters.task_enum.predict, true);
        var y_pred = _surrogateModel.Predict(SurrogateModelPredictDatasetPath);
        File.Delete(SurrogateModelPredictDatasetPath);

        double surrogateModelTrainingRmse = _surrogateModel.ComputeRmse(y_true, y_pred);
        Log.Info($"Surrogate model Training RMSE: {surrogateModelTrainingRmse} (trained on {x.Shape[0]} samples)");

        return xRows.Count;
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
