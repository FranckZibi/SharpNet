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
    [NotNull] private readonly string _workingDirectory;
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
    /// 
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
    /// number of model trained in parallel (in separate threads)
    /// </summary>
    private readonly int _numModelTrainingInParallel;

    private readonly int _numberOfNewSamplesWithCostToRetrainSurrogateModel;
    private readonly int _numberOfSampleCandidatesToPredictAtTheSameTime;

    /// <summary>
    /// Number of samples that have been used to train the model '_surrogateModel'
    /// 0 means the model has not been trained so far
    /// The value is in the range [0, SamplesWithScore.Length] 
    /// </summary>
    private int _samplesUsedForModelTraining = 0;
    #endregion

    public BayesianSearchHPO(IDictionary<string, object> searchSpace, Func<T> createDefaultSample, Action<T> postBuild,
        Func<T, bool> isValidSample, [NotNull] string workingDirectory,
        AbstractHyperParameterSearchSpace.RANDOM_SEARCH_OPTION randomSearchOption, 
        int numModelTrainingInParallel, 
        int numberOfNewSamplesWithCostToRetrainSurrogateModel,
        int numberOfSampleCandidatesToPredictAtTheSameTime,
        Action<string> log, 
        int maxSamplesToProcess) :
        base(searchSpace, createDefaultSample, postBuild, isValidSample, log, maxSamplesToProcess)
    {
        Debug.Assert(numModelTrainingInParallel >= 1);

        if (!Directory.Exists(workingDirectory))
        {
            Directory.CreateDirectory(workingDirectory);
        }

        _workingDirectory = workingDirectory;
        _randomSearchOption = randomSearchOption;
        _numModelTrainingInParallel = numModelTrainingInParallel;
        _numberOfNewSamplesWithCostToRetrainSurrogateModel = numberOfNewSamplesWithCostToRetrainSurrogateModel;
        _numberOfSampleCandidatesToPredictAtTheSameTime = numberOfSampleCandidatesToPredictAtTheSameTime;

        var surrogateModelParameters = new Parameters();
        var categoricalFeaturesFieldValue = (SurrogateModelCategoricalFeature().Length>= 1) ? ("name:" + string.Join(',', SurrogateModelCategoricalFeature())) : "";
        ClassFieldSetter.Set(surrogateModelParameters, new Dictionary<string, object> {
            { "num_threads", 2},
            { "device_type", "cpu" },
            { "num_iterations", 200 },
            { "boosting", "rf" },
            { "verbosity", -1 },
            { "early_stopping_round", 0 },
            { "bagging_fraction", 0.9 },
            { "bagging_freq", 1 },
            { "colsample_bytree", 0.9 },
            { "min_data_in_leaf", 5 },
            //{ "num_leaves", 50 },
            { "categorical_feature", categoricalFeaturesFieldValue },
        });
        _surrogateModel = new LightGBMModel(surrogateModelParameters, workingDirectory, "");
    }

    private string SurrogateModelTrainingDatasetPath => Path.Combine(_workingDirectory, "Dataset", "surrogate_train.csv");
    private string SurrogateModelPredictDatasetPath => Path.Combine(_workingDirectory, "Dataset", "surrogate_predict.csv");

    protected override (T,int,string) Next
    {
        get
        {
            lock (_lockObject)
            {
                if (_nextSamplesToCompute.Count == 0)
                {
                    _log("No more samples available, computing new ones");
                    _nextSamplesToCompute.AddRange(GetNextSamplesForObjectiveFunction());
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

    private IEnumerable<Tuple<T, float[], float, string>> GetNextSamplesForObjectiveFunction()
    {
        var result = new List<Tuple<T, float[], float, string>>();

        using var x = NewSamplesForPrediction();
        using var dataset = new InMemoryDataSet(x, null, "", Objective_enum.Regression, null, new[] { "NONE" }, SurrogateModelFeatureNames(), false);


        LightGBMModel.Save(dataset, SurrogateModelPredictDatasetPath, Parameters.task_enum.predict, true);
        var y = _samplesUsedForModelTraining == 0 
                
                // the model has not been trained so far, we can not use it for now
                ? new CpuTensor<float>(new [] { x.Shape[0], 1 }) 

                // the model has been already trained, we can use it
                : _surrogateModel.Predict(SurrogateModelPredictDatasetPath);

        File.Delete(SurrogateModelPredictDatasetPath);


        var ySpan = y.AsFloatCpuSpan;
        var surrogateModelCostAndIndex = new List<Tuple<float, int>>();
        for (var index = 0; index < ySpan.Length; index++)
        {
            surrogateModelCostAndIndex.Add(Tuple.Create(ySpan[index], index));
        }

        foreach (var (surrogateModelCost, index) in surrogateModelCostAndIndex.OrderBy(e => e.Item1))
        {
            var sampleAsFloatArray = x.RowSlice(index, 1).ContentAsFloatArray();
            var (currentSample,currentSampleDescription) = FromFloatVectorToSampleAndDescription(sampleAsFloatArray);
            if (currentSample == null)
            {
                continue;
            }
            result.Add(Tuple.Create(currentSample, sampleAsFloatArray, surrogateModelCost, currentSampleDescription));
            if (result.Count >= _numModelTrainingInParallel)
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
        var sampleDescription = ToSampleDescription(searchSpaceHyperParameters);

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
        var t = CreateDefaultSample();
        ClassFieldSetter.Set(t, FromString2String_to_String2Object(searchSpaceHyperParameters));
        PostBuild(t);
        return (IsValidSample(t) ? t : null, sampleDescription);
    }



    private CpuTensor<float> NewSamplesForPrediction()
    {

        var orderedValues = SearchSpace.OrderBy(t => t.Key).Where(t=>!t.Value.IsConstant).Select(t=>t.Value).ToList();

        float[] NextSampleForPrediction()
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
        while (rows.Count < _numberOfSampleCandidatesToPredictAtTheSameTime)
        {
            rows.Add(NextSampleForPrediction());
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

            _log($"Registering actual cost ({cost}) of sample {sampleId} (surrogate cost estimate: {surrogateCostEstimate})");
            _samplesWithScoreIfAvailable[sampleId] = Tuple.Create(sampleTuple.Item1, sampleTuple.Item2, surrogateCostEstimate, cost, sampleId, sampleDescription);
            RegisterSampleCost(SearchSpace, sample, cost, elapsedTimeInSeconds);

            if (SamplesWithScore.Count >= (_samplesUsedForModelTraining + _numberOfNewSamplesWithCostToRetrainSurrogateModel))
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
            _log($"No samples to train the surrogate model");
            return 0;
        }
        using var x = NewCpuTensor(xRows);
        var yData = samplesWithScore.Select(t => t.Item4).ToArray();
        // ReSharper disable once InconsistentNaming
        using var y_true = new CpuTensor<float>(new[] { x.Shape[0], 1 }, yData);
        using var dataset = new InMemoryDataSet(x, y_true, "", Objective_enum.Regression, null, null, SurrogateModelFeatureNames(), false);
        _log($"Training surrogate model with {x.Shape[0]} samples");

        LightGBMModel.Save(dataset, SurrogateModelTrainingDatasetPath, Parameters.task_enum.train, true);
        _surrogateModel.Train(SurrogateModelTrainingDatasetPath);
        //File.Delete(SurrogateModelTrainingDatasetPath);

        LightGBMModel.Save(dataset, SurrogateModelPredictDatasetPath, Parameters.task_enum.predict, true);
        var y_pred = _surrogateModel.Predict(SurrogateModelPredictDatasetPath);
        File.Delete(SurrogateModelPredictDatasetPath);

        double surrogateModelTrainingRmse = _surrogateModel.ComputeRmse(y_true, y_pred);
        _log($"Surrogate model Training RMSE: {surrogateModelTrainingRmse} (trained on {x.Shape[0]} samples)");

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