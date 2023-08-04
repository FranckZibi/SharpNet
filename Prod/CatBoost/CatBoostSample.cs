using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using SharpNet.Datasets;
using SharpNet.GPU;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.Models;
// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable FieldCanBeMadeReadOnly.Global
// ReSharper disable NotAccessedField.Global

namespace SharpNet.CatBoost;

[SuppressMessage("ReSharper", "UnusedMember.Global")]
[SuppressMessage("ReSharper", "IdentifierTypo")]
public class CatBoostSample : AbstractModelSample
{
    #region Constructors
    public CatBoostSample() :base(CategoricalHyperParameters)
    {
    }
    #endregion

    // ReSharper disable once MemberCanBeMadeStatic.Global
    public void AddExtraMetricToComputeForTraining()
    {
    }
    [SuppressMessage("ReSharper", "ExpressionIsAlwaysNull")]
    public (IScore trainLossIfAvailable, IScore validationLossIfAvailable, IScore trainRankingMetricIfAvailable, IScore validationRankingMetricIfAvailable) ExtractScores(IEnumerable<string> linesFromLog)
    {
        List<string> tokenAndMandatoryTokenAfterToken = new() { "learn:", null, "test:", null, "best:", null };
        var extractedScores = Utils.ExtractValuesFromOutputLog(linesFromLog, 0, tokenAndMandatoryTokenAfterToken.ToArray());
        var trainValue = extractedScores[0];
        var validationValue = extractedScores[use_best_model ? 2 : 1];
        var trainLossIfAvailable = double.IsNaN(trainValue) ? null : new Score((float)trainValue, GetLoss());
        var validationLossIfAvailable = double.IsNaN(validationValue) ? null : new Score((float)validationValue, GetLoss());
        IScore trainRankingMetricIfAvailable = null;
        IScore validationRankingMetricIfAvailable = null;
        return (trainLossIfAvailable, validationLossIfAvailable, trainRankingMetricIfAvailable, validationRankingMetricIfAvailable);
    }
    #region Common parameters

    /// <summary>
    /// The metric to use in training. The specified value also determines the machine learning problem to solve.
    /// aliases: objective
    /// </summary>
    public enum loss_function_enum { RMSE, Logloss, MAE, CrossEntropy, Quantile, LogLinQuantile, Lq, MultiRMSE, MultiClass, MultiClassOneVsAll, MultiLogloss, MultiCrossEntropy, MAPE, Poisson, PairLogit, PairLogitPairwise, QueryRMSE, QuerySoftMax, Tweedie, YetiRank, YetiRankPairwise, StochasticFilter, StochasticRank, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE}
    public loss_function_enum loss_function = loss_function_enum.DEFAULT_VALUE;

    public enum metric_enum { RMSE, Logloss, MAE, CrossEntropy, Quantile, LogLinQuantile, Lq, MultiClass, MultiClassOneVsAll, ultiLogloss, MultiCrossEntropy, MAPE, Poisson, PairLogit, PairLogitPairwise, QueryRMSE, QuerySoftMax, Tweedie, SMAPE, Recall, Precision, F1, TotalF1, Accuracy, BalancedAccuracy, BalancedErrorRate, Kappa, WKappa, LogLikelihoodOfPrediction, AUC, QueryAUC, R2, FairLoss, NumErrors, MCC, BrierScore, HingeLoss, HammingLoss, ZeroOneLoss, MSLE, edianAbsoluteError, Huber, Expectile, MultiRMSE, PairAccuracy, AverageGain, PFound, NDCG, DCG, FilteredDCG, NormalizedGini, PrecisionAt, RecallAt, MAP, CtrFactor, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } 
    public metric_enum eval_metric = metric_enum.DEFAULT_VALUE;

    /// <summary>
    /// The maximum number of trees that can be built when solving machine learning problems.
    /// aliases: num_boost_round, n_estimators, num_trees
    /// </summary>
    public int iterations = 1000;


    /// <summary>
    /// The learning rate.
    /// alias: eta
    /// </summary>
    public double learning_rate = DEFAULT_VALUE;

    /// <summary>
    /// The random seed used for training.
    /// alias: random_state
    /// </summary>
    public int random_seed = DEFAULT_VALUE;

    /// <summary>
    /// Coefficient at the L2 regularization term of the cost function.
    /// Any positive value is allowed.
    /// alias: reg_lambda
    /// </summary>
    public float l2_leaf_reg = DEFAULT_VALUE;

    /// <summary>
    /// Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes.
    /// </summary>
    public float bagging_temperature = DEFAULT_VALUE;

    /// <summary>
    /// Sample rate for bagging.
    /// This parameter can be used if one of the following bootstrap types is selected:
    /// Poisson, Bernoulli, MVS
    /// </summary>
    public float subsample = DEFAULT_VALUE;


    /// <summary>
    /// Frequency to sample weights and objects when building trees.
    /// </summary>
    public enum sampling_frequency_enum { PerTree, PerTreeLevel, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } 
    public sampling_frequency_enum sampling_frequency = sampling_frequency_enum.DEFAULT_VALUE;

    /// <summary>
    /// The amount of randomness to use for scoring splits when the tree structure is selected.
    /// Use this parameter to avoid over fitting the model.
    /// </summary>
    public float random_strength = DEFAULT_VALUE;

    /// <summary>
    /// If this parameter is set, the number of trees that are saved in the resulting model is defined as follows:
    ///    1. Build the number of trees defined by the training parameters.
    ///    2. Use the validation dataset to identify the iteration with the optimal value of the metric specified in  --eval-metric(--eval-metric).
    /// </summary>
    public bool use_best_model = true;


    /// <summary>
    /// Depth of the tree (any integer up to 16)
    /// alias: max_depth
    /// </summary>
    public int depth = DEFAULT_VALUE;

    /// <summary>
    /// The tree growing policy. Defines how to perform greedy tree construction.
    /// </summary>
    public enum grow_policy_enum { SymmetricTree, Depthwise, Lossguide, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } 
    public grow_policy_enum grow_policy = grow_policy_enum.DEFAULT_VALUE;

    /// <summary>
    /// The minimum number of training samples in a leaf.
    /// CatBoost does not search for new splits in leaves with samples count less than the specified value.
    /// Can be used only with the Lossguide and Depthwise growing policies.
    /// alias: min_child_samples
    /// </summary>
    public int min_data_in_leaf = DEFAULT_VALUE;

    /// <summary>
    /// The maximum number of leafs in the resulting tree.
    /// Can be used only with the Lossguide growing policy.
    /// It is not recommended to use values greater than 64, since it can significantly slow down the training process.
    /// alias: num_leaves 
    /// </summary>
    public int max_leaves = DEFAULT_VALUE;


    /// <summary>
    /// Random subspace method.
    /// The percentage of features to use at each split selection,
    /// when features are selected over again at random.
    /// The value must be in the range (0;1]
    /// alias: colsample_bylevel
    /// </summary>
    public float rsm = DEFAULT_VALUE;

    /// <summary>
    /// Coefficient for changing the length of folds.
    ///The value must be greater than 1. The best validation result is achieved with minimum values.
    /// With values close to 1 (for example, 1+\epsilon1+ϵ), each iteration takes a quadratic amount of memory
    /// and time for the number of objects in the iteration.
    /// Thus, low values are possible only when there is a small number of objects.
    /// </summary>
    public float fold_len_multiplier = DEFAULT_VALUE;

    /// <summary>
    /// The principles for calculating the approximated values.
    /// </summary>
    public bool approx_on_full_history = false;


    /// <summary>
    /// Boosting scheme.
    /// Possible values:
    ///     Ordered — Usually provides better quality on small datasets, but it may be slower than the Plain scheme.
    ///     Plain — The classic gradient boosting scheme.
    /// </summary>
    public enum boosting_type_enum { Ordered, Plain, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } 
    public boosting_type_enum boosting_type = boosting_type_enum.DEFAULT_VALUE;

    /// <summary>
    /// Enables the Stochastic Gradient Langevin Boosting mode.
    /// </summary>
    public bool langevin = false;

    /// <summary>
    /// The diffusion temperature of the Stochastic Gradient Langevin Boosting mode.
    /// Only non-negative values are supported.
    /// </summary>
    public float diffusion_temperature = DEFAULT_VALUE;

    /// <summary>
    /// The score type used to select the next split during the tree construction.
    /// </summary>
    public enum score_function_enum { Cosine, L2, NewtonCosine, NewtonL2, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } 
    public score_function_enum score_function = score_function_enum.DEFAULT_VALUE;
    #endregion

    #region Performance settings

    /// <summary>
    /// The number of threads to use during the training.
    /// The default value -1 is not accepted by the CLI
    /// </summary>
    public int thread_count = DEFAULT_VALUE;
    #endregion

    #region Processing unit settings
    /// <summary>
    /// The processing unit type to use for training.
    /// </summary>
    public enum task_type_enum { CPU, GPU}
    public task_type_enum task_type = task_type_enum.CPU;

    /// <summary>
    /// IDs of the GPU devices to use for training (indices are zero-based).
    /// </summary>
    public string devices;
    #endregion

    #region Overfitting detection settings
    /// <summary>
    /// Sets the over fitting detector type to Iter and stops the training after the specified
    /// number of iterations since the iteration with the optimal metric value.
    /// </summary>
    public void set_early_stopping_rounds(int count)
    {
        od_type = od_type_enum.Iter;
        od_wait = count;
    }

    /// <summary>
    /// The type of the over fitting detector to use.
    /// </summary>
    public enum od_type_enum {IncToDec, Iter, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE }
    public od_type_enum od_type = od_type_enum.DEFAULT_VALUE;


    /// <summary>
    /// The threshold for the IncToDec over fitting detector type.
    /// The training is stopped when the specified value is reached.
    /// Requires that a validation dataset was input.
    /// </summary>
    public float od_pval = DEFAULT_VALUE;

    /// <summary>
    /// The number of iterations to continue the training after the iteration with the optimal metric value.
    ///The purpose of this parameter differs depending on the selected over fitting detector type:
    ///     IncToDec — Ignore the over fitting detector when the threshold is reached and continue learning for the specified number of iterations after the iteration with the optimal metric value.
    ///     Iter — Consider the model over fitted and stop training after the specified number of iterations since the iteration with the optimal metric value.
    /// </summary>
    public int od_wait = DEFAULT_VALUE;
    #endregion

    #region Output settings
    public enum logging_level_enum { Silent, Verbose, Info, Debug, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } 
    public logging_level_enum logging_level = logging_level_enum.DEFAULT_VALUE;

    /// <summary>
    /// The logging level to output to stdout.
    /// Should be set to iterations/10
    /// </summary>
    public int verbose = DEFAULT_VALUE;

    /// <summary>
    /// The directory for storing the files generated during training.
    /// </summary>
    public string train_dir;

    /// <summary>
    /// The model size regularization coefficient.
    /// The larger the value, the smaller the model size. 
    /// </summary>
    public float model_size_reg = DEFAULT_VALUE;

    /// <summary>
    /// Allow to write analytical and snapshot files during training.
    /// If set to False, the snapshot and data visualization tools are unavailable.
    /// </summary>
    public bool allow_writing_files = true;
    #endregion

    public override string ToPath(string workingDirectory, string sampleName)
    {
        return Path.Combine(workingDirectory, sampleName + "_conf." + GetType().Name + ".json");
    }


    public override void Use_All_Available_Cores()
    {
        thread_count = Utils.CoreCount;
        devices = null;
        task_type = MustUseGPU ? task_type_enum.GPU : task_type_enum.CPU;
    }

    protected override string ToConfigContent(Func<string, object, bool> accept)
    {
        var result = new List<string>();
        foreach (var (parameterName, fieldValue) in ToDictionaryConfigContent(accept).OrderBy(f => f.Key))
        {
            var fieldValueAsJsonString = Utils.FieldValueToJsonString(fieldValue);
            result.Add($"\t\"{parameterName}\": {fieldValueAsJsonString}");
        }
        return "{" + Environment.NewLine + string.Join("," + Environment.NewLine, result) + Environment.NewLine + "}";
    }


    public override bool FixErrors()
    {
        // On GPU: grow policy Depthwise can't be used with ordered boosting
        if (grow_policy == grow_policy_enum.Depthwise && boosting_type == boosting_type_enum.Ordered && task_type == task_type_enum.GPU)
        {
            grow_policy = grow_policy_enum.SymmetricTree;
        }
        if (grow_policy != grow_policy_enum.SymmetricTree && grow_policy != grow_policy_enum.Depthwise)
        {
            min_data_in_leaf = DEFAULT_VALUE;
        }
        if (!langevin)
        {
            diffusion_temperature = DEFAULT_VALUE;
        }

        if (loss_function == loss_function_enum.DEFAULT_VALUE)
        {
            throw new ArgumentException("loss_function must always be specified");
        }
        if (eval_metric == metric_enum.DEFAULT_VALUE)
        {
            SetRankingEvaluationMetric(GetLoss());
        } 
        return true;
    }
    public override EvaluationMetricEnum GetLoss()
    {
        switch (loss_function)
        {
            case loss_function_enum.DEFAULT_VALUE:
                return EvaluationMetricEnum.DEFAULT_VALUE;
            case loss_function_enum.RMSE:
                return EvaluationMetricEnum.Rmse;
            case loss_function_enum.MAE:
                return EvaluationMetricEnum.Mae;
            case loss_function_enum.Logloss:
                return EvaluationMetricEnum.BinaryCrossentropy;
            case loss_function_enum.MultiClass:
                return EvaluationMetricEnum.CategoricalCrossentropy;
            default:
                throw new NotImplementedException($"can't manage metric {loss_function}");
        }
    }
    public override EvaluationMetricEnum GetRankingEvaluationMetric()
    {
        switch (eval_metric)
        {
            case metric_enum.DEFAULT_VALUE:
                return EvaluationMetricEnum.DEFAULT_VALUE;
            case metric_enum.MAE:
                return EvaluationMetricEnum.Mae;
            case metric_enum.Logloss:
                return EvaluationMetricEnum.BinaryCrossentropy;
            case metric_enum.MultiClass:
                return EvaluationMetricEnum.CategoricalCrossentropy;
            default:
                throw new NotImplementedException($"can't manage {nameof(eval_metric)} {eval_metric}");
        }
    }

    public override List<EvaluationMetricEnum> GetAllEvaluationMetrics()
    {
        return new List<EvaluationMetricEnum> { GetRankingEvaluationMetric() };
    }

    private void SetRankingEvaluationMetric(EvaluationMetricEnum rankingEvaluationMetric)
    {
        switch (rankingEvaluationMetric)
        {
            case EvaluationMetricEnum.DEFAULT_VALUE:
                eval_metric = metric_enum.DEFAULT_VALUE;
                return;
            case EvaluationMetricEnum.Mae:
                eval_metric = metric_enum.MAE;
                return;
            case EvaluationMetricEnum.BinaryCrossentropy:
                eval_metric = metric_enum.Logloss;
                return;
            case EvaluationMetricEnum.CategoricalCrossentropy:
                eval_metric = metric_enum.MultiClass;
                return;
            default:
                throw new NotImplementedException($"can't set {nameof(eval_metric)} {rankingEvaluationMetric}");
        }
    }

    private static readonly HashSet<string> CategoricalHyperParameters = new()
    {
        "use_best_model",
        "approx_on_full_history",
        "langevin",
        "loss_function",
        "eval_metric",
        "sampling_frequency",
        "grow_policy",
        "boosting_type",
        "score_function",
        "task_type",
        "od_type"
    };


    public override bool MustUseGPU => GPUWrapper.GetDeviceCount() >= 1;

    public override void SetTaskId(int taskId)
    {
        devices = MustUseGPU ? taskId.ToString() : null;
    }

    public override void FillSearchSpaceWithDefaultValues(IDictionary<string, object> existingHyperParameterValues)
    {
        const string taskTypeName = nameof(task_type);
        if (!existingHyperParameterValues.ContainsKey(taskTypeName))
        {
            const string devicesKeyName = nameof(devices);
            const string threadCountName = nameof(thread_count);
            existingHyperParameterValues.Remove(devicesKeyName); //need to be set after for GPU
            if (MustUseGPU)
            {
                //Each GPU will be run in parallel, and will have some CPU dedicated to him
                //So if we have 4 GPU, we'll run 4 tasks in parallel, and each task will have 1/4 of the total CPU resources
                existingHyperParameterValues[taskTypeName] = nameof(task_type_enum.GPU);
                existingHyperParameterValues[threadCountName] = Math.Max(1, Utils.CoreCount / GPUWrapper.GetDeviceCount());
            }
            else
            {
                //Each CPU will be run in parallel (no GPU)
                //So if we have 4 CPU, we'll run 4 tasks in parallel
                existingHyperParameterValues[taskTypeName] = nameof(task_type_enum.CPU);
                existingHyperParameterValues[threadCountName] = 1;
            }
        }
    }

    public override Model NewModel(AbstractDatasetSample datasetSample, string workingDirectory, string modelName)
    {
        return new CatBoostModel(this, workingDirectory, modelName);
    }
    
    /// <summary>
    /// The default Search Space for CatBoost Model
    /// </summary>
    /// <returns></returns>
    // ReSharper disable once UnusedMember.Global
    public static Dictionary<string, object> DefaultSearchSpace(int iterations)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //uncomment appropriate one
            ////for Regression Tasks: RMSE, etc.
            //{"loss_function", nameof(loss_function_enum.Logloss)},
            ////no need to set 'eval_metric', it will be the same as 'loss_function'
            ////for binary classification:
            //{"loss_function", nameof(loss_function_enum.Logloss)},
            //{"eval_metric", nameof(metric_enum.Accuracy)},
            ////for multi class classification:
            //{"loss_function", nameof(loss_function_enum.MultiClass)},
            //{"eval_metric", nameof(metric_enum.Accuracy)},

            { "logging_level", nameof(logging_level_enum.Verbose)},
            { "allow_writing_files",false},
            { "thread_count",1},
            { "iterations", iterations },
            //{ "od_type", "Iter"},
            //{ "od_wait",iterations/10},
            { "depth", AbstractHyperParameterSearchSpace.Range(2, 10) },
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.01f, 1.00f)},
            { "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
            { "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
            { "l2_leaf_reg",AbstractHyperParameterSearchSpace.Range(0, 10)},
            //{"grow_policy", new []{ "SymmetricTree", "Depthwise" /*, "Lossguide"*/}},
        };
        return searchSpace;
    }

}