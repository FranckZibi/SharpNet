﻿using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using SharpNet.HyperParameters;

namespace SharpNet.CatBoost
{
    [SuppressMessage("ReSharper", "UnusedMember.Global")]
    [SuppressMessage("ReSharper", "IdentifierTypo")]
    [SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
    public class CatBoostSample : AbstractSample, IMetricFunction
    {
        public CatBoostSample() :base(_categoricalHyperParameters)
        {
        }

        public static CatBoostSample ValueOf(string workingDirectory, string modelName)
        {
            return (CatBoostSample) ISample.LoadConfigIntoSample(() => new CatBoostSample(), workingDirectory, modelName);
        }

        public override void Save(string path)
        {
            var configContent = ToJsonConfigContent(DefaultAcceptForConfigContent);
            File.WriteAllText(path, configContent);
        }


        private static readonly HashSet<string> _categoricalHyperParameters = new()
        {
            "use_best_model", "approx_on_full_history", "langevin", "loss_function", "eval_metric", "sampling_frequency",  "grow_policy",  "boosting_type", "score_function", "task_type", "od_type"
        };
        public override bool PostBuild()
        {
            return true; //TODO
        }


        public string DeviceName()
        {
            if (task_type == task_type_enum.CPU)
            {
                return thread_count + "cpu";
            }
            else
            {
                return "gpu";
            }
        }

        #region Common parameters

        /// <summary>
        /// The metric to use in training. The specified value also determines the machine learning problem to solve.
        /// aliases: objective
        /// </summary>
        public enum loss_function_enum { RMSE, Logloss, MAE, CrossEntropy, Quantile, LogLinQuantile, Lq, MultiRMSE, MultiClass, MultiClassOneVsAll, MultiLogloss, MultiCrossEntropy, MAPE, Poisson, PairLogit, PairLogitPairwise, QueryRMSE, QuerySoftMax, Tweedie, YetiRank, YetiRankPairwise, StochasticFilter, StochasticRank, DEFAULT_VALUE }
        public loss_function_enum loss_function = loss_function_enum.DEFAULT_VALUE;

        public enum metric_enum { RMSE, Logloss, MAE, CrossEntropy, Quantile, LogLinQuantile, Lq, MultiClass, MultiClassOneVsAll, ultiLogloss, MultiCrossEntropy, MAPE, Poisson, PairLogit, PairLogitPairwise, QueryRMSE, QuerySoftMax, Tweedie, SMAPE, Recall, Precision, F1, TotalF1, Accuracy, BalancedAccuracy, BalancedErrorRate, Kappa, WKappa, LogLikelihoodOfPrediction, AUC, QueryAUC, R2, FairLoss, NumErrors, MCC, BrierScore, HingeLoss, HammingLoss, ZeroOneLoss, MSLE, edianAbsoluteError, Huber, Expectile, MultiRMSE, PairAccuracy, AverageGain, PFound, NDCG, DCG, FilteredDCG, NormalizedGini, PrecisionAt, RecallAt, MAP, CtrFactor, DEFAULT_VALUE }
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
        public enum sampling_frequency_enum { PerTree, PerTreeLevel, DEFAULT_VALUE }
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
        public enum grow_policy_enum { SymmetricTree, Depthwise, Lossguide, DEFAULT_VALUE }
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
        public enum boosting_type_enum { Ordered, Plain, DEFAULT_VALUE }
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
        public enum score_function_enum { Cosine, L2, NewtonCosine, NewtonL2, DEFAULT_VALUE }
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
        public string devices = DEFAULT_VALUE_STR;
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
        public enum od_type_enum {IncToDec, Iter, DEFAULT_VALUE}
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
        public enum logging_level_enum { Silent, Verbose, Info, Debug, DEFAULT_VALUE }
        public logging_level_enum logging_level = logging_level_enum.DEFAULT_VALUE;

        /// <summary>
        /// The logging level to output to stdout.
        /// Should be set to iterations/10
        /// </summary>
        public int verbose = DEFAULT_VALUE;

        /// <summary>
        /// The directory for storing the files generated during training.
        /// </summary>
        public string train_dir = DEFAULT_VALUE_STR;

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

        public MetricEnum GetMetric()
        {
            switch (eval_metric)
            {
                case metric_enum.Accuracy:
                    return MetricEnum.Accuracy;
                case metric_enum.RMSE:
                    return MetricEnum.Rmse;
                case metric_enum.Logloss:
                    return MetricEnum.Loss; //we'll use the same metric as the loss
                case metric_enum.AUC:
                    return MetricEnum.Accuracy; //AUC is not implemented : we use accuracy instead
                default:
                    return MetricEnum.Loss; //we'll use the same metric as the loss
            }
        }
        public override List<string> SampleFiles(string workingDirectory, string modelName)
        {
            return new List<string> { ISample.ToJsonPath(workingDirectory, modelName) };
        }
        public LossFunctionEnum GetLoss()
        {
            switch (loss_function)
            {
                case loss_function_enum.RMSE:
                    return LossFunctionEnum.Rmse;
                case loss_function_enum.Logloss:
                    return LossFunctionEnum.BinaryCrossentropy;
                default:
                case loss_function_enum.DEFAULT_VALUE:
                    throw new NotImplementedException($"can't manage metric {loss_function}");
            }
        }
    }
}