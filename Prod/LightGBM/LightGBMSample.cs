// ReSharper disable UnusedMember.Global
// ReSharper disable InconsistentNaming
// ReSharper disable IdentifierTypo
// ReSharper disable CommentTypo

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using SharpNet.Datasets;
using SharpNet.HPO;
using SharpNet.HyperParameters;

namespace SharpNet.LightGBM;

[SuppressMessage("ReSharper", "MemberCanBePrivate.Global")]
public class LightGBMSample : AbstractSample, IModelSample
{
    public LightGBMSample() :base(_categoricalHyperParameters)
    {
    }
    public EvaluationMetricEnum GetLoss()
    {
        switch (objective)
        {
            case objective_enum.regression:
                return EvaluationMetricEnum.Rmse;
            case objective_enum.binary:
                return EvaluationMetricEnum.BinaryCrossentropy;
            case objective_enum.multiclass:
            case objective_enum.cross_entropy: //TODO to check
                return EvaluationMetricEnum.CategoricalCrossentropy; 
            default:
                throw new NotImplementedException($"can't manage metric {objective}");
        }
    }
    public override bool FixErrors()
    {
        if (boosting == boosting_enum.rf)
        {
            if (bagging_freq <= 0 || bagging_fraction >= 1.0f || bagging_fraction <= 0.0f)
            {
                return false;
            }
        }
        if (boosting != boosting_enum.dart)
        {
            drop_rate = DEFAULT_VALUE;
            max_drop = DEFAULT_VALUE;
            skip_drop = DEFAULT_VALUE;
            xgboost_dart_mode = false;
            uniform_drop = false;
            drop_seed = DEFAULT_VALUE;
        }

        if (path_smooth > 0 && min_data_in_leaf<2)
        {
            min_data_in_leaf = 2;
        }


        if (objective == objective_enum.multiclass || objective == objective_enum.multiclassova)
        {
            if (num_class < 2)
            {
                return false;
            }
        }
        else
        {
            //no need of 'num_class' field
            num_class = DEFAULT_VALUE;
        }

        if (bagging_freq <= 0)
        {
            //bagging is disabled
            //bagging_fraction must be equal to 1.0 (100%)
            if (bagging_fraction < 1)
            {
                return false;
            }
        }
        else
        {
            //bagging is enabled
            //bagging_fraction must be stricly less then 1.0 (100%)
            if (bagging_fraction >= 1)
            {
                return false;
            }
        }
        return true;
    }

    public void Use_All_Available_Cores()
    {
        num_threads = Utils.CoreCount;
    }

    public string DeviceName()
    {

        if (device_type == device_type_enum.cpu)
        {
            return num_threads + "cpu";
        }
        else
        {
            return "gpu";
        }
    }

    public void UpdateForDataset(DataSet dataset)
    {
        var categoricalFeatures = dataset.CategoricalFeatures;
        categorical_feature = (categoricalFeatures.Length >= 1) ? ("name:" + string.Join(',', categoricalFeatures)) : "";
    }


    #region Core Parameters

    #region CLI specific

    /// <summary>
    /// path of config file
    /// aliases: config_file
    /// </summary>
    public string config;

    /// <summary>
    ///  train: for training, aliases: training
    ///  predict: for prediction, aliases: prediction, test
    ///  convert_model: for converting model file into if-else format, see more information in Convert Parameters
    ///  refit: for refitting existing models with new data, aliases: refit_tree
    ///         save_binary, load train(and validation) data then save dataset to binary file.Typical usage: save_binary first, then run multiple train tasks in parallel using the saved binary file
    /// aliases: task_type
    /// </summary>
    public enum task_enum { train, predict, convert_model, refit, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } ;
    // ReSharper disable once MemberCanBePrivate.Global
    public task_enum task = task_enum.DEFAULT_VALUE;

    //path of training data, LightGBM will train from this data
    //aliases: train, train_data, train_data_file, data_filename
    public string data;

    //path(s) of validation/test data, LightGBM will output metrics for these data
    //support multiple validation data, separated by ,
    //aliases: test, valid_data, valid_data_file, test_data, test_data_file, valid_filenames
    public string valid;
    #endregion


   

    public enum objective_enum { regression, regression_l1, huber, fair, poisson, quantile, mape, gamma, tweedie, binary, multiclass, multiclassova, cross_entropy, cross_entropy_lambda, lambdarank, rank_xendcg, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } 
    //aliases: objective_type, app, application, loss
    // ReSharper disable once MemberCanBePrivate.Global
    public objective_enum objective = objective_enum.DEFAULT_VALUE;

    /// <summary>
    /// true if we face a classification problem
    /// </summary>
    public bool IsClassification => 
        objective == objective_enum.binary
        || objective == objective_enum.multiclass
        || objective == objective_enum.multiclassova;

    public enum boosting_enum { gbdt, rf, dart, goss, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE } 
    //gbdt:  traditional Gradient Boosting Decision Tree, aliases: gbrt
    //rf:    Random Forest, aliases: random_forest
    //dart:  Dropouts meet Multiple Additive Regression Trees
    //goss:  Gradient-based One-Side Sampling
    //Note: internally, LightGBM uses gbdt mode for the first 1 / learning_rate iterations
    public boosting_enum boosting = boosting_enum.DEFAULT_VALUE;

    //number of boosting iterations
    //Note: internally, LightGBM constructs num_class* num_boost_round trees for multi-class classification problems
    //aliases: num_iteration, n_iter, num_tree, num_trees, num_round, num_rounds, nrounds, num_boost_round, n_estimators, max_iter
    //constraints: num_iterations >= 0
    public int num_iterations = DEFAULT_VALUE;

    //shrinkage rate
    //in dart, it also affects on normalization weights of dropped trees
    //aliases: shrinkage_rate, eta, 
    //constraints: learning_rate > 0.0
    public double learning_rate = DEFAULT_VALUE;

    //max number of leaves in one tree
    //aliases: num_leaf, max_leaves, max_leaf, max_leaf_nodes
    //constraints: 1 < num_leaves <= 131072
    //default: 31
    public int num_leaves = DEFAULT_VALUE;

    //used only in train, prediction and refit tasks or in correspondent functions of language-specific packages
    //number of threads for LightGBM
    //0 means default number of threads in OpenMP
    //for the best speed, set this to the number of real CPU cores, not the number of threads (most CPUs use hyper-threading to generate 2 threads per CPU core)
    //do not set it too large if your dataset is small (for instance, do not use 64 threads for a dataset with 10,000 rows)
    //be aware a task manager or any similar CPU monitoring tool might report that cores not being fully utilized.
    //This is normal for distributed learning, do not use all CPU cores because this will cause poor performance for the network communication
    //    Note: please don’t change this during training, especially when running multiple jobs simultaneously by external packages, otherwise it may cause undesirable errors
    //aliases: num_thread, nthread, nthreads, n_jobs
    public int num_threads = DEFAULT_VALUE;

    //device for the tree learning, you can use GPU to achieve the faster learning
    //Note: it is recommended to use the smaller max_bin(e.g. 63) to get the better speed up
    //Note: for the faster speed, GPU uses 32-bit float point to sum up by default, so this may affect the accuracy for some tasks.You can set gpu_use_dp= true to enable 64-bit float point, but it will slow down the training
    //Note: refer to Installation Guide to build LightGBM with GPU support
    //aliases: device
    public enum device_type_enum {cpu,gpu, DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE}
    public device_type_enum device_type = device_type_enum.DEFAULT_VALUE;


    //this seed is used to generate other seeds, e.g. data_random_seed, feature_fraction_seed, etc.
    //by default, this seed is unused in favor of default values of other seeds
    //this seed has lower priority in comparison with other seeds, which means that it will be overridden, if you set other seeds explicitly
    //aliases: random_seed, seed 
    public int random_state = DEFAULT_VALUE;

    //used only with cpu device type
    //setting this to true should ensure the stable results when using the same data and the same parameters (and different num_threads)
    //when you use the different seeds, different LightGBM versions, the binaries compiled by different compilers, or in different systems, the results are expected to be different
    //you can raise issues in LightGBM GitHub repo when you meet the unstable results
    //Note: setting this to true may slow down the training
    //Note: to avoid potential instability due to numerical issues, please set force_col_wise=true or force_row_wise=true when setting deterministic=true
    public bool deterministic = false;

    //public enum tree_learner_enum { serial, feature, data, voting }
    ////serial:   single machine tree learner
    ////feature:  feature parallel tree learner, aliases: feature_parallel
    ////data:     data parallel tree learner, aliases: data_parallel
    ////voting:   voting parallel tree learner, aliases: voting_parallel
    ////refer to Distributed Learning Guide to get more details
    ////aliases: tree, tree_type, tree_learner_type
    //public tree_learner_enum tree_learner = tree_learner_enum.serial;

    #endregion

    #region Learning Control Parameters

    #region CLI specific
    //filename of input model
    //for prediction task, this model will be applied to prediction data
    //for train task, training will be continued from this model
    //aliases: model_input, model_in
    public string input_model;

    //filename of output model in training
    //aliases: model_output, model_out
    public string output_model;

    //the feature importance type in the saved model file
    //0: count-based feature importance (numbers of splits are counted);
    //1: gain-based feature importance (values of gain are counted)
    public int saved_feature_importance_type = DEFAULT_VALUE;

    //frequency of saving model file snapshot
    //set this to positive value to enable this function. For example, the model file will be snapshotted at each iteration if snapshot_freq=1
    //aliases: save_period
    public int snapshot_freq = DEFAULT_VALUE;
    #endregion

    //used only with cpu device type
    //set this to true to force col-wise histogram building
    //enabling this is recommended when:
    //  the number of columns is large, or the total number of bins is large
    //  num_threads is large, e.g. > 20
    //  you want to reduce memory cost
    //Note: when both force_col_wise and force_row_wise are false, LightGBM will firstly try them both,
    //and then use the faster one.
    //To remove the overhead of testing set the faster one to true manually
    //Note: this parameter cannot be used at the same time with force_row_wise, choose only one of them
    public bool force_col_wise = false;

    //used only with cpu device type
    //set this to true to force row-wise histogram building
    //enabling this is recommended when:
    //  the number of data points is large, and the total number of bins is relatively small
    //  num_threads is relatively small, e.g. <= 16
    //  you want to use small bagging_fraction or goss boosting to speed up
    //Note: setting this to true will double the memory cost for Dataset object.
    //If you have not enough memory, you can try setting force_col_wise=true
    //Note: when both force_col_wise and force_row_wise are false, LightGBM will firstly try them both,
    //and then use the faster one.
    //To remove the overhead of testing set the faster one to true manually
    //Note: this parameter cannot be used at the same time with force_col_wise, choose only one of them
    public bool force_row_wise = false;

    //max cache size in MB for historical histogram
    //< 0 means no limit
    public double histogram_pool_size = DEFAULT_VALUE;

    //limit the max depth for tree model.
    //This is used to deal with over-fitting when #data is small.
    //Tree still grows leaf-wise
    //<= 0 means no limit
    public int max_depth = DEFAULT_VALUE;

    //minimal number of data in one leaf. Can be used to deal with over-fitting
    //aliases: min_data_per_leaf, min_data, min_child_samples, min_samples_leaf, constraints: min_data_in_leaf >= 0
    //Note: this is an approximation based on the Hessian, so occasionally you may observe splits
    //      which produce leaf nodes that have less than this many observations
    // default: 20
    public int min_data_in_leaf = DEFAULT_VALUE;

    // minimal sum hessian in one leaf.Like min_data_in_leaf, it can be used to deal with over-fitting
    // aliases:     min_sum_hessian_per_leaf, min_sum_hessian, min_hessian, min_child_weight,
    // constraints: min_sum_hessian_in_leaf >= 0.0
    public double min_sum_hessian_in_leaf = DEFAULT_VALUE;

    //like feature_fraction, but this will randomly select part of data without resampling
    //can be used to speed up training
    //can be used to deal with over-fitting
    //Note: to enable bagging, bagging_freq should be set to a non zero value as well
    // aliases: sub_row, subsample, bagging
    // constraints: 0.0 < bagging_fraction <= 1.0
    public double bagging_fraction = DEFAULT_VALUE;

    // aliases: pos_sub_row, pos_subsample, pos_bagging
    // constraints: 0.0 < pos_bagging_fraction <= 1.0
    //used only in binary application
    //used for imbalanced binary classification problem, will randomly sample #pos_samples * pos_bagging_fraction positive samples in bagging
    //should be used together with neg_bagging_fraction
    //set this to 1.0 to disable
    //Note: to enable this, you need to set bagging_freq and neg_bagging_fraction as well
    //Note: if both pos_bagging_fraction and neg_bagging_fraction are set to 1.0, balanced bagging is disabled
    //Note: if balanced bagging is enabled, bagging_fraction will be ignored
    public double pos_bagging_fraction = DEFAULT_VALUE;

    //used only in binary application
    //used for imbalanced binary classification problem, will randomly sample #neg_samples * neg_bagging_fraction negative samples in bagging
    //should be used together with pos_bagging_fraction
    //set this to 1.0 to disable
    //Note: to enable this, you need to set bagging_freq and pos_bagging_fraction as well
    //Note: if both pos_bagging_fraction and neg_bagging_fraction are set to 1.0, balanced bagging is disabled
    //Note: if balanced bagging is enabled, bagging_fraction will be ignored
    // aliases: neg_sub_row, neg_subsample, neg_bagging, constraints: 0.0 < neg_bagging_fraction <= 1.0
    public double neg_bagging_fraction = DEFAULT_VALUE;

    //frequency for bagging
    //0 means disable bagging; k means perform bagging at every k iteration.
    //Every k-th iteration, LightGBM will randomly select bagging_fraction * 100 % of the data to use for the next k iterations
    //Note: to enable bagging, bagging_fraction should be set to value smaller than 1.0 as well
    //aliases: subsample_freq
    public int bagging_freq = DEFAULT_VALUE;

    //random seed for bagging
    //aliases: bagging_fraction_seed
    public int bagging_seed = DEFAULT_VALUE;

    //LightGBM will randomly select a subset of features on each iteration (tree) if feature_fraction is smaller than 1.0.
    //For example, if you set it to 0.8, LightGBM will select 80% of features before training each tree
    //can be used to speed up training
    //can be used to deal with over-fitting
    //aliases:      sub_feature, feature_fraction 
    //constraints:  0.0 < feature_fraction <= 1.0
    public double colsample_bytree= DEFAULT_VALUE;

    //LightGBM will randomly select a subset of features on each tree node if feature_fraction_bynode is smaller than 1.0.
    //For example, if you set it to 0.8, LightGBM will select 80% of features at each tree node
    //can be used to deal with over-fitting
    //Note: unlike feature_fraction, this cannot speed up training
    //Note: if both feature_fraction and feature_fraction_bynode are smaller than 1.0, the final fraction of each node is feature_fraction * feature_fraction_bynode
    // aliases: sub_feature_bynode, feature_fraction_bynode  
    // constraints: 0.0 < feature_fraction_bynode <= 1.0
    public double colsample_bynode = DEFAULT_VALUE;

    //random seed for feature_fraction
    public int feature_fraction_seed = DEFAULT_VALUE;

    //use extremely randomized trees
    //if set to true, when evaluating node splits LightGBM will check only one randomly-chosen threshold for each feature
    //can be used to speed up training
    //can be used to deal with over-fitting
    // aliases: extra_tree
    public bool extra_trees = false;

    //random seed for selecting thresholds when extra_trees is true
    public int extra_seed = DEFAULT_VALUE;

    //will stop training if one metric of one validation data doesn’t improve in last early_stopping_round rounds
    //<= 0 means disable
    //can be used to speed up training
    // aliases: early_stopping_rounds, early_stopping, n_iter_no_change
    public int early_stopping_round = DEFAULT_VALUE;

    //LightGBM allows you to provide multiple evaluation metrics.
    //Set this to true, if you want to use only the first metric for early stopping
    public bool first_metric_only = false;

    //used to limit the max output of tree leaves
    //<= 0 means no constraint
    //the final max output of leaves is learning_rate * max_delta_step
    //aliases: max_tree_output, max_leaf_output
    public double max_delta_step = DEFAULT_VALUE;

    // L1 regularization
    // aliases: reg_alpha, l1_regularization
    // constraints: lambda_l1 >= 0.0
    public double lambda_l1 = DEFAULT_VALUE;

    //L2 regularization
    // aliases: reg_lambda, lambda, l2_regularization
    // constraints: lambda_l2 >= 0.0
    public double lambda_l2 = DEFAULT_VALUE;


    //linear tree regularization, corresponds to the parameter lambda in Eq. 3 of Gradient Boosting with Piece-Wise Linear Regression Trees
    //constraints: linear_lambda >= 0.0
    public double linear_lambda = DEFAULT_VALUE;

    //the minimal gain to perform split
    //can be used to speed up training
    // aliases: min_split_gain
    // constraints: min_gain_to_split >= 0.0
    public double min_gain_to_split = DEFAULT_VALUE;

    #region used only in dart
    // dropout rate: a fraction of previous trees to drop during the dropout
    // used only in dart
    // aliases: rate_drop
    // constraints: 0.0 <= drop_rate <= 1.0
    // default: 0.1
    public double drop_rate = DEFAULT_VALUE;

    //max number of dropped trees during one boosting iteration
    //<=0 means no limit
    // default: 50
    public int max_drop = DEFAULT_VALUE;

    //probability of skipping the dropout procedure during a boosting iteration
    //constraints: 0.0 <= skip_drop <= 1.0 
    // default: 0.50
    public double skip_drop = DEFAULT_VALUE;

    //set this to true, if you want to use xgboost dart mode
    public bool xgboost_dart_mode = false;

    //set this to true, if you want to use uniform drop
    public bool uniform_drop = false;

    //random seed to choose dropping models
    public int drop_seed = DEFAULT_VALUE;
    #endregion


    //used only in goss
    //the retain ratio of large gradient data
    //constraints: 0.0 <= top_rate <= 1.0
    public double top_rate = DEFAULT_VALUE;

    // used only in goss
    // the retain ratio of small gradient data
    // constraints: 0.0 <= other_rate <= 1.0
    public double other_rate = DEFAULT_VALUE;

    // minimal number of data per categorical group
    // constraints: min_data_per_group > 0
    public int min_data_per_group = DEFAULT_VALUE;

    //used for the categorical features
    //limit number of split points considered for categorical features. See the documentation on how LightGBM finds optimal splits for categorical features for more details
    //can be used to speed up training
    // constraints: max_cat_threshold > 0
    public int max_cat_threshold = DEFAULT_VALUE;

    // used for the categorical features
    // L2 regularization in categorical split
    // constraints: cat_l2 >= 0.0
    public double cat_l2 = DEFAULT_VALUE;

    // used for the categorical features
    // this can reduce the effect of noises in categorical features, especially for categories with few data
    // constraints: cat_smooth >= 0.0
    public double cat_smooth = DEFAULT_VALUE;

    //when number of categories of one feature smaller than or equal to max_cat_to_onehot, 
    //one-vs-other split algorithm will be used
    //constraints: max_cat_to_onehot > 0
    public int max_cat_to_onehot = DEFAULT_VALUE;

    //// used only in voting tree learner, refer to Voting parallel
    //// set this to larger value for more accurate result, but it will slow down the training speed
    //// aliases: topk
    //// constraints: top_k > 0
    //public int top_k = DEFAULT_VALUE;

    ////used for constraints of monotonic features
    ////1 means increasing, -1 means decreasing, 0 means non-constraint
    ////you need to specify all features in order. For example, mc=-1,0,1 means decreasing for 1st feature, non-constraint for 2nd feature and increasing for the 3rd feature
    ////aliases: mc, monotone_constraint, monotonic_cst
    //public int[] monotone_constraints = null;

    //used only if monotone_constraints is set
    //monotone constraints method
    //basic, the most basic monotone constraints method. It does not slow the library at all, but over-constrains the predictions
    //intermediate, a more advanced method, which may slow the library very slightly. However, this method is much less constraining than the basic method and should significantly improve the results
    //advanced, an even more advanced method, which may slow the library. However, this method is even less constraining than the intermediate method and should again significantly improve the results
    // options: basic, intermediate, advanced, aliases: monotone_constraining_method, mc_method
    //enum monotone_constraints_method = basic; 

    ////used only if monotone_constraints is set
    ////monotone penalty: a penalization parameter X forbids any monotone splits on the first X (rounded down) level(s) of the tree. The penalty applied to monotone splits on a given depth is a continuous, increasing function the penalization parameter
    //// if 0.0 (the default), no penalization is applied
    //// aliases: monotone_splits_penalty, ms_penalty, mc_penalty, 
    //// constraints: monotone_penalty >= 0.0
    //public double monotone_penalty = DEFAULT_VALUE;

    ////used to control feature’s split gain, will use gain[i] = max(0, feature_contri[i]) * gain[i] to replace the split gain of i-th feature
    ////you need to specify all features in order
    ////aliases: feature_contrib, fc, fp, feature_penalty
    //public double[] feature_contri = null;

    ////path to a .json file that specifies splits to force at the top of every decision tree before best-first learning commences
    ////.json file can be arbitrarily nested, and each split contains feature, threshold fields, as well as left and right fields representing subsplits
    ////categorical splits are forced in a one-hot fashion, with left representing the split containing the feature value and right representing other values
    ////Note: the forced split logic will be ignored, if the split makes gain worse
    ////see this file as an example
    //// aliases: fs, forced_splits_filename, forced_splits_file, forced_splits
    //public string forcedsplits_filename;

    ////decay rate of refit task, will use leaf_output = refit_decay_rate * old_leaf_output + (1.0 - refit_decay_rate) * new_leaf_output to refit trees
    ////used only in refit task in CLI version or as argument in refit function in language-specific package
    ////constraints: 0.0 <= refit_decay_rate <= 1.0
    //public double refit_decay_rate = 0.9;

    #region cost-effective gradient boosting
    //cost-effective gradient boosting multiplier for all penalties
    //constraints: cegb_tradeoff >= 0.0
    public double cegb_tradeoff = DEFAULT_VALUE;

    //cost-effective gradient-boosting penalty for splitting a node
    //constraints: cegb_penalty_split >= 0.0
    public double cegb_penalty_split = DEFAULT_VALUE;

    //cost-effective gradient boosting penalty for using a feature
    //applied per data point
    //default = 0,0,...,0
    public double[] cegb_penalty_feature_lazy = null;

    //cost-effective gradient boosting penalty for using a feature
    //applied once per forest
    //default = 0,0,...,0
    public double[] cegb_penalty_feature_coupled = null;
    #endregion

    //controls smoothing applied to tree nodes
    //helps prevent overfitting on leaves with few samples
    //if set to zero, no smoothing is applied
    //if path_smooth > 0 then min_data_in_leaf must be at least 2
    //larger values give stronger regularization
    //the weight of each node is (n / path_smooth) * w + w_p / (n / path_smooth + 1), where
    //  n is the number of samples in the node,
    //  w is the optimal node weight to minimise the loss (approximately -sum_gradients / sum_hessians),
    //  w_p is the weight of the parent node
    //note that the parent output w_p itself has smoothing applied, unless it is the root node,
    //so that the smoothing effect accumulates with the tree depth
    //constraints: path_smooth >=  0.0
    public double path_smooth = DEFAULT_VALUE;

    ////controls which features can appear in the same branch
    ////by default interaction constraints are disabled, to enable them you can specify:
    ////  for CLI, lists separated by commas, e.g. [0,1,2],[2,3]
    ////  for Python-package, list of lists, e.g. [[0, 1, 2], [2, 3]]
    ////  for R-package, list of character or numeric vectors, e.g. list(c("var1", "var2", "var3"), c("var3", "var4")) or list(c(1L, 2L, 3L), c(3L, 4L)). Numeric vectors should use 1-based indexing, where 1L is the first feature, 2L is the second feature, etc
    ////any two features can only appear in the same branch only if there exists a constraint containing both features
    //public string interaction_constraints = "";

    //controls the level of LightGBM’s verbosity
    //< 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug
    //aliases: verbose
    public int verbosity = DEFAULT_VALUE;
    #endregion

    #region IO Parameters / Dataset Parameters

    #region CLI specific
    //if true, LightGBM will save the dataset (including validation data) to a binary file. This speed ups the data loading for the next time
    //Note: init_score is not saved in binary file
    //Note: can be used only in CLI version; for language-specific packages you can use the correspondent function
    //aliases: is_save_binary, is_save_binary_file
    public bool save_binary = false;
    #endregion

    //max number of bins that feature values will be bucketed in
    //small number of bins may reduce training accuracy but may increase general power (deal with over-fitting)
    //LightGBM will auto compress memory according to max_bin. 
    //For example, LightGBM will use uint8_t for feature value if max_bin=255
    // aliases: max_bins
    // constraints: max_bin > 1
    public int max_bin = DEFAULT_VALUE;

    //max number of bins for each feature
    //if not specified, will use max_bin for all features
    //public int[] max_bin_by_feature = null;

    //minimal number of data inside one bin
    //use this to avoid one-data-one-bin (potential over-fitting)
    //constraints: min_data_in_bin > 0
    public int min_data_in_bin = DEFAULT_VALUE;

    //number of data that sampled to construct feature discrete bins
    //setting this to larger value will give better training result, but may increase data loading time
    //set this to larger value if data is very sparse
    //Note: don’t set this to small values, otherwise, you may encounter unexpected errors and poor accuracy        //aliases: subsample_for_bin
    // constraints: bin_construct_sample_cnt > 0
    public int bin_construct_sample_cnt = DEFAULT_VALUE;

    //random seed for sampling data to construct histogram bins
    //aliases: data_seed
    public int data_random_seed = DEFAULT_VALUE;

    //used to enable/disable sparse optimization
    //aliases: is_sparse, enable_sparse, sparse
    public bool is_enable_sparse = true;

    //set this to false to disable Exclusive Feature Bundling (EFB), which is described in LightGBM: A Highly Efficient Gradient Boosting Decision Tree
    //Note: disabling this may cause the slow training speed for sparse datasets
    //aliases: is_enable_bundle, bundle
    public bool enable_bundle = true;

    // set this to false to disable the special handle of missing value
    public bool use_missing = true;

    //set this to true to treat all zero as missing values (including the unshown values in LibSVM / sparse matrices)
    //set this to false to use na for representing missing values
    public bool zero_as_missing = false;

    //set this to true (the default) to tell LightGBM to ignore the features that are unsplittable based on min_data_in_leaf
    //as dataset object is initialized only once and cannot be changed after that, you may need to set this to false when searching parameters with min_data_in_leaf, otherwise features are filtered by min_data_in_leaf firstly if you don’t reconstruct dataset object
    //Note: setting this to false may slow down the training
    public bool feature_pre_filter = true;

    //used for distributed learning (excluding the feature_parallel mode)
    //true if training data are pre-partitioned, and different machines use different partitions
    //aliases: is_pre_partition
    public bool pre_partition = false;

    //set this to true if data file is too big to fit in memory
    //by default, LightGBM will map data file to memory and load features from memory. This will provide faster data loading speed, but may cause run out of memory error when the data file is very big
    //Note: works only in case of loading data directly from text file
    //aliases: two_round_loading, use_two_round_loading
    public bool two_round = false;

    //set this to true if input data has header
    //Note: works only in case of loading data directly from text file
    // aliases: has_header
    public bool header = false;

    //used to specify the label column
    //use number for index, e.g. label=0 means column_0 is the label
    //add a prefix name: for column name, e.g. label=name:is_click
    //if omitted, the first column in the training data is used as the label
    //Note: works only in case of loading data directly from text file
    // aliases: label
    public string label_column;

    //used to specify the weight column
    //use number for index, e.g. weight=0 means column_0 is the weight
    //add a prefix name: for column name, e.g. weight=name:weight
    //Note: works only in case of loading data directly from text file
    //Note: index starts from 0 and it doesn’t count the label column when passing type is int, e.g. when label is column_0, and weight is column_1, the correct parameter is weight=0
    //aliases: weight
    public string weight_column;

    //used to specify the query/group id column
    //use number for index, e.g. query=0 means column_0 is the query id
    //add a prefix name: for column name, e.g. query=name:query_id
    //Note: works only in case of loading data directly from text file
    //Note: data should be grouped by query_id, for more information, see Query Data
    //Note: index starts from 0 and it doesn’t count the label column when passing type is int, e.g. when label is column_0 and query_id is column_1, the correct parameter is query=0
    //aliases: group, group_id, query_column, query, query_id
    public string group_column;

    //used to specify some ignoring columns in training
    //use number for index, e.g. ignore_column=0,1,2 means column_0, column_1 and column_2 will be ignored
    //add a prefix name: for column name, e.g. ignore_column=name:c1,c2,c3 means c1, c2 and c3 will be ignored
    //Note: works only in case of loading data directly from text file
    //Note: index starts from 0 and it doesn’t count the label column when passing type is int
    //Note: despite the fact that specified columns will be completely ignored during the training, they still should have a valid format allowing LightGBM to load file successfully
    //aliases: ignore_feature, blacklist
    public string ignore_column;

    //used to specify categorical features
    //use number for index, e.g. categorical_feature=0,1,2 means column_0, column_1 and column_2 are categorical features
    //add a prefix name: for column name, e.g. categorical_feature=name:c1,c2,c3 means c1, c2 and c3 are categorical features
    //Note: only supports categorical with int type (not applicable for data represented as pandas DataFrame in Python-package)
    //Note: index starts from 0 and it doesn’t count the label column when passing type is int
    //Note: all values should be less than Int32.MaxValue (2147483647)
    //Note: using large values could be memory consuming. Tree decision rule works best when categorical features are presented by consecutive integers starting from zero
    //Note: all negative values will be treated as missing values
    //Note: the output cannot be monotonically constrained with respect to a categorical feature
    //aliases: cat_feature, categorical_column, cat_column, categorical_features
    public string categorical_feature;

    ////path to a .json file that specifies bin upper bounds for some or all features
    ////.json file should contain an array of objects, each containing the word feature (integer feature index) and bin_upper_bound (array of thresholds for binning)
    ////see this file as an example
    //public string forcedbins_filename;

    ////use precise floating point number parsing for text parser (e.g. CSV, TSV, LibSVM input)
    ////Note: setting this to true may lead to much slower text parsing
    //public bool precise_float_parser = false;

    ////path to a .json file that specifies customized parser initialized configuration
    ////see lightgbm-transform for usage examples
    ////Note: lightgbm-transform is not maintained by LightGBM’s maintainers. 
    ////B.ug reports or feature requests should go to issues page
    //public string parser_config_file;

    //fit piecewise linear gradient boosting tree
    //tree splits are chosen in the usual way, but the model at each leaf is linear instead of constant
    //the linear model at each leaf includes all the numerical features in that leaf’s branch
    //categorical features are used for splits as normal but are not used in the linear models
    //missing values should not be encoded as 0. Use np.nan for Python, NA for the CLI, and NA, NA_real_, or NA_integer_ for R
    //it is recommended to rescale data before training so that features have similar mean and standard deviation
    //Note: only works with CPU and serial tree learner
    //Note: regression_l1 objective is not supported with linear tree boosting
    //Note: setting linear_tree=true significantly increases the memory use of LightGBM
    //Note: if you specify monotone_constraints, constraints will be enforced when choosing the split points, but not when fitting the linear models on leaves
    //aliases: linear_trees    
    //public bool linear_tree = false;

    #endregion

    #region Prediction tasks specific

    //used to specify from which iteration to start the prediction
    //<= 0 means from the first iteration    
    public int start_iteration_predict = DEFAULT_VALUE;

    //used to specify how many trained iterations will be used in prediction
    //<= 0 means no limit
    public int num_iteration_predict = DEFAULT_VALUE;

    //set this to true to predict only the raw scores
    //set this to false to predict transformed scores
    //aliases: is_predict_raw_score, predict_rawscore, raw_score
    public bool predict_raw_score = false;

    //set this to true to predict with leaf index of all trees
    //aliases: is_predict_leaf_index, leaf_index
    public bool predict_leaf_index = false;

    //set this to true to estimate SHAP values, which represent how each feature contributes to each prediction
    //produces #features + 1 values where the last value is the expected value of the model output over the training data
    //Note: if you want to get more explanation for your model’s predictions using SHAP values like SHAP interaction values, you can install shap package
    //Note: unlike the shap package, with predict_contrib we return a matrix with an extra column, where the last column is the expected value
    //Note: this feature is not implemented for linear trees
    // aliases: is_predict_contrib, contrib
    public bool predict_contrib = false;

    //control whether or not LightGBM raises an error when you try to predict on data with a different number of features than the training data
    //if false (the default), a fatal error will be raised if the number of features in the dataset you predict on differs from the number seen during training
    //if true, LightGBM will attempt to predict on whatever data you provide. This is dangerous because you might get incorrect predictions, but you could use it in situations where it is difficult or expensive to generate some features and you are very confident that they were never chosen for splits in the model
    //Note: be very careful setting this parameter to true
    public bool predict_disable_shape_check = false;

    //used only in classification and ranking applications
    //used only for predicting normal or raw scores
    //if true, will use early-stopping to speed up the prediction. May affect the accuracy
    //Note: cannot be used with rf boosting type or custom objective function
    public bool pred_early_stop = false;

    //the frequency of checking early-stopping prediction
    public int pred_early_stop_freq = DEFAULT_VALUE;

    //the threshold of margin in early-stopping prediction
    public double pred_early_stop_margin = DEFAULT_VALUE;

    //filename of prediction result
    //Note: can be used only in CLI version
    //aliases: predict_result, output_result , predict_name, prediction_name, pred_name, name_pred
    public string prediction_result;
    #endregion

    //#region IO Parameters / Convert Parameters

    ////used only in convert_model task
    ////only cpp is supported yet; for conversion model to other languages consider using m2cgen utility
    ////if convert_model_language is set and task=train, the model will be also converted
    ////Note: can be used only in CLI version    
    //public string convert_model_language;

    ////used only in convert_model task
    ////output filename of converted model
    ////Note: can be used only in CLI version
    ////aliases: convert_model_file
    //public string convert_model;

    //#endregion

    #region Objective Parameters

    //used only in rank_xendcg objective
    //random seed for objectives, if random process is needed
    public int objective_seed = DEFAULT_VALUE;

    // used only in multi-class classification application
    // aliases: num_classes
    // constraints: num_class > 0
    public int num_class = DEFAULT_VALUE;

    //used only in binary and multiclassova applications
    //set this to true if training data are unbalanced
    //Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
    //Note: this parameter cannot be used at the same time with scale_pos_weight, choose only one of them    // aliases: unbalance, unbalanced_sets
    public bool is_unbalance = false;

    //used only in binary and multiclassova applications
    //weight of labels with positive class
    //Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
    //Note: this parameter cannot be used at the same time with is_unbalance, choose only one of them
    // constraints: scale_pos_weight > 0.0
    public double scale_pos_weight = DEFAULT_VALUE;

    //used only in binary and multiclassova classification and in lambdarank applications
    //parameter for the sigmoid function
    // constraints: sigmoid > 0.0
    public double sigmoid = DEFAULT_VALUE;

    //used only in regression, binary, multiclassova and cross-entropy applications
    //adjusts initial score to the mean of labels for faster convergence
    public bool boost_from_average = true;

    //used only in regression application
    //used to fit sqrt(label) instead of original values and prediction result will be also automatically converted to prediction^2
    //might be useful in case of large-range labels
    public bool reg_sqrt = false;

    // used only in huber and quantile regression applications
    // parameter for Huber loss and Quantile regression
    // constraints: alpha > 0.0
    public double alpha = DEFAULT_VALUE;

    //used only in fair regression application
    //parameter for Fair loss
    // constraints: fair_c > 0.0
    public double fair_c = DEFAULT_VALUE;

    //used only in poisson regression application
    //parameter for Poisson regression to safeguard optimization
    //constraints: poisson_max_delta_step > 0.0
    public double poisson_max_delta_step = DEFAULT_VALUE;

    //used only in tweedie regression application
    //used to control the variance of the tweedie distribution
    //set this closer to 2 to shift towards a Gamma distribution
    //set this closer to 1 to shift towards a Poisson distribution
    // constraints: 1.0 <= tweedie_variance_power < 2.0
    public double tweedie_variance_power = DEFAULT_VALUE;

    //used only in lambdarank application
    //controls the number of top-results to focus on during training, refer to “truncation level” in the Sec. 3 of LambdaMART paper
    //this parameter is closely related to the desirable cutoff k in the metric NDCG@k that we aim at optimizing the ranker for. The optimal setting for this parameter is likely to be slightly higher than k (e.g., k + 3) to include more pairs of documents to train on, 
    //but perhaps not too high to avoid deviating too much from the desired target metric NDCG@k
    // constraints: lambdarank_truncation_level > 0
    //public int lambdarank_truncation_level = DEFAULT_VALUE;

    //used only in lambdarank application
    //set this to true to normalize the lambdas for different queries, and improve the performance for unbalanced data
    //set this to false to enforce the original lambdarank algorithm
    //public bool lambdarank_norm = true;

    //used only in lambdarank application
    //relevant gain for labels. For example, the gain of label 2 is 3 in case of default label gains
    //separate by ,
    //default = 0,1,3,7,15,31,63,...,2^30-1
    //public double[] label_gain = null; 
    #endregion

    #region Metric Parameters

    //metric(s) to be evaluated on the evaluation set(s)
    // see: https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric
    //support multiple metrics, separated by ,
    //type = multi-enum
    //aliases: metrics, metric_types
    public string metric;

    #region CLI specific
    //frequency for metric output
    //aliases: output_freq
    //constraints: metric_freq > 0
    public int metric_freq = 1;

    //set this to true to output metric result over training dataset
    //aliases: training_metric, is_training_metric, train_metric
    public bool is_provide_training_metric = false;
    #endregion

    //used only with ndcg and map metrics
    //NDCG and MAP evaluation positions, separated by ,
    //aliases: ndcg_eval_at, ndcg_at, map_eval_at, map_at
    //public string eval_at; 

    //used only with multi_error metric
    //threshold for top-k multi-error metric
    //the error on each sample is 0 if the true class is among the top multi_error_top_k predictions, and 1 otherwise
    //more precisely, the error on a sample is 0 if there are at least num_classes - multi_error_top_k predictions strictly less than the prediction on the true class
    //when multi_error_top_k=1 this is equivalent to the usual multi-error metric
    // constraints: multi_error_top_k > 0
    public int multi_error_top_k = DEFAULT_VALUE;

    //used only with auc_mu metric
    //list representing flattened matrix (in row-major order) giving loss weights for classification errors
    //list should have n * n elements, where n is the number of classes
    //the matrix co-ordinate [i, j] should correspond to the i * n + j-th element of the list
    //if not specified, will use equal weights for all classes
    //public double[] auc_mu_weights = null;
    #endregion

    //#region Network Parameters
    ////the number of machines for distributed learning application
    ////this parameter is needed to be set in both socket and mpi versions
    //// aliases: num_machine, constraints: num_machines > 0
    //public int num_machines = DEFAULT_VALUE;

    ////TCP listen port for local machines
    ////Note: don’t forget to allow this port in firewall settings before training
    //// aliases: local_port, port, constraints: local_listen_port > 0
    //public int local_listen_port = DEFAULT_VALUE;

    //// socket time-out in minutes
    //// constraints: time_out > 0
    //public int time_out = DEFAULT_VALUE;

    ////path of file that lists machines for this distributed learning application
    ////each line contains one IP and one port for one machine. The format is ip port (space as a separator)
    ////Note: can be used only in CLI version
    //// aliases: machine_list_file, machine_list, mlist
    //public string machine_list_filename;

    ////list of machines in the following format: ip1:port1,ip2:port2
    //// aliases: workers, nodes
    //public string machines;
    //#endregion

    #region GPU Parameters

    //OpenCL platform ID. Usually each GPU vendor exposes one OpenCL platform
    //-1 means the system-wide default platform
    //Note: refer to GPU Targets for more details
    public int gpu_platform_id = DEFAULT_VALUE;

    //OpenCL device ID in the specified platform. Each GPU in the selected platform has a unique device ID
    //-1 means the default device in the selected platform
    //Note: refer to GPU Targets for more details    
    public int gpu_device_id = DEFAULT_VALUE;

    //set this to true to use double precision math on GPU (by default single precision is used)
    //Note: can be used only in OpenCL implementation, in CUDA implementation only double precision is currently supported
    public bool gpu_use_dp = false;

    //number of GPUs
    //Note: can be used only in CUDA implementation
    // constraints: num_gpu > 0
    public int num_gpu = DEFAULT_VALUE;

    #endregion

    /// <summary>
    /// The default Search Space for CatBoost Model
    /// </summary>
    /// <returns></returns>
    // ReSharper disable once UnusedMember.Global
    public static Dictionary<string, object> DefaultSearchSpace(int num_iterations)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //uncomment appropriate one
            //{"objective", "regression"},      //for Regression Tasks
            //{"objective", "binary"},          //for binary classification
            //{"objective", "multiclass"},      //for multi class classification
            //{"num_class", number_of_class },  //for multi class classification

            //high priority
            { "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
            { "bagging_freq", new[]{0, 1} },
            { "boosting", new []{"gbdt", "dart"}},
            { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.3f, 1.0f)},
            { "early_stopping_round", num_iterations/10 },
            { "lambda_l1",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.005f, 0.2f)},
            { "max_depth", new[]{10, 20, 50, 100, 255} },
            { "min_data_in_leaf", new[]{20, 50 /*,100*/} },
            { "num_iterations", num_iterations },
            { "num_leaves", AbstractHyperParameterSearchSpace.Range(3, 50) },
            { "num_threads", 1},
            { "verbosity", "0" },

            //medium priority
            { "drop_rate", new[]{0.05, 0.1, 0.2}},                               //specific to dart mode
            { "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            { "min_data_in_bin", new[]{3, 10, 100, 150}  },
            { "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            { "max_drop", new[]{40, 50, 60}},                                   //specific to dart mode
            { "skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},  //specific to dart mode

            //low priority
            { "extra_trees", new[] { true , false } }, //low priority 
            //{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)}, //very low priority
            { "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) }, //low priority
            { "min_sum_hessian_in_leaf", AbstractHyperParameterSearchSpace.Range(1e-3f, 1.0f) },

        };

        return searchSpace;
    }



    private static readonly HashSet<string> _categoricalHyperParameters = new()
    {
        "saved_feature_importance_type",
        "verbosity",
        "task",
        "objective",
        "boosting",
        "device_type",
        "tree_learner",
    };
}