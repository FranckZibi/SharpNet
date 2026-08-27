using SharpNet.Hyperparameters;

namespace SharpNet
{
    public enum Objective_enum
    {
        Regression,
        Classification
    }

    /// <summary>
    /// the loss function (= the objective function)
    /// the goal of the model will be to reduce the value returned by this loss function
    /// (lower is always better)
    /// </summary>
    public enum EvaluationMetricEnum
    {

        /// <summary>
        /// y true : a matrix of shape (batch_size, numClass) with, for each row the 'true' proba of each class
        /// y_predicted: a matrix of shape (batch_size, numClass) with, for each row the predicted proba of each class
        /// works only for metric (to rank submission), do not work as a loss function, higher s better
        /// </summary>
        Accuracy,

        /// <summary>
        /// y true : a sparse matrix of shape (batch_size, 1) with the index of the 'true' class
        /// y_predicted: a matrix of shape (batch_size, numClass) with, for each row the predicted proba of each class
        /// works only for metric (to rank submission), do not work as a loss function, higher s better
        /// </summary>
        SparseAccuracy,

        AccuracyCategoricalCrossentropyWithHierarchy,   // works only for metric (to rank submission), do not work as a loss function, higher s better

        /// <summary>
        /// To be used with sigmoid activation layer.
        /// In a single row, each value will be in [0,1] range
        /// Support of multi labels (one element can belong to several numClass at the same time)
        /// The expected Y value is a binary value: 0 or 1
        /// </summary>
        BinaryCrossentropy, // ok for loss, lower is better

        /// <summary>
        /// To be used with sigmoid activation layer.
        /// In a single row, each value will be in [0,1] range
        /// Support of multi labels (one element can belong to several numClass at the same time)
        /// The expected Y value is a binary value: 0 or 1
        /// </summary>
        BCEWithFocalLoss, // ok for loss, lower is better


        /// <summary>
        /// To be used with sigmoid activation layer.
        /// In a single row, each value will be in [0,1] range
        /// Support of multi labels (one element can belong to several numClass at the same time)
        /// The expected Y value is a continuous value in [0, 1] range (not a binary value: 0 or 1)
        /// </summary>
        BCEContinuousY, // ok for loss, lower is better

        /// <summary>
        /// To be used with softmax activation layer.
        /// In a single row, each value will be in [0,1] range, and the sum of all values wil be equal to 1.0 (= 100%)
        /// Do not support multi labels (each element can belong to exactly 1 category)
        /// </summary>
        CategoricalCrossentropy, // ok for loss, lower is better


        /* Hierarchical Category:
                              Object
                          /           \
                         /             \
                        /               \
                     Fruit             Flower
                      75%                25%
                   /   |   \            |    \
             Cherry  Apple  Orange    Rose    Tulip 
              70%     20%    10%      50%      50%
                     /   \            
                   Fuji  Golden
                    15%   85%
        */
        /// <summary>
        /// To be used with SoftmaxWithHierarchy activation layer.
        /// Each category (parent node) can be divided into several sub categories (children nodes)
        /// For any parent node: all children will have a proba in [0,1] range, and the sum of all children proba will be equal to 1.0 (= 100%)
        /// </summary>
        CategoricalCrossentropyWithHierarchy, // ok for loss, lower is better

        /*
         * Huber loss, see  https://en.wikipedia.org/wiki/Huber_loss
         * */
        Huber, // ok for loss, lower is better

        /*
        * Mean Squared Error loss, see https://en.wikipedia.org/wiki/Mean_squared_error
        * loss = ( predicted - expected ) ^2
        * */
        Mse, // ok for loss, lower is better

        /*
        * Mean Squared Error of log loss,
        * loss = ( log( max(predicted,epsilon) ) - log(expected) ) ^2
        * */
        MseOfLog, // ok for loss, lower is better

        /*
        * Mean Absolute Error loss, see https://en.wikipedia.org/wiki/Mean_absolute_error
        * loss = abs( predicted - expected )
        * */
        Mae, // ok for loss, lower is better

        /*
         * RootMean Squared Error loss, see https://en.wikipedia.org/wiki/Mean_squared_error
         * loss = ( predicted - expected ) ^2
         * */
        Rmse, // ok for loss, lower is better

        F1Micro, // ok for loss, higher is better

        PearsonCorrelation, // works only for metric (to rank submission), do not work as a loss function, higher s better
        SpearmanCorrelation, // works only for metric (to rank submission), do not work as a loss function, higher s better

        //Mean Squared Log Error, see: https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error
        //loss = (log(1+predicted) - log(1+expected)) ^ 2
        MeanSquaredLogError, // ok for loss, lower is better

        /// <summary>
        /// To be used with softmax activation layer.
        /// For the prediction:
        ///     In a single row, each value will be in [0,1] range, and the sum of all values wil be equal to 1.0 (= 100%)
        /// For the y_true:
        ///     In a single row, each value will be a scalar integer in the range [0, number_of_categories-1]
        /// Do not support multi labels (each element can belong to exactly 1 category)
        /// </summary>
        SparseCategoricalCrossentropy, // ok for loss, lower is better


        //Area Under the Curve, see: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
        AUC, // works only for metric (to rank submission), do not work as a loss function, higher s better

        //Average Precision Score, see : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
        AveragePrecisionScore, // works only for metric (to rank submission), do not work as a loss function, higher s better


        DEFAULT_VALUE = AbstractSample.DEFAULT_VALUE, // default value, do not use
    }
}
