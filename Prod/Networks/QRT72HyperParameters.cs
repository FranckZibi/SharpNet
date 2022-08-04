using System.Collections.Generic;
using SharpNet.Data;
using SharpNet.HyperParameters;

namespace SharpNet.Networks;

public class QRT72HyperParameters : AbstractSample
{
    public QRT72HyperParameters() : base(new HashSet<string>())
    {
    }

    #region Hyper-Parameters

    public int NumEpochs = 100;
    public int BatchSize = Tensor.CosineSimilarity504_TimeSeries_Length * 25;
    //public double InitialLearningRate = 100000.0;
    public double InitialLearningRate = 1.0;
    public double PercentageInTraining = 0.5;
    public double lambdaL2Regularization_matrix = 0;
    public double lambdaL2Regularization_vector = 0;
    public bool RandomizeOrder = true;
    //public double SGD_momentum = 0.9;
    //public bool SGD_useNesterov = false;
    #endregion
}
