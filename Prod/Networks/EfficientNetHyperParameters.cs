using System.Collections.Generic;
using SharpNet.GPU;
using SharpNet.HyperParameters;

namespace SharpNet.Networks;

public class EfficientNetHyperParameters : AbstractSample
{
    public EfficientNetHyperParameters() : base(new HashSet<string>())
    {
    }

    #region Hyper-Parameters
    public cudnnActivationMode_t DefaultActivation = cudnnActivationMode_t.CUDNN_ACTIVATION_SWISH;
    public double BatchNormMomentum = 0.99;
    public double BatchNormEpsilon = 0.001;
    /// <summary>
    /// name of the trained network to load the weights from.
    /// used for transfer learning
    /// </summary>
    public string WeightForTransferLearning = "";
    #endregion
}