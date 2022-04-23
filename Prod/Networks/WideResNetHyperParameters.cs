using System.Collections.Generic;
using SharpNet.HyperParameters;

namespace SharpNet.Networks;

public class WideResNetHyperParameters : AbstractSample
{

    public WideResNetHyperParameters() : base(new HashSet<string>())
    {
    }
    
    #region Hyper-Parameters
    /// <summary>
    /// 0 to disable dropout
    /// any value > 0 will enable dropout
    /// </summary>
    public double WRN_DropOut;
    public double WRN_DropOutAfterDenseLayer;
    public NetworkSample.POOLING_BEFORE_DENSE_LAYER WRN_PoolingBeforeDenseLayer = NetworkSample.POOLING_BEFORE_DENSE_LAYER.AveragePooling_2;
    #endregion
}