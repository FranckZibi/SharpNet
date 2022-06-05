using System.Collections.Generic;

namespace SharpNet.HyperParameters;

public class PredictionsSample : AbstractSample
{
    public PredictionsSample() : base(new HashSet<string>())
    {
    }

    #region Hyper-Parameters
    public string Train_PredictionsPath;
    public string Validation_PredictionsPath;
    public string Test_PredictionsPath;
    #endregion
}