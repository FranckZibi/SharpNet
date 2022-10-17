using System.Collections.Generic;

namespace SharpNet.HyperParameters;

public class PredictionsSample : AbstractSample
{
    public PredictionsSample() : base(new HashSet<string>())
    {
    }

    #region Hyper-Parameters
    public string Train_PredictionsFileName;
    public string Validation_PredictionsFileName;
    public string Test_PredictionsFileName;
    public string Train_PredictionsFileName_InModelFormat;
    public string Validation_PredictionsFileName_InModelFormat;
    public string Test_PredictionsFileName_InModelFormat;
    #endregion
}
