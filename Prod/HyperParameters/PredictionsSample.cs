namespace SharpNet.Hyperparameters;

public class PredictionsSample : AbstractSample
{
    // ReSharper disable once EmptyConstructor
    public PredictionsSample()
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
