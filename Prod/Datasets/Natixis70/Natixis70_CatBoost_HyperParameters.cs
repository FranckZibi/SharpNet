using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.HyperParameters;

namespace SharpNet.Datasets.Natixis70;

public class Natixis70_CatBoost_HyperParameters : MultiSamples
{
    public Natixis70_CatBoost_HyperParameters() : this(new ISample[] { new CatBoostSample(), new Natixis70DatasetHyperParameters() }) { }
    public Natixis70_CatBoost_HyperParameters(ISample[] samples) : base(samples) { }

    public CatBoostSample CatBoostSample => (CatBoostSample)Samples[0];
    public Natixis70DatasetHyperParameters DatasetHyperParameters => (Natixis70DatasetHyperParameters)Samples[1];

    public static Natixis70_CatBoost_HyperParameters ValueOf(string workingDirectory, string modelName)
    {
        return new Natixis70_CatBoost_HyperParameters(new ISample[]
        {
            CatBoostSample.ValueOf(workingDirectory, modelName+"_0"),
            Natixis70DatasetHyperParameters.ValueOf(workingDirectory, modelName+"_1")
        });
    }

    public override CpuTensor<float> Y_Train_dataset_to_Perfect_Predictions(string y_train_dataset)
    {
        //TODO
        return DatasetHyperParameters.LightGBM_2_ExpectedPredictionFormat(y_train_dataset);
    }

    public override bool PostBuild()
    {
        if (!base.PostBuild())
        {
            return false;
        }
        return true;
    }
}