using System;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;

namespace SharpNet.Datasets.AmazonEmployeeAccessChallenge;

public class AmazonEmployeeAccessChallenge_CatBoost_HyperParameters : MultiSamples
{
    public AmazonEmployeeAccessChallenge_CatBoost_HyperParameters() : this(new ISample[] { new CatBoostSample(), new AmazonEmployeeAccessChallengeDatasetHyperParameters() }) { }
    private AmazonEmployeeAccessChallenge_CatBoost_HyperParameters(ISample[] samples) : base(samples) { }

    public CatBoostSample CatBoostSample => (CatBoostSample)Samples[0];
    public AmazonEmployeeAccessChallengeDatasetHyperParameters DatasetHyperParameters => (AmazonEmployeeAccessChallengeDatasetHyperParameters)Samples[1];

    public static AmazonEmployeeAccessChallenge_CatBoost_HyperParameters ValueOf(string workingDirectory, string modelName)
    {
        return new AmazonEmployeeAccessChallenge_CatBoost_HyperParameters(new ISample[]
        {
            LightGBMSample.ValueOf(workingDirectory, modelName+"_0"),
            AmazonEmployeeAccessChallengeDatasetHyperParameters.ValueOf(workingDirectory, modelName+"_1")
        });
    }

    public override CpuTensor<float> Y_Train_dataset_to_Perfect_Predictions(string y_train_dataset)
    {
        throw new NotImplementedException();
        //return DatasetHyperParameters.LightGBM_2_ExpectedPredictionFormat(y_train_dataset);
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
