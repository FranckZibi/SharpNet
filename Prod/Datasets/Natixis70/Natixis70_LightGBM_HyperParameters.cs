using System.Collections.Generic;
using System.IO;
using SharpNet.CPU;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;

// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.Datasets.Natixis70
{
    /// <summary>
    /// Network support for Natixis70 challenge
    /// </summary>
    public class Natixis70_LightGBM_HyperParameters : MultiSamples
    {
        public Natixis70_LightGBM_HyperParameters() : this(new ISample[]{new LightGBMSample(), new Natixis70DatasetHyperParameters()}) { }
        public Natixis70_LightGBM_HyperParameters(ISample[] samples) : base(samples) { }

        public LightGBMSample LightGbmSample => (LightGBMSample)Samples[0];
        public Natixis70DatasetHyperParameters DatasetHyperParameters => (Natixis70DatasetHyperParameters)Samples[1];

        public static Natixis70_LightGBM_HyperParameters ValueOf(string workingDirectory, string modelName)
        {
            //TODO TO REMOVE : old way of storing all configs in same file
            if (File.Exists(ISample.ToPath(workingDirectory, modelName)))
            {
                var content = ISample.LoadConfig(workingDirectory, modelName);
                var res = new Natixis70_LightGBM_HyperParameters();
                res.Set(Utils.FromString2String_to_String2Object(content));
                return res;
            }

            return new Natixis70_LightGBM_HyperParameters(new ISample[]
            {
                LightGBMSample.ValueOf(workingDirectory, modelName+"_0"),
                Natixis70DatasetHyperParameters.ValueOf(workingDirectory, modelName+"_1")
            });
        }

        public override CpuTensor<float> Y_Train_dataset_to_Perfect_Predictions(string y_train_dataset)
        {
            return DatasetHyperParameters.LightGBM_2_ExpectedPredictionFormat(y_train_dataset);
        }

        public override bool PostBuild()
        {
            if (!base.PostBuild())
            {
                return false;
            }
            var categoricalFeaturesFieldValue = (CategoricalFeatures().Count >= 1) ? ("name:" + string.Join(',', DatasetHyperParameters.CategoricalFeatures())) : "";
            LightGbmSample.categorical_feature = categoricalFeaturesFieldValue;
            return true;
        }

        public List<string> CategoricalFeatures() => DatasetHyperParameters.CategoricalFeatures();
        public List<string> TargetFeatures() => DatasetHyperParameters.TargetFeatures();
    }
}
