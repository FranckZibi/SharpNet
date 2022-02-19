using System.IO;
using SharpNet.CPU;
using SharpNet.Datasets.Natixis70;
using SharpNet.LightGBM;

// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.HyperParameters
{
    /// <summary>
    /// Network support for Natixis70 challenge
    /// </summary>
    public class Model_and_Dataset_Sample : MultiSamples
    {
        //public Natixis70_LightGBM_HyperParameters() : this(new ISample[]{new LightGBMSample(), new Natixis70DatasetHyperParameters()}) { }
        public Model_and_Dataset_Sample(IModelSample modelSample, AbstractDatasetSample abstractDatasetSample) : base(new ISample[] { modelSample, abstractDatasetSample }) { }

        public IModelSample ModelSample => (IModelSample)Samples[0];
        public AbstractDatasetSample DatasetSample => (AbstractDatasetSample)Samples[1];

        public static Model_and_Dataset_Sample ValueOf(string workingDirectory, string modelName)
        {
            //TODO TO REMOVE : old way of storing all configs in same file
            if (File.Exists(ISample.ToPath(workingDirectory, modelName)))
            {
                var content = ISample.LoadConfig(workingDirectory, modelName);
                var res = new Model_and_Dataset_Sample(new LightGBMSample(), new Natixis70DatasetSample());
                res.Set(Utils.FromString2String_to_String2Object(content));
                return res;
            }

            return new Model_and_Dataset_Sample(
                ISample.ValueOfModelSample(workingDirectory, modelName + "_0"),
                AbstractDatasetSample.ValueOf(workingDirectory, modelName + "_1"));
        }

        public override CpuTensor<float> Y_Train_dataset_to_Perfect_Predictions(string y_train_dataset)
        {
            return DatasetSample.ModelPrediction_2_TargetPredictionFormat(y_train_dataset);
        }

        public override bool PostBuild()
        {
            if (!base.PostBuild())
            {
                return false;
            }

            if (ModelSample is LightGBMSample lightGBMSample)
            {
                var categoricalFeatures = DatasetSample.CategoricalFeatures();
                var categoricalFeaturesFieldValue = (categoricalFeatures.Count >= 1)
                    ? ("name:" + string.Join(',', categoricalFeatures))
                    : "";
                lightGBMSample.categorical_feature = categoricalFeaturesFieldValue;
            }

            return true;
        }
    }
}
