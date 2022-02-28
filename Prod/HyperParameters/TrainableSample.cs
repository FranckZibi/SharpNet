using SharpNet.CPU;
using SharpNet.LightGBM;
using SharpNet.Models;

// ReSharper disable MemberCanBePrivate.Global

namespace SharpNet.HyperParameters
{
    public class TrainableSample : MultiSamples, ITrainableSample
    {
        #region Constructor

        public TrainableSample(IModelSample modelSample, AbstractDatasetSample abstractDatasetSample)
            : base(new ISample[] { modelSample, abstractDatasetSample })
        {
        }
        public static TrainableSample ValueOf(string workingDirectory, string modelName)
        {
            return new TrainableSample(
                IModelSample.ValueOfModelSample(workingDirectory, modelName),
                AbstractDatasetSample.ValueOf(workingDirectory, modelName + "_1"));
        }
        #endregion

        public IModelSample ModelSample => (IModelSample)Samples[0];
        public AbstractDatasetSample DatasetSample => (AbstractDatasetSample)Samples[1];
        public AbstractModel NewUntrainedModel(string workingDirectory)
        {
            return AbstractModel.NewUntrainedModel(ModelSample, workingDirectory, ComputeHash());
        }
        public override CpuTensor<float> Y_Train_dataset_to_Perfect_Predictions(string y_train_dataset)
        {
            return DatasetSample.PredictionsInModelFormat_2_PredictionsInTargetFormat(y_train_dataset);
        }
        public override bool PostBuild()
        {
            if (!base.PostBuild())
            {
                return false;
            }

            if (ModelSample is LightGBMSample lightGbmSample)
            {
                var categoricalFeatures = DatasetSample.CategoricalFeatures();
                var categoricalFeaturesFieldValue = (categoricalFeatures.Count >= 1)
                    ? ("name:" + string.Join(',', categoricalFeatures))
                    : "";
                lightGbmSample.categorical_feature = categoricalFeaturesFieldValue;
            }

            return true;
        }
    }
}
