using SharpNet.LightGBM;

namespace SharpNet.HyperParameters
{
    public class ModelAndDatasetSample : MultiSamples
    {
        #region Constructor
        public ModelAndDatasetSample(IModelSample modelSample, AbstractDatasetSample abstractDatasetSample)
            : base(new ISample[] { modelSample, abstractDatasetSample })
        {
        }
        public static ModelAndDatasetSample LoadModelAndDatasetSample(string workingDirectory, string modelName)
        {
            return new ModelAndDatasetSample(
                IModelSample.LoadModelSample(workingDirectory, SampleIndexToSampleName(0, 2, modelName)),
                AbstractDatasetSample.ValueOf(workingDirectory, SampleIndexToSampleName(1, 2, modelName)));
        }
        #endregion

        public IModelSample ModelSample => (IModelSample)Samples[0];
        public AbstractDatasetSample DatasetSample => (AbstractDatasetSample)Samples[1];

        public override bool FixErrors()
        {
            if (!base.FixErrors())
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
