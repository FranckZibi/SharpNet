using System;
using SharpNet.LightGBM;

namespace SharpNet.HyperParameters
{
    public class ModelAndDatasetSample : MultiSamples
    {
        #region Constructor

        // ReSharper disable once MemberCanBePrivate.Global
        public ModelAndDatasetSample(ISample[] samples) : base(samples) { }
        public ModelAndDatasetSample(IModelSample modelSample, AbstractDatasetSample abstractDatasetSample)
            : this(new ISample[] { modelSample, abstractDatasetSample })
        {
        }
        public static ModelAndDatasetSample LoadModelAndDatasetSample(string workingDirectory, string modelName)
        {
            return new ModelAndDatasetSample(
                IModelSample.LoadModelSample(workingDirectory, ModelAndDatasetSampleIndexToSampleName(modelName, 0)),
                AbstractDatasetSample.ValueOf(workingDirectory, ModelAndDatasetSampleIndexToSampleName(modelName, 1)));
        }
        #endregion

        private static string ModelAndDatasetSampleIndexToSampleName(string modelName, int sampleIndex)
        {
            switch (sampleIndex)
            {
                case 0: return modelName;
                case 1: return modelName + "_dataset";
                default: throw new ArgumentException($"invalid index {sampleIndex} for {nameof(ModelAndDatasetSample)}");
            }
        }

        protected override string SampleIndexToSampleName(string modelName, int sampleIndex)
        {
            return ModelAndDatasetSampleIndexToSampleName(modelName, sampleIndex);
        }

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
