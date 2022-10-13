using System;
using System.Collections.Generic;
using SharpNet.LightGBM;

namespace SharpNet.HyperParameters
{
    public class ModelAndDatasetPredictionsSample : MultiSamples
    {
        #region Constructor

        // ReSharper disable once MemberCanBePrivate.Global
        public ModelAndDatasetPredictionsSample(ISample[] samples) : base(samples) { }

        public static ModelAndDatasetPredictionsSample Load(string workingDirectory, string modelName)
        {
            return new ModelAndDatasetPredictionsSample(
                new ISample[]{
                IModelSample.LoadModelSample(workingDirectory, ModelAndDatasetSampleIndexToSampleName(modelName, 0)),
                AbstractDatasetSample.ValueOf(workingDirectory, ModelAndDatasetSampleIndexToSampleName(modelName, 1)),
                ISample.LoadSample<PredictionsSample>(workingDirectory, ModelAndDatasetSampleIndexToSampleName(modelName, 2))
                });
        }

        public ModelAndDatasetPredictionsSample CopyWithNewPercentageInTrainingAndKFold(double newPercentageInTraining, int newKFold)
        {
            var clonedSamples = new List<ISample>();
            foreach (var s in Samples)
            {
                if (s is AbstractDatasetSample datasetSample)
                {
                    clonedSamples.Add(datasetSample.CopyWithNewPercentageInTrainingAndKFold(newPercentageInTraining, newKFold));
                }
                else
                {
                    clonedSamples.Add(s.Clone());
                }
            }
            return new ModelAndDatasetPredictionsSample(clonedSamples.ToArray());
        }

        public static ModelAndDatasetPredictionsSample New(IModelSample modelSample, AbstractDatasetSample abstractDatasetSample)
        {
            return new ModelAndDatasetPredictionsSample(new ISample[]{ modelSample, abstractDatasetSample, new PredictionsSample()});
        }
        #endregion

        private static string ModelAndDatasetSampleIndexToSampleName(string modelName, int sampleIndex)
        {
            switch (sampleIndex)
            {
                case 0: return modelName;
                case 1: return modelName + "_dataset";
                case 2: return modelName + "_predictions";
                default: throw new ArgumentException($"invalid index {sampleIndex} for {nameof(ModelAndDatasetPredictionsSample)}");
            }
        }

        protected override string SampleIndexToSampleName(string modelName, int sampleIndex)
        {
            return ModelAndDatasetSampleIndexToSampleName(modelName, sampleIndex);
        }

        public IModelSample ModelSample => (IModelSample)Samples[0];
        public AbstractDatasetSample DatasetSample => (AbstractDatasetSample)Samples[1];
        public PredictionsSample PredictionsSample  => (PredictionsSample) Samples[2];
        public override bool FixErrors()
        {
            if (!base.FixErrors())
            {
                return false;
            }

            if (ModelSample is LightGBMSample lightGbmSample)
            {
                var categoricalFeatures = DatasetSample.CategoricalFeatures;
                var categoricalFeaturesFieldValue = (categoricalFeatures.Length >= 1)
                    ? ("name:" + string.Join(',', categoricalFeatures))
                    : "";
                lightGbmSample.categorical_feature = categoricalFeaturesFieldValue;
            }

            return true;
        }
    }
}
