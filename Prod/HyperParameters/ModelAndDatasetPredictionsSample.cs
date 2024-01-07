using System;
using System.Collections.Generic;
using SharpNet.Datasets;

namespace SharpNet.Hyperparameters
{
    public sealed class ModelAndDatasetPredictionsSample : MultiSamples, IDisposable
    {
        #region Constructor

        // ReSharper disable once MemberCanBePrivate.Global
        public ModelAndDatasetPredictionsSample(ISample[] samples) : base(samples)
        {
            FixErrors();
        }
    
        public static AbstractDatasetSample LoadDatasetSample(string workingDirectory, string modelName, Action<IDictionary<string, string>> contentUpdater = null)
        {
            return (AbstractDatasetSample)ISample.Load(workingDirectory, ModelAndDatasetSampleIndexToSampleName(modelName, 1), contentUpdater);
        }

        public static PredictionsSample LoadPredictions(string workingDirectory, string modelName)
        {
            //We try to load the prediction sample file
            //If it is missing, we'll just use an empty prediction file
            try
            {
                return (PredictionsSample)ISample.Load(workingDirectory, ModelAndDatasetSampleIndexToSampleName(modelName, 2));
            }
            catch
            {
                //Prediction file is missing, we'll use a default prediction file
                return new PredictionsSample();
            }
        }

        public static ModelAndDatasetPredictionsSample Load(string workingDirectory, string modelName, bool useAllAvailableCores, Action<IDictionary<string, string>> contentUpdater = null)
        {
            var modelSample = AbstractModelSample.LoadModelSample(workingDirectory, ModelAndDatasetSampleIndexToSampleName(modelName, 0), useAllAvailableCores, contentUpdater);
            var datasetSample = LoadDatasetSample(workingDirectory, modelName, contentUpdater);
            var predictionsSample = LoadPredictions(workingDirectory, modelName);

            return new ModelAndDatasetPredictionsSample(
                new ISample[]{
                modelSample,
                datasetSample,
                predictionsSample
                });
        }

        public ModelAndDatasetPredictionsSample CopyWithNewModelSample(AbstractModelSample newModelSample)
        {
            var clonedSamples = new List<ISample>();
            foreach (var s in Samples)
            {
                clonedSamples.Add(s is AbstractModelSample ? newModelSample : s.Clone());
            }
            return new ModelAndDatasetPredictionsSample(clonedSamples.ToArray());
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

        public static ModelAndDatasetPredictionsSample New(AbstractModelSample modelSample, AbstractDatasetSample abstractDatasetSample)
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

        #region Filling Search Space with Default Values for Model
        public override void FillSearchSpaceWithDefaultValues(IDictionary<string, object> HyperparameterSearchSpace)
        {
            ModelSample.FillSearchSpaceWithDefaultValues(HyperparameterSearchSpace);
        }
        #endregion

        public AbstractModelSample ModelSample => (AbstractModelSample)Samples[0];
        public AbstractDatasetSample DatasetSample => (AbstractDatasetSample)Samples[1];
        public PredictionsSample PredictionsSample  => (PredictionsSample) Samples[2];


        #region Dispose pattern
        private bool disposed = false;
        private void Dispose(bool disposing)
        {
            if (disposed)
            {
                return;
            }
            disposed = true;
            //Release Unmanaged Resources
            if (disposing)
            {
                //Release Managed Resources
                DatasetSample?.Dispose();
                Samples[1] = null;
            }
        }
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        ~ModelAndDatasetPredictionsSample()
        {
            Dispose(false);
        }
        #endregion

        public override bool FixErrors()
        {
            if (!base.FixErrors())
            {
                return false;
            }
            //if (ModelSample.GetObjective() != DatasetSample.GetObjective())
            //{
            //    return false;
            //}
            return true;
        }
    }
}
