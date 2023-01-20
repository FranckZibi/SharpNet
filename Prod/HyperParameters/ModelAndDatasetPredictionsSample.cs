using System;
using System.Collections.Generic;
using SharpNet.Datasets;

namespace SharpNet.HyperParameters
{
    public class ModelAndDatasetPredictionsSample : MultiSamples
    {
        #region Constructor

        // ReSharper disable once MemberCanBePrivate.Global
        public ModelAndDatasetPredictionsSample(ISample[] samples) : base(samples)
        {
            FixErrors();
        }
    
        public static AbstractDatasetSample LoadDatasetSample(string workingDirectory, string modelName)
        {
            return (AbstractDatasetSample)ISample.Load(workingDirectory, ModelAndDatasetSampleIndexToSampleName(modelName, 1));
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

        public static ModelAndDatasetPredictionsSample Load(string workingDirectory, string modelName, bool useAllAvailableCores)
        {
            var modelSample = IModelSample.LoadModelSample(workingDirectory, ModelAndDatasetSampleIndexToSampleName(modelName, 0), useAllAvailableCores);
            var datasetSample = LoadDatasetSample(workingDirectory, modelName);
            var predictionsSample = LoadPredictions(workingDirectory, modelName);

            return new ModelAndDatasetPredictionsSample(
                new ISample[]{
                modelSample,
                datasetSample,
                predictionsSample
                });
        }

        public ModelAndDatasetPredictionsSample CopyWithNewModelSample(IModelSample newModelSample)
        {
            var clonedSamples = new List<ISample>();
            foreach (var s in Samples)
            {
                clonedSamples.Add(s is IModelSample ? newModelSample : s.Clone());
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

        public override bool FixErrors()
        {
            if (ModelSample is Networks.NetworkSample networkSample)
            {
                networkSample.ApplyDataset(DatasetSample);
            }
            if (!base.FixErrors())
            {
                return false;
            }
            return true;
        }

        #region Filling Search Space with Default Values for Model
        public override void FillSearchSpaceWithDefaultValues(IDictionary<string, object> hyperParameterSearchSpace)
        {
            ModelSample.FillSearchSpaceWithDefaultValues(hyperParameterSearchSpace, DatasetSample);
        }

        #endregion


        public IModelSample ModelSample => (IModelSample)Samples[0];
        public AbstractDatasetSample DatasetSample => (AbstractDatasetSample)Samples[1];
        public PredictionsSample PredictionsSample  => (PredictionsSample) Samples[2];
    }
}
