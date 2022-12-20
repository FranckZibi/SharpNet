using System;
using System.Collections.Generic;
using SharpNet.CatBoost;
using SharpNet.Datasets;
using SharpNet.LightGBM;
using SharpNet.Networks;

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

        public static ModelAndDatasetPredictionsSample Load(string workingDirectory, string modelName)
        {
            var modelSample = IModelSample.LoadModelSample(workingDirectory, ModelAndDatasetSampleIndexToSampleName(modelName, 0));
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
            if (ModelSample is NetworkSample networkSample)
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
            switch (ModelSample)
            {
                case LightGBMSample:
                    FillWithDefaultLightGBMHyperParameterValues(hyperParameterSearchSpace);
                    break;
                case CatBoostSample:
                    FillWithDefaultCatBoostHyperParameterValues(hyperParameterSearchSpace);
                    break;
                case NetworkSample:
                    FillWithDefaultNetworkHyperParameterValues(hyperParameterSearchSpace);
                    break;
            }
        }
        private void FillWithDefaultLightGBMHyperParameterValues(IDictionary<string, object> existingHyperParameterValues)
        {
            const string objectiveKeyName = nameof(LightGBMSample.objective);
            if (!existingHyperParameterValues.ContainsKey(objectiveKeyName))
            {
                existingHyperParameterValues[objectiveKeyName] = GetDefaultHyperParameterValueForLightGBM(objectiveKeyName);
            }
            const string numClassKeyName = nameof(LightGBMSample.num_class);
            if (!existingHyperParameterValues.ContainsKey(numClassKeyName) && DatasetSample.GetObjective() == Objective_enum.Classification)
            {
                existingHyperParameterValues[numClassKeyName] = GetDefaultHyperParameterValueForLightGBM(numClassKeyName);
            }
        }
        private object GetDefaultHyperParameterValueForLightGBM(string hyperParameterName)
        {
            switch (hyperParameterName)
            {
                case nameof(LightGBMSample.objective):
                    if (DatasetSample.GetObjective() == Objective_enum.Regression)
                    {
                        return nameof(LightGBMSample.objective_enum.regression);
                    }
                    if (DatasetSample.GetObjective() == Objective_enum.Classification)
                    {
                        if (DatasetSample.NumClass >= 2)
                        {
                            return nameof(LightGBMSample.objective_enum.multiclass);
                        }
                        return nameof(LightGBMSample.objective_enum.binary);
                    }
                    break;
                case nameof(LightGBMSample.num_class):
                    return DatasetSample.NumClass;
            }
            var errorMsg = $"do not know default value for Hyper Parameter {hyperParameterName} for model {typeof(LightGBMModel)}";
            ISample.Log.Error(errorMsg);
            throw new ArgumentException(errorMsg);
        }
        private void FillWithDefaultCatBoostHyperParameterValues(IDictionary<string, object> existingHyperParameterValues)
        {
            const string lossFunctionKeyName = nameof(CatBoostSample.loss_function);
            if (!existingHyperParameterValues.ContainsKey(lossFunctionKeyName))
            {
                existingHyperParameterValues[lossFunctionKeyName] = GetDefaultHyperParameterValueForCatBoost(lossFunctionKeyName);
            }
        }
        private object GetDefaultHyperParameterValueForCatBoost(string hyperParameterName)
        {
            switch (hyperParameterName)
            {
                case nameof(CatBoostSample.loss_function):
                    if (DatasetSample.GetObjective() == Objective_enum.Regression)
                    {
                        return nameof(CatBoostSample.loss_function_enum.RMSE);
                    }
                    if (DatasetSample.GetObjective() == Objective_enum.Classification)
                    {
                        if (DatasetSample.NumClass >= 2)
                        {
                            return nameof(CatBoostSample.loss_function_enum.MultiClass);
                        }
                        return nameof(CatBoostSample.loss_function_enum.Logloss);
                    }
                    break;
            }
            var errorMsg = $"do not know default value for Hyper Parameter {hyperParameterName} for model {typeof(CatBoostModel)}";
            ISample.Log.Error(errorMsg);
            throw new ArgumentException(errorMsg);
        }
        private void FillWithDefaultNetworkHyperParameterValues(IDictionary<string, object> existingHyperParameterValues)
        {
            //!D TO change
            ((NetworkSample)ModelSample).ApplyDataset(DatasetSample);

            const string lossFunctionKeyName = nameof(NetworkSample.LossFunction);
            if (!existingHyperParameterValues.ContainsKey(lossFunctionKeyName))
            {
                existingHyperParameterValues[lossFunctionKeyName] = GetDefaultHyperParameterValueForNetwork(lossFunctionKeyName);
            }
        }
        private object GetDefaultHyperParameterValueForNetwork(string hyperParameterName)
        {
            switch (hyperParameterName)
            {
                case nameof(NetworkSample.LossFunction):
                    return DatasetSample.DefaultLossFunction.ToString();
            }
            var errorMsg = $"do not know default value for Hyper Parameter {hyperParameterName} for model {typeof(Network)}";
            ISample.Log.Error(errorMsg);
            throw new ArgumentException(errorMsg);
        }
        #endregion


        public IModelSample ModelSample => (IModelSample)Samples[0];
        public AbstractDatasetSample DatasetSample => (AbstractDatasetSample)Samples[1];
        public PredictionsSample PredictionsSample  => (PredictionsSample) Samples[2];
    }
}
