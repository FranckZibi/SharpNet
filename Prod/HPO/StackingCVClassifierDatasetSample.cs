using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.HPO
{
    public class StackingCVClassifierDatasetSample : DelegatedDatasetSample
    {
        #region private fields
        private readonly InMemoryDataSet TrainingDataSet;
        private readonly InMemoryDataSet InferenceDataSet;
        #endregion

        #region constructor

        // ReSharper disable once UnusedMember.Global
        private StackingCVClassifierDatasetSample(AbstractDatasetSample embeddedDatasetSample, InMemoryDataSet trainingDataSet, InMemoryDataSet inferenceDataSet, int cv) : base(embeddedDatasetSample)
        {
            TrainingDataSet = trainingDataSet;
            InferenceDataSet= inferenceDataSet;
            KFold = cv;
            EmbeddedDatasetSample.KFold = cv;
            if (KFold == 1)
            {
                if (PercentageInTraining >= 1.0)
                {
                    PercentageInTraining = 0.8;
                }
            }
            else
            {
                PercentageInTraining = 1.0;
            }
        }

        public static StackingCVClassifierDatasetSample New(IReadOnlyList<Tuple<string, string, string>> workingDirectoryAndModelNames, [NotNull] string workingDirectory, bool use_features_in_secondary = true, int cv = 2)
        {
            List<DataFrame> y_preds_for_training_InModelFormat = new();
            List<DataFrame> y_preds_for_inference_InModelFormat = new();
            CpuTensor<float> y_true_training_InTargetFormat = null;
            InMemoryDataSet validationDataset = null, testDataset = null;

            if (!Directory.Exists(workingDirectory))
            {
                Directory.CreateDirectory(workingDirectory);
            }

            AbstractDatasetSample EmbeddedDatasetSample = null;

            for (int i = 0; i < workingDirectoryAndModelNames.Count; ++i)
            {
                var (embeddedModelWorkingDirectory, embeddedModelNameForTraining, embeddedModelNameForInference) = workingDirectoryAndModelNames[i];
                var embeddedModelAndDatasetPredictionsForTraining = ModelAndDatasetPredictions.Load(embeddedModelWorkingDirectory, embeddedModelNameForTraining);

                var embeddedModelAndDatasetPredictionsForInference = embeddedModelAndDatasetPredictionsForTraining;
                if (!string.IsNullOrEmpty(embeddedModelNameForInference) && embeddedModelNameForInference != embeddedModelNameForTraining)
                {
                    embeddedModelAndDatasetPredictionsForInference = ModelAndDatasetPredictions.Load(embeddedModelWorkingDirectory, embeddedModelNameForInference); ;
                }


                EmbeddedDatasetSample = embeddedModelAndDatasetPredictionsForTraining.DatasetSample;
                //var y_pred_train_InModelFormat = datasetSample.LoadPredictionsInModelFormat(embeddedModelWorkingDirectory, embeddedModelAndDatasetPredictionsForTraining.PredictionsSample.Train_PredictionsFileName_InModelFormat);
                var y_pred_training_InModelFormat = EmbeddedDatasetSample.LoadPredictionsInModelFormat(embeddedModelWorkingDirectory, embeddedModelAndDatasetPredictionsForTraining.PredictionsSample.Validation_PredictionsFileName_InModelFormat);
                var y_pred_inference_InModelFormat = EmbeddedDatasetSample.LoadPredictionsInModelFormat(embeddedModelWorkingDirectory, embeddedModelAndDatasetPredictionsForInference.PredictionsSample.Test_PredictionsFileName_InModelFormat);
                if (y_pred_training_InModelFormat == null)
                {
                    y_pred_training_InModelFormat = EmbeddedDatasetSample.LoadPredictionsInModelFormat(embeddedModelWorkingDirectory, embeddedModelAndDatasetPredictionsForTraining.PredictionsSample.Train_PredictionsFileName_InModelFormat);
                }

                //_y_preds_train_InModelFormat.Add(y_pred_train_InModelFormat);
                y_preds_for_training_InModelFormat.Add(y_pred_training_InModelFormat);
                y_preds_for_inference_InModelFormat.Add(y_pred_inference_InModelFormat);

                if (i == 0)
                {
                    validationDataset = EmbeddedDatasetSample.LoadValidationDataset()?? EmbeddedDatasetSample.LoadTrainDataset();
                    if (validationDataset == null)
                    {
                        throw new ArgumentException($"no Validation/Training DataSet found");
                    }
                    testDataset = EmbeddedDatasetSample.LoadTestDataset();
                    if (testDataset == null)
                    {
                        throw new ArgumentException($"no Test DataSet found");
                    }

                    y_true_training_InTargetFormat = validationDataset.Y_InTargetFormat().FloatCpuTensor();
                }
            }

            //we load the validation predictions done by the source models
            y_preds_for_training_InModelFormat.RemoveAll(t => t == null);
            Debug.Assert(y_preds_for_training_InModelFormat.All(t => t != null));
            Debug.Assert(DataFrame.SameShape(y_preds_for_training_InModelFormat));
            if (y_preds_for_training_InModelFormat.Count == 0)
            {
                throw new Exception($"no Validation Predictions found");
            }

            //we load the test predictions done by the source models (if any)
            y_preds_for_inference_InModelFormat.RemoveAll(t => t == null);
            //Debug.Assert(_TestPredictions_INModelFormat.Count == 0 || _TestPredictions_INModelFormat.Count == _ValidationPredictions_InModelFormat.Count);
            Debug.Assert(DataFrame.SameShape(y_preds_for_inference_InModelFormat));
            if (y_preds_for_inference_InModelFormat.Count == 0)
            {
                throw new Exception($"no Test Predictions found");
            }
            if (y_preds_for_training_InModelFormat.Count != y_preds_for_inference_InModelFormat.Count)
            {
                throw new Exception($"Not the same number of predictions between Validation {y_preds_for_training_InModelFormat.Count} and Test {y_preds_for_inference_InModelFormat.Count}");
            }

            var x_training_InModelFormat = CpuTensor<float>.MergeHorizontally(y_preds_for_training_InModelFormat.Select(df => df.FloatCpuTensor()).ToArray());
            var x_inference_InModelFormat = CpuTensor<float>.MergeHorizontally(y_preds_for_inference_InModelFormat.Select(df => df.FloatCpuTensor()).ToArray());

            // the predictions files never had Id Columns, we add it.
            var x_training_InModelFormat_df = DataFrame.New(x_training_InModelFormat);
            var x_inference_InModelFormat_df = DataFrame.New(x_inference_InModelFormat);

            if (use_features_in_secondary)
            {
                x_training_InModelFormat_df = DataFrame.MergeHorizontally(validationDataset.XDataFrame, x_training_InModelFormat_df);
                x_inference_InModelFormat_df = DataFrame.MergeHorizontally(testDataset.XDataFrame, x_inference_InModelFormat_df);
            }
            x_training_InModelFormat_df = validationDataset.AddIdColumnsAtLeftIfNeeded(x_training_InModelFormat_df);
            x_inference_InModelFormat_df = testDataset.AddIdColumnsAtLeftIfNeeded(x_inference_InModelFormat_df);


            if (x_training_InModelFormat_df.Shape[1] != x_inference_InModelFormat_df.Shape[1])
            {
                throw new Exception($"training and inference dataset must have the same number of columns");
            }

            var datasetSampleCategoricalFeatures = Utils.Intersect(x_training_InModelFormat_df.Columns, EmbeddedDatasetSample.CategoricalFeatures).ToArray();
            var datasetSampleIdColumns = Utils.Intersect(x_training_InModelFormat_df.Columns, EmbeddedDatasetSample.IdColumns).ToArray();
            var TrainingDataSet = new InMemoryDataSet(
                x_training_InModelFormat_df.FloatCpuTensor(),
                y_true_training_InTargetFormat,
                EmbeddedDatasetSample.Name,
                EmbeddedDatasetSample.GetObjective(),
                null,
                x_training_InModelFormat_df.Columns,
                datasetSampleCategoricalFeatures,
                datasetSampleIdColumns,
                EmbeddedDatasetSample.TargetLabels,
                false,
                EmbeddedDatasetSample.GetSeparator());

            var InferenceDataSet = new InMemoryDataSet(
                x_inference_InModelFormat_df.FloatCpuTensor(),
                null,
                EmbeddedDatasetSample.Name,
                EmbeddedDatasetSample.GetObjective(),
                null,
                x_inference_InModelFormat_df.Columns,
                datasetSampleCategoricalFeatures,
                datasetSampleIdColumns,
                EmbeddedDatasetSample.TargetLabels,
                false,
                EmbeddedDatasetSample.GetSeparator());


            TrainingDataSet.to_csv_in_directory(workingDirectory, true, true, true);
            InferenceDataSet.to_csv_in_directory(workingDirectory, true, true, true);

        
            if (EmbeddedDatasetSample.PercentageInTraining > 0.99)
            {
                EmbeddedDatasetSample.PercentageInTraining = 0.8;
            }
            return new StackingCVClassifierDatasetSample(EmbeddedDatasetSample, TrainingDataSet, InferenceDataSet, cv);

        }
        #endregion


        public override ISample Clone()
        {
            return new StackingCVClassifierDatasetSample(EmbeddedDatasetSample, TrainingDataSet, InferenceDataSet, KFold);
        }


        public override DataSet TestDataset()
        {
            return InferenceDataSet;
        }

        public override DataSet FullTrainingAndValidation()
        {
            return TrainingDataSet;
        }
    }
}