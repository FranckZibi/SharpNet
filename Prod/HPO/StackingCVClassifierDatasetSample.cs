using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;

namespace SharpNet.HPO
{
    public class StackingCVClassifierDatasetSample : WrappedDatasetSample
    {
        #region private fields
        private readonly DataFrameDataSet _trainingDataFrameDataSet;
        private readonly DataFrameDataSet _inferenceDataFrameDataSet;
        #endregion

        #region constructor

        // ReSharper disable once UnusedMember.Global
        private StackingCVClassifierDatasetSample(DataFrameDataSet trainingDataFrameDataSet, DataFrameDataSet inferenceDataFrameDataSet, int cv) : base(trainingDataFrameDataSet.DatasetSample)
        {
            _trainingDataFrameDataSet = trainingDataFrameDataSet;
            _inferenceDataFrameDataSet= inferenceDataFrameDataSet;
            KFold = cv;
            Original.KFold = cv;
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
            DataFrame y_true_training_InModelFormat = null;
            DataFrameDataSet validationDataset = null, testDataset = null;

            if (!Directory.Exists(workingDirectory))
            {
                Directory.CreateDirectory(workingDirectory);
            }

            AbstractDatasetSample embeddedDatasetSample = null;

            for (int i = 0; i < workingDirectoryAndModelNames.Count; ++i)
            {
                var (embeddedModelWorkingDirectory, embeddedModelNameForTraining, embeddedModelNameForInference) = workingDirectoryAndModelNames[i];
                if (embeddedDatasetSample == null)
                {
                    embeddedDatasetSample = ModelAndDatasetPredictionsSample.LoadDatasetSample(embeddedModelWorkingDirectory, embeddedModelNameForTraining);
                }
                var trainingPredictionsSample = ModelAndDatasetPredictionsSample.LoadPredictions(embeddedModelWorkingDirectory, embeddedModelNameForTraining);
                var inferencePredictionsSample = trainingPredictionsSample;
                if (!string.IsNullOrEmpty(embeddedModelNameForInference) && embeddedModelNameForInference != embeddedModelNameForTraining)
                {
                    inferencePredictionsSample = ModelAndDatasetPredictionsSample.LoadPredictions(embeddedModelWorkingDirectory, embeddedModelNameForInference);
                }

                var y_pred_training_InModelFormat = embeddedDatasetSample.LoadPredictionsInModelFormat(embeddedModelWorkingDirectory, trainingPredictionsSample.Validation_PredictionsFileName_InModelFormat);
                var y_pred_inference_InModelFormat = embeddedDatasetSample.LoadPredictionsInModelFormat(embeddedModelWorkingDirectory, inferencePredictionsSample.Test_PredictionsFileName_InModelFormat);
                if (y_pred_training_InModelFormat == null)
                {
                    y_pred_training_InModelFormat = embeddedDatasetSample.LoadPredictionsInModelFormat(embeddedModelWorkingDirectory, trainingPredictionsSample.Train_PredictionsFileName_InModelFormat);
                }

                //_y_preds_train_InModelFormat.Add(y_pred_train_InModelFormat);
                y_preds_for_training_InModelFormat.Add(y_pred_training_InModelFormat);
                y_preds_for_inference_InModelFormat.Add(y_pred_inference_InModelFormat);

                if (i == 0)
                {
                    validationDataset = embeddedDatasetSample.LoadValidationDataset()?? embeddedDatasetSample.LoadTrainDataset();
                    if (validationDataset == null)
                    {
                        throw new ArgumentException($"no Validation/Training DataSet found");
                    }
                    testDataset = embeddedDatasetSample.LoadTestDataset();
                    if (testDataset == null)
                    {
                        throw new ArgumentException($"no Test DataSet found");
                    }

                    y_true_training_InModelFormat = validationDataset.Y_InModelFormat();
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

            if (x_training_InModelFormat_df.Shape[1] != x_inference_InModelFormat_df.Shape[1])
            {
                throw new Exception($"training and inference dataset must have the same number of columns");
            }

            var TrainingDataSet = new DataFrameDataSet(
                embeddedDatasetSample,
                x_training_InModelFormat_df,
                y_true_training_InModelFormat, //y_true_training_InTargetFormat,
                null); //TODO


            var InferenceDataSet = new DataFrameDataSet(
                embeddedDatasetSample,
                x_inference_InModelFormat_df,
                null,
                null); //TODO

            //TrainingDataSet.to_csv_in_directory(workingDirectory, true, true, true);
            //InferenceDataSet.to_csv_in_directory(workingDirectory, true, true, true);

            if (embeddedDatasetSample.PercentageInTraining > 0.99)
            {
                embeddedDatasetSample.PercentageInTraining = 0.8;
            }
            return new StackingCVClassifierDatasetSample(TrainingDataSet, InferenceDataSet, cv);

        }
        #endregion


        public override ISample Clone()
        {
            return new StackingCVClassifierDatasetSample(_trainingDataFrameDataSet, _inferenceDataFrameDataSet, KFold);
        }

        public override int[] X_Shape(int batchSize) => throw new NotImplementedException(); //!D TODO
        public override int[] Y_Shape(int batchSize) => throw new NotImplementedException(); //!D TODO

        public override int NumClass => _trainingDataFrameDataSet.DatasetSample.NumClass;

        public override DataSet TestDataset()
        {
            return _inferenceDataFrameDataSet;
        }

        public override DataSet FullTrainingAndValidation()
        {
            return _trainingDataFrameDataSet;
        }
    }
}