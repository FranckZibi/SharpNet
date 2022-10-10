using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using JetBrains.Annotations;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.LightGBM
{
    public class LightGBMModel : AbstractModel
    {
        #region public fields & properties
        public LightGBMSample LightGbmSample => (LightGBMSample)ModelSample;
        #endregion

        #region constructor
        /// <summary>
        /// 
        /// </summary>
        /// <param name="lightGbmModelSample"></param>
        /// <param name="workingDirectory"></param>
        /// <param name="modelName">the name of the model to use</param>
        /// <exception cref="Exception"></exception>
        [SuppressMessage("ReSharper", "VirtualMemberCallInConstructor")]
        public LightGBMModel(LightGBMSample lightGbmModelSample, string workingDirectory, [JetBrains.Annotations.NotNull] string modelName): base(lightGbmModelSample, workingDirectory, modelName)
        {
            if (!File.Exists(ExePath))
            {
                throw new Exception($"Missing exe {ExePath}");
            }
            if (!Directory.Exists(RootDatasetPath))
            {
                Directory.CreateDirectory(RootDatasetPath);
            }
            if (!Directory.Exists(TempPath))
            {
                Directory.CreateDirectory(TempPath);
            }
        }
        #endregion

        public override (string train_XDatasetPath, string train_YDatasetPath, string train_XYDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath, string validation_XYDatasetPath) 
            Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny)
        {
            const bool addTargetColumnAsFirstColumn = true;
            const bool includeIdColumns = false;
            const bool overwriteIfExists = false;
            string trainDatasetPath = trainDataset.to_csv_in_directory(RootDatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            string validationDatasetPath = null;
            if (validationDatasetIfAny != null)
            {
                validationDatasetPath = validationDatasetIfAny.to_csv_in_directory(RootDatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            }
            return Fit(trainDatasetPath, validationDatasetPath);
        }

     
        public override DataFrame Predict(IDataSet dataset, bool addIdColumnsAtLeft, bool removeAllTemporaryFilesAtEnd)
        {
            if (!File.Exists(ModelPath))
            {
                throw new Exception($"missing model {ModelPath} for inference");
            }
            const bool addTargetColumnAsFirstColumn = false;
            const bool includeIdColumns = false;
            const bool overwriteIfExists = false;
            string datasetPath = dataset.to_csv_in_directory(RootDatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            var predictionResultPath = Path.Combine(TempPath, ModelName + "_predict_" + Path.GetFileNameWithoutExtension(datasetPath) + ".txt");

            //we save in 'tmpLightGbmSamplePath' the model sample used for prediction
            var tmpLightGBMSamplePath = predictionResultPath.Replace(".txt", ".conf");
            var tmpLightGBMSample = (LightGBMSample)LightGbmSample.Clone();
            tmpLightGBMSample.Set(new Dictionary<string, object> {
                {"task", LightGBMSample.task_enum.predict},
                {"data", datasetPath},
                {"input_model", ModelPath},
                {"prediction_result", predictionResultPath},
                {"header", true}
            });
            tmpLightGBMSample.Save(tmpLightGBMSamplePath);

            Utils.Launch(WorkingDirectory, ExePath, "config=" + tmpLightGBMSamplePath, IModel.Log);

            var predictionsDf = LoadProbaFile(predictionResultPath, false, false, dataset, addIdColumnsAtLeft);
            Utils.TryDelete(tmpLightGBMSamplePath);
            Utils.TryDelete(predictionResultPath);
            if (removeAllTemporaryFilesAtEnd)
            {
                Utils.TryDelete(datasetPath);
            }
            return predictionsDf;
        }
        public override void Save(string workingDirectory, string modelName)
        {
            //No need to save model : it is already saved
            LightGbmSample.Save(workingDirectory, modelName);
        }

        public override int GetNumEpochs()
        {
            return LightGbmSample.num_iterations;
        }
        public override string DeviceName()
        {
            return LightGbmSample.DeviceName();
        }
        public override int TotalParams()
        {
            return -1; //TODO
        }

        public override double GetLearningRate()
        {
            return LightGbmSample.learning_rate;
        }

        public override void Use_All_Available_Cores()
        {
            LightGbmSample.num_threads = Utils.CoreCount;
        }

        public override List<string> ModelFiles()
        {
            return new List<string> { ModelPath };
        }
        //public static LightGBMModel LoadTrainedLightGBMModel(string workingDirectory, string modelName)
        //{
        //    var sample = ISample.LoadSample<LightGBMSample>(workingDirectory, modelName);
        //    return new LightGBMModel(sample, workingDirectory, modelName);
        //}

        private (string train_XDatasetPath, string train_YDatasetPath, string train_XYDatasetPath, string validation_XDatasetPath, string validation_YDatasetPath, string validation_XYDatasetPath) 
            Fit([JetBrains.Annotations.NotNull] string trainDatasetPath, [CanBeNull] string validationDatasetPathIfAny)
        {
            //we save in 'tmpLightGBMSamplePath' the model sample used for training
            var tmpLightGBMSamplePath = ISample.ToPath(TempPath, ModelName);
            var tmpLightGBMSample = (LightGBMSample)LightGbmSample.Clone();
            tmpLightGBMSample.Set(new Dictionary<string, object> {
                {"task", LightGBMSample.task_enum.train},
                {"data", trainDatasetPath},
                {"valid", validationDatasetPathIfAny??""},
                {"output_model", ModelPath},
                {"input_model", ""},
                {"prediction_result", ""},
                {"header", true},
                {"save_binary", false},
            });
            var logMsg = $"Training model '{ModelName}' with training dataset {Path.GetFileNameWithoutExtension(trainDatasetPath)}" +(string.IsNullOrEmpty(validationDatasetPathIfAny) ? "" : $" and validation dataset {Path.GetFileNameWithoutExtension(validationDatasetPathIfAny)}");
            if (LoggingForModelShouldBeDebug(ModelName))
            {
                LogDebug(logMsg);
            }
            else
            {
                LogInfo(logMsg);
            }
            tmpLightGBMSample.Save(tmpLightGBMSamplePath);

            Utils.Launch(WorkingDirectory, ExePath, "config=" + tmpLightGBMSamplePath, IModel.Log);
            Utils.TryDelete(tmpLightGBMSamplePath);
            return (null, null, trainDatasetPath, null, null, validationDatasetPathIfAny);
        }

        private static string ExePath => Path.Combine(Utils.ChallengesPath, "bin", "lightgbm.exe");
        private string TempPath => Path.Combine(WorkingDirectory, "Temp");

        /// <summary>
        /// path of a trained model.
        /// null if no trained model is available
        /// </summary>
        private string ModelPath => Path.Combine(WorkingDirectory, ModelName + ".txt");
    }
}
