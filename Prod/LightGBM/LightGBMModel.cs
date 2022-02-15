using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.LightGBM
{
    public class LightGBMModel : AbstractModel
    {
        #region private fields & properties
        private const char separator = ',';
        #endregion

        #region public fields & properties
        /// <summary>
        /// path of a trained model.
        /// null if no trained model is available
        /// </summary>
        [NotNull] public string ModelPath => Path.Combine(WorkingDirectory, ModelName+".txt");
        [NotNull] public string ModelConfigPath => ISample.ToPath(WorkingDirectory, ModelName);
        public override string WorkingDirectory { get; }
        public override string ModelName { get; }
        public LightGBMSample LightGbmSample => (LightGBMSample)Sample;
        #endregion

        #region constructor
        /// <summary>
        /// 
        /// </summary>
        /// <param name="lightGBMSample"></param>
        /// <param name="workingDirectory"></param>
        /// <param name="modelName">the name of the model to use</param>
        /// <exception cref="Exception"></exception>
        public LightGBMModel(LightGBMSample lightGBMSample, string workingDirectory, [NotNull] string modelName): base(lightGBMSample)
        {
            WorkingDirectory = workingDirectory;
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
            ModelName = modelName;
        }
        #endregion

        public override void Fit(IDataSet trainDataset, IDataSet validationDatasetIfAny)
        {
            string trainDatasetPath = DatasetPath(trainDataset, true);
            trainDataset.to_csv(trainDatasetPath, separator, true, false);

            string validationDatasetPath = "";
            if (validationDatasetIfAny != null)
            {
                validationDatasetPath = DatasetPath(validationDatasetIfAny, true);
                validationDatasetIfAny.to_csv(validationDatasetPath, separator, true, false);
            }
            Fit(trainDatasetPath, validationDatasetPath);
        }
        public override CpuTensor<float> Predict(IDataSet dataset)
        {
            const bool addTargetColumnAsFirstColumn = false;
            string predictionDatasetPath = DatasetPath(dataset, addTargetColumnAsFirstColumn);
            dataset.to_csv(predictionDatasetPath, separator, addTargetColumnAsFirstColumn, addTargetColumnAsFirstColumn);
            var predictions = Predict(predictionDatasetPath);
            if (predictions.Shape[0]!= dataset.Count)
            {
                throw new Exception($"Invalid number of predictions, received {predictions.Shape[0]} but expected {dataset.Count}");
            }
            return predictions;
        }
        public override void Save(string workingDirectory, string modelName)
        {
            //No need to save model : it is already saved
            LightGbmSample.Save(workingDirectory, modelName);
        }

        protected override List<string> ModelFiles()
        {
            return new List<string> { ModelPath };
        }

        private void Fit([NotNull] string trainDatasetPath, [CanBeNull] string validationDatasetPathIfAny)
        {
            LastDatasetPathUsedForTraining = trainDatasetPath;
            LastDatasetPathUsedForValidation = validationDatasetPathIfAny ?? "";
            LightGbmSample.Set(new Dictionary<string, object> {
                {"task", LightGBMSample.task_enum.train},
                {"data", trainDatasetPath},
                {"valid", validationDatasetPathIfAny??""},
                {"output_model", ""}, //this will be set below
                {"input_model", ""},
                {"prediction_result", ""},
                {"header", true},
                {"save_binary", false},
            });
            LightGbmSample.Set("output_model", ModelPath);
            Log.Info($"Training model {ModelName} with training dataset {Path.GetFileNameWithoutExtension(trainDatasetPath)}");
            LightGbmSample.Save(ModelConfigPath);
            Utils.Launch(WorkingDirectory, ExePath, "config=" + ModelConfigPath, Log);
        }
        private CpuTensor<float> Predict(string predictionDatasetPath)
        {
            if (!File.Exists(ModelPath))
            {
                throw new Exception($"missing model {ModelPath} for inference");
            }
            LastDatasetPathUsedForPrediction = predictionDatasetPath;
            const LightGBMSample.task_enum task = LightGBMSample.task_enum.predict;
            var predictionResultPath = Path.Combine(TempPath, ModelName + "_predict_" + Path.GetFileNameWithoutExtension(predictionDatasetPath) + ".txt");
            var configFilePath = predictionResultPath.Replace(".txt", ".conf");
            LightGbmSample.Set(new Dictionary<string, object> {
                {"task", task},
                {"data", predictionDatasetPath},
                {"input_model", ModelPath},
                {"prediction_result", predictionResultPath},
            });
            LightGbmSample.Save(configFilePath);
            Utils.Launch(WorkingDirectory, ExePath, "config=" + configFilePath, Log);
            var predictions = File.ReadAllLines(predictionResultPath).Select(float.Parse).ToArray();
            File.Delete(configFilePath);
            File.Delete(predictionResultPath);
            return new CpuTensor<float>(new[] { predictions.Length, 1 }, predictions);
        }
        private static string ExePath => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SharpNet", "bin", "lightgbm.exe");
        private string RootDatasetPath => Path.Combine(WorkingDirectory, "Dataset");
        private string TempPath => Path.Combine(WorkingDirectory, "Temp");
        private string DatasetPath(IDataSet dataset, bool addTargetColumnAsFirstColumn) => DatasetPath(dataset, addTargetColumnAsFirstColumn, RootDatasetPath);
    }
}
