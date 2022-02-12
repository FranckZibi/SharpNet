using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.LightGBM
{
    public class LightGBMModel : AbstractModel
    {
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
            const LightGBMSample.task_enum task = LightGBMSample.task_enum.train;
            string trainDatasetPath = DatasetPath(trainDataset, task);
            Save_in_LightGBM_format(trainDataset, trainDatasetPath, task);

            string validationDatasetPath = "";
            if (validationDatasetIfAny != null)
            {
                validationDatasetPath = DatasetPath(validationDatasetIfAny, task);
                Save_in_LightGBM_format(validationDatasetIfAny, validationDatasetPath, task);
            }
            Fit(trainDatasetPath, validationDatasetPath);
        }
        public void Fit([NotNull] string trainDatasetPath, [CanBeNull] string validationDatasetPathIfAny)
        {
            const LightGBMSample.task_enum task = LightGBMSample.task_enum.train;
            LightGbmSample.Set(new Dictionary<string, object> {
                {"task", task},
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
            Launch(ModelConfigPath);
        }

       // ReSharper disable once MemberCanBePrivate.Global
        public override CpuTensor<float> Predict(IDataSet dataset)
        {
            const LightGBMSample.task_enum task = LightGBMSample.task_enum.predict;
            string predictionDatasetPath = DatasetPath(dataset, task);
            Save_in_LightGBM_format(dataset, predictionDatasetPath, task);
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
        public CpuTensor<float> Predict(string predictionDatasetPath)
        {
            if (!File.Exists(ModelPath))
            {
                throw new Exception($"missing model {ModelPath} for inference");
            }
            const LightGBMSample.task_enum task = LightGBMSample.task_enum.predict;
            var predictionResultPath = Path.Combine(TempPath, ModelName + "_predict_" + Path.GetFileNameWithoutExtension(predictionDatasetPath) + ".txt");
            var configFilePath = predictionResultPath.Replace(".txt", ".conf");
            LightGbmSample.Set(new Dictionary<string, object> {
                {"task", task},
                {"data", predictionDatasetPath},
                {"input_model", ModelPath},
                {"prediction_result", predictionResultPath},
            });
            Launch(configFilePath);
            var predictions = File.ReadAllLines(predictionResultPath).Select(float.Parse).ToArray();
            File.Delete(configFilePath);
            File.Delete(predictionResultPath);
            return new CpuTensor<float>(new[] { predictions.Length, 1 }, predictions);
        }

        // ReSharper disable once MemberCanBePrivate.Global
        // ReSharper disable once UnusedMember.Global
        //public double ComputeAccuracy(Tensor y_true, Tensor y_pred)
        //{
        //    using var buffer = new CpuTensor<float>(new[] { y_true.Shape[0] });
        //    return y_true.ComputeAccuracy(y_pred, NetworkConfig.LossFunctionEnum.BinaryCrossentropy, buffer);
        //}
        //// ReSharper disable once MemberCanBePrivate.Global
        public override float ComputeScore(CpuTensor<float> y_true, CpuTensor<float> y_pred)
        {
            return (float)ComputeRmse(y_true, y_pred); //?D TODO
        }

        public static string DatasetPath(IDataSet dataset, LightGBMSample.task_enum task, string rootDatasetPath) => Path.Combine(rootDatasetPath, ComputeUniqueDatasetName(dataset, task) + ".csv");
        /// <summary>
        /// save the dataset in path 'path' in 'LightGBM' format.
        /// if task == train or refit:
        ///     first column is the label 'y' (to predict)
        ///     all other columns are the features
        /// else
        ///     save only 'x' (feature) tensor
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="path">the path where to save the dataset</param>
        /// <param name="task">the task to perform</param>
        /// <param name="overwriteIfExists">overwrite the file if it already exists</param>
        public static void Save_in_LightGBM_format([NotNull] IDataSet dataset, [NotNull] string path, LightGBMSample.task_enum task, bool overwriteIfExists = false)
        {
            if (File.Exists(path) && !overwriteIfExists)
            {
                Log.Debug($"No need to save dataset {dataset.Name} in path {path} : it already exists");
                return;
            }
            Log.Debug($"Saving dataset {dataset.Name} in path {path} for task {task}");
            if (dataset.X_if_available == null)
            {
                throw new NotImplementedException($"Save only works if X_if_available or not null");
            }
            var X = dataset.X_if_available;
            Debug.Assert(dataset.FeatureNamesIfAny.Length == X.Shape[1]);
            Debug.Assert(X.Shape.Length == 2);
            const char separator = ',';
            var sb = new StringBuilder();
            if (ShouldSaveLabel(task))
            {
                sb.Append("y" + separator);
            }
            sb.Append(string.Join(separator, dataset.FeatureNamesIfAny) + Environment.NewLine);
            var xDataAsSpan = X.AsFloatCpuSpan;
            // ReSharper disable once PossibleNullReferenceException
            var yDataAsSpan = ShouldSaveLabel(task) ? dataset.Y.AsFloatCpuSpan : null;

            for (int row = 0; row < dataset.Count; ++row)
            {
                if (ShouldSaveLabel(task))
                {
                    sb.Append(yDataAsSpan[row].ToString(CultureInfo.InvariantCulture) + separator);
                }
                for (int featureId = 0; featureId < X.Shape[1]; ++featureId)
                {
                    sb.Append(xDataAsSpan[row * X.Shape[1] + featureId].ToString(CultureInfo.InvariantCulture));
                    if (featureId == X.Shape[1] - 1)
                    {
                        sb.Append(Environment.NewLine);
                    }
                    else
                    {
                        sb.Append(separator);
                    }
                }
            }
            File.WriteAllText(path, sb.ToString());
            Log.Debug($"Dataset {dataset.Name} saved in path {path} for task {task}");
        }

        protected override List<string> ModelFiles()
        {
            return new List<string> { ModelPath };
        }



        private static string ExePath => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SharpNet", "bin", "lightgbm.exe");
        private string RootDatasetPath => Path.Combine(WorkingDirectory, "Dataset");
        private string TempPath => Path.Combine(WorkingDirectory, "Temp");
        private string DatasetPath(IDataSet dataset, LightGBMSample.task_enum task) => DatasetPath(dataset, task, RootDatasetPath);
        private static bool ShouldSaveLabel(LightGBMSample.task_enum task)
        {
            return task == LightGBMSample.task_enum.train || task == LightGBMSample.task_enum.refit;
        }
        private static string ComputeUniqueDatasetName(IDataSet dataset, LightGBMSample.task_enum task)
        {
            if (dataset.X_if_available == null)
            {
                throw new NotImplementedException();
            }
            var desc = ComputeDescription(dataset.X_if_available);
            if (ShouldSaveLabel(task))
            {
                desc += '_' + ComputeDescription(dataset.Y);
            }
            return dataset.Name + '_'+ Utils.ComputeHash(desc, 10);
        }
        private static string ComputeDescription(Tensor tensor)
        {
            Debug.Assert(tensor.Shape.Length == 2);
            var xDataSpan = tensor.AsReadonlyFloatCpuContent;
            var desc = string.Join('_', tensor.Shape);
            for (int col = 0; col < tensor.Shape[1]; ++col)
            {
                int row = ((tensor.Shape[0] - 1) * col) / Math.Max(1, tensor.Shape[1] - 1);
                var val = xDataSpan[row * tensor.Shape[1] + col];
                desc += '_' + Math.Round(val, 6).ToString(CultureInfo.InvariantCulture);
            }
            return desc;
        }
        private void Launch(string configFilePath)
        {
            LightGbmSample.Save(configFilePath);
            var errorDataReceived = "";
            var psi = new ProcessStartInfo(ExePath)
            {
                WorkingDirectory = WorkingDirectory,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                Arguments = "config=" + configFilePath,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden
            };
            var process = Process.Start(psi);
            if (process == null)
            {
                const string errorMsg = "Fail to start LightGBM Engine";
                Log.Fatal(errorMsg);
                throw new Exception(errorMsg);
            }
            process.ErrorDataReceived += (_, e) =>
            {
                if (e.Data != null)
                {
                    errorDataReceived = e.Data;
                }
            };
            process.OutputDataReceived += (_, e) =>
            {
                if (string.IsNullOrEmpty(e.Data))
                {
                    return;
                }
                if (e.Data.Contains("[Warning] "))
                {
                    Log.Debug(e.Data.Replace("[Warning] ", ""));
                }
                else if (e.Data.Contains("[Info] "))
                {
                    Log.Info(e.Data.Replace("[Info] ", ""));
                }
                else 
                {
                    Log.Info(e.Data);
                }
            };
            process.BeginErrorReadLine();
            process.BeginOutputReadLine();
            process.WaitForExit();
            if (!string.IsNullOrEmpty(errorDataReceived)|| process.ExitCode != 0)
            {
                var errorMsg = "Error in LightGBM " + errorDataReceived;
                Log.Fatal(errorMsg);
                throw new Exception(errorMsg);
            }
        }
        private static double ComputeRmse(CpuTensor<float> y_true, CpuTensor<float> y_pred)
        {
            using var buffer = new CpuTensor<float>(new[] { y_true.Shape[0] });
            return Math.Sqrt(y_true.ComputeMse(y_pred, buffer));
        }
    }
}
