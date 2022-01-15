using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using log4net;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.Networks;

namespace SharpNet.LightGBM
{
    public class LightGBMModel
    {
        #region private fields & properties
        private readonly TensorMemoryPool _memoryPool = new TensorMemoryPool(null);
        private readonly Parameters _parameters;
        /// <summary>
        /// parameters that must be present in the config file, even if they have the default value
        /// </summary>
        private static readonly HashSet<string> MandatoryParametersInConfigFile = new HashSet<string> { "objective", "task" };
        private static string DefaultLogDirectory => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SharpNet", "CFM63");
        private readonly string _workingDirectory;
        private readonly string _modelName;
        private Tensor _buffer;
        private string _modelPath;
        #endregion

        #region public fields & properties
        /// <summary>
        /// path of a trained model.
        /// null if no trained model is available
        /// </summary>
        public static readonly ILog Log = LogManager.GetLogger(typeof(LightGBMModel));
        #endregion

        #region constructor
        public LightGBMModel(Parameters p, string modelPrefix)
        {
            _workingDirectory = DefaultLogDirectory;
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
            _parameters = p.Clone();
            var modelHash = _parameters.ComputeHash();
            _modelName = "model_" + modelPrefix + (string.IsNullOrEmpty(modelPrefix) ? "" : "_") + modelHash;
            Utils.ConfigureGlobalLog4netProperties(DefaultLogDirectory, _modelName);
            Utils.ConfigureThreadLog4netProperties(DefaultLogDirectory, _modelName);
        }
        #endregion

        public void Train(IDataSet trainDataset, IDataSet validationDataset = null)
        {
            const Parameters.task_enum task = Parameters.task_enum.train;
            var trainDatasetPath = DatasetPath(trainDataset, task);
            Save(trainDataset, trainDatasetPath, task);
            _modelPath = Path.Combine(_workingDirectory, _modelName + "_" + Path.GetFileNameWithoutExtension(trainDatasetPath) + ".txt");

            ClassFieldSetter.Set(_parameters, new Dictionary<string, object> {
                {"task", task},
                {"data", trainDatasetPath},
                {"valid", ""},
                {"output_model", _modelPath},
                {"input_model", ""},
                {"prediction_result", ""},
                {"header", true},
                {"save_binary", false},
                {"is_provide_training_metric", true},
                });
            if (validationDataset != null)
            {
                var validationDatasetPath = DatasetPath(validationDataset, task);
                Save(validationDataset, validationDatasetPath, task);
                ClassFieldSetter.Set(_parameters, "valid", validationDatasetPath);
            }
            var configFilePath = Path.Combine(TempPath, Path.GetFileNameWithoutExtension(_modelPath) + ".conf");
            Launch(configFilePath);
        }
        public CpuTensor<float> Predict(IDataSet dataset)
        {
            if (string.IsNullOrEmpty(_modelPath))
            {
                throw new Exception("missing model for inference");
            }
            const Parameters.task_enum task = Parameters.task_enum.predict;
            var predictionDatasetPath = DatasetPath(dataset, task);
            var predictionResultPath = PredictionTempResultPath(dataset);
            var configFilePath = predictionResultPath.Replace(".txt", ".conf");

            Save(dataset, predictionDatasetPath, task);

            ClassFieldSetter.Set(_parameters, new Dictionary<string, object> {
                {"task", task},
                {"data", predictionDatasetPath},
                {"input_model", _modelPath},
                {"prediction_result", predictionResultPath},
                });
            Launch(configFilePath);
            var predictions = File.ReadAllLines(predictionResultPath).Select(float.Parse).ToArray();
            if (predictions.Length != dataset.Count)
            {
                throw new Exception($"Invalid number of predictions, received {predictions.Length} but expected {dataset.Count}");
            }
            //File.Delete(configFilePath);
            //File.Delete(predictionResultPath);
            return new CpuTensor<float>(new[] { predictions.Length, 1 }, predictions);
        }
        public double ComputeAccuracy(Tensor y_true, Tensor y_predicted)
        {
            _memoryPool.GetFloatTensor(ref _buffer, new[] { y_true.Shape[0] });
            return y_true.ComputeAccuracy(y_predicted, NetworkConfig.LossFunctionEnum.BinaryCrossentropy, _buffer);
        }

        #region private methods
        private string PredictionTempResultPath(IDataSet dataset)
        {
            if (string.IsNullOrEmpty(_modelPath))
            {
                throw new Exception("missing model");
            }
            return Path.Combine(TempPath,
                Path.GetFileNameWithoutExtension(_modelPath)
                + "_predict_"
                + ComputeUniqueDatasetName(dataset, Parameters.task_enum.predict)
                + ".txt");
        }
        private static string ExePath => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SharpNet", "bin", "lightgbm.exe");
        private string RootDatasetPath => Path.Combine(_workingDirectory, "Dataset");
        private string TempPath => Path.Combine(_workingDirectory, "Temp");
        private string DatasetPath(IDataSet dataset, Parameters.task_enum task) => Path.Combine(RootDatasetPath, ComputeUniqueDatasetName(dataset, task) + ".csv");
        private static bool ShouldSaveLabel(Parameters.task_enum task)
        {
            return task == Parameters.task_enum.train || task == Parameters.task_enum.refit;
        }
        private static string ComputeUniqueDatasetName(IDataSet dataset, Parameters.task_enum task)
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
        /// <param name="overwriteIfExists">overwrite the file if ti already exists</param>
        private static void Save(IDataSet dataset, string path, Parameters.task_enum task, bool overwriteIfExists = false)
        {
            if (File.Exists(path) && !overwriteIfExists)
            {
                Log.Debug($"No need to save dataset {dataset.Name} in path {path} : it already exists");
                return;
            }
            Log.Debug($"Saving dataset {dataset.Name} in path {path} for task {task}");
            if (dataset.X_if_available == null)
            {
                throw new NotImplementedException($"Save only works if X_if_available os not null");
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
            var yDataAsSpan = dataset.Y.AsFloatCpuSpan;

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
        private void Launch(string configFilePath)
        {
            _parameters.Save(configFilePath, true, MandatoryParametersInConfigFile);
            var errorDataReceived = "";
            var psi = new ProcessStartInfo(ExePath)
            {
                WorkingDirectory = _workingDirectory,
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
            process.ErrorDataReceived += (s, e) =>
            {
                if (e.Data != null)
                {
                    errorDataReceived = e.Data;
                }
            };
            process.OutputDataReceived += (s, e) =>
            {
                if (string.IsNullOrEmpty(e.Data))
                {
                    return;
                }
                if (e.Data.Contains("[Warning] "))
                {
                    Log.Warn(e.Data.Replace("[Warning] ", ""));
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
        #endregion
    }
}
