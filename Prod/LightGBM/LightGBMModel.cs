using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using JetBrains.Annotations;
using log4net;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Networks;

namespace SharpNet.LightGBM
{
    public class LightGBMModel
    {
        #region private fields & properties
        private readonly Parameters _parameters;
        private readonly string _workingDirectory;
        /// <summary>
        /// path of a trained model.
        /// null if no trained model is available
        /// </summary>
        [NotNull] private string ModelPath => Path.Combine(_workingDirectory, _modelName+".txt");
        [NotNull] private string ModelConfigPath => ISample.ToPath(_workingDirectory, _modelName);
        private readonly string _modelName;
        private static readonly object LockUpdateFileObject = new();
        #endregion

        #region public fields & properties
        public static readonly ILog Log = LogManager.GetLogger(typeof(LightGBMModel));
        #endregion
       
        #region constructor
        /// <summary>
        /// 
        /// </summary>
        /// <param name="p"></param>
        /// <param name="workingDirectory"></param>
        /// <param name="mandatoryModelNameIfAny">
        /// the name of the model to use if any.
        ///  if null or empty, we'll compute the new model name based on the hash of the associated parameters
        /// </param>
        /// <exception cref="Exception"></exception>
        public LightGBMModel(Parameters p, string workingDirectory, string mandatoryModelNameIfAny = "")
        {
            _workingDirectory = workingDirectory;
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
            _parameters = p;
            _modelName = string.IsNullOrEmpty(mandatoryModelNameIfAny) ?
                _parameters.ComputeHash():
                mandatoryModelNameIfAny;
        }
        #endregion

        public void Train(IDataSet trainDataset, IDataSet validationDataset = null)
        {
            const Parameters.task_enum task = Parameters.task_enum.train;
            string trainDatasetPath = DatasetPath(trainDataset, task);
            Save(trainDataset, trainDatasetPath, task);

            string validationDatasetPath = "";
            if (validationDataset != null)
            {
                validationDatasetPath = DatasetPath(validationDataset, task);
                Save(validationDataset, validationDatasetPath, task);
            }
            Train(trainDatasetPath, validationDatasetPath);
        }
        public void Train(string trainDatasetPath, string validationDatasetPath = "")
        {
            const Parameters.task_enum task = Parameters.task_enum.train;
            _parameters.Set(new Dictionary<string, object> {
                {"task", task},
                {"data", trainDatasetPath},
                {"valid", validationDatasetPath},
                {"output_model", ""}, //this will be set below
                {"input_model", ""},
                {"prediction_result", ""},
                {"header", true},
                {"save_binary", false},
            });
            _parameters.Set("output_model", ModelPath);
            Log.Info($"Training model {_modelName} with training dataset {Path.GetFileNameWithoutExtension(trainDatasetPath)}");
            Launch(ModelConfigPath);
        }

        /// <param name="savePredictions"></param>
        /// <param name="UnNormalizeYIfNeeded"></param>
        /// <param name="shouldSavePredictionsBasedOnValidationScore"></param>
        /// <param name="trainingTimeInSeconds"></param>
        /// <param name="totalParams"></param>
        /// <param name="trainDataset"></param>
        /// <param name="validationDataset"></param>
        /// <param name="testDataset"></param>
        /// <returns>the cost associated with the model</returns>
        public float CreateModelResults(Action<CpuTensor<float>, string> savePredictions, Func<CpuTensor<float>, CpuTensor<float>> UnNormalizeYIfNeeded, Func<double,bool> shouldSavePredictionsBasedOnValidationScore,double trainingTimeInSeconds, int totalParams, IDataSet trainDataset, IDataSet validationDataset = null, IDataSet testDataset = null)
        {
            Log.Debug($"Computing Model '{_modelName}' predictions for Training Dataset");
            var trainPredictions = UnNormalizeYIfNeeded(Predict(trainDataset));
            Log.Debug("Computing Model score on Training");
            var scoreTrain = ComputeRmse(UnNormalizeYIfNeeded(trainDataset.Y), trainPredictions);
            Log.Debug($"Model '{_modelName}' score on training: {scoreTrain}");

            bool shouldSavePredictionsAndModelBasedOnValidationScore = true;
            float scoreValidation = float.NaN;
            CpuTensor<float> validationPredictions = null;
            if (validationDataset != null)
            {
                Log.Debug($"Computing Model '{_modelName}' predictions for Validation Dataset");
                validationPredictions = UnNormalizeYIfNeeded(Predict(validationDataset));
                Log.Debug($"Computing Model '{_modelName}' score on Validation");
                scoreValidation = (float)ComputeRmse(UnNormalizeYIfNeeded(validationDataset.Y), validationPredictions);
                Log.Debug($"Model '{_modelName}' score on Validation: {scoreValidation}");
                if (!shouldSavePredictionsBasedOnValidationScore(scoreValidation))
                {
                    Log.Debug($"No need to save Model '{_modelName}' predictions because of a validation score {scoreValidation}");
                    shouldSavePredictionsAndModelBasedOnValidationScore = false;
                }
            }

            if (shouldSavePredictionsAndModelBasedOnValidationScore)
            {
                var trainPredictionsFileName = _modelName + "_predict_train_" + Math.Round(scoreTrain, 5) + ".csv";
                Log.Info($"Saving Model '{_modelName}' predictions for Training Dataset");
                savePredictions(trainPredictions, Path.Combine(_workingDirectory, trainPredictionsFileName));

                if (!float.IsNaN(scoreValidation))
                {
                    Log.Info($"Saving Model '{_modelName}' predictions for Validation Dataset");
                    var validationPredictionsFileName = _modelName
                                                        + "_predict_valid_" + Math.Round(scoreValidation, 5)
                                                        + ".csv";
                    savePredictions(validationPredictions,  Path.Combine(_workingDirectory, validationPredictionsFileName));
                }

                if (testDataset != null)
                {
                    Log.Debug($"Computing Model '{_modelName}' predictions for Test Dataset");
                    var testPredictions = Predict(testDataset);
                    Log.Info("Saving predictions for Test Dataset");
                    var testPredictionsFileName = _modelName
                                                  + "_predict_test_.csv";
                    savePredictions(testPredictions, Path.Combine(_workingDirectory, testPredictionsFileName));
                }
            }
            else
            {
                Log.Debug($"Removing model '{_modelName}' files: {Path.GetFileName(ModelPath)} and {Path.GetFileName(ModelConfigPath)} because of low score ({scoreValidation})");
                File.Delete(ModelPath);
                File.Delete(ModelConfigPath);
            }

            string line = "";
            
            try
            {
                int numEpochs = _parameters.num_iterations;
                //We save the results of the net
                line = DateTime.Now.ToString("F", CultureInfo.InvariantCulture) + ";"
                    + _modelName.Replace(';', '_') + ";"
                    + _parameters.DeviceName() + ";"
                    + totalParams + ";"
                    + numEpochs + ";"
                    + "-1" + ";"
                    + _parameters.learning_rate + ";"
                    + trainingTimeInSeconds + ";"
                    + (trainingTimeInSeconds / numEpochs) + ";"
                    + scoreTrain + ";"
                    + "NaN" + ";"
                    + scoreValidation + ";"
                    + "NaN" + ";"
                    + Environment.NewLine;
                var testsCsv = string.IsNullOrEmpty(trainDataset.Name) ? "Tests.csv" : ("Tests_" + trainDataset.Name + ".csv");
                lock (LockUpdateFileObject)
                {
                    File.AppendAllText(Utils.ConcatenatePathWithFileName(_workingDirectory, testsCsv), line);
                }
            }
            catch (Exception e)
            {
                Log.Error("fail to add line in file:" + Environment.NewLine + line + Environment.NewLine + e);
            }

            return scoreValidation;
        }

        // ReSharper disable once MemberCanBePrivate.Global
        public CpuTensor<float> Predict(IDataSet dataset)
        {
            const Parameters.task_enum task = Parameters.task_enum.predict;
            string predictionDatasetPath = DatasetPath(dataset, task);
            Save(dataset, predictionDatasetPath, task);
            var predictions = Predict(predictionDatasetPath);
            if (predictions.Shape[0]!= dataset.Count)
            {
                throw new Exception($"Invalid number of predictions, received {predictions.Shape[0]} but expected {dataset.Count}");
            }
            return predictions;
        }
        public CpuTensor<float> Predict(string predictionDatasetPath)
        {
            if (!File.Exists(ModelPath))
            {
                throw new Exception($"missing model {ModelPath} for inference");
            }
            const Parameters.task_enum task = Parameters.task_enum.predict;
            var predictionResultPath = Path.Combine(TempPath, _modelName + "_predict_" + Path.GetFileNameWithoutExtension(predictionDatasetPath) + ".txt");
            var configFilePath = predictionResultPath.Replace(".txt", ".conf");
            _parameters.Set(new Dictionary<string, object> {
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
        public double ComputeAccuracy(Tensor y_true, Tensor y_predicted)
        {
            using var buffer = new CpuTensor<float>(new[] { y_true.Shape[0] });
            return y_true.ComputeAccuracy(y_predicted, NetworkConfig.LossFunctionEnum.BinaryCrossentropy, buffer);
        }
        // ReSharper disable once MemberCanBePrivate.Global
        public double ComputeRmse(CpuTensor<float> y_true, CpuTensor<float> y_predicted)
        {
            using var buffer = new CpuTensor<float>(new[] { y_true.Shape[0] });
            return Math.Sqrt(y_true.ComputeMse(y_predicted, buffer));
        }
        public static string DatasetPath(IDataSet dataset, Parameters.task_enum task, string rootDatasetPath) => Path.Combine(rootDatasetPath, ComputeUniqueDatasetName(dataset, task) + ".csv");
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
        public static void Save([NotNull] IDataSet dataset, [NotNull] string path, Parameters.task_enum task, bool overwriteIfExists = false)
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

        private static string ExePath => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SharpNet", "bin", "lightgbm.exe");
        private string RootDatasetPath => Path.Combine(_workingDirectory, "Dataset");
        private string TempPath => Path.Combine(_workingDirectory, "Temp");
        private string DatasetPath(IDataSet dataset, Parameters.task_enum task) => DatasetPath(dataset, task, RootDatasetPath);
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
        private void Launch(string configFilePath)
        {
            _parameters.Save(configFilePath);
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
    }
}
