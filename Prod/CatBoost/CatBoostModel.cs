﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using JetBrains.Annotations;
using SharpNet.CPU;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.CatBoost
{
    public class CatBoostModel : AbstractModel
    {
        #region prrivate fields & properties
        private const char separator = ',';
        private static readonly object LockToColumnDescription = new();
        #endregion

        #region public fields & properties
        [NotNull] public string ModelPath => Path.Combine(WorkingDirectory, ModelName+".json");
        [NotNull] public string ModelConfigPath => ISample.ToJsonPath(WorkingDirectory, ModelName);
        public override string WorkingDirectory { get; }
        public override string ModelName { get; }
        public CatBoostSample CatBoostSample => (CatBoostSample)Sample;
        #endregion

        #region constructor
        /// <summary>
        /// 
        /// </summary>
        /// <param name="CatBoostSample"></param>
        /// <param name="workingDirectory"></param>
        /// <param name="modelName">the name of the model to use</param>
        /// <exception cref="Exception"></exception>
        public CatBoostModel(CatBoostSample CatBoostSample, string workingDirectory, [NotNull] string modelName): base(CatBoostSample)
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

            string datasetColumnDescriptionPath = trainDatasetPath + ".co";
            to_column_description(datasetColumnDescriptionPath, trainDataset, trainDataset.CategoricalFeatures, true, false);

            Log.Info($"Training model {ModelName} with training dataset {Path.GetFileNameWithoutExtension(trainDatasetPath)}");

            string arguments = "fit " +
                               " --learn-set " + trainDatasetPath +
                               " --delimiter=\"" + separator + "\"" +
                               " --has-header" +
                               " --params-file " + ModelConfigPath +
                               " --column-description " + datasetColumnDescriptionPath +
                               " --allow-writing-files false " + //to disable the creation of tmp files
                               " --model-file " + ModelPath;

            if (!string.IsNullOrEmpty(validationDatasetPath))
            {
                arguments += " --test-set " + validationDatasetPath;
            }
            else
            {
                CatBoostSample.use_best_model = false;
            }

            CatBoostSample.Save(ModelConfigPath);


            //if (CatBoostSample.allow_writing_files)
            //{
            //    var tmpDirectory = Path.Combine(WorkingDirectory, ModelName + "_temp");
            //    Directory.CreateDirectory(tmpDirectory);
            //    arguments +=
            //        " --learn-err-log " + Path.Combine(tmpDirectory, "learn_error.tsv") +
            //        "  --test-err-log " + Path.Combine(tmpDirectory, "test_error.tsv") +
            //        " --json-log " + Path.Combine(tmpDirectory, "catboost_training.json") +
            //        " --profile-log " + Path.Combine(tmpDirectory, "profile-log") +
            //        " --trace-log " + Path.Combine(tmpDirectory, "trace.log");
            //}

            Utils.Launch(WorkingDirectory, ExePath, arguments, Log);
        }
        public override CpuTensor<float> Predict(IDataSet dataset)
        {
            const bool targetColumnIsFirstColumn = true;
            string predictionDatasetPath = DatasetPath(dataset, targetColumnIsFirstColumn);
            dataset.to_csv(predictionDatasetPath, separator, targetColumnIsFirstColumn, false);

            string datasetColumnDescriptionPath = predictionDatasetPath + ".co";
            to_column_description(datasetColumnDescriptionPath, dataset, dataset.CategoricalFeatures, targetColumnIsFirstColumn, true);


            if (!File.Exists(ModelPath))
            {
                throw new Exception($"missing model {ModelPath} for inference");
            }
            var predictionResultPath = Path.Combine(TempPath, ModelName + "_predict_" + Path.GetFileNameWithoutExtension(predictionDatasetPath) + ".tsv");
            var configFilePath = predictionResultPath.Replace(".txt", "_conf.json");
            CatBoostSample.Save(configFilePath);

            string modelFormat = ModelPath.EndsWith("json") ? "json" : "CatboostBinary";
            var arguments = "calc " +
                            " --input-path " + predictionDatasetPath +
                            " --output-path " + predictionResultPath +
                            " --delimiter=\"" + separator + "\"" +
                            " --has-header" +
                            " --column-description " + datasetColumnDescriptionPath +
                            " --model-file " + ModelPath +
                            " --model-format " + modelFormat
                            
                            ;

            //+ " --prediction-type Probability,Class,RawFormulaVal,Exponent,LogProbability "
            if (dataset.Objective == Objective_enum.Classification)
            {
                arguments += " --prediction-type Probability ";
            }
            else
            {
                arguments += " --prediction-type RawFormulaVal ";
            }

            Utils.Launch(WorkingDirectory, ExePath, arguments, Log);
            var predictions1 = File.ReadAllLines(predictionResultPath).Skip(1).Select(l => l.Split()[1]).Select(float.Parse).ToArray();
            File.Delete(configFilePath);
            File.Delete(predictionResultPath);
            var predictions = new CpuTensor<float>(new[] { predictions1.Length, 1 }, predictions1);
            if (predictions.Shape[0]!= dataset.Count)
            {
                throw new Exception($"Invalid number of predictions, received {predictions.Shape[0]} but expected {dataset.Count}");
            }
            return predictions;
        }
        public override void Save(string workingDirectory, string modelName)
        {
            //No need to save model : it is already saved in json format
            CatBoostSample.Save(workingDirectory, modelName);
        }
        protected override List<string> ModelFiles()
        {
            return new List<string> { ModelPath };
        }

        private static void to_column_description([NotNull] string path, IDataSet dataset, IList<string> categoricalColumns, bool hasTargetColumnAsFirstColumn, bool overwriteIfExists = false)
        {
            if (File.Exists(path) && !overwriteIfExists)
            {
                Log.Debug($"No need to save dataset column description in path {path} : it already exists");
                return;
            }
            var sb = new StringBuilder();
            var datasetFeatureNamesIfAny = dataset.FeatureNamesIfAny;
            for (int featureId = 0; featureId < datasetFeatureNamesIfAny.Length; ++featureId)
            {
                var featureName = datasetFeatureNamesIfAny[featureId];
                if (featureId == 0 && hasTargetColumnAsFirstColumn)
                {
                    //the first column contains the target
                    sb.Append($"{featureId}\tLabel" + Environment.NewLine);
                }
                else if (categoricalColumns.Contains(featureName))
                {
                    sb.Append($"{featureId}\tCateg"+Environment.NewLine);
                }
            }
            Log.Debug($"Saving dataset column description in path {path}");
            var fileContent = sb.ToString().Trim();
            lock (LockToColumnDescription)
            {
                File.WriteAllText(path, fileContent);
            }
            Log.Debug($"Dataset column description saved in path {path}");
        }
        private static string ExePath => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SharpNet", "bin", "catboost.exe");

        private string RootDatasetPath => Path.Combine(WorkingDirectory, "Dataset");
        private string TempPath => Path.Combine(WorkingDirectory, "Temp");
        private string DatasetPath(IDataSet dataset, bool addTargetColumnAsFirstColumn) => DatasetPath(dataset, addTargetColumnAsFirstColumn, RootDatasetPath);
    }
}
