using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Text;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.CatBoost
{
    public class CatBoostModel : Model
    {
        #region prrivate fields & properties
        private static readonly object LockToColumnDescription = new();
        #endregion
        #region constructor
        /// <summary>
        /// 
        /// </summary>
        /// <param name="catBoostModelSample"></param>
        /// <param name="workingDirectory"></param>
        /// <param name="modelName">the name of the model to use</param>
        /// <exception cref="Exception"></exception>
        [SuppressMessage("ReSharper", "VirtualMemberCallInConstructor")]
        public CatBoostModel(CatBoostSample catBoostModelSample, string workingDirectory, [JetBrains.Annotations.NotNull] string modelName): base(catBoostModelSample, workingDirectory, modelName)
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

        public override (string train_XDatasetPath_InModelFormat, string train_YDatasetPath_InModelFormat, string train_XYDatasetPath_InModelFormat, string validation_XDatasetPath_InModelFormat, string validation_YDatasetPath_InModelFormat, string validation_XYDatasetPath_InModelFormat) 
            Fit(DataSet trainDataset, DataSet validationDatasetIfAny)
        {
            const bool addTargetColumnAsFirstColumn = true;
            const bool includeIdColumns = false;
            const bool overwriteIfExists = false;
            string trainDatasetPath_InModelFormat = trainDataset.to_csv_in_directory(RootDatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            char separator = trainDataset.Separator;

            string validationDatasetPathIfAny_InModelFormat = "";
            if (validationDatasetIfAny != null)
            {
                validationDatasetPathIfAny_InModelFormat = validationDatasetIfAny.to_csv_in_directory(RootDatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            }

            string datasetColumnDescriptionPath = trainDatasetPath_InModelFormat + ".co";
            to_column_description(datasetColumnDescriptionPath, trainDataset, addTargetColumnAsFirstColumn, false);

            var logMsg = $"Training model '{ModelName}' with training dataset {Path.GetFileNameWithoutExtension(trainDatasetPath_InModelFormat)}";
            if (LoggingForModelShouldBeDebug(ModelName))
            {
                LogDebug(logMsg);
            }
            else
            {
                LogInfo(logMsg);
            }


            var tempModelSamplePath = ISample.ToJsonPath(TempPath, ModelName);
            string arguments = "fit " +
                               " --learn-set " + trainDatasetPath_InModelFormat +
                               " --delimiter=\"" + separator + "\"" +
                               " --has-header" +
                               " --params-file " + tempModelSamplePath +
                               " --column-description " + datasetColumnDescriptionPath +
                               " --allow-writing-files false " + //to disable the creation of tmp files
                               " --model-file " + ModelPath +
                               " --logging-level Silent "+
                               " --verbose false "
                             ;

            if (!string.IsNullOrEmpty(validationDatasetPathIfAny_InModelFormat))
            {
                arguments += " --test-set " + validationDatasetPathIfAny_InModelFormat;
            }
            else
            {
                CatBoostSample.use_best_model = false;
            }

            CatBoostSample.Save(tempModelSamplePath);


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
            return (null, null, trainDatasetPath_InModelFormat, null, null, validationDatasetPathIfAny_InModelFormat);
        }

        public override (DataFrame,string) PredictWithPath(DataSet dataset, bool removeAllTemporaryFilesAtEnd)
        {
            if (!File.Exists(ModelPath))
            {
                throw new Exception($"missing model {ModelPath} for inference");
            }
            const bool addTargetColumnAsFirstColumn = true;
            const bool includeIdColumns = false;
            const bool overwriteIfExists = false;
            string datasetPath = dataset.to_csv_in_directory(RootDatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);

            string datasetColumnDescriptionPath = datasetPath + ".co";
            to_column_description(datasetColumnDescriptionPath, dataset, addTargetColumnAsFirstColumn, true);


            var predictionResultPath = Path.Combine(TempPath, ModelName + "_predict_" + Path.GetFileNameWithoutExtension(datasetPath) + ".tsv");
            var configFilePath = predictionResultPath.Replace(".txt", "_conf.json");
            CatBoostSample.Save(configFilePath);

            string modelFormat = ModelPath.EndsWith("json") ? "json" : "CatboostBinary";
            var arguments = "calc " +
                            " --input-path " + datasetPath +
                            " --output-path " + predictionResultPath +
                            " --delimiter=\"" + dataset.Separator + "\"" +
                            " --has-header" +
                            " --column-description " + datasetColumnDescriptionPath +
                            " --model-file " + ModelPath +
                            " --model-format " + modelFormat
                            ;

            //+ " --prediction-type Probability,Class,RawFormulaVal,Exponent,LogProbability "
            if (dataset.IsRegressionProblem)
            {
                arguments += " --prediction-type RawFormulaVal ";
            }
            else
            {
                arguments += " --prediction-type Probability ";
            }

            Utils.Launch(WorkingDirectory, ExePath, arguments, Log);
            var predictionsDf = LoadProbaFile(predictionResultPath, true, true, null, dataset);
            Utils.TryDelete(configFilePath);
            Utils.TryDelete(predictionResultPath);
            if (removeAllTemporaryFilesAtEnd)
            {
                Utils.TryDelete(datasetPath);
                datasetPath = "";
                Utils.TryDelete(datasetColumnDescriptionPath);
            }
            return (predictionsDf, datasetPath);
        }
        public override void Save(string workingDirectory, string modelName)
        {
            //No need to save model : it is already saved in json format
            CatBoostSample.Save(workingDirectory, modelName);
        }
        public override int GetNumEpochs()
        {
            return CatBoostSample.iterations;
        }
        public override string DeviceName()
        {
            return CatBoostSample.DeviceName();
        }
        public override int TotalParams()
        {
            return -1; //TODO
        }
        public override double GetLearningRate()
        {
            return CatBoostSample.learning_rate;
        }
        public override List<string> ModelFiles()
        {
            return new List<string> { ModelPath };
        }
        public override void Use_All_Available_Cores()
        {
            CatBoostSample.thread_count = Utils.CoreCount;
        }
        //public static CatBoostModel LoadTrainedCatBoostModel(string workingDirectory, string modelName)
        //{
        //    var sample = ISample.LoadSample<CatBoostSample>(workingDirectory, modelName);
        //    return new CatBoostModel(sample, workingDirectory, modelName);
        //}

        private static void to_column_description([JetBrains.Annotations.NotNull] string path, DataSet dataset, bool addTargetColumnAsFirstColumn, bool overwriteIfExists = false)
        {
            if (File.Exists(path) && !overwriteIfExists)
            {
                Log.Debug($"No need to save dataset column description in path {path} : it already exists");
                return;
            }
            var categoricalColumns = dataset.CategoricalFeatures;
            var sb = new StringBuilder();
            int nextColumnIdx = 0;
            if (addTargetColumnAsFirstColumn)
            {
                foreach(var _ in dataset.TargetLabels)
                {
                    sb.Append($"{nextColumnIdx++}\tLabel" + Environment.NewLine); //this column is the target
                }
            }
            foreach (var columnName in dataset.ColumnNames)
            {
                if (categoricalColumns.Contains(columnName))
                {
                    sb.Append($"{nextColumnIdx}\tCateg"+Environment.NewLine); //this column is a categorical feature
                }
                ++nextColumnIdx;
            }
            Log.Debug($"Saving dataset column description in path {path}");
            var fileContent = sb.ToString().Trim();
            lock (LockToColumnDescription)
            {
                File.WriteAllText(path, fileContent);
            }
            Log.Debug($"Dataset column description saved in path {path}");
        }
        private static string ExePath => Path.Combine(Utils.ChallengesPath, "bin", "catboost.exe");

        private string TempPath => Path.Combine(WorkingDirectory, "Temp");
        [JetBrains.Annotations.NotNull] private string ModelPath => Path.Combine(WorkingDirectory, ModelName + ".json");
        private CatBoostSample CatBoostSample => (CatBoostSample)ModelSample;
    }
}

