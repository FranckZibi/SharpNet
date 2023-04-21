using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Text;
using SharpNet.Datasets;
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
            if (!Directory.Exists(TempPath))
            {
                Directory.CreateDirectory(TempPath);
            }
        }
        #endregion

        public override (string train_XDatasetPath_InModelFormat, string train_YDatasetPath_InModelFormat, string train_XYDatasetPath_InModelFormat, string validation_XDatasetPath_InModelFormat, string validation_YDatasetPath_InModelFormat, string validation_XYDatasetPath_InModelFormat, IScore trainLossIfAvailable, IScore validationLossIfAvailable, IScore trainRankingMetricIfAvailable, IScore validationRankingMetricIfAvailable)
            Fit(DataSet trainDataset, DataSet validationDatasetIfAny)
        {
            var sw = Stopwatch.StartNew();
            const bool addTargetColumnAsFirstColumn = true;
            const bool includeIdColumns = false;
            const bool overwriteIfExists = false;
            string trainDatasetPath_InModelFormat = trainDataset.to_csv_in_directory(DatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            char separator = trainDataset.Separator;

            string validationDatasetPathIfAny_InModelFormat = "";
            if (validationDatasetIfAny != null)
            {
                validationDatasetPathIfAny_InModelFormat = validationDatasetIfAny.to_csv_in_directory(DatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            }

            string datasetColumnDescriptionPath = trainDatasetPath_InModelFormat + ".co";
            to_column_description(datasetColumnDescriptionPath, trainDataset, addTargetColumnAsFirstColumn, false);

            var tempModelSamplePath = CatBoostSample.ToPath(TempPath, ModelName);
            string arguments = "fit " +
                               " --learn-set " + trainDatasetPath_InModelFormat +
                               " --delimiter=\"" + separator + "\"" +
                               " --has-header" +
                               " --params-file " + tempModelSamplePath +
                               " --column-description " + datasetColumnDescriptionPath +
                               " --allow-writing-files false " + //to disable the creation of tmp files
                               " --model-file " + ModelPath
                              +" --logging-level Verbose "
                ;

            if (!string.IsNullOrEmpty(validationDatasetPathIfAny_InModelFormat))
            {
                CatBoostSample.use_best_model = true;
                arguments += " --test-set " + validationDatasetPathIfAny_InModelFormat;
            }
            else
            {
                CatBoostSample.use_best_model = false;
            }

            CatBoostSample.AddExtraMetricToComputeForTraining();
            CatBoostSample.Save(tempModelSamplePath);
            LogForModel($"Training model '{ModelName}' with training dataset '{Path.GetFileNameWithoutExtension(trainDatasetPath_InModelFormat)}'");
            var linesFromLog = Utils.Launch(WorkingDirectory, ExePath, arguments, Log, true);
            var (trainLossIfAvailable, validationLossIfAvailable, trainRankingMetricIfAvailable, validationRankingMetricIfAvailable) = CatBoostSample.ExtractScores(linesFromLog);
            LogForModel($"Model '{ModelName}' trained with dataset '{Path.GetFileNameWithoutExtension(trainDatasetPath_InModelFormat)}' in {sw.Elapsed.TotalSeconds}s (trainScore = {trainLossIfAvailable} / validationScore = {validationLossIfAvailable})");
            return (null, null, trainDatasetPath_InModelFormat, null, null, validationDatasetPathIfAny_InModelFormat,
                trainLossIfAvailable, validationLossIfAvailable, trainRankingMetricIfAvailable, validationRankingMetricIfAvailable);
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
            string datasetPath = dataset.to_csv_in_directory(DatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);

            string datasetColumnDescriptionPath = datasetPath + ".co";
            to_column_description(datasetColumnDescriptionPath, dataset, addTargetColumnAsFirstColumn, false);


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

            Utils.Launch(WorkingDirectory, ExePath, arguments, Log, false);
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
        public override double GetLearningRate()
        {
            return CatBoostSample.learning_rate;
        }
        public override List<string> AllFiles()
        {
            return new List<string> { ModelPath };
        }
        //public static CatBoostModel LoadTrainedCatBoostModel(string workingDirectory, string modelName)
        //{
        //    var sample = ISample.LoadSample<CatBoostSample>(workingDirectory, modelName);
        //    return new CatBoostModel(sample, workingDirectory, modelName);
        //}

        private static readonly object Lock_to_column_description = new();


        private static void to_column_description([JetBrains.Annotations.NotNull] string path, DataSet dataset, bool addTargetColumnAsFirstColumn, bool overwriteIfExists = false)
        {
            lock (Lock_to_column_description)
            {
                if (File.Exists(path) && !overwriteIfExists)
                {
                    //Log.Debug($"No need to save dataset column description in path {path} : it already exists");
                    return;
                }
                var categoricalColumns = dataset.CategoricalFeatures;
                var sb = new StringBuilder();
                int nextColumnIdx = 0;
                if (addTargetColumnAsFirstColumn)
                {
                    //this first column is the target
                    sb.Append($"{nextColumnIdx++}\tLabel" + Environment.NewLine);
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
        }
        private static string ExePath => Path.Combine(Utils.ChallengesPath, "bin", "catboost.exe");

        [JetBrains.Annotations.NotNull] private string ModelPath => Path.Combine(WorkingDirectory, ModelName + ".json");
        //[JetBrains.Annotations.NotNull] private string ModelPath => Path.Combine(WorkingDirectory, ModelName + ".bin");
        private CatBoostSample CatBoostSample => (CatBoostSample)ModelSample;
    }
}

