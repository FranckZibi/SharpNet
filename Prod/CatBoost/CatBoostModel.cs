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
            if (ModelSample.GetLoss() == EvaluationMetricEnum.DEFAULT_VALUE)
            {
                throw new ArgumentException("Loss Function not set");
            }
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
                //CatBoostSample.use_best_model = true;
                arguments += " --test-set " + validationDatasetPathIfAny_InModelFormat;
            }
            else
            {
                //without validation dataset there is nothing such as 'best model'
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

        private static List<int> LabelIndexFromColumnDescriptionFile(string datasetColumnDescriptionPath)
        {
            if (string.IsNullOrEmpty(datasetColumnDescriptionPath) || !File.Exists(datasetColumnDescriptionPath))
            {
                return null;
            }
            List<int> res = new();
            foreach (var l in File.ReadAllLines(datasetColumnDescriptionPath))
            {
                var splitted = l.Split('\t');
                if (splitted.Length == 2 && string.Equals(splitted[1], "Label", StringComparison.OrdinalIgnoreCase))
                {
                    res.Add(int.Parse(splitted[0]));
                }
            }
            return res;
        }

        public override DataFrame ComputeFeatureImportance(AbstractDatasetSample datasetSample, AbstractDatasetSample.DatasetType datasetType)
        {
            var sw = Stopwatch.StartNew();
            Log.Info($"Computing feature importance for {datasetType} Dataset...");
            try
            {
                if (!File.Exists(ModelPath))
                {
                    Log.Error($"missing model {ModelPath} for computing Feature Importance");
                    return null;
                }
                var datasetPath = datasetSample.ExtractDatasetPath_InModelFormat(datasetType);
                if (string.IsNullOrEmpty(datasetPath) || !File.Exists(datasetPath))
                {
                    Log.Error($"missing {datasetType} Dataset {datasetPath} for computing Feature Importance");
                    return null;
                }
                string datasetColumnDescriptionPath = datasetPath + ".co";
                if (!File.Exists(datasetColumnDescriptionPath))
                {
                    Log.Error($"missing {datasetColumnDescriptionPath} file for computing Feature Importance");
                    return null;
                }
                var contribPath = Path.Combine(TempPath, ModelName + "_contrib_" + Path.GetFileNameWithoutExtension(datasetPath) + ".txt");
                var arguments = "fstr " +
                                " --model-file " + ModelPath +
                                " --delimiter=\"" + datasetSample.GetSeparator()+ "\"" +
                                " --has-header" +
                                " --fstr-type ShapValues" +
                                " --column-description " + datasetColumnDescriptionPath +
                                " --input-path " + datasetPath +
                                " --output-path " + contribPath;

                Utils.Launch(WorkingDirectory, ExePath, arguments, Log, false);

                //we retrieve the column names, removing any label columns we may find
                var labelIndexes = LabelIndexFromColumnDescriptionFile(datasetColumnDescriptionPath);
                var columnsWithoutLabels = Utils.ReadCsv(datasetPath).First();
                foreach (var labelIndex in labelIndexes.OrderByDescending(x => x).ToList())
                {
                    var tmp = columnsWithoutLabels.ToList();
                    tmp.RemoveAt(labelIndex);
                    columnsWithoutLabels = tmp.ToArray();
                }
                var featureImportance_df = LoadFeatureImportance(contribPath, columnsWithoutLabels);
                Utils.TryDelete(contribPath);
                Log.Info($"Feature importance for {datasetType} Dataset has been computed in {sw.ElapsedMilliseconds}ms");
                return featureImportance_df;
            }
            catch (Exception e)
            {
                Log.Error($"fail to compute feature importance for {datasetType} Dataset: {e}");
                return null;
            }
        }

        private static DataFrame LoadFeatureImportance(string featureImportancePath, string[] columns)
        {
            Log.Info($"Loading Feature Importance file {featureImportancePath}...");
            var raw_df = DataFrame.read_csv_normalized(featureImportancePath, '\t', false, x => typeof(float));
            var entireFeatureImportance = raw_df.FloatTensor.SpanContent;
            // the last column of the TSV SHAP file is the expected prediction: it can be ignored
            int cols = raw_df.Shape[1];
            if ((cols-1) != columns.Length)
            {
                var errorMsg = $"Feature Importance file {featureImportancePath} has {cols-1} feature columns instead of {columns.Length}";
                Log.Error(errorMsg);
                throw new ArgumentException(errorMsg);
            }

            var featureImportance = new float[cols];
            for (int i = 0; i < entireFeatureImportance.Length; ++i)
            {
                featureImportance[i % cols] += Math.Abs(entireFeatureImportance[i]);
            }
            featureImportance = featureImportance.Take(columns.Length).ToArray();
            var totalFeatureImportance = Math.Max(featureImportance.Sum(), 0.01f);
            featureImportance = featureImportance.Select(i => (100f * i / totalFeatureImportance)).ToArray();
            var featureName_df = DataFrame.New(columns, new[] { "Feature" });
            var featureImportance_df = DataFrame.New(featureImportance, new[] { "Importance" });

            var finalDf = DataFrame.MergeHorizontally(featureName_df, featureImportance_df);
            finalDf = finalDf.sort_values("Importance", ascending: false);
            return finalDf;
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

            var modelFormat = ModelPath.EndsWith("json") ? "json" : "CatboostBinary";
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
            if (ModelSample.IsRegressionProblem)
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

