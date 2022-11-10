using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.LightGBM
{
    public class LightGBMModel : Model
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

        public override (string train_XDatasetPath_InModelFormat, string train_YDatasetPath_InModelFormat, string train_XYDatasetPath_InModelFormat, string validation_XDatasetPath_InModelFormat, string validation_YDatasetPath_InModelFormat, string validation_XYDatasetPath_InModelFormat) 
            Fit(DataSet trainDataset, DataSet validationDatasetIfAny)
        {
            const bool addTargetColumnAsFirstColumn = true;
            const bool includeIdColumns = false;
            const bool overwriteIfExists = false;
            var train_XYDatasetPath_InModelFormat = trainDataset.to_csv_in_directory(RootDatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            var validation_XYDatasetPath_InModelFormat = validationDatasetIfAny?.to_csv_in_directory(RootDatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            LightGbmSample.UpdateForDataset(trainDataset);
            //we save in 'tmpLightGBMSamplePath' the model sample used for training
            var tmpLightGBMSamplePath = ISample.ToPath(TempPath, ModelName);
            var tmpLightGBMSample = (LightGBMSample)LightGbmSample.Clone();
            tmpLightGBMSample.Set(new Dictionary<string, object> {
                {"task", LightGBMSample.task_enum.train},
                {"data", train_XYDatasetPath_InModelFormat},
                {"valid", validation_XYDatasetPath_InModelFormat??""},
                {"output_model", ModelPath},
                {"input_model", ""},
                {"prediction_result", ""},
                {"header", true},
                {"save_binary", false},
            });
            var logMsg = $"Training model '{ModelName}' with training dataset {Path.GetFileNameWithoutExtension(train_XYDatasetPath_InModelFormat)}" +(string.IsNullOrEmpty(validation_XYDatasetPath_InModelFormat) ? "" : $" and validation dataset {Path.GetFileNameWithoutExtension(validation_XYDatasetPath_InModelFormat)}");
            if (LoggingForModelShouldBeDebug(ModelName))
            {
                LogDebug(logMsg);
            }
            else
            {
                LogInfo(logMsg);
            }
            tmpLightGBMSample.Save(tmpLightGBMSamplePath);

            Utils.Launch(WorkingDirectory, ExePath, "config=" + tmpLightGBMSamplePath, Log);
            Utils.TryDelete(tmpLightGBMSamplePath);
            return (null, null, train_XYDatasetPath_InModelFormat, null, null, validation_XYDatasetPath_InModelFormat);
        }

        public override DataFrame ComputeFeatureImportance(AbstractDatasetSample datasetSample, AbstractDatasetSample.DatasetType datasetType)
        {
            var sw = Stopwatch.StartNew();
            DataFrame featureImportance_df = null;
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
                var columns = Utils.ReadCsv(datasetPath).First();
                columns = Utils.Without(columns, datasetSample.IdColumns).ToArray();
                columns = Utils.Without(columns, datasetSample.TargetLabels).ToArray();

                var contribPath = Path.Combine(TempPath, ModelName + "_contrib_" + Path.GetFileNameWithoutExtension(datasetPath) + ".txt");
                //we save in 'tmpLightGbmSamplePath' the model sample used for computing Feature Importance
                var tmpLightGBMSamplePath = contribPath.Replace(".txt", ".conf");
                var tmpLightGBMSample = (LightGBMSample)LightGbmSample.Clone();
                tmpLightGBMSample.Set(new Dictionary<string, object>
                {
                    { "task", LightGBMSample.task_enum.predict },
                    { "data", datasetPath },
                    { "input_model", ModelPath },
                    { "predict_contrib", true },
                    { "prediction_result", contribPath },
                    { "header", true }
                });
                tmpLightGBMSample.Use_All_Available_Cores();
                tmpLightGBMSample.Save(tmpLightGBMSamplePath);
                Utils.Launch(WorkingDirectory, ExePath, "config=" + tmpLightGBMSamplePath, Log);

                featureImportance_df = LoadFeatureImportance(contribPath, columns, datasetSample);
                Utils.TryDelete(tmpLightGBMSamplePath);
                Utils.TryDelete(contribPath);
                Log.Info($"Feature importance for {datasetType} Dataset has been computed in {sw.ElapsedMilliseconds}ms");
            }
            catch (Exception e)
            {
                Log.Error($"fail to compute feature importance for {datasetType} Dataset: {e}");
            }
            return featureImportance_df;
        }


        private static DataFrame LoadFeatureImportance(string featureImportancePath, string[] columns, AbstractDatasetSample datasetSample)
        {
            int num_class = datasetSample.NumClass;
            var entireFeatureImportance = new float[num_class* (columns.Length+1)];

            Log.Info($"Loading LightGBM Feature Importance file {featureImportancePath}...");
            var stringsEnumerable = File.ReadAllLines(featureImportancePath);

            Log.Info($"Parsing file content...");
            object lockLoadFeatureImportance = new();
            void ProcessLine(int lineIdx)
            {
                float[] elements = stringsEnumerable[lineIdx].Split('\t').Select(s=> Math.Abs(float.Parse(s))).ToArray();
                if (entireFeatureImportance != null && elements.Length != entireFeatureImportance.Length)
                {
                    throw new ArgumentException($"found line with {elements.Length} elements but expecting {entireFeatureImportance.Length}");
                }
                lock (lockLoadFeatureImportance)
                {
                    for (int i = 0; i < elements.Length; ++i)
                    {
                        entireFeatureImportance[i % entireFeatureImportance.Length] += elements[i];
                    }
                }
            }
            Parallel.For(0, stringsEnumerable.Length, ProcessLine);

            var featureImportance = new float[columns.Length + 1];
            for (int i = 0; i < entireFeatureImportance.Length;++i)
            {
                featureImportance[i % featureImportance.Length] += entireFeatureImportance[i];
            }
            featureImportance = featureImportance.Take(columns.Length).ToArray();
            var totalFeatureImportance = Math.Max(featureImportance.Sum(),0.01f);
            featureImportance = featureImportance.Select(i => (100f * i / totalFeatureImportance)).ToArray();
            var featureName_df = DataFrame.New(columns, new[] { "Feature" });
            var featureImportance_df = DataFrame.New(featureImportance, new[] { "Importance" });

            var finalDf = DataFrame.MergeHorizontally(featureName_df, featureImportance_df);

            if (num_class >= 2) //Multi class Classification
            {
                //we'll display 1 extra column for each class to predict
                for (int numClass = 0; numClass < num_class; numClass++)
                {
                    var featureImportanceNumClass = new float[columns.Length];
                    for (int i = 0; i < columns.Length;++i)
                    {
                        featureImportanceNumClass[i] = entireFeatureImportance[numClass * (columns.Length + 1)+i];
                    }
                    featureImportanceNumClass = featureImportanceNumClass.Select(i=>(100f*i / totalFeatureImportance)).ToArray();
                    var numClass_df = DataFrame.New(featureImportanceNumClass, new[] { datasetSample.TargetLabelDistinctValues[numClass]});
                    finalDf = DataFrame.MergeHorizontally(finalDf, numClass_df);
                }
            }
            finalDf = finalDf.sort_values("Importance", ascending: false);

            return finalDf;
        }
        public override (DataFrame,string)  PredictWithPath(DataSet dataset, bool removeAllTemporaryFilesAtEnd)
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

            Utils.Launch(WorkingDirectory, ExePath, "config=" + tmpLightGBMSamplePath, Log);

            var predictionsDf = LoadProbaFile(predictionResultPath, false, false, null, dataset);
            Utils.TryDelete(tmpLightGBMSamplePath);
            Utils.TryDelete(predictionResultPath);
            if (removeAllTemporaryFilesAtEnd)
            {
                Utils.TryDelete(datasetPath);
                datasetPath = "";
            }
            return (predictionsDf, datasetPath);
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

        public override List<string> AllFiles()
        {
            return new List<string> { ModelPath };
        }
        //public static LightGBMModel LoadTrainedLightGBMModel(string workingDirectory, string modelName)
        //{
        //    var sample = ISample.LoadSample<LightGBMSample>(workingDirectory, modelName);
        //    return new LightGBMModel(sample, workingDirectory, modelName);
        //}

        private static string ExePath => Path.Combine(Utils.ChallengesPath, "bin", "lightgbm.exe");
        private string TempPath => Path.Combine(WorkingDirectory, "Temp");

        /// <summary>
        /// path of a trained model.
        /// null if no trained model is available
        /// </summary>
        private string ModelPath => Path.Combine(WorkingDirectory, ModelName + ".txt");
    }
}
