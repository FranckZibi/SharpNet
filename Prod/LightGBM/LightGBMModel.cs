using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
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
            string trainDatasetPath = trainDataset.to_csv_in_directory(RootDatasetPath, true, false);
            string validationDatasetPath = "";
            if (validationDatasetIfAny != null)
            {
                validationDatasetPath = validationDatasetIfAny.to_csv_in_directory(RootDatasetPath, true, false);
            }
            return Fit(trainDatasetPath, validationDatasetPath);
        }

        public override DataFrame Predict(IDataSet dataset)
        {
            return PredictWithPath(dataset).predictions;
        }

        public override (DataFrame predictions, string predictionPath) PredictWithPath(IDataSet dataset)
        {
            const bool addTargetColumnAsFirstColumn = false;
            string predictionDatasetPath = dataset.to_csv_in_directory(RootDatasetPath, addTargetColumnAsFirstColumn, addTargetColumnAsFirstColumn);
            if (!File.Exists(ModelPath))
            {
                throw new Exception($"missing model {ModelPath} for inference");
            }
            var predictionResultPath = Path.Combine(TempPath, ModelName + "_predict_" + Path.GetFileNameWithoutExtension(predictionDatasetPath) + ".txt");

            //we save in 'tmpLightGbmSamplePath' the model sample used for prediction
            var tmpLightGBMSamplePath = predictionResultPath.Replace(".txt", ".conf");
            var tmpLightGBMSample = (LightGBMSample)LightGbmSample.Clone();
            tmpLightGBMSample.Set(new Dictionary<string, object> {
                {"task", LightGBMSample.task_enum.predict},
                {"data", predictionDatasetPath},
                {"input_model", ModelPath},
                {"prediction_result", predictionResultPath},
                {"header", true}
            });
            tmpLightGBMSample.Save(tmpLightGBMSamplePath);

            Utils.Launch(WorkingDirectory, ExePath, "config=" + tmpLightGBMSamplePath, IModel.Log);
            float[][] predictionResultContent = File.ReadAllLines(predictionResultPath).Select(l => l.Split().Select(float.Parse).ToArray()).ToArray();
            File.Delete(tmpLightGBMSamplePath);
            File.Delete(predictionResultPath);
            int rows = predictionResultContent.Length;
            int columns = predictionResultContent[0].Length;
            var content = new float[rows*columns];
            for (int row = 0; row < rows; row++)
            {
                if (columns != predictionResultContent[row].Length)
                {
                    var errorMsg = $"invalid number of predictions in line {row} of file {predictionResultPath}, expecting {columns} ";
                    LogError(errorMsg);
                    throw new Exception(errorMsg);
                }
                Array.Copy(predictionResultContent[row], 0, content, row* columns, columns);
            }

            var cpuTensor =  new CpuTensor<float>(new[] { rows, columns}, content);
            string[] predictionLabels = dataset.TargetLabels;
            if (predictionLabels.Length != columns)
            {
                predictionLabels = Enumerable.Range(0, columns).Select(x => x.ToString()).ToArray();
            }

            var predictions = DataFrame.New(cpuTensor, predictionLabels, Array.Empty<string>());
            if (dataset.IdFeatures.Length != 0)
            {
                var idDataFrame = (DataFrameT<float>)dataset.ExtractIdDataFrame();
                predictions = (DataFrameT<float>)DataFrame.MergeHorizontally(idDataFrame, predictions);
            }

            if (predictions.Shape[0]!= dataset.Count)
            {
                throw new Exception($"Invalid number of predictions, received {predictions.Shape[0]} but expected {dataset.Count}");
            }
            return (predictions, predictionDatasetPath);
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
            LogInfo($"Training model '{ModelName}' with training dataset {Path.GetFileNameWithoutExtension(trainDatasetPath)}" + (string.IsNullOrEmpty(validationDatasetPathIfAny)?"":$" and validation dataset {Path.GetFileNameWithoutExtension(validationDatasetPathIfAny)}")
            );
            tmpLightGBMSample.Save(tmpLightGBMSamplePath);

            Utils.Launch(WorkingDirectory, ExePath, "config=" + tmpLightGBMSamplePath, IModel.Log);
            File.Delete(tmpLightGBMSamplePath);
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
