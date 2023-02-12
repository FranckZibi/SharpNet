using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO;
using SharpNet.Datasets;
using SharpNet.HyperParameters;
using SharpNet.Models;

namespace SharpNet.Svm
{
    // ReSharper disable once InconsistentNaming
    public class SVMModel : Model
    {
        #region public fields & properties
        private SVMSample SVMSample => (SVMSample)ModelSample;
        #endregion

        #region constructor
        /// <summary>
        /// 
        /// </summary>
        /// <param name="SVMModelSample"></param>
        /// <param name="workingDirectory"></param>
        /// <param name="modelName">the name of the model to use</param>
        /// <exception cref="Exception"></exception>
        [SuppressMessage("ReSharper", "VirtualMemberCallInConstructor")]
        public SVMModel(SVMSample SVMModelSample, string workingDirectory, [JetBrains.Annotations.NotNull] string modelName) : base(SVMModelSample, workingDirectory, modelName)
        {
            if (!File.Exists(TrainExePath))
            {
                throw new Exception($"Missing exe {TrainExePath}");
            }
            if (!File.Exists(PredictExePath))
            {
                throw new Exception($"Missing exe {PredictExePath}");
            }
            if (!Directory.Exists(TempPath))
            {
                Directory.CreateDirectory(TempPath);
            }
        }
        #endregion

        public override (string train_XDatasetPath_InModelFormat, string train_YDatasetPath_InModelFormat, string
            train_XYDatasetPath_InModelFormat, string validation_XDatasetPath_InModelFormat, string
            validation_YDatasetPath_InModelFormat, string validation_XYDatasetPath_InModelFormat, IScore
            trainScoreIfAvailable, IScore validationScoreIfAvailable, IScore trainMetricIfAvailable, IScore
            validationMetricIfAvailable)
            Fit(DataSet trainDataset, DataSet validationDatasetIfAny)
        {
            var sw = Stopwatch.StartNew();
            const bool overwriteIfExists = false;
            var train_XYDatasetPath_InModelFormat = trainDataset.to_libsvm_in_directory(DatasetPath, overwriteIfExists);
            //var validation_XYDatasetPath_InModelFormat = validationDatasetIfAny?.to_csv_in_directory(DatasetPath, addTargetColumnAsFirstColumn, includeIdColumns, overwriteIfExists);
            //SVMSample.UpdateForDataset(trainDataset);

            //SVMSample.AddExtraMetricToComputeForTraining();
            //svm-train [options] training_set_file [model_file]
            string parameters = "";

            parameters += " -s " + (int)SVMSample.svm_type;
            parameters += " -t " + (int)SVMSample.kernel_type;
            if (SVMSample.degree != AbstractSample.DEFAULT_VALUE) { parameters += " -d " + SVMSample.degree.ToString(CultureInfo.InvariantCulture); }
            if (SVMSample.gamma != AbstractSample.DEFAULT_VALUE) { parameters += " -g " + SVMSample.gamma.ToString(CultureInfo.InvariantCulture); }
            if (SVMSample.coef0 != AbstractSample.DEFAULT_VALUE) { parameters += " -r " + SVMSample.coef0.ToString(CultureInfo.InvariantCulture); }
            if (SVMSample.cost != AbstractSample.DEFAULT_VALUE) { parameters += " -c " + SVMSample.cost.ToString(CultureInfo.InvariantCulture); }
            if (SVMSample.nu != AbstractSample.DEFAULT_VALUE) { parameters += " -n " + SVMSample.nu.ToString(CultureInfo.InvariantCulture); }
            if (SVMSample.epsilon_SVR != AbstractSample.DEFAULT_VALUE) { parameters += " -p " + SVMSample.epsilon_SVR.ToString(CultureInfo.InvariantCulture); }
            if (SVMSample.cachesize != AbstractSample.DEFAULT_VALUE) { parameters += " -m " + SVMSample.cachesize.ToString(CultureInfo.InvariantCulture); }
            if (SVMSample.epsilon != AbstractSample.DEFAULT_VALUE) { parameters += " -e " + SVMSample.epsilon.ToString(CultureInfo.InvariantCulture); }
            if (SVMSample.shrinking != AbstractSample.DEFAULT_VALUE) { parameters += " -h " + SVMSample.shrinking.ToString(CultureInfo.InvariantCulture); }
            if (SVMSample.n_fold_svm != AbstractSample.DEFAULT_VALUE) { parameters += " -v " + SVMSample.n_fold_svm.ToString(CultureInfo.InvariantCulture); }
            //parameters += " -q "; //quiet mode (no outputs)
            parameters += " " + train_XYDatasetPath_InModelFormat;
            parameters += " " + ModelPath;

            LogForModel($"Training model '{ModelName}' with training dataset '{Path.GetFileNameWithoutExtension(train_XYDatasetPath_InModelFormat)}'");
            var linesFromLog = Utils.Launch(WorkingDirectory, TrainExePath, parameters, Log, true);
            (IScore trainLossIfAvailable, IScore validationLossIfAvailable, IScore trainMetricIfAvailable, IScore validationMetricIfAvailable) = SVMSample.ExtractScores(linesFromLog);

            LogForModel($"Model '{ModelName}' trained with dataset '{Path.GetFileNameWithoutExtension(train_XYDatasetPath_InModelFormat)}' in {sw.Elapsed.TotalSeconds}s (trainScore = {trainLossIfAvailable} / validationScore = {validationLossIfAvailable} / trainMetric = {trainMetricIfAvailable} / validationMetric = {validationMetricIfAvailable})");
            return (null, null, train_XYDatasetPath_InModelFormat, null, null, null, trainLossIfAvailable, validationLossIfAvailable, trainMetricIfAvailable, validationMetricIfAvailable);
        }

        public override (DataFrame, string) PredictWithPath(DataSet dataset, bool removeAllTemporaryFilesAtEnd)
        {
            if (!File.Exists(ModelPath))
            {
                throw new Exception($"missing model {ModelPath} for inference");
            }
            const bool overwriteIfExists = false;
            string datasetPath = dataset.to_libsvm_in_directory(DatasetPath, overwriteIfExists);
            var predictionResultPath = Path.Combine(TempPath, ModelName + "_predict_" + Path.GetFileNameWithoutExtension(datasetPath) + ".txt");

            string parameters = "";

            //parameters += " -q "; //quiet mode (no outputs)
            parameters += " " + datasetPath;
            parameters += " " + ModelPath;
            parameters += " " + predictionResultPath;
           

            Utils.Launch(WorkingDirectory, PredictExePath, parameters, Log, false);

            var predictionsDf = LoadProbaFile(predictionResultPath, false, false, null, dataset);
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
            SVMSample.Save(workingDirectory, modelName);
        }
     
        public override List<string> AllFiles()
        {
            return new List<string> { ModelPath };
        }
       
        private static string TrainExePath => Path.Combine(Utils.ChallengesPath, "bin", "libsvm", "svm-train.exe");
        private static string PredictExePath => Path.Combine(Utils.ChallengesPath, "bin", "libsvm", "svm-predict.exe");

        /// <summary>
        /// path of a trained model.
        /// null if no trained model is available
        /// </summary>
        private string ModelPath => Path.Combine(WorkingDirectory, ModelName + ".txt");
    }
}
