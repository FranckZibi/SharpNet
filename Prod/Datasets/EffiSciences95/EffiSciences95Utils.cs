using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using log4net;
using SharpNet.DataAugmentation;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.Networks;

namespace SharpNet.Datasets.EffiSciences95;

// ReSharper disable once UnusedType.Global
public static class EffiSciences95Utils
{
    public const string NAME = "EffiSciences95";


    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(EffiSciences95Utils));
    #endregion

    #region load of datasets

    public static readonly int[] Shape_CHW = { 3, 218, 178 };

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    // ReSharper disable once MemberCanBePrivate.Global

    public static string IDMDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "ImageDatabaseManagement");
    public static string LabeledPath = Path.Combine(IDMDirectory, "EffiSciences95_Labeled.csv");
    public static string UnlabeledPath = Path.Combine(IDMDirectory, "EffiSciences95_Unlabeled.csv");


    public static string Labeled_TextTargetPath => Path.Combine(DataDirectory, "Labeled_TextTarget.csv");
    public static string Unlabeled_TextTargetPath => Path.Combine(DataDirectory, "Unlabeled_TextTarget.csv");

    public static string LabeledDirectory => Path.Combine(DataDirectory, "Labeled");
    public static string UnlabeledDirectory => Path.Combine(DataDirectory, "Unlabeled");

    #endregion

    public static string IdToPath(int id, bool isLabeled)
    {
        return isLabeled ? Path.Combine(LabeledDirectory, $"{id}.jpg") : Path.Combine(UnlabeledDirectory, $"{id}.jpg");
    }

    public static int MaxId(bool isLabeled)
    {
        return isLabeled ? 19999 : 69729;
    }

    public static Dictionary<int, int> IdToTextTarget(bool isLabeled)
    {
        Dictionary<int, int> res = new();
        foreach (var l in Utils.ReadCsv(isLabeled?Labeled_TextTargetPath:Unlabeled_TextTargetPath).Skip(1))
        {
            res[int.Parse(l[0])] = int.Parse(l[1]);
        }
        return res;
    }


    public static void Run()
    {
        BoxFinder.FindBox(false);
    }



    public static void InferenceUnlabeledEffiSciences95(string modelName)
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, "log");
        const bool isLabeled = false;
        using var network = Network.LoadTrainedNetworkModel(WorkingDirectory, modelName);
        using var unlabeledDataset = EffiSciences95DirectoryDataSet.ValueOf(isLabeled);
        Log.Info($"computing predictions of model {modelName} on dataset of {unlabeledDataset.Count} rows");
        var p = network.Predict(unlabeledDataset, 64);
        var sb = new StringBuilder();
        sb.Append("index,labels");

        for (int id = 0; id < p.Shape[0]; ++id)
        {
            var rowWithPrediction = p.RowSlice(id, 1);
            var predictionWithProba = rowWithPrediction.ContentAsFloatArray();
            int prediction = predictionWithProba[0] > predictionWithProba[1] ? 0 : 1;
            sb.Append(Environment.NewLine + id + "," + prediction);
        }

        var predictionPaths = Path.Combine(WorkingDirectory, "predictions_unlabeled_" + modelName + "_" + DateTime.Now.Ticks + ".csv");
        Log.Info($"saving predictions in file {predictionPaths}");
        File.WriteAllText(predictionPaths, sb.ToString());
    }


    /// <summary>
    /// The default EfficientNet Hyper-Parameters for CIFAR10
    /// </summary>
    /// <returns></returns>
    public static EfficientNetNetworkSample DefaultEfficientNetNetworkSample()
    {
        var config = (EfficientNetNetworkSample)new EfficientNetNetworkSample()
        {
            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
            CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
            lambdaL2Regularization = 0.0005,
            //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, CIFAR10DataSet.NAME),
            NumEpochs = 10,
            BatchSize = 64,
            InitialLearningRate = 0.01,

            //Data augmentation
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
            WidthShiftRangeInPercentage = 0.0,
            HeightShiftRangeInPercentage = 0.0,
            HorizontalFlip = true,
            VerticalFlip = false,
            FillMode = ImageDataGenerator.FillModeEnum.Reflect,
            AlphaMixup = 0.0,
            AlphaCutMix = 1.0,
            CutoutPatchPercentage = 0.0
        }
            .WithSGD(0.9, false)
            .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
        return config;

    }

    public static void Launch_HPO(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, "log");
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            {"PercentageInTraining", 0.9}, //will be automatically set to 1 if KFold is enabled

            { "BatchSize", new[] {96} },
            { "NumEpochs", new[] { numEpochs } },
            
            {"LossFunction", "CategoricalCrossentropy"},  //for multi class classification


            // DataAugmentation
            { "HorizontalFlip", new[] { true /*, false*/} }, //true
            { "VerticalFlip", new[] { false /*, true*/} }, //false
            { "AlphaMixup", new[] { 0.0 /*, 0.5, 1.0*/ } }, //0.0
            //{ "CutoutPatchPercentage", new[] { 0.0, 0.1, 0.3} },
            //{ "RotationRangeInDegrees", new[] { 0.0, 5, 10 /*, 20*/} },
            { "RotationRangeInDegrees", new[]{5.0, 7.0 } },
            //{ "ZoomRange", new[] { 0.0, 0.1, 0.2} },
            //{ "EqualizeOperationProbability", new[] { 0.0, 0.2} },
            //{ "AutoContrastOperationProbability", new[] { 0.0, 0.2} },


      
            // Optimizer 
            //{ "OptimizerType", new[] { "AdamW"} },
            //{ "SGD_usenesterov", new[] { true, false } },
            { "lambdaL2Regularization", new[] { 0.0005, /*0.001, 0.00005*/ } },
            //{"DefaultMobileBlocksDescriptionCount", new[]{5}},
            // Learning Rate
            { "InitialLearningRate", new []{0.01 , 0.015 /* , 0.02, 0.005*/}},
            //{ "InitialLearningRate", 0.001f },
            // Learning Rate Scheduler
            //{ "LearningRateSchedulerType", new[] { "OneCycle" } },
            //{ "LearningRateSchedulerType", "CyclicCosineAnnealing" },
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new EffiSciences95DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }
}