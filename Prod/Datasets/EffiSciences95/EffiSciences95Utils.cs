using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using log4net;
using SharpNet.DataAugmentation;
using SharpNet.HPO;
using SharpNet.Hyperparameters;
using SharpNet.MathTools;
using SharpNet.Networks;

namespace SharpNet.Datasets.EffiSciences95;


/// <summary>
/// this is the main class of the project, it launches the different steps needed to produce the prediction file
/// </summary>
public static class EffiSciences95Utils
{
    public static void Run()
    {
        //find the coordinates of the box with the labels (old or young) for each picture in directory 'Labeled' (training dataset)
        //and stores those coordinates in a file 'EffiSciences95_Labeled.csv'
        LabelFinder.FindLabelCoordinates("Labeled");
        //find the coordinates of the box with the labels (old or young) for each pictures in directory 'Unlabeled' (test dataset)
        //and stores those coordinates in a file 'EffiSciences95_Unlabeled.csv'
        LabelFinder.FindLabelCoordinates("Unlabeled");
        // launch a Hyperparameters search (for 1 hour) using an EfficientNet Neural Network (with 30 epochs) to train on the Labeled dataset
        // Each picture will have the box with the label removed before being used as a training picture
        // (see method 'OriginalElementContent' in class 'EffiSciences95DirectoryDataSet')
        Launch_HPO(30, 3600);
        //create the prediction file for the test dataset (Unlabeled) using the best deep learning model found at the previous step (here: 'F1040C26F7')
        InferenceUnlabeledEffiSciences95(WorkingDirectory, "F1040C26F7", true);
    }


    public const string NAME = "EffiSciences95";

    #region public fields & properties
    private static readonly ILog Log = LogManager.GetLogger(typeof(EffiSciences95Utils));
    #endregion

    #region load of datasets
    public static readonly string[] TargetLabelDistinctValues = { "old", "young" };
    private static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    // ReSharper disable once MemberCanBePrivate.Global

    public static readonly string IDMDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "ImageDatabaseManagement");
    #endregion

    public static string PictureIdToPath(int pictureId, string directory)
    {
        return Path.Combine(DataDirectory, directory, $"{pictureId}.jpg");
    }
    public static int MaxPictureId(string directory)
    {
        return (directory=="Labeled") ? 19999 : 69729;
    }
    public static Dictionary<int, int> LoadPredictionFile(string predictionFile)
    {
        Dictionary<int, int> res = new();
        foreach (var l in Utils.ReadCsv(Path.Combine(DataDirectory, predictionFile)).Skip(1))
        {
            res[int.Parse(l[0])] = int.Parse(l[1]);
        }
        return res;
    }
    /// <summary>
    /// this method creates a prediction file for the test dataset (Unlabeled) using the deep learning model 'modelName' in directory 'modelDirectory'
    /// </summary>
    private static void InferenceUnlabeledEffiSciences95(string modelDirectory, string modelName, bool useAllAvailableCores)
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, NAME);
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, NAME);
        using var network = Network.LoadTrainedNetworkModel(modelDirectory, modelName, useAllAvailableCores: useAllAvailableCores);
        using var datasetSample = new EffiSciences95DatasetSample();
        using var unlabeledDataset = EffiSciences95DirectoryDataSet.ValueOf(datasetSample, "Unlabeled");
        Log.Info($"computing predictions of model {modelName} on dataset of {unlabeledDataset.Count} rows");
        var p = network.Predict(unlabeledDataset, 64);
        var sb = new StringBuilder();
        sb.Append("index,labels"+ Environment.NewLine);


        var idToTextTarget = LoadPredictionFile("Unlabeled_TextTarget.csv");
        LinearRegression lr = new();
        for (int id = 0; id < p.Shape[0]; ++id)
        {
            var rowWithPrediction = p.RowSlice(id, 1);
            var predictionWithProba = rowWithPrediction.ContentAsFloatArray();
            int prediction = predictionWithProba[0] > predictionWithProba[1] ? 0 : 1;

            if (idToTextTarget.TryGetValue(id, out var value))
            {
                lr.Add(value, prediction);
            }
            sb.Append(id + "," + prediction+ Environment.NewLine);
        }
        var predictionPaths = Path.Combine(WorkingDirectory, modelName +"_predict_tests.csv");
        Log.Info($"Saving predictions in file {predictionPaths}");
        File.WriteAllText(predictionPaths, sb.ToString());
    }
    /// <summary>
    /// The default EfficientNet Hyperparameters for the Deep Learning model
    /// </summary>
    /// <returns></returns>
    private static EfficientNetNetworkSample DefaultEfficientNetNetworkSample()
    {
        var config = (EfficientNetNetworkSample)new EfficientNetNetworkSample()
        {
            LossFunction = EvaluationMetricEnum.CategoricalCrossentropy,
            CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
            //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, CIFAR10DataSet.NAME),
            num_epochs = 10,
            BatchSize = 64,

            //Data augmentation
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
            WidthShiftRangeInPercentage = 0.0,
            HeightShiftRangeInPercentage = 0.0,
            HorizontalFlip = true,
            VerticalFlip = false,
            FillMode = ImageDataGenerator.FillModeEnum.Reflect,
            AlphaMixUp = 0.0,
            AlphaCutMix = 0.0,
            CutoutPatchPercentage = 0.0
        }
            .WithSGD(lr:0.01, 0.9, weight_decay: 0.0005, false)
            .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
        return config;

    }
    /// <summary>
    /// the method uses a Bayesian Search to find the best Hyperparameters for the EfficientNet Deep Learning model
    /// </summary>
    /// <param name="num_epochs"></param>
    /// <param name="maxAllowedSecondsForAllComputation"></param>
    private static void Launch_HPO(int num_epochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, NAME);
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, NAME);
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.9}, //will be automatically set to 1 if KFold is enabled

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.CategoricalCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)},
            { nameof(NetworkSample.BatchSize), new[] {64 /*, 96*/} },
            { nameof(NetworkSample.num_epochs), new[] { num_epochs } },
            // Optimizer 
            //{ nameof(NetworkSample.OptimizerType), new[] { "AdamW"} },
            //{ nameof(NetworkSample.nesterov), new[] { true, false } },
            //{ nameof(NetworkSample.weight_decay), new[] { 0.0005, 0.001, 0.00005 } },
            { nameof(NetworkSample.weight_decay), new[] { 0.0005 /*, 0.001, 0.00005*/ } },
            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), new[]{4, 5}},
            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), new []{0.01 /*, 0.015, 0.005*/}},
            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), new[] { "OneCycle" } },
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },


            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[] {0.1}},
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[] {0.1}},
            { "MinEnlargeForBox", new[] {3/*,3*/}},
            { "MaxEnlargeForBox", new[] {10}},
            { "EnlargeOldBoxToYoungBoxShape", new[] { /*false,*/ true} },
            { "AddNewBoxOfOtherCategory", new[] { false /*, true*/} },
            // DataAugmentation
            //{ nameof(NetworkSample.HorizontalFlip), new[] { true /*, false*/} }, //true
            //{ nameof(NetworkSample.VerticalFlip), new[] { false /*, true*/} }, //false
            //{ nameof(NetworkSample.AlphaMixUp), new[] { 0.0 /*, 0.5, 1.0*/ } }, //0.0
            { nameof(NetworkSample.AlphaCutMix), new[] { 1.0, 2.0 } }, //0.0
            //{ nameof(NetworkSample.CutoutPatchPercentage), new[] { 0.0, 0.15, 0.3} },
            //{ "RotationRangeInDegrees", new[] { 0.0, 5.0, 10.0} },
            //{ nameof(NetworkSample.ZoomRange), new[] { 0.05} },
            //{ "EqualizeOperationProbability", new[] { 0.0, 0.2} },
            //{ "AutoContrastOperationProbability", new[] { 1.0} },
      
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new EffiSciences95DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperparameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }
}
