using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using log4net;
using SharpNet.CPU;
using SharpNet.DataAugmentation;
using SharpNet.DataAugmentation.Operations;
using SharpNet.GPU;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.MathTools;
using SharpNet.Networks;
using SharpNet.Networks.Transformers;
using static SharpNet.Networks.NetworkSample;

namespace SharpNet.Datasets.Biosonar85;

public static class Biosonar85Utils
{
    private const string NAME = "Biosonar85";


    #region public fields & properties
    private static readonly ILog Log = LogManager.GetLogger(typeof(Biosonar85Utils));
    #endregion

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    public static readonly string[] TargetLabelDistinctValues = new string[]{"y"};

    //public static EfficientNetNetworkSample DefaultEfficientNetNetworkSample()
    //{
    //    var config = (EfficientNetNetworkSample)new EfficientNetNetworkSample()
    //        {
    //            LossFunction = EvaluationMetricEnum.BinaryCrossentropy,
    //            CompatibilityMode = NetworkSample.CompatibilityModeEnum.TensorFlow,
    //            lambdaL2Regularization = 0.0005,
    //            //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, CIFAR10DataSet.NAME),
    //            NumEpochs = 10,
    //            BatchSize = 256,
    //            InitialLearningRate = 0.01,

    //            //Data augmentation
    //            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.NO_AUGMENTATION,
    //            WidthShiftRangeInPercentage = 0.0,
    //            HeightShiftRangeInPercentage = 0.0,
    //            HorizontalFlip = false,
    //            VerticalFlip = false,
    //            // FillMode = ImageDataGenerator.FillModeEnum.Reflect,
    //            AlphaMixup = 0.0,
    //            AlphaCutMix = 0.0,
    //            CutoutPatchPercentage = 0.0
    //        }
    //        .WithSGD(0.9, false)
    //        .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
    //    return config;

    //}

    //const float x_train_mean = -37.149295f;       //for train dataset only 
    public const float x_train_mean = -37.076445f;         // for train+test dataset
    
    //const float x_train_volatility = 14.906728f;  //for train dataset only 
    public const float x_train_volatility = 14.84446f;     // for train+test dataset


    public const float x_train_mean_PNG_1CHANNEL_V2 = -57.210965f;    // for train dataset
    public const float x_test_mean_PNG_1CHANNEL_V2 = -54.409614f;     // for test dataset
    public const float x_train_volatility_PNG_1CHANNEL_V2 = 13.373847f;            // for train dataset
    public const float x_test_volatility_PNG_1CHANNEL_V2 = 10.493092f;             // for test dataset

    public static void Run()
    {
        //var trainDataset = Load("X_train_23168_101_64_1024_512.bin", "Y_train_23168_1_64_1024_512.bin", true);
        //var testDataset = Load("X_test_950_101_64_1024_512.bin", "Y_test_950_1_64_1024_512.bin", false);

        //var bin_file = Path.Combine(DataDirectory, "Y_train_ofTdMHi.csv.bin");
        //var tensor = CpuTensor<float>.LoadFromBinFile(bin_file, new[] { -1, 101, 64});

        ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "569C5C14D2", null, percentageInTraining:0.8, retrainOnFullDataset:false, useAllAvailableCores:true);return;

        //ComputeAverage_avg();return;

        //Launch_HPO_Spectogram(10); return;
        //Launch_HPO_Transformers(10); return;
        //Launch_HPO(10);return;
    }

    public static void ComputeAverage_avg()
    {
        var dfs = new List<DataFrame>();
        var path = @"\\RYZEN2700X-DEV\Challenges\Biosonar85\Submit\";
        foreach (var file in new[]
                 {
                     @"7E45F84676_predict_test_0,9353867531264475.csv",
                     @"569C5C14D2_predict_test_0.936063704706595.csv",
                 })
        {
            dfs.Add(DataFrame.read_csv(Path.Combine(path, file), true, x => x == "id" ? typeof(string) : typeof(float)));
        }
        DataFrame.Average(dfs.ToArray()).to_csv(Path.Combine(path, "7E45F84676_569C5C14D2_avg.csv"));
    }


    public static (int[] shape, int n_fft, int hop_len, int f_min, int f_max, int top_db) ProcessXFileName(string xPath)
    {
        var xSplitted = Path.GetFileNameWithoutExtension(xPath).Split("_");
        var xShape = new[] { int.Parse(xSplitted[^8]), int.Parse(xSplitted[^7]), int.Parse(xSplitted[^6]) };
        var n_fft = int.Parse(xSplitted[^5]);
        var hop_len = int.Parse(xSplitted[^4]);
        var f_min = int.Parse(xSplitted[^3]);
        var f_max = int.Parse(xSplitted[^2]);
        var top_db = int.Parse(xSplitted[^1]);
        return (xShape, n_fft, hop_len, f_min, f_max, top_db);
    }



    public static DirectoryDataSet LoadPng(string pngDirectory, string csvPath, bool hasLabels, List<Tuple<float, float>> meanAndVolatilityForEachChannel)
    {
        const int numClass = 1;
        var df = DataFrame.read_csv(csvPath, columnNameToType: (s => s == "id" ? typeof(string) : typeof(float)));
        var y_IDs = df.StringColumnContent("id");
        var elementIdToPaths = new List<List<string>>();
        foreach (var wavFilename in y_IDs)
        {
            elementIdToPaths.Add(new List<string> { Path.Combine(pngDirectory, wavFilename.Replace("wav", "png")) });
        }
        var elementIdToCategoryIndex = Enumerable.Repeat(0, y_IDs.Length).ToList();
        var labels = hasLabels ? df.FloatColumnContent("pos_label") : new float[y_IDs.Length];

        //y_IDs = y_IDs.Take(128).ToArray();
        //elementIdToCategoryIndex = elementIdToCategoryIndex.Take(y_IDs.Length).ToList();
        //elementIdToPaths = elementIdToPaths.Take(y_IDs.Length).ToList();
        //labels = labels.Take(y_IDs.Length).ToArray();

        var expectedYIfAny = new CpuTensor<float>(new[] { y_IDs.Length, numClass }, labels);

        var dataset = new Biosonar85DirectoryDataSet(
            elementIdToPaths,
            elementIdToCategoryIndex,
            expectedYIfAny,
            NAME,
            Objective_enum.Classification,
            1, // channels
            numClass,
            meanAndVolatilityForEachChannel,
            ResizeStrategyEnum.None,
            new string[0], //featureNames
            y_IDs
        );
        return dataset;
    }
   
    public static InMemoryDataSet Load(string xFileName, [CanBeNull] string yFileNameIfAny, string csvPath, float mean = 0f, float stdDev = 1f)
    {
        var xPath = Path.Join(DataDirectory, xFileName);
        (int[] xShape, int _, int _, int _, int _, int _) = ProcessXFileName(xPath);

        var xTensor = CpuTensor<float>.LoadFromBinFile(xPath, xShape);
        var yTensor = string.IsNullOrEmpty(yFileNameIfAny)
            ?null //no Y available for Dataset
            :CpuTensor<float>.LoadFromBinFile(Path.Join(DataDirectory, yFileNameIfAny), new []{ xShape[0], 1 });


        var xAccBefore = new DoubleAccumulator();
        xAccBefore.Add(xTensor.SpanContent);
        Log.Info($"Stats for {xFileName} before standardization: {xAccBefore}");

        //We standardize the input
        Log.Info($"Mean: {mean}, StdDev: {stdDev}");
        xTensor.LinearFunction(1f / stdDev, xTensor, -mean / stdDev);

        var xAccAfter = new DoubleAccumulator();
        xAccAfter.Add(xTensor.SpanContent);
        Log.Info($"Stats for {xFileName} after standardization: {xAccAfter}");


        var yID = DataFrame.read_string_csv(csvPath).StringColumnContent("id");

        var dataset = new InMemoryDataSet(
            xTensor,
            yTensor,
            xFileName,
            Objective_enum.Classification,
            null, //meanAndVolatilityForEachChannel
            null, //columnNames
            new string[0], //categoricalFeatures,
            "id", //idColumn,
            yID,
            ',' //separator,
        );
        return dataset;
    }

    
    // ReSharper disable once UnusedMember.Global
    public static void Launch_HPO_Transformers(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, "log");
        var searchSpace = new Dictionary<string, object>
        {
            //{"KFold", 2},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            
            //related to model
            { "LossFunction", nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { "EvaluationMetrics", nameof(EvaluationMetricEnum.Accuracy)+","+nameof(EvaluationMetricEnum.AUC)},
            { "BatchSize", new[] {1024} },
            { "NumEpochs", new[] { numEpochs } },
            { "ShuffleDatasetBeforeEachEpoch", true},
            // Optimizer 
            { "OptimizerType", new[] { "AdamW"} },
            //{ "OptimizerType", new[] { "SGD"} },
            //{ "AdamW_L2Regularization", new[]{0.005f , 0.01f } }, //0.005 or 0.01
            //{ "AdamW_L2Regularization", AbstractHyperParameterSearchSpace.Range(0.0005f,0.05f) },
            { "AdamW_L2Regularization", 0.005 },

            //Dataset

            { "ShuffleDatasetBeforeSplit", true},
            { "InputDataType", nameof(Biosonar85DatasetSample.InputDataTypeEnum.TRANSFORMERS_3D)},

            {"embedding_dim", 64},
            {"input_is_already_embedded", true },

            {"encoder_num_transformer_blocks", new[]{4} }, //!D 2
            
            {"encoder_num_heads", new[]{8} },

            {"encoder_mha_use_bias_Q_V_K", new[]{false /*,true*/ } },
            {"encoder_mha_use_bias_O", true  }, // must be true

            {"encoder_mha_dropout", new[]{0.2 } },
            {"encoder_feed_forward_dim", 4*64},
            {"encoder_feed_forward_dropout", new[]{/*0,*/ 0.2 }}, //0.2

            {"encoder_use_causal_mask", true},
            {"output_shape_must_be_scalar", true},
            {"LastActivationLayer", nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},

            {"pooling_before_dense_layer", new[]{ nameof(POOLING_BEFORE_DENSE_LAYER.NONE) /*,nameof(POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling), nameof(POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling)*/ } }, //must be NONE
            {"layer_norm_before_last_dense", false}, // must be false

            // DataAugmentation
            { "DataAugmentationType", nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { "AlphaCutMix", new[] { 0, 0.5, 1.0} },
            { "AlphaMixup", new[] { 0, 0.5, 1.0} },
            { "CutoutPatchPercentage", new[] { 0, 0.1, 0.2} },
            { "RowsCutoutPatchPercentage", new[] { 0, 0.1, 0.2} },
            { "ColumnsCutoutPatchPercentage", new[] { 0, 0.1, 0.2} },
            //{ "HorizontalFlip",new[]{true,false } },
            //{ "VerticalFlip",new[]{true,false } },
            //{ "Rotate180Degrees",new[]{true,false } },
            //{ "FillMode",new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect), nameof(ImageDataGenerator.FillModeEnum.Nearest), nameof(ImageDataGenerator.FillModeEnum.Modulo) } },
            { "FillMode",nameof(ImageDataGenerator.FillModeEnum.Reflect) },
            { "WidthShiftRangeInPercentage", new[] { 0, 0.1, 0.2} },
            { "HeightShiftRangeInPercentage", new[] { 0, 0.1, 0.2 } },
            //{ "ZoomRange", new[] { 0.0 , 0.05,0.1 } },
            
            
            
            // Learning Rate
            //{ "InitialLearningRate", AbstractHyperParameterSearchSpace.Range(0.01f,0.2f,AbstractHyperParameterSearchSpace.range_type.normal)}, 
            //{ "InitialLearningRate", AbstractHyperParameterSearchSpace.Range(0.02f,0.05f)}, //0.02 to 0.05
            { "InitialLearningRate", new[]{0.01, 0.05, 0.1 } },
            //{ "InitialLearningRate", AbstractHyperParameterSearchSpace.Range(0.002f,0.2f)}, //0.02 to 0.05
            // Learning Rate Scheduler
            //{ "LearningRateSchedulerType", new[] { "OneCycle" } },
            { "LearningRateSchedulerType", "CyclicCosineAnnealing" },
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new TransformerNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Global
    public static void Launch_HPO(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, "log");

        var searchSpace = new Dictionary<string, object>
        {
            //{"KFold", 2},
            {"PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            
            //Related to model
            { "LossFunction", nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { "EvaluationMetrics", nameof(EvaluationMetricEnum.AUC)},
            { "BatchSize", new[] {256} },
            { "NumEpochs", new[] { numEpochs } },
            { "ShuffleDatasetBeforeEachEpoch", true},
            // Optimizer 
            { "OptimizerType", new[] { "AdamW" } },
            //{ "OptimizerType", new[] { "SGD"} },
            //{ "AdamW_L2Regularization", AbstractHyperParameterSearchSpace.Range(0.00001f,0.01f, AbstractHyperParameterSearchSpace.range_type.normal) },
            //{ "AdamW_L2Regularization", AbstractHyperParameterSearchSpace.Range(0.00001f,0.01f, AbstractHyperParameterSearchSpace.range_type.normal) },
            { "AdamW_L2Regularization", 0.01 },

            //Dataset
            { "ShuffleDatasetBeforeSplit", true},
            { "InputDataType", nameof(Biosonar85DatasetSample.InputDataTypeEnum.NETWORK_4D)},


            //{ "Use_MaxPooling", new[]{true,false}},
            //{ "Use_AvgPooling", new[]{/*true,*/false}}, //should be false
                

            // DataAugmentation
            { "DataAugmentationType", nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { "AlphaCutMix", 0.5}, //must be > 0
            { "AlphaMixup", new[] { 0 /*, 0.25*/} }, // must be 0
            { "CutoutPatchPercentage", new[] {0, 0.1,0.2} },
            { "RowsCutoutPatchPercentage", 0.2 },
            { "ColumnsCutoutPatchPercentage", new[] {0.1, 0.2} },
            //{ "HorizontalFlip",new[]{true,false } },
            //{ "VerticalFlip",new[]{true,false } },
            //{ "Rotate180Degrees",new[]{true,false } },
            { "FillMode",nameof(ImageDataGenerator.FillModeEnum.Modulo) },
            { "WidthShiftRangeInPercentage", 0.1 },
            //{ "HeightShiftRangeInPercentage", new[] { 0.0 , 0.1,0.2 } }, //0
            //{ "ZoomRange", new[] { 0.0 , 0.05 } },

            

            //{ "SGD_usenesterov", new[] { true, false } },
            //{ "lambdaL2Regularization", new[] { 0.0005, 0.001, 0.00005 } },
            //{ "lambdaL2Regularization", new[] {0.001, 0.0005, 0.0001, 0.00005 } }, // 0.0001 or 0.001
            //{"DefaultMobileBlocksDescriptionCount", new[]{5}},
            //{"LastActivationLayer", nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},

            // Learning Rate
            //{ "InitialLearningRate", new []{0.01, 0.1 }}, //SGD: 0.01 //AdamW: 0.01 or 0.001
            //{ "InitialLearningRate", AbstractHyperParameterSearchSpace.Range(0.001f,0.2f,AbstractHyperParameterSearchSpace.range_type.normal)},
            { "InitialLearningRate", 0.005}, 
            // Learning Rate Scheduler
            //{ "LearningRateSchedulerType", new[] { "OneCycle" } },
            { "LearningRateSchedulerType", "CyclicCosineAnnealing" },
        };

        //var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        //var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
         var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new Biosonar85NetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    public static void Launch_HPO_Spectogram(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, "log");
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, "log");

        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            { "PercentageInTraining", 0.8}, //will be automatically set to 1 if KFold is enabled
            { "ShuffleDatasetBeforeSplit", true},
            //{ "InputDataType", nameof(Biosonar85DatasetSample.InputDataTypeEnum.PNG_1CHANNEL)},
            { "InputDataType", nameof(Biosonar85DatasetSample.InputDataTypeEnum.PNG_1CHANNEL_V2)},

            //related to model
            { "LossFunction", nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { "EvaluationMetrics", nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { "BatchSize", new[] {128} },
            
            { "NumEpochs", new[] { numEpochs } },
            
            { "ShuffleDatasetBeforeEachEpoch", true},
            // Optimizer 
            { "OptimizerType", new[] { "AdamW" /*, "SGD"*/ } },
            //{ "SGD_usenesterov", new[] { true, false } },
            { "lambdaL2Regularization", new[] { 0.0005 /*, 0.001*/} },
            { "AdamW_L2Regularization", new[] { /*0.005, 0.05,*/ 0.0005 } }, // to discard: 0.005, 0.05
            //{ "DefaultMobileBlocksDescriptionCount", 4},
            // Learning Rate
            //{ "InitialLearningRate", AbstractHyperParameterSearchSpace.Range(0.003f, 0.03f)},
            { "InitialLearningRate", new[]{0.01,0.025 } },
            // Learning Rate Scheduler
            //{ "LearningRateSchedulerType", new[] { "OneCycle" } },
            //{ "LearningRateSchedulerType", "CyclicCosineAnnealing" },
            { "LastActivationLayer", nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            { "LearningRateSchedulerType", "CyclicCosineAnnealing" },
            //{"LearningRateSchedulerType", new[]{"OneCycle"/*, "CyclicCosineAnnealing"*/} },
            {"DisableReduceLROnPlateau", true},
            {"OneCycle_DividerForMinLearningRate", 20},
            {"OneCycle_PercentInAnnealing", new[]{ 0.1, 0.4 } },
            {"CyclicCosineAnnealing_nbEpochsInFirstRun", 10},
            {"CyclicCosineAnnealing_nbEpochInNextRunMultiplier", 2},
            {"CyclicCosineAnnealing_MinLearningRate", 1e-5},


            // DataAugmentation
            { "DataAugmentationType", nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { "AlphaCutMix", new[] { 0.0,  1.0} },
            { "AlphaMixup", new[] { 0.0, 1.0} },
            //{ "UseMaxCutMix", new[] { true, false} },
            //{ "UseMaxMixup", new[] { true, false} },
            { "CutoutPatchPercentage", new[] {/* 0 ,*/ 0.1} }, //0
            { "RowsCutoutPatchPercentage", new[] { 0 /*, 0.1, 0.2*/} }, //0 is better
            { "ColumnsCutoutPatchPercentage", new[] { 0 /*, 0.1*/} }, // must be 0
            { "HorizontalFlip",new[]{true,false } },
            //{ "VerticalFlip",new[]{true,false } },
            //{ "Rotate180Degrees",new[]{true,false } },
            { "FillMode",new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            //{ "HeightShiftRangeInPercentage", AbstractHyperParameterSearchSpace.Range(0.05f, 0.30f) }, // must be > 0 , 0.1 seems good default
            { "HeightShiftRangeInPercentage", 0.1},
            { "WidthShiftRangeInPercentage", 0 }, //0

};

        //model: B6E9B1D5CE
        /*
        searchSpace["InitialLearningRate"] = 0.005;
        searchSpace["lambdaL2Regularization"] = 0.0005;
        searchSpace["RowsCutoutPatchPercentage"] = 0;
        searchSpace["HeightShiftRangeInPercentage"] = 0.1;
        searchSpace["AlphaCutMix"] = 0;
        searchSpace["AlphaMixup"] = 1;
        searchSpace["HorizontalFlip"] = false;
        searchSpace["LearningRateSchedulerType"] = "OneCycle";
        searchSpace["OptimizerType"] = "AdamW";
        //searchSpace["AdamW_L2Regularization"] = 0.005;
        */


        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }



    /// <summary>
    /// The default EfficientNet Hyper-Parameters for CIFAR10
    /// </summary>
    /// <returns></returns>
    public static EfficientNetNetworkSample DefaultEfficientNetNetworkSample()
    {
        var config = (EfficientNetNetworkSample)new EfficientNetNetworkSample()
        {
            LossFunction = EvaluationMetricEnum.BinaryCrossentropy,
            EvaluationMetrics = new List<EvaluationMetricEnum> {EvaluationMetricEnum.AUC},
            CompatibilityMode = CompatibilityModeEnum.TensorFlow,
            lambdaL2Regularization = 0.0005,
            //!D WorkingDirectory = Path.Combine(NetworkSample.DefaultWorkingDirectory, CIFAR10DataSet.NAME),
            NumEpochs = 10,
            BatchSize = 64,
            InitialLearningRate = 0.01,


            //Data augmentation
            DataAugmentationType = ImageDataGenerator.DataAugmentationEnum.DEFAULT,
            FillMode = ImageDataGenerator.FillModeEnum.Reflect,
            AlphaMixup = 0.0,
            AlphaCutMix = 0.0,
            CutoutPatchPercentage = 0.0
        }
            .WithSGD(0.9, false)
            .WithCyclicCosineAnnealingLearningRateScheduler(10, 2);
        return config;

    }
}