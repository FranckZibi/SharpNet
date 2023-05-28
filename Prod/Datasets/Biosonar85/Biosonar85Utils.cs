using System.Collections.Generic;
using System.IO;
using JetBrains.Annotations;
using log4net;
using SharpNet.CPU;
using SharpNet.DataAugmentation;
using SharpNet.GPU;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.MathTools;
using SharpNet.Networks.Transformers;
using static SharpNet.Networks.NetworkSample;

namespace SharpNet.Datasets.Biosonar85;

public static class Biosonar85Utils
{
    public const string NAME = "Biosonar85";


    #region public fields & properties
    public static readonly ILog Log = LogManager.GetLogger(typeof(Biosonar85Utils));
    #endregion

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    public static string SubmitDirectory => Path.Combine(WorkingDirectory, "Submit");
    // ReSharper disable once MemberCanBePrivate.Global


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
    const float x_train_mean = -37.076445f;         // for train+test dataset
    
    //const float x_train_volatility = 14.906728f;  //for train dataset only 
    const float x_train_volatility = 14.84446f;     // for train+test dataset

    public static void Run()
    {
        //var trainDataset = Load("X_train_23168_101_64_1024_512.bin", "Y_train_23168_1_64_1024_512.bin", true);
        //var testDataset = Load("X_test_950_101_64_1024_512.bin", "Y_test_950_1_64_1024_512.bin", false);

        //var bin_file = Path.Combine(DataDirectory, "Y_train_ofTdMHi.csv.bin");
        //var tensor = CpuTensor<float>.LoadFromBinFile(bin_file, new[] { -1, 101, 64});

        ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "0908F47370", null, percentageInTraining:0.8, retrainOnFullDataset:false, useAllAvailableCores:true);return;

        //Launch_HPO_Transformers(30); return;
        //Launch_HPO(1);return;
    }



    public static (int[] shape, int n_fft, int hop_len) ProcessXFileName(string xPath)
    {
        var xSplitted = Path.GetFileNameWithoutExtension(xPath).Split("_");
        //var xShape = new [] { int.Parse(xSplitted[^5]), 1, int.Parse(xSplitted[^4]), int.Parse(xSplitted[^3]) };
        var xShape = new[] { int.Parse(xSplitted[^5]), int.Parse(xSplitted[^4]), int.Parse(xSplitted[^3]) };
        var n_fft = int.Parse(xSplitted[^2]);
        var hop_len = int.Parse(xSplitted[^1]);
        return (xShape, n_fft, hop_len);
    }

    public static InMemoryDataSet Load(string xFileName, [CanBeNull] string yFileNameIfAny, string csvFileName)
    {
        var xPath = Path.Join(DataDirectory, xFileName);
        (int[] xShape, int _, int _) = ProcessXFileName(xPath);

        var xTensor = CpuTensor<float>.LoadFromBinFile(xPath, xShape);
        var yTensor = string.IsNullOrEmpty(yFileNameIfAny)
            ?null //no Y available for Dataset
            :CpuTensor<float>.LoadFromBinFile(Path.Join(DataDirectory, yFileNameIfAny), new []{ xShape[0], 1 });

        //We standardize the input
        xTensor.LinearFunction(1f / x_train_volatility, xTensor, -x_train_mean / x_train_volatility);

        var xAcc = new DoubleAccumulator();
        xAcc.Add(xTensor.ContentAsFloatArray());
        Log.Info($"Stats for {xFileName} after standardization: {xAcc}");


        var yID = DataFrame.read_string_csv(Path.Join(DataDirectory, csvFileName)).StringColumnContent("id");

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
            { "RankingEvaluationMetric", nameof(EvaluationMetricEnum.AUC)},
            { "BatchSize", new[] {256} },
            { "NumEpochs", new[] { numEpochs } },
            { "ShuffleDatasetBeforeEachEpoch", true},
            // Optimizer 
            { "OptimizerType", new[] { "AdamW"} },
            //{ "OptimizerType", new[] { "SGD"} },
            //{ "AdamW_L2Regularization", new[]{0.005f , 0.01f } }, //0.005 or 0.01
            //{ "AdamW_L2Regularization", AbstractHyperParameterSearchSpace.Range(0.005f,0.01f) },
            { "AdamW_L2Regularization", 0.005 },

            //Dataset
            { "ShuffleDatasetBeforeSplit", true},
            { "UseTransformers", true},

            {"embedding_dim", 64},
            {"input_is_already_embedded", true },

            {"encoder_num_transformer_blocks", new[]{2} }, //!D 2
            
            {"encoder_num_heads", new[]{8} },

            {"encoder_mha_use_bias_Q_V_K", new[]{false /*,true*/ } },
            {"encoder_mha_use_bias_O", true  }, // must be true

            {"encoder_mha_dropout", new[]{0.2 } },
            {"encoder_feed_forward_dim", 4*64},
            {"encoder_feed_forward_dropout", new[]{/*0,*/ 0.2 }}, //0.2

            {"encoder_use_causal_mask", true},
            {"output_shape_must_be_scalar", true},
            {"lastActivationLayer", nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},

            {"pooling_before_dense_layer", new[]{ nameof(POOLING_BEFORE_DENSE_LAYER.NONE) /*,nameof(POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling), nameof(POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling)*/ } }, //must be NONE
            {"layer_norm_before_last_dense", false}, // must be false

            // DataAugmentation
            { "DataAugmentationType", nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { "AlphaCutMix", new[] { /*0, 0.25, 0.5,*/ 1.0} }, //must be > 0
            { "AlphaMixup", new[] { 0 /*, 0.25*/} }, // must be 0
            //{ "CutoutPatchPercentage", new[] {0, 0.25} },
            //{ "RowsCutoutPatchPercentage", new[] {0, 0.25, 0.5} },
            //{ "ColumnsCutoutPatchPercentage", new[] {0, 0.25, 0.5 } },
            //{ "HorizontalFlip",new[]{true,false } },
            //{ "VerticalFlip",new[]{true,false } },
            //{ "Rotate180Degrees",new[]{true,false } },
            //{ "FillMode",new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect), nameof(ImageDataGenerator.FillModeEnum.Nearest), nameof(ImageDataGenerator.FillModeEnum.Modulo) } },
            { "FillMode",nameof(ImageDataGenerator.FillModeEnum.Modulo) },
            //{ "WidthShiftRangeInPercentage", new[] { 0.0 , 0.25 } },
            //{ "HeightShiftRangeInPercentage", new[] { 0.0 , 0.01,0.05,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 } },
            //{ "HeightShiftRangeInPercentage", AbstractHyperParameterSearchSpace.Range(0,1.0f) },
            //{ "ZoomRange", new[] { 0.0 , 0.05 } },
            
            
            
            // Learning Rate
            //{ "InitialLearningRate", AbstractHyperParameterSearchSpace.Range(0.01f,0.2f,AbstractHyperParameterSearchSpace.range_type.normal)}, 
            //{ "InitialLearningRate", AbstractHyperParameterSearchSpace.Range(0.02f,0.05f)}, //0.02 to 0.05
            { "InitialLearningRate", 0.02},
            // Learning Rate Scheduler
            //{ "LearningRateSchedulerType", new[] { "OneCycle" } },
            { "LearningRateSchedulerType", "CyclicCosineAnnealing" },
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new TransformerNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


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
            { "RankingEvaluationMetric", nameof(EvaluationMetricEnum.AUC)},
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
}