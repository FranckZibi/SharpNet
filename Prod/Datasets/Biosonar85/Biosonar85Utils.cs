using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using JetBrains.Annotations;
using log4net;
using SharpNet.CatBoost;
using SharpNet.CPU;
using SharpNet.Data;
using SharpNet.DataAugmentation;

using SharpNet.GPU;
using SharpNet.HPO;
using SharpNet.HyperParameters;
using SharpNet.LightGBM;
using SharpNet.Networks;
using SharpNet.Networks.Transformers;
using static SharpNet.Networks.NetworkSample;

namespace SharpNet.Datasets.Biosonar85;

public static class Biosonar85Utils
{
    public const string NAME = "Biosonar85";


    #region public fields & properties
    private static readonly ILog Log = LogManager.GetLogger(typeof(Biosonar85Utils));
    #endregion

    public static string WorkingDirectory => Path.Combine(Utils.ChallengesPath, NAME);
    public static string DataDirectory => Path.Combine(WorkingDirectory, "Data");
    public static readonly string[] TargetLabelDistinctValues = { "y" };


    public static void Run()
    {
        Utils.ConfigureGlobalLog4netProperties(WorkingDirectory, NAME);
        Utils.ConfigureThreadLog4netProperties(WorkingDirectory, NAME);

        //Mixup.DisplayStatsForAlphaMixup();return;

        //var trainDataset = Load("X_train_23168_101_64_1024_512.bin", "Y_train_23168_1_64_1024_512.bin", true);
        //var testDataset = Load("X_test_950_101_64_1024_512.bin", "Y_test_950_1_64_1024_512.bin", false);

        //var bin_file = Path.Combine(DataDirectory, "Y_train_ofTdMHi.csv.bin");
        //var tensor = CpuTensor<float>.LoadFromBinFile(bin_file, new[] { -1, 101, 64});
        //ParseLogFile(); return;

        //Log.Info(AllCombinations(0.4, 0.7));return;
        //Log.Info(AllCombinations(0.701, 0.85));return;

        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "2A4F619211", null, percentageInTraining:0.5, retrainOnFullDataset:false, useAllAvailableCores:true);return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "2E3950406D_19", null, percentageInTraining: 0.5, retrainOnFullDataset: false, useAllAvailableCores: true); return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "FB0927A468_17", null, percentageInTraining: 0.5, retrainOnFullDataset: false, useAllAvailableCores: true); return;
        //ChallengeTools.Retrain(Path.Combine(WorkingDirectory, "Dump"), "FB0927A468_17", 2, null, retrainOnFullDataset: false, useAllAvailableCores: true); return;


        //ChallengeTools.ComputeAndSaveFeatureImportance(WorkingDirectory, "674CD08C52", true); return;


        //ComputeAverage_avg();return;

        //Launch_HPO_spectrogram(20); return;

        //Launch_HPO_MEL_SPECTROGRAM_256_801(10); return;
        //Launch_HPO_MEL_SPECTROGRAM_64_401(10); return;

        //Launch_HPO_MEL_SPECTROGRAM_64_401(10); return;
        Launch_HPO_MEL_SPECTROGRAM_128_401(10); return;

        //LaunchCatBoostHPO(1000); return;

        //LaunchLightGBMHPO(350);return;

        //Launch_HPO_Transformers(10); return;
        //Launch_HPO(10);return;
    }

    // ReSharper disable once UnusedMember.Global
    public static string AllCombinations(double minPercentage, double maxPercentage)
    {
        var siteToCount = SiteToCount();

        double totalCount = siteToCount.Select(m => m.Value).Sum();
        List<Tuple<string, double>> allSites = new();
        foreach( var (site,count) in siteToCount)
        {
            allSites.Add(Tuple.Create(site,count/totalCount));
        }
        allSites = allSites.OrderByDescending(s => s.Item2).ToList();

        List<Tuple<string, double>> res = new();
        AllCombinations_Helper(0, 0, new List<string>(), allSites, res);
        res = res.Where(s => s.Item2 >= minPercentage && s.Item2 <= maxPercentage).ToList();
        var allAvailableCombinations = res.OrderBy(s=>s.Item2).Select(s=>s.Item1).ToList();
        return string.Join(", ", allAvailableCombinations);
    }

    private static Dictionary<string, int> SiteToCount()
    {
        var siteToCount = new Dictionary<string, int>();
        var ids = DataFrame.read_string_csv(Biosonar85DatasetSample.Y_train_path).StringColumnContent("id");
        foreach (var id in ids)
        {
            var site = IdToSite(id);
            if (!siteToCount.ContainsKey(site))
            {
                siteToCount[site] = 0;
            }
            ++siteToCount[site];
        }
        return siteToCount;
    }

    private static void ParseLogFile()
    {
        const string log_directory = @"C:\Projects\Challenges\Biosonar85\MandatorySitesForTraining";
        var csvPath = Path.Combine(log_directory, "data.csv");
        Dictionary<string, int> siteToCount = SiteToCount();
        foreach (var logPath in Directory.GetFiles(log_directory, "*.log"))
        {
            var lines = File.ReadAllLines(logPath);
            var currentBlock = new List<string>();

            for (int i = 0; i < lines.Length; ++i)
            {
                if (lines[i].Contains("sites for training"))
                {
                    ProcessBlock(currentBlock, siteToCount, csvPath);
                    currentBlock.Clear();
                    currentBlock.Add(lines[i]);
                }
                else
                {
                    currentBlock.Add(lines[i]);
                    if (i == lines.Length - 1)
                    {
                        ProcessBlock(currentBlock, siteToCount, csvPath);
                        currentBlock.Clear();
                    }
                }
            }
        }

    }
    private static void ProcessBlock(List<string> lines, Dictionary<string, int> siteToCount, string csvPath)
    {
        var sites = lines.FirstOrDefault(l => l.Contains("sites for training"));
        if (string.IsNullOrEmpty(sites))
        {
            return; 
        }
        sites = sites.Split().Last();
        double totalCount = siteToCount.Select(m => m.Value).Sum();
        double siteCount = 0;
        foreach (var site in sites.Split(','))
        {
            siteCount += siteToCount[site];
        }   
        var networkName = lines.FirstOrDefault(l => l.Contains("Network Name:"));
        if (string.IsNullOrEmpty(networkName))
        {
            return ;
        }
        networkName = networkName.Split().Last();
        int bestEpoch = -1;
        float bestScore = 0;
        for (int epoch = 1;; ++epoch)
        {
            var epochLine = lines.FirstOrDefault(l => l.Contains($"Epoch {epoch}/20"));
            if (string.IsNullOrEmpty(epochLine))
            {
                break;
            }
            const string toFind = "val_accuracy:";
            var index = epochLine.IndexOf(toFind);
            if (index < 0)
            {
                break;
            }
            var score = float.Parse(epochLine.Substring(index + toFind.Length).Trim().Split()[0]);
            if (bestEpoch == -1 || score > bestScore)
            {
                bestEpoch = epoch;
                bestScore = score;
            }
        }

        if (bestEpoch == -1)
        {
            return;
        }

        if (!File.Exists(csvPath))
        {
            File.AppendAllText(csvPath, "Sep=;"+Environment.NewLine);
            File.AppendAllText(csvPath, $"NetworkName;Sites;% in training;BestEpoch;BestScore" + Environment.NewLine);
        }
        File.AppendAllText(csvPath, $"{networkName};{sites};{siteCount / totalCount};{bestEpoch};{bestScore}" + Environment.NewLine);
    }



    private static void AllCombinations_Helper(int depth, double totalCount, List<string> currentPath, List<Tuple<string, double>> allSites, List<Tuple<string, double>> res)
    {
        if (depth>=allSites.Count)
        {
            res.Add(Tuple.Create("\""+string.Join(",", currentPath)+"\"", totalCount));
            return;
        }

        //without site at index 'depth'
        AllCombinations_Helper(depth + 1, totalCount, currentPath, allSites, res);

        //with site at index 'depth'
        currentPath.Add(allSites[depth].Item1);
        AllCombinations_Helper(depth + 1, totalCount+ allSites[depth].Item2, currentPath, allSites, res);
        currentPath.RemoveAt(currentPath.Count - 1);
    }

    public static string IdToSite(string id) 
    {
        return id.Split(new[] { '-', '.' })[1]; 
    }



    // ReSharper disable once UnusedMember.Global
    public static void ComputeAverage_avg()
    {
        var dfs = new List<DataFrame>();
        const string path = @"\\RYZEN2700X-DEV\Challenges\Biosonar85\Submit\";
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
        
        var meanAndVolatilityForEachChannel = new List<Tuple<float, float>> { Tuple.Create(mean, stdDev) };
        
        var xPath = Path.Join(DataDirectory, xFileName);
        (int[] xShape, int _, int _, int _, int _, int _) = ProcessXFileName(xPath);

        ISample.Log.Info($"Loading {xShape[0]} tensors from {xPath} with shape {Tensor.ShapeToString(xShape)} (Y file:{yFileNameIfAny})");


        var xTensor = CpuTensor<float>.LoadFromBinFile(xPath, xShape);
        var yTensor = string.IsNullOrEmpty(yFileNameIfAny)
            ?null //no Y available for Dataset
            :CpuTensor<float>.LoadFromBinFile(Path.Join(DataDirectory, yFileNameIfAny), new []{ xShape[0], 1 });


        mean = 0f; stdDev = 1f; //!D //no standardization
        /*
        // we disable standardization
        var xAccBefore = new DoubleAccumulator();
        xAccBefore.Add(xTensor.SpanContent);
        Log.Info($"Stats for {xFileName} before standardization: {xAccBefore}");

        //We standardize the input
        Log.Info($"Mean: {mean}, StdDev: {stdDev}");
        xTensor.LinearFunction(1f / stdDev, xTensor, -mean / stdDev);

        var xAccAfter = new DoubleAccumulator();
        xAccAfter.Add(xTensor.SpanContent);
        Log.Info($"Stats for {xFileName} after standardization: {xAccAfter}");
        */

        var yID = DataFrame.read_string_csv(csvPath).StringColumnContent("id");

        var dataset = new Biosonar85InMemoryDataSet(
            xTensor,
            yTensor,
            xFileName,
            meanAndVolatilityForEachChannel,
            yID);
        return dataset;
    }


    public static TensorListDataSet LoadTensorListDataSet(string xFileName, [CanBeNull] string yFileNameIfAny, string csvPath, float mean, float stdDev)
    {
        //!D we disable standardization
        mean = 0;
        stdDev = 1;

        List<Tuple<float, float>> meanAndVolatilityForEachChannel = null;
        var xPath = Path.Join(DataDirectory, xFileName);
        (int[] xShape, int _, int _, int _, int _, int _) = ProcessXFileName(xPath);

        var xTensorList = CpuTensor<float>.LoadTensorListFromBinFileAndStandardizeIt(xPath, xShape , mean, stdDev);
        var yTensor = string.IsNullOrEmpty(yFileNameIfAny)
            ? null //no Y available for Dataset
            : CpuTensor<float>.LoadFromBinFile(Path.Join(DataDirectory, yFileNameIfAny), new[] { xShape[0], 1 });

        var yID = DataFrame.read_string_csv(csvPath).StringColumnContent("id");

        var dataset = new TensorListDataSet(
            xTensorList,
            yTensor,
            xFileName,
            Objective_enum.Classification,
            meanAndVolatilityForEachChannel,
            null /* columnNames*/, 
            new string[0],
            yID,
            "id",
            ',');
        return dataset;
    }


    // ReSharper disable once UnusedMember.Global
    private static void Launch_HPO_Transformers(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //{"KFold", 2},
            {nameof(AbstractDatasetSample.PercentageInTraining), 0.8}, //will be automatically set to 1 if KFold is enabled
            
            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)+","+nameof(EvaluationMetricEnum.AUC)},
            { nameof(NetworkSample.BatchSize), new[] {1024} },
            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },
            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW"} },
            //{ nameof(NetworkSample.OptimizerType), new[] { "SGD"} },
            //{ nameof(NetworkSample.AdamW_L2Regularization), new[]{0.005f , 0.01f } }, //0.005 or 0.01
            //{ nameof(NetworkSample.AdamW_L2Regularization), AbstractHyperParameterSearchSpace.Range(0.0005f,0.05f) },
            { nameof(NetworkSample.AdamW_L2Regularization), 0.005 },

            //Dataset

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.TRANSFORMERS_3D)},

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
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},

            {"pooling_before_dense_layer", new[]{ nameof(POOLING_BEFORE_DENSE_LAYER.NONE) /*,nameof(POOLING_BEFORE_DENSE_LAYER.GlobalAveragePooling), nameof(POOLING_BEFORE_DENSE_LAYER.GlobalMaxPooling)*/ } }, //must be NONE
            {"layer_norm_before_last_dense", false}, // must be false

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), new[] { 0, 0.5, 1.0} },
            { nameof(NetworkSample.AlphaMixup), new[] { 0, 0.5, 1.0} },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] { 0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] { 0, 0.1, 0.2} },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage), new[] { 0, 0.1, 0.2} },
            //{ nameof(NetworkSample.HorizontalFlip),new[]{true,false } },
            //{ nameof(NetworkSample.VerticalFlip),new[]{true,false } },
            //{ nameof(NetworkSample.Rotate180Degrees),new[]{true,false } },
            //{ nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect), nameof(ImageDataGenerator.FillModeEnum.Nearest), nameof(ImageDataGenerator.FillModeEnum.Modulo) } },
            { nameof(NetworkSample.FillMode),nameof(ImageDataGenerator.FillModeEnum.Reflect) },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[] { 0, 0.1, 0.2} },
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[] { 0, 0.1, 0.2 } },
            //{ nameof(NetworkSample.ZoomRange), new[] { 0.0 , 0.05,0.1 } },
            
            
            
            // Learning Rate
            //{ nameof(NetworkSample.InitialLearningRate), AbstractHyperParameterSearchSpace.Range(0.01f,0.2f,AbstractHyperParameterSearchSpace.range_type.normal)}, 
            //{ nameof(NetworkSample.InitialLearningRate), AbstractHyperParameterSearchSpace.Range(0.02f,0.05f)}, //0.02 to 0.05
            { nameof(NetworkSample.InitialLearningRate), new[]{0.01, 0.05, 0.1 } },
            //{ nameof(NetworkSample.InitialLearningRate), AbstractHyperParameterSearchSpace.Range(0.002f,0.2f)}, //0.02 to 0.05
            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), new[] { "OneCycle" } },
            { nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new TransformerNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Global
    private static void Launch_HPO(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //{"KFold", 2},
            {nameof(AbstractDatasetSample.PercentageInTraining), 0.8}, //will be automatically set to 1 if KFold is enabled
            
            //Related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.AUC)},
            { nameof(NetworkSample.BatchSize), new[] {256} },
            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },
            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            //{ nameof(NetworkSample.OptimizerType), new[] { "SGD"} },
            //{ nameof(NetworkSample.AdamW_L2Regularization), AbstractHyperParameterSearchSpace.Range(0.00001f,0.01f, AbstractHyperParameterSearchSpace.range_type.normal) },
            //{ nameof(NetworkSample.AdamW_L2Regularization), AbstractHyperParameterSearchSpace.Range(0.00001f,0.01f, AbstractHyperParameterSearchSpace.range_type.normal) },
            { nameof(NetworkSample.AdamW_L2Regularization), 0.01 },

            //Dataset
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.NETWORK_4D)},


            //{ "Use_MaxPooling", new[]{true,false}},
            //{ "Use_AvgPooling", new[]{/*true,*/false}}, //should be false
                

            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), 0.5}, //must be > 0
            { nameof(NetworkSample.AlphaMixup), new[] { 0 /*, 0.25*/} }, // must be 0
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1,0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), 0.2 },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage), new[] {0.1, 0.2} },
            //{ nameof(NetworkSample.HorizontalFlip),new[]{true,false } },
            //{ nameof(NetworkSample.VerticalFlip),new[]{true,false } },
            //{ nameof(NetworkSample.Rotate180Degrees),new[]{true,false } },
            { nameof(NetworkSample.FillMode),nameof(ImageDataGenerator.FillModeEnum.Modulo) },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), 0.1 },
            //{ nameof(NetworkSample.HeightShiftRangeInPercentage), new[] { 0.0 , 0.1,0.2 } }, //0
            //{ nameof(NetworkSample.ZoomRange), new[] { 0.0 , 0.05 } },

            

            //{ nameof(NetworkSample.SGD_usenesterov), new[] { true, false } },
            //{ nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005, 0.001, 0.00005 } },
            //{ nameof(NetworkSample.lambdaL2Regularization), new[] {0.001, 0.0005, 0.0001, 0.00005 } }, // 0.0001 or 0.001
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), new[]{5}},
            //{ nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},

            // Learning Rate
            //{ nameof(NetworkSample.InitialLearningRate), new []{0.01, 0.1 }}, //SGD: 0.01 //AdamW: 0.01 or 0.001
            //{ nameof(NetworkSample.InitialLearningRate), AbstractHyperParameterSearchSpace.Range(0.001f,0.2f,AbstractHyperParameterSearchSpace.range_type.normal)},
            { nameof(NetworkSample.InitialLearningRate), 0.005}, 
            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), new[] { "OneCycle" } },
            { nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
        };

        //var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        //var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
         var hpo = new RandomSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new Biosonar85NetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Global
    private static void Launch_HPO_spectrogram(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.8}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            //{ nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.PNG_1CHANNEL)},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.PNG_1CHANNEL_V2)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.94},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BatchSize), new[] {128} },
            
            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },
            
            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" /*, "SGD"*/ } },
            //{ nameof(NetworkSample.SGD_usenesterov), new[] { true, false } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005 /*, 0.001*/} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] { /*0.005,0.001,, 0.05*/ 0.0005, 0.00025 } }, // to discard: 0.005, 0.05 0.001
            
            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), -1 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 4 },

            // Learning Rate
            //{ nameof(NetworkSample.InitialLearningRate), AbstractHyperParameterSearchSpace.Range(0.003f, 0.03f)},
            
            { nameof(NetworkSample.InitialLearningRate), new[]{0.0025, 0.005 , 0.01} }, //0.005 or 0.01

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), new[] { "OneCycle" } },
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[]{ 0.1} }, //discard: 0.4
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), new[] { 0.0,  1.0} }, //0
            { nameof(NetworkSample.AlphaMixup), new[] { 0.0,  1.0} }, // 1 or 0 , 1 seems better
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {/*0,*/ 0.05, 0.1} }, //0.1 or 0.2
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {/*0 ,*/ 0.1} }, //0 or 0.1
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 }, // must be 0            
            { nameof(NetworkSample.HorizontalFlip),new[]{true,false } },
            
            //{ nameof(NetworkSample.VerticalFlip),new[]{true,false } },
            //{ nameof(NetworkSample.Rotate180Degrees),new[]{true,false } },
            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            //{ nameof(NetworkSample.HeightShiftRangeInPercentage), AbstractHyperParameterSearchSpace.Range(0.05f, 0.30f) }, // must be > 0 , 0.1 seems good default
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0.05, 0.1   } }, //to discard: 0.2
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}}, // must be 0

        };

        //model: FB0927A468_17
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.00025;
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.0025;
        searchSpace[nameof(NetworkSample.HorizontalFlip)] = true;
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.5;
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.MinimumRankingScoreToSaveModel)] = 0.93;


        //!D
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = AbstractHyperParameterSearchSpace.Range(0.025f, 0.075f);
        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = AbstractHyperParameterSearchSpace.Range(0.0002f, 0.0003f);
        searchSpace[nameof(NetworkSample.AlphaMixup)] = AbstractHyperParameterSearchSpace.Range(0.75f, 1.25f);
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = AbstractHyperParameterSearchSpace.Range(0.002f, 0.03f);
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = AbstractHyperParameterSearchSpace.Range(0.05f, 0.15f);
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = AbstractHyperParameterSearchSpace.Range(0.05f, 0.15f);
        searchSpace[nameof(NetworkSample.MinimumRankingScoreToSaveModel)] = 0.94;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = new[] { -1,5,6 };


        //searchSpace["MandatorySitesForTraining"] = new string[] { "GUA,StEUS", "BON,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,ARUBA", "StMARTIN,JAM,ARUBA,StEUS,BERMUDE", "StMARTIN,JAM,ARUBA,StEUS,BAHAMAS", "BON,JAM,ARUBA,StEUS", "BON,StMARTIN", "GUA,StEUS,BERMUDE", "GUA,StEUS,BAHAMAS", "GUA,ARUBA,BERMUDE", "GUA,ARUBA,BAHAMAS", "StMARTIN,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "BON,JAM,ARUBA,StEUS,BERMUDE", "BON,JAM,ARUBA,StEUS,BAHAMAS", "BON,StMARTIN,BERMUDE", "BON,StMARTIN,BAHAMAS", "GUA,StEUS,BAHAMAS,BERMUDE", "GUA,ARUBA,BAHAMAS,BERMUDE", "BON,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,ARUBA,StEUS", "BON,StMARTIN,BAHAMAS,BERMUDE", "BON,StMARTIN,StEUS", "BON,StMARTIN,ARUBA", "GUA,ARUBA,StEUS,BERMUDE", "GUA,ARUBA,StEUS,BAHAMAS", "BON,StMARTIN,StEUS,BERMUDE", "BON,StMARTIN,StEUS,BAHAMAS", "BON,StMARTIN,ARUBA,BERMUDE", "BON,StMARTIN,ARUBA,BAHAMAS", "GUA,ARUBA,StEUS,BAHAMAS,BERMUDE", "BON,StMARTIN,StEUS,BAHAMAS,BERMUDE", "BON,StMARTIN,ARUBA,BAHAMAS,BERMUDE", "BON,StMARTIN,ARUBA,StEUS", "GUA,JAM", "BON,StMARTIN,ARUBA,StEUS,BERMUDE", "BON,StMARTIN,ARUBA,StEUS,BAHAMAS", "GUA,JAM,BERMUDE", "GUA,JAM,BAHAMAS", "BON,StMARTIN,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,JAM,BAHAMAS,BERMUDE", "GUA,JAM,StEUS", "GUA,JAM,ARUBA", "BON,StMARTIN,JAM", "GUA,JAM,StEUS,BERMUDE", "GUA,JAM,StEUS,BAHAMAS", "GUA,JAM,ARUBA,BERMUDE", "GUA,JAM,ARUBA,BAHAMAS", "BON,StMARTIN,JAM,BERMUDE", "BON,StMARTIN,JAM,BAHAMAS", "GUA,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,JAM,ARUBA,StEUS", "BON,StMARTIN,JAM,BAHAMAS,BERMUDE", "GUA,StMARTIN", "BON,StMARTIN,JAM,StEUS", "BON,StMARTIN,JAM,ARUBA", "GUA,JAM,ARUBA,StEUS,BERMUDE", "GUA,JAM,ARUBA,StEUS,BAHAMAS", "GUA,StMARTIN,BERMUDE", "GUA,StMARTIN,BAHAMAS", "GUA,BON", "BON,StMARTIN,JAM,StEUS,BERMUDE", "BON,StMARTIN,JAM,StEUS,BAHAMAS", "BON,StMARTIN,JAM,ARUBA,BERMUDE", "BON,StMARTIN,JAM,ARUBA,BAHAMAS", "GUA,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,BAHAMAS,BERMUDE", "GUA,BON,BERMUDE", "GUA,BON,BAHAMAS", "BON,StMARTIN,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,StEUS", "BON,StMARTIN,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA", "GUA,BON,BAHAMAS,BERMUDE", "BON,StMARTIN,JAM,ARUBA,StEUS", "GUA,StMARTIN,StEUS,BERMUDE", "GUA,StMARTIN,StEUS,BAHAMAS", "GUA,StMARTIN,ARUBA,BERMUDE", "GUA,StMARTIN,ARUBA,BAHAMAS", "GUA,BON,StEUS", "GUA,BON,ARUBA", "BON,StMARTIN,JAM,ARUBA,StEUS,BERMUDE", "BON,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS", "GUA,StMARTIN,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA,BAHAMAS,BERMUDE", "GUA,BON,StEUS,BERMUDE", "GUA,BON,StEUS,BAHAMAS", "GUA,BON,ARUBA,BERMUDE", "GUA,BON,ARUBA,BAHAMAS", "BON,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA,StEUS", "GUA,BON,StEUS,BAHAMAS,BERMUDE", "GUA,BON,ARUBA,BAHAMAS,BERMUDE", "GUA,StMARTIN,ARUBA,StEUS,BERMUDE", "GUA,StMARTIN,ARUBA,StEUS,BAHAMAS", "GUA,BON,ARUBA,StEUS", "GUA,StMARTIN,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,BON,ARUBA,StEUS,BERMUDE", "GUA,BON,ARUBA,StEUS,BAHAMAS", "GUA,BON,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM" };
        // the 39 missings
        //searchSpace["MandatorySitesForTraining"] = new string[] { "GUA,StEUS", "GUA,ARUBA","StMARTIN,JAM,ARUBA,StEUS,BAHAMAS","BON,JAM,ARUBA,StEUS","GUA,ARUBA,BERMUDE","BON,JAM,ARUBA,StEUS,BERMUDE","BON,StMARTIN,BERMUDE","BON,StMARTIN,StEUS","GUA,ARUBA,StEUS,BERMUDE","BON,StMARTIN,StEUS,BERMUDE","BON,StMARTIN,ARUBA,BERMUDE","BON,StMARTIN,ARUBA,BAHAMAS,BERMUDE","BON,StMARTIN,ARUBA,StEUS","BON,StMARTIN,ARUBA,StEUS,BERMUDE","BON,StMARTIN,ARUBA,StEUS,BAHAMAS","GUA,JAM,BAHAMAS","GUA,JAM,BAHAMAS,BERMUDE","GUA,JAM,StEUS","GUA,JAM,ARUBA,BERMUDE","GUA,JAM,ARUBA,BAHAMAS","BON,StMARTIN,JAM,BERMUDE","GUA,JAM,ARUBA,BAHAMAS,BERMUDE","BON,StMARTIN,JAM,BAHAMAS,BERMUDE","BON,StMARTIN,JAM,StEUS","GUA,JAM,ARUBA,StEUS,BAHAMAS","GUA,BON","GUA,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE","GUA,StMARTIN,StEUS,BERMUDE","GUA,StMARTIN,StEUS,BAHAMAS","GUA,StMARTIN,ARUBA,BAHAMAS","GUA,BON,StEUS","GUA,BON,ARUBA","BON,StMARTIN,JAM,ARUBA,StEUS,BERMUDE","GUA,StMARTIN,ARUBA,BAHAMAS,BERMUDE","GUA,BON,ARUBA,BAHAMAS,BERMUDE","GUA,BON,ARUBA,StEUS","GUA,BON,ARUBA,StEUS,BAHAMAS", "GUA,StMARTIN,JAM" };
        //searchSpace["MandatorySitesForTraining"] = new string[] { "GUA,ARUBA","StMARTIN,JAM,ARUBA,StEUS,BAHAMAS","BON,JAM,ARUBA,StEUS","BON,StMARTIN,BERMUDE","BON,StMARTIN,StEUS","BON,StMARTIN,StEUS,BERMUDE","BON,StMARTIN,ARUBA,StEUS","BON,StMARTIN,ARUBA,StEUS,BERMUDE","BON,StMARTIN,ARUBA,StEUS,BAHAMAS","GUA,JAM,BAHAMAS","GUA,JAM,BAHAMAS,BERMUDE","BON,StMARTIN,JAM","BON,StMARTIN,JAM,BERMUDE","BON,StMARTIN,JAM,BAHAMAS,BERMUDE","BON,StMARTIN,JAM,StEUS","GUA,JAM,ARUBA,StEUS,BAHAMAS","GUA,BON","GUA,StMARTIN,StEUS,BERMUDE","GUA,StMARTIN,StEUS,BAHAMAS","GUA,StMARTIN,ARUBA,BAHAMAS","GUA,BON,StEUS","GUA,BON,ARUBA,BAHAMAS,BERMUDE","GUA,BON,ARUBA,StEUS", "GUA,StMARTIN,JAM,BERMUDE", "GUA,StMARTIN,JAM,BAHAMAS", "GUA,BON,JAM", "GUA,StMARTIN,JAM,BAHAMAS,BERMUDE", "GUA,BON,JAM,BERMUDE", "GUA,BON,JAM,BAHAMAS", "GUA,StMARTIN,JAM,StEUS", "GUA,StMARTIN,JAM,ARUBA", "GUA,BON,JAM,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM,StEUS,BERMUDE", "GUA,StMARTIN,JAM,StEUS,BAHAMAS", "GUA,StMARTIN,JAM,ARUBA,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,BAHAMAS", "GUA,BON,JAM,StEUS", "GUA,BON,JAM,ARUBA", "GUA,StMARTIN,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,BON,JAM,StEUS,BERMUDE", "GUA,BON,JAM,StEUS,BAHAMAS", "GUA,BON,JAM,ARUBA,BERMUDE", "GUA,BON,JAM,ARUBA,BAHAMAS", "GUA,StMARTIN,JAM,ARUBA,StEUS", "GUA,BON,JAM,StEUS,BAHAMAS,BERMUDE", "GUA,BON,JAM,ARUBA,BAHAMAS,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,StEUS,BERMUDE", "GUA,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS", "GUA,BON,JAM,ARUBA,StEUS", "GUA,BON,StMARTIN", "GUA,StMARTIN,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,BON,JAM,ARUBA,StEUS,BERMUDE", "GUA,BON,JAM,ARUBA,StEUS,BAHAMAS", "GUA,BON,StMARTIN,BERMUDE", "GUA,BON,StMARTIN,BAHAMAS", "GUA,BON,JAM,ARUBA,StEUS,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,StEUS", "GUA,BON,StMARTIN,ARUBA", "GUA,BON,StMARTIN,StEUS,BERMUDE", "GUA,BON,StMARTIN,StEUS,BAHAMAS", "GUA,BON,StMARTIN,ARUBA,BERMUDE", "GUA,BON,StMARTIN,ARUBA,BAHAMAS", "GUA,BON,StMARTIN,StEUS,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,ARUBA,BAHAMAS,BERMUDE", "GUA,BON,StMARTIN,ARUBA,StEUS" };
        //searchSpace["MandatorySitesForTraining"] = "GUA,BON,ARUBA";
        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    private static void Launch_HPO_MEL_SPECTROGRAM_128_401(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        // stats from FindBestLearningRate
        //     DefaultMobileBlocksDescriptionCount     BatchSize   Best learning rate                           Free GPU Memory     
        //     B0 + -1                                 64          0.002 => 0.030 or 0.08                       12700MB             A305BB2D2F
        //     B0 + 5                                  64          0.003 => 0.025 or 0.06                       14587MB             93898B4B73  
        //     B0 + 5                                  128         ?     => 0.002                               7268MB              C05FD055A3
        //     B0 + -1                                 128         ?     => 0.01 or 0.1                         5500MB             A305BB2D2F                                       


        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_128_401)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.93},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BatchSize), new[] {128} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] {0.00015,  0.00025,  0.0005} },

            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), -1 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 5 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), new[]{0.00125, 0.0025, 0.005 } },

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[]{ 0.1} },
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixup), new[] { 1.0, 1.2} },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {0.1, 0.2} },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            { nameof(NetworkSample.HorizontalFlip),new[]{true,false } },

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0, 0.05} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };

        //model: FB0927A468_17
        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.00025;
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = -1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.HorizontalFlip)] = true;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.0025;
        searchSpace[nameof(NetworkSample.MinimumRankingScoreToSaveModel)] = 0.94;
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 0.5;
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;



        //searchSpace[nameof(NetworkSample.InitialLearningRate)] = new []{ 0.002, 0.01, 0.03, 0.02, 0.025, 0.08, 0.06 };
        //searchSpace[nameof(NetworkSample.NumEpochs)] = 10;
        //searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = new[]{0,0.1};
        //searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = new[]{5, -1};
        //best: DAA0AA4E45 0.9381
        //LossFunction = BinaryCrossentropy 
        //InitialLearningRate = 0.01
        //DefaultMobileBlocksDescriptionCount = 5
        //OneCycle_PercentInAnnealing = 0


        //searchSpace[nameof(NetworkSample.InitialLearningRate)] = new[] { 0.001, 0.002, 0.01, 0.02, 0.05 };
        //searchSpace[nameof(NetworkSample.NumEpochs)] = 10;
        //searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = 0; //new[]{0,0.1};
        //searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; // new[]{5, -1};
        //searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BCEWithFocalLoss);
        //searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = new[]{0, 0.5 ,1 ,2 ,3, 5};
        //best: AA88481AB3 0.9444
        //LossFunction = BCEWithFocalLoss 
        //BCEWithFocalLoss_Gamma = 0.5
        //InitialLearningRate = 0.02
        //.../...
        //DefaultMobileBlocksDescriptionCount = 5
        //OneCycle_PercentInAnnealing = 0


        //searchSpace[nameof(NetworkSample.InitialLearningRate)] = new[] { 0.015, 0.02, 0.025, 0.03 };
        //searchSpace[nameof(NetworkSample.NumEpochs)] = 10;
        //searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = 0; //new[]{0,0.1};
        //searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; // new[]{5, -1};
        //searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BCEWithFocalLoss);
        //searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = new[] { 0.3, 0.4, 0.5, 0.6, 0.7};
        //best: C05BE7F68B 0.9441
        //BCEWithFocalLoss_Gamma = 0.7
        //InitialLearningRate = 0.02


        //searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = new[] { 0.0001, 0.00025, 0.0005 };
        //searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = new[] { 0.5, 0.7, 1.0 };
        //searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.02;
        //searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BCEWithFocalLoss);
        //searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = new[] { 20, 25 };
        //searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = new[] { 0.5, 0.8 };
        //best: D64580081B 0.9448601
        //AdamW_L2Regularization = 0.0005
        //PercentageInTraining = 0.5
        //OneCycle_DividerForMinLearningRate = 25
        //BCEWithFocalLoss_Gamma = 1

        //searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.0005;
        //searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.02;
        //searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = 0.7;
        //searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.0005;
        //searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = new[] { 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8 };
        //searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 25;
        //PercentageInTraining = 0.50 or 0.55 or 0.60

        //searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = 0.0005;
        //searchSpace[nameof(NetworkSample.AlphaMixup)] = new[] { 1, 1.2 };
        //searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = new[] { 0, 0.05, 0.1, 0.2 };
        //searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = new[] { 0, 1, 2 };
        //searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = new[] { 0, 0.05, 0.1, 0.2 };
        //searchSpace[nameof(NetworkSample.CutoutCount)] = new[] { 0, 1, 2 };
        //searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = new[] { 0, 0.05, 0.1, 0.15 };
        //searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = new[] { 0, 0.05, 0.1 };
        //searchSpace[nameof(NetworkSample.RowsCutoutCount)] = new[] { 0, 1, 2 };
        //searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = new[] { 0, 0.025 };
        //best: 0.9420
        //CutoutCount = 1
        //CutoutPatchPercentage = 0.1
        //ColumnsCutoutCount = 0
        //ColumnsCutoutPatchPercentage = 0
        //RowsCutoutCount = 1
        //RowsCutoutPatchPercentage = 0.1
        //AlphaMixup = 1.2
        //HeightShiftRangeInPercentage = 0
        //WidthShiftRangeInPercentage = 0

        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = new[] { 0.00025, 0.0005 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = 1.0; //new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = new[] {0, 0.7};
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass)] = new[]{0.41,0.5};
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = 0.05;
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.02;
        searchSpace[nameof(NetworkSample.LossFunction)] = nameof(EvaluationMetricEnum.BinaryCrossentropy); //new[]{nameof(EvaluationMetricEnum.BinaryCrossentropy), nameof(EvaluationMetricEnum.BCEWithFocalLoss)};
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = new[]{0,0.1};
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = new[]{5,-1}; //new[]{2,3,4,5,6, -1};
        //best: 0.9440      393C258A23
        //AdamW_L2Regularization = 0.00025
        //DefaultMobileBlocksDescriptionCount = 5
        //OneCycle_PercentInAnnealing = 0




        searchSpace[nameof(NetworkSample.AdamW_L2Regularization)] = new[] { 0.00025, 0.0005 };
        searchSpace[nameof(NetworkSample.AlphaCutMix)] = 0;
        searchSpace[nameof(NetworkSample.AlphaMixup)] = new[] { 1, 1.2, 1.5, 2.0,3 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_Gamma)] = new[] { 0, 0.7 };
        searchSpace[nameof(NetworkSample.BCEWithFocalLoss_PercentageInTrueClass)] = new[] { 0.4, 0.5, 0.6 };
        searchSpace[nameof(NetworkSample.ColumnsCutoutPatchPercentage)] = 0;
        searchSpace[nameof(NetworkSample.ColumnsCutoutCount)] = 0;
        searchSpace[nameof(NetworkSample.CutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.CutoutCount)] = 1;
        searchSpace[nameof(NetworkSample.HeightShiftRangeInPercentage)] = new[] { 0.05, 0.1 };
        searchSpace[nameof(NetworkSample.InitialLearningRate)] = 0.02; //new[]{ 0.01, 0.02,0.05};
        searchSpace[nameof(NetworkSample.LossFunction)] = new[]{nameof(EvaluationMetricEnum.BinaryCrossentropy), nameof(EvaluationMetricEnum.BCEWithFocalLoss)};
        searchSpace[nameof(NetworkSample.NumEpochs)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_DividerForMinLearningRate)] = 20;
        searchSpace[nameof(NetworkSample.OneCycle_PercentInAnnealing)] = new[]{0,0.1};
        searchSpace[nameof(NetworkSample.RowsCutoutPatchPercentage)] = 0.1;
        searchSpace[nameof(NetworkSample.RowsCutoutCount)] = 1; //new[] { 1, 10 };
        searchSpace[nameof(NetworkSample.WidthShiftRangeInPercentage)] = 0;
        searchSpace[nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount)] = 5; //new[]{4,5,6, -1};



        //TO find the best learning rate
        //searchSpace[nameof(NetworkSample.BatchSize)] = 8;
        //searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 1.0;

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    private static void Launch_HPO_MEL_SPECTROGRAM_256_801(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        // stats from FindBestLearningRate
        //     DefaultMobileBlocksDescriptionCount     BatchSize   Best learning rate                           Free GPU Memory
        //     B0 + -1                                 8           0.0001   0.001
        //     B0 + -1                                 32          0.006    0.02 (between 0.01 and 0.025)
        //     B2 + -1                                 8           ?        0.045                               16000 MB/25GB   D58CB83204
        //     B2 + -1                                 16          ?        0.05                                10171 MB/25GB   32093F4126
        //     B2 + -1                                 32          ?        0.05                                  666 MB/25GB   9D677FD756

        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_256_801)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.94},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},
            { nameof(NetworkSample.BatchSize), new[] {32} }, //because of memory issues, we have to use small batches

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] {0.00015,  0.00025,  0.0005} },
            
            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), -1 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 5 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), new[]{0.00125, 0.0025, 0.005 } },

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[]{ 0.1} },
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixup), new[] { 1.0, 1.2} },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {0.1, 0.2} },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            { nameof(NetworkSample.HorizontalFlip),new[]{true,false } },
            
            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0, 0.05} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };


        //TO find the best learning rate
        //searchSpace[nameof(NetworkSample.BatchSize)] = 8;
        //searchSpace[nameof(AbstractDatasetSample.PercentageInTraining)] = 1.0;

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }



    private static void Launch_HPO_MEL_SPECTROGRAM_64_401(int numEpochs = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        // stats from FindBestLearningRate
        //     DefaultMobileBlocksDescriptionCount     BatchSize   Best learning rate:
        //     B0 + -1                                 8           0.0001   0.015
        //     B0 + -1                                 32          0.001    0.025
        //     B0 + -1                                 64          0.0005   0.03
        //     B0 + -1                                 128         0.002    0.05

        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled

            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.MEL_SPECTROGRAM_64_401)},
            { nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.93},

            //related to model
            { nameof(NetworkSample.LossFunction), nameof(EvaluationMetricEnum.BinaryCrossentropy)},
            { nameof(NetworkSample.EvaluationMetrics), nameof(EvaluationMetricEnum.Accuracy)/*+","+nameof(EvaluationMetricEnum.AUC)*/},


            { nameof(NetworkSample.BatchSize), new[] {128} },

            { nameof(NetworkSample.NumEpochs), new[] { numEpochs } },

            { nameof(NetworkSample.ShuffleDatasetBeforeEachEpoch), true},
            // Optimizer 
            { nameof(NetworkSample.OptimizerType), new[] { "AdamW" } },
            { nameof(NetworkSample.lambdaL2Regularization), new[] { 0.0005} },
            { nameof(NetworkSample.AdamW_L2Regularization), new[] {0.00015,  0.00025,  0.0005} },
            
            { nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), -1 },
            //{ nameof(EfficientNetNetworkSample.DefaultMobileBlocksDescriptionCount), 5 },

            // Learning Rate
            { nameof(NetworkSample.InitialLearningRate), 0.05 },

            // Learning Rate Scheduler
            //{ nameof(NetworkSample.LearningRateSchedulerType), "CyclicCosineAnnealing" },
            {nameof(NetworkSample.LearningRateSchedulerType), new[]{"OneCycle"} },
            {nameof(EfficientNetNetworkSample.LastActivationLayer), nameof(cudnnActivationMode_t.CUDNN_ACTIVATION_SIGMOID)},
            {nameof(NetworkSample.DisableReduceLROnPlateau), true},
            {nameof(NetworkSample.OneCycle_DividerForMinLearningRate), 20},
            {nameof(NetworkSample.OneCycle_PercentInAnnealing), new[]{ 0.1} },
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochsInFirstRun), 10},
            {nameof(NetworkSample.CyclicCosineAnnealing_nbEpochInNextRunMultiplier), 2},
            {nameof(NetworkSample.CyclicCosineAnnealing_MinLearningRate), 1e-5},


            // DataAugmentation
            { nameof(NetworkSample.DataAugmentationType), nameof(ImageDataGenerator.DataAugmentationEnum.DEFAULT) },
            { nameof(NetworkSample.AlphaCutMix), 0 },
            { nameof(NetworkSample.AlphaMixup), new[] { 1.0, 1.2} },
            { nameof(NetworkSample.CutoutPatchPercentage), new[] {0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutPatchPercentage), new[] {0, 0.1, 0.2} },
            { nameof(NetworkSample.RowsCutoutCount), new[] {1, 3, 5} },
            { nameof(NetworkSample.ColumnsCutoutPatchPercentage),  0 },
            { nameof(NetworkSample.HorizontalFlip),new[]{true,false } },

            { nameof(NetworkSample.FillMode),new[]{ nameof(ImageDataGenerator.FillModeEnum.Reflect) /*, nameof(ImageDataGenerator.FillModeEnum.Modulo)*/ } }, //Reflect
            { nameof(NetworkSample.HeightShiftRangeInPercentage), new[]{0, 0.05, 0.10} },
            { nameof(NetworkSample.WidthShiftRangeInPercentage), new[]{0}},

        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(DefaultEfficientNetNetworkSample(), new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }


    

    /// <summary>
    /// The default EfficientNet Hyper-Parameters for CIFAR10
    /// </summary>
    /// <returns></returns>
    private static EfficientNetNetworkSample DefaultEfficientNetNetworkSample()
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



    private static void LaunchCatBoostHPO(int iterations = 10, int maxAllowedSecondsForAllComputation = 0)
    {
        // ReSharper disable once ConvertToConstant.Local
        var searchSpace = new Dictionary<string, object>
          {

              //related to Dataset 
              //{"KFold", 2},
              { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled
              { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
              { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.LIBROSA_FEATURES)},
              //{ nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.92},

              //related to model
              {"loss_function", nameof(CatBoostSample.loss_function_enum.Logloss)},
              {"eval_metric", nameof(CatBoostSample.metric_enum.Accuracy)},
              { "thread_count",1},
              { "task_type","GPU"},
              { "logging_level", nameof(CatBoostSample.logging_level_enum.Verbose)},
              { "allow_writing_files",false},
              { "iterations", iterations },
              //{ "od_type", "Iter"},
              //{ "od_wait",iterations/10},
              { "depth", new[]{5,6,7,8,9,10} },
              { "learning_rate", new[]{0.01,0.02, 0.03}},
              //{ "random_strength",AbstractHyperParameterSearchSpace.Range(1e-9f, 10f, AbstractHyperParameterSearchSpace.range_type.normal)},
              //{ "bagging_temperature",AbstractHyperParameterSearchSpace.Range(0.0f, 2.0f)},
              { "l2_leaf_reg",new[]{0,1,5,10,20}},
              //{"grow_policy", new []{ "SymmetricTree", "Depthwise" /*, "Lossguide"*/}},
          };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new CatBoostSample{loss_function = CatBoostSample.loss_function_enum.Logloss , eval_metric = CatBoostSample.metric_enum.Accuracy }, new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

    // ReSharper disable once UnusedMember.Global
    private static void LaunchLightGBMHPO(int num_iterations = 100, int maxAllowedSecondsForAllComputation = 0)
    {
        var searchSpace = new Dictionary<string, object>
        {
            //related to Dataset 
            //{"KFold", 2},
            { nameof(AbstractDatasetSample.PercentageInTraining), 0.5}, //will be automatically set to 1 if KFold is enabled
            { nameof(AbstractDatasetSample.ShuffleDatasetBeforeSplit), true},
            { nameof(Biosonar85DatasetSample.InputDataType), nameof(Biosonar85DatasetSample.InputDataTypeEnum.LIBROSA_FEATURES)},
            //{ nameof(NetworkSample.MinimumRankingScoreToSaveModel), 0.92},
          
            //uncomment appropriate one
            ////for regression:
            //{"objective", "regression"},      

            ////for binary classification:
            {"objective", "binary"},
            {"metric", "accuracy"},

            ////for multi class classification:
            //{"objective", "multiclass"},      
            //{"num_class", number_of_class },

            //high priority
            
            //!D { "bagging_fraction", new[]{0.8f, 0.9f, 1.0f} },
            { "bagging_fraction", 0.8 },
            //!D
            { "bagging_freq", 5 },

            //{ "boosting", new []{"gbdt", "dart"}},
            { "boosting", new []{"dart"}},
            
            //!D { "colsample_bytree",AbstractHyperParameterSearchSpace.Range(0.3f, 1.0f)},
            { "colsample_bytree",0.8},

            //{ "early_stopping_round", num_iterations/10 },


            //against over fitting
            { "extra_trees", new[] { true , false } },
            { "min_sum_hessian_in_leaf", new[]{0,0.1,1.0} },
            { "path_smooth", new[]{0,0.1,1.0} },
            { "min_gain_to_split", new[]{0,0.1,1.0} },
            { "max_bin", new[]{10,50, 255} },
            { "lambda_l1",new[]{0,1,5,10,50}},
            { "lambda_l2",new[]{0,1,5,10,50}},
            { "min_data_in_leaf", new[]{10,100,500} },
            
            
            //!D { "learning_rate",AbstractHyperParameterSearchSpace.Range(0.005f, 0.1f)},
            { "learning_rate", new[]{0.001, 0.005, 0.01, 0.05, 0.1}},
            
            //!D{ "max_depth", new[]{10, 20, 50, 100, 255} },
            { "max_depth", new[]{10, 50} },

            { "num_iterations", num_iterations },
            { "num_leaves", new[]{3, 10, 50} },
            { "num_threads", -1},
            { "verbosity", "0" },

            ////medium priority
            //{ "drop_rate", new[]{0.05, 0.1, 0.2}},                               //specific to dart mode
            //{ "lambda_l2",AbstractHyperParameterSearchSpace.Range(0f, 2f)},
            //{ "min_data_in_bin", new[]{3, 10, 100, 150}  },
            //{ "max_bin", AbstractHyperParameterSearchSpace.Range(10, 255) },
            //{ "max_drop", new[]{40, 50, 60}},                                   //specific to dart mode
            //{ "skip_drop",AbstractHyperParameterSearchSpace.Range(0.1f, 0.6f)},  //specific to dart mode

            ////low priority
            ////{ "colsample_bynode",AbstractHyperParameterSearchSpace.Range(0.5f, 1.0f)}, //very low priority
            //{ "path_smooth", AbstractHyperParameterSearchSpace.Range(0f, 1f) }, //low priority
        };

        var hpo = new BayesianSearchHPO(searchSpace, () => ModelAndDatasetPredictionsSample.New(new LightGBMSample{objective =  LightGBMSample.objective_enum.binary}, new Biosonar85DatasetSample()), WorkingDirectory);
        IScore bestScoreSoFar = null;
        const bool retrainOnFullDatasetIfBetterModelFound = false;
        hpo.Process(t => SampleUtils.TrainWithHyperParameters((ModelAndDatasetPredictionsSample)t, WorkingDirectory, retrainOnFullDatasetIfBetterModelFound, ref bestScoreSoFar), maxAllowedSecondsForAllComputation);
    }

}